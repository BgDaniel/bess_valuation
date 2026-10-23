r"""Backward Bellman induction (LSMC): regression, control variate, concavity projection.

Notation: days :math:`t = 0..D-1`, SoC grid :math:`s_0..s_{G-1}` [MWh], paths
:math:`n`, state :math:`X_t` (price-model factors or price features of day t),
daily value :math:`F(s_i, s_j, p_t)` [€] (:mod:`bess_valuation.daily`).

For :math:`t = D-1, \dots, 0`:

1. **Continuation value.** :math:`\hat C_t(s_j, X) \approx E[V_{t+1}(s_j) \mid X_t = X]`
   by least squares of the realised :math:`V_{t+1}(s_j, n)` on basis functions
   :math:`\varphi(X_t)` (all monomials of the standardised features up to ``degree``),
   one regression per grid point :math:`s_j`. On the last day
   :math:`\hat C_{D-1}(s) = v_{term}\,s` exactly.
2. **Control variate** (if the scenarios carry :math:`\mu(X_t) = E[p_{t+1} \mid X_t]`):
   the intrinsic part

   .. math::

       h_t(s_j, X_t) = \max_k \big[F(s_j, s_k, \mu(X_t)) + \bar C_{t+1}(s_k)\big],
       \qquad \bar C_{t+1}(s_k) = \tfrac1N \textstyle\sum_n V_{t+2}(s_k, n),

   is computed exactly (vertex library) and only the residual
   :math:`V_{t+1} - h_t` is regressed; :math:`\hat C_t = h_t + \beta_t(s_j)^{\mathsf T}\varphi`.
3. **Concavity.** Per path the vector :math:`(\hat C_t(s_j, X_t^n))_j` is projected onto
   concave sequences: decreasing isotonic regression (pool adjacent violators) of its
   slopes, then the level is restored by matching the mean (PAV-type projection).
4. **Decision and propagation** (Longstaff-Schwartz): for every start point
   :math:`s_i`, :math:`j^* = \arg\max_j F(s_i, s_j, p^n_t) + \hat C_t(s_j, X^n_t)` and
   :math:`V_t(s_i, n) = F(s_i, s_{j^*}, p^n_t) + V_{t+1}(s_{j^*}, n)` (realised, not
   regressed, values).

The in-sample value is the mean of :math:`V_0(s_{init}, n)`; it is biased upwards
(the policy is fitted to the same paths), so the forward simulation on independent
paths (:mod:`bess_valuation.forward`) gives the out-of-sample value.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSettings:
    """Numerical settings of the valuation (not part of the battery config).

    Attributes:
        n_regression_paths: Paths used for the regression (the rest are out-of-sample);
            default half of the paths.
        degree: Total degree of the polynomial basis.
        features: "state" (``prices.X``), "prices" (daily mean and max - min of the DA
            prices) or "auto" (state if available).
        control_variate: Use the intrinsic control variate if ``conditional_mean`` exists.
            Off by default: on the synthetic scenarios it reduced the regression variance
            by a factor of only ~1.004 but cost about a quarter of the runtime.
        concave_projection: Project the continuation value onto concave functions of SoC.
        library_dir: Folder of the persisted vertex libraries (None: do not persist).
        value_cache_dir: Folder of cached daily-value tables F (None: no cache); a run
            on the same battery physics, SoC grid and price scenarios loads them instead
            of solving the daily problems again.
        max_vertices: Cap of cached vertices per SoC pair.
        seed: Seed of the library processing order.
        plot_days: Number of days whose continuation curves are kept for plot 9.
    """

    n_regression_paths: int | None = None
    degree: int = 2
    features: str = "auto"
    control_variate: bool = False
    concave_projection: bool = True
    library_dir: str | None = "library_cache"
    value_cache_dir: str | None = "value_cache"
    max_vertices: int = 1000
    seed: int = 0
    plot_days: int = 4


@dataclass
class RegressionModel:
    """Fitted continuation values, applicable to new paths.

    Attributes:
        exponents: Monomial exponents of the basis, shape (B, m).
        mean: Feature means per day, shape (D, m).
        std: Feature standard deviations per day, shape (D, m).
        beta: Coefficients per day and SoC grid point, shape (D, G, B).
        cbar: Mean realised value after day t+1, used by the control variate, shape (D, G).
        terminal: Terminal value per grid point [€], shape (G,).
        control_variate: Whether h_t is added.
    """

    exponents: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    beta: np.ndarray
    cbar: np.ndarray
    terminal: np.ndarray
    control_variate: bool


@dataclass
class BackwardResult:
    """Output of :func:`backward_induction`.

    Attributes:
        model: Fitted continuation values.
        value: Realised values V_0(s, n) on the regression paths [€], shape (N, G).
        r2: R^2 of the regression per day and grid point, shape (D, G) (NaN on the last day).
        var_ratio: Var(V_{t+1}) / Var(V_{t+1} - h_t) per day and grid point (variance
            reduction by the control variate; 1 without it), shape (D, G).
        violation_share: Share of paths whose continuation curve was changed by the
            concavity projection, per day, shape (D,).
        curve_days: Days kept for plot 9, shape (K,).
        curve_raw: Mean over paths of C_t(s) before projection, shape (K, G).
        curve_projected: Same after projection, shape (K, G).
        curve_samples_raw: C_t(s) of up to 5 sample paths before projection, shape (K, S, G).
        curve_samples_projected: Same after projection, shape (K, S, G).
    """

    model: RegressionModel
    value: np.ndarray
    r2: np.ndarray
    var_ratio: np.ndarray
    violation_share: np.ndarray
    curve_days: np.ndarray
    curve_raw: np.ndarray
    curve_projected: np.ndarray
    curve_samples_raw: np.ndarray
    curve_samples_projected: np.ndarray
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ building blocks


def monomial_exponents(n_features: int, degree: int) -> np.ndarray:
    """Exponents of all monomials of total degree <= ``degree``, shape (B, m)."""
    rows = [e for e in itertools.product(range(degree + 1), repeat=n_features) if sum(e) <= degree]
    rows.sort(key=lambda e: (sum(e), [-v for v in e]))
    return np.array(rows, dtype=np.int64).reshape(-1, n_features)


def basis_matrix(z: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Basis functions phi(z) for standardised features z of shape (N, m) -> (N, B)."""
    return np.prod(z[:, None, :] ** exponents[None, :, :], axis=2)


def price_features(prices: np.ndarray) -> np.ndarray:
    """Daily mean and spread (max - min) of the prices, shape (..., 2)."""
    return np.stack([prices.mean(axis=-1), prices.max(axis=-1) - prices.min(axis=-1)], axis=-1)


@njit(cache=True)
def _concave_row(y, out):
    g = y.shape[0]
    if g < 3:
        out[:] = y
        return
    n = g - 1
    level = np.empty(n)
    weight = np.empty(n)
    size = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):  # PAV: slopes must be non-increasing
        level[k] = y[i + 1] - y[i]
        weight[k] = 1.0
        size[k] = 1
        k += 1
        while k > 1 and level[k - 2] < level[k - 1]:
            w = weight[k - 2] + weight[k - 1]
            level[k - 2] = (weight[k - 2] * level[k - 2] + weight[k - 1] * level[k - 1]) / w
            weight[k - 2] = w
            size[k - 2] += size[k - 1]
            k -= 1
    out[0] = 0.0
    i = 0
    for b in range(k):
        for _ in range(size[b]):
            out[i + 1] = out[i] + level[b]
            i += 1
    shift = 0.0
    for i in range(g):
        shift += y[i] - out[i]
    shift /= g
    for i in range(g):
        out[i] += shift


@njit(parallel=True, cache=True)
def concave_projection(values):
    """PAV-type concave projection of each row of ``values`` (N, G) over the SoC grid.

    The slopes are replaced by their decreasing isotonic regression (pool adjacent
    violators) and the level is chosen to keep the row mean.
    """
    out = np.empty_like(values)
    for n in prange(values.shape[0]):
        _concave_row(values[n], out[n])
    return out


def intrinsic_offset(F_mu: np.ndarray, cbar: np.ndarray) -> np.ndarray:
    """h(s_j) = max_k F(s_j, s_k, mu) + cbar(s_k) for F_mu of shape (N, G, G) -> (N, G)."""
    return np.max(F_mu + cbar[None, None, :], axis=2)


def _features(prices_da: np.ndarray, X: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "auto":
        mode = "state" if X is not None else "prices"
    if mode == "state":
        if X is None:
            raise ValueError("features='state' needs prices.X")
        return np.asarray(X, dtype=np.float64)
    if mode == "prices":
        return price_features(np.asarray(prices_da, dtype=np.float64))
    raise ValueError(f"unknown features {mode!r}")


def continuation(model: RegressionModel, t: int, feats: np.ndarray,
                 F_mu: np.ndarray | None) -> np.ndarray:
    """Raw continuation values C_t(s_j, X) for features of shape (N, m) -> (N, G)."""
    if t == model.beta.shape[0] - 1:
        return np.broadcast_to(model.terminal, (feats.shape[0], model.terminal.size)).copy()
    z = (feats - model.mean[t]) / model.std[t]
    C = basis_matrix(z, model.exponents) @ model.beta[t].T
    if model.control_variate:
        C += intrinsic_offset(F_mu, model.cbar[t])
    return C


# ------------------------------------------------------------------ backward induction


def backward_induction(F: np.ndarray, feats: np.ndarray, grid: np.ndarray,
                       terminal_soc_value: float, settings: EngineSettings,
                       F_mu: np.ndarray | None = None,
                       progress: Callable[[float], None] | None = None) -> BackwardResult:
    """LSMC backward induction on the regression paths.

    Args:
        F: Daily values F(s_i, s_j, p^n_t) [€], shape (N, D, G, G).
        feats: Regression features per path and day, shape (N, D, m).
        grid: SoC grid [MWh], shape (G,).
        terminal_soc_value: Value of the SoC left after the last day [€/MWh].
        settings: Degree, projection and control-variate switches.
        F_mu: F(s_j, s_k, mu(X_t)) for the control variate, shape (N, D, G, G), or None.
        progress: Optional callback with the finished fraction (0..1) after each day.

    Returns:
        Fitted model, realised values and diagnostics.
    """
    n_paths, n_days, g, _ = F.shape
    use_cv = settings.control_variate and F_mu is not None
    exponents = monomial_exponents(feats.shape[-1], settings.degree)
    terminal = terminal_soc_value * grid
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-12] = 1.0
    beta = np.zeros((n_days, g, exponents.shape[0]))
    cbar = np.zeros((n_days, g))
    r2 = np.full((n_days, g), np.nan)
    var_ratio = np.ones((n_days, g))
    violation = np.zeros(n_days)
    # the last day's continuation value is the terminal value, so it is not shown
    plot_days = np.unique(np.linspace(0, max(n_days - 2, 0), settings.plot_days).round().astype(int))
    n_s = min(5, n_paths)
    curve_raw = np.zeros((plot_days.size, g))
    curve_proj = np.zeros((plot_days.size, g))
    samp_raw = np.zeros((plot_days.size, n_s, g))
    samp_proj = np.zeros((plot_days.size, n_s, g))

    v_next = np.broadcast_to(terminal, (n_paths, g)).copy()   # V_{t+1}
    v_after = v_next.copy()                                    # V_{t+2}
    model = RegressionModel(exponents=exponents, mean=mean, std=std, beta=beta, cbar=cbar,
                            terminal=terminal, control_variate=use_cv)
    rows = np.arange(n_paths)
    logger.info("backward induction: %d paths, %d days, %d SoC grid points, %d basis functions%s",
                n_paths, n_days, g, exponents.shape[0], " + control variate" if use_cv else "")
    with tqdm(total=n_days * g, desc="backward induction (day x SoC)", unit="step",
              leave=False) as bar:
        for t in range(n_days - 1, -1, -1):
            if t < n_days - 1:
                z = (feats[:, t] - mean[t]) / std[t]
                phi = basis_matrix(z, exponents)
                target = v_next
                if use_cv:
                    cbar[t] = v_after.mean(axis=0)
                    h = intrinsic_offset(F_mu[:, t], cbar[t])
                    resid = target - h
                else:
                    h = 0.0
                    resid = target
                beta[t] = np.linalg.lstsq(phi, resid, rcond=None)[0].T
                C = phi @ beta[t].T + h
                var_t = target.var(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r2[t] = np.where(var_t > 0, 1.0 - (target - C).var(axis=0) / var_t, np.nan)
                    if use_cv:
                        var_ratio[t] = np.where(resid.var(axis=0) > 0, var_t / resid.var(axis=0), 1.0)
            else:
                C = np.broadcast_to(terminal, (n_paths, g)).copy()
            C_proj = concave_projection(C) if settings.concave_projection else C
            violation[t] = float(np.mean(np.any(np.abs(C_proj - C) > 1e-9 * (1 + np.abs(C)), axis=1)))
            k = np.searchsorted(plot_days, t)
            if k < plot_days.size and plot_days[k] == t:
                curve_raw[k], curve_proj[k] = C.mean(axis=0), C_proj.mean(axis=0)
                samp_raw[k], samp_proj[k] = C[:n_s], C_proj[:n_s]
            v_t = np.empty((n_paths, g))
            for i in range(g):
                j = np.argmax(F[:, t, i, :] + C_proj, axis=1)
                v_t[:, i] = F[rows, t, i, j] + v_next[rows, j]
                bar.update()
            v_after, v_next = v_next, v_t
            if progress is not None:
                progress((n_days - t) / n_days)
    return BackwardResult(model=model, value=v_next, r2=r2, var_ratio=var_ratio,
                          violation_share=violation, curve_days=plot_days, curve_raw=curve_raw,
                          curve_projected=curve_proj, curve_samples_raw=samp_raw,
                          curve_samples_projected=samp_proj)
