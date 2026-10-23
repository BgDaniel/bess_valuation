r"""Forward simulation, intrinsic value and perfect-foresight bound.

* **Forward simulation** on the out-of-sample paths: starting at :math:`s_{init}`, each
  day chooses :math:`j^* = \arg\max_j F(s_i, s_j, p^n_t) + \hat C_t(s_j, X^n_t)` with the
  continuation values fitted on the regression paths (concave projection as in the
  backward pass), and books the realised DA cashflow :math:`F(s_i, s_{j^*}, p^n_t)` [€].
* **Intrinsic value**: the same grid Bellman recursion, deterministic, on the mean price
  path :math:`\bar p_t` (mean over paths); since the value of a fixed schedule is linear
  in the prices, it is also the expected value of that fixed schedule.
* **Perfect foresight**: the deterministic grid recursion on each path's own prices; an
  upper bound for every policy that ends each day on the SoC grid.

Cycles are full equivalent cycles per day: discharged energy on the battery side divided
by the usable capacity, :math:`\sum_t (d_t/\eta_d) / (e_{max} - e_{min})`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from bess_valuation.config import BatteryConfig
from bess_valuation.daily import daily_schedules
from bess_valuation.library import VertexLibrary
from bess_valuation.lsmc import RegressionModel, concave_projection, continuation

logger = logging.getLogger(__name__)


@dataclass
class ForwardResult:
    """Realised dispatch on the out-of-sample paths.

    Attributes:
        soc_index: Grid index of the SoC at the start of each day and after the last,
            shape (N, D+1).
        cash: Daily DA cashflow [€], shape (N, D).
        terminal: Terminal value of the final SoC [€], shape (N,).
        charge: Charge energy [MWh], shape (N, D, T), float32.
        discharge: Discharge energy [MWh], shape (N, D, T), float32.
        soc: SoC at the end of each period [MWh], shape (N, D, T), float32.
        cycles: Full equivalent cycles per day, shape (N, D).
    """

    soc_index: np.ndarray
    cash: np.ndarray
    terminal: np.ndarray
    charge: np.ndarray | None = None
    discharge: np.ndarray | None = None
    soc: np.ndarray | None = None
    cycles: np.ndarray | None = None

    @property
    def value(self) -> np.ndarray:
        """Total realised value per path [€], shape (N,)."""
        return self.cash.sum(axis=1) + self.terminal


def forward_simulation(model: RegressionModel, F: np.ndarray, feats: np.ndarray,
                       F_mu: np.ndarray | None, i0: int, project: bool = True,
                       progress: Callable[[float], None] | None = None) -> ForwardResult:
    """Apply the fitted policy on (out-of-sample) paths.

    Args:
        model: Continuation values from the backward induction.
        F: Daily values F(s_i, s_j, p^n_t) [€], shape (N, D, G, G).
        feats: Regression features, shape (N, D, m).
        F_mu: F(s_j, s_k, mu(X_t)) for the control variate, shape (N, D, G, G), or None.
        i0: Grid index of the initial SoC.
        project: Apply the concave projection to the continuation values.
        progress: Optional callback with the finished fraction (0..1) after each day.

    Returns:
        SoC indices and daily cashflows (schedules are attached by :func:`attach_schedules`).
    """
    n_paths, n_days, g, _ = F.shape
    idx = np.empty((n_paths, n_days + 1), dtype=np.int64)
    idx[:, 0] = i0
    cash = np.empty((n_paths, n_days))
    rows = np.arange(n_paths)
    for t in tqdm(range(n_days), desc="forward simulation", unit="day", leave=False):
        C = continuation(model, t, feats[:, t], None if F_mu is None else F_mu[:, t])
        if project:
            C = concave_projection(C)
        cand = F[rows, t, idx[:, t], :] + C
        j = np.argmax(cand, axis=1)
        cash[:, t] = F[rows, t, idx[:, t], j]
        idx[:, t + 1] = j
        if progress is not None:
            progress((t + 1) / n_days)
    terminal = model.terminal[idx[:, -1]]
    return ForwardResult(soc_index=idx, cash=cash, terminal=terminal)


def grid_bellman(F: np.ndarray, i0: int, terminal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Bellman recursion over the SoC grid for each path.

    Args:
        F: Daily values [€], shape (N, D, G, G).
        i0: Grid index of the initial SoC.
        terminal: Terminal value per grid point [€], shape (G,).

    Returns:
        (value per path [€] (N,), daily cash of the optimal path (N, D), SoC indices (N, D+1)).
    """
    n_paths, n_days, g, _ = F.shape
    W = np.broadcast_to(terminal, (n_paths, g)).copy()
    choice = np.empty((n_paths, n_days, g), dtype=np.int64)
    for t in range(n_days - 1, -1, -1):
        total = F[:, t] + W[:, None, :]
        choice[:, t] = np.argmax(total, axis=2)
        W = np.max(total, axis=2)
    rows = np.arange(n_paths)
    idx = np.empty((n_paths, n_days + 1), dtype=np.int64)
    idx[:, 0] = i0
    cash = np.empty((n_paths, n_days))
    for t in range(n_days):
        j = choice[rows, t, idx[:, t]]
        cash[:, t] = F[rows, t, idx[:, t], j]
        idx[:, t + 1] = j
    return W[:, i0], cash, idx


def attach_schedules(result: ForwardResult, library: VertexLibrary, prices: np.ndarray,
                     grid: np.ndarray, config: BatteryConfig, seed: int = 0,
                     progress: Callable[[float], None] | None = None) -> None:
    """Fill charge, discharge, SoC per period and cycles of a forward result in place.

    Args:
        result: Forward result with SoC indices.
        library: Vertex library (the schedules are looked up again per realised pair).
        prices: DA prices [€/MWh], shape (N, D, T).
        grid: SoC grid [MWh].
        config: Battery parameters.
        seed: Library processing order.
        progress: Optional callback with the finished fraction (0..1).
    """
    n_paths, n_days, n_t = prices.shape
    s_start = grid[result.soc_index[:, :-1]].ravel()
    s_end = grid[result.soc_index[:, 1:]].ravel()
    revenue, c, d = daily_schedules(library, prices.reshape(-1, n_t), s_start, s_end, seed=seed,
                                    progress=progress)
    gap = np.max(np.abs(revenue - result.cash.ravel()), initial=0.0)
    if gap > 1e-6 * max(1.0, float(np.abs(result.cash).max(initial=0.0))):
        raise RuntimeError(f"schedule revenue differs from the daily value by {gap:.3g} EUR")
    delta = config.eta_charge * c - d / config.eta_discharge
    soc = s_start[:, None] + np.cumsum(delta, axis=1)
    result.charge = c.reshape(n_paths, n_days, n_t).astype(np.float32)
    result.discharge = d.reshape(n_paths, n_days, n_t).astype(np.float32)
    result.soc = soc.reshape(n_paths, n_days, n_t).astype(np.float32)
    usable = config.e_max - config.e_min
    result.cycles = (d / config.eta_discharge).sum(axis=1).reshape(n_paths, n_days) / usable
