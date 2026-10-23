r"""Daily lookup over the SoC grid (DA only so far; IDA1-3 rolling intrinsic in milestone 4).

For a day with price vector :math:`p` [€/MWh] and SoC grid points :math:`s_i` [MWh] the
daily value is

.. math::

    F(s_i, s_j, p) = \max_{x \in P(s_i, s_j)} p^{\mathsf T}(d - c) - \lambda\,\mathbf 1^{\mathsf T} d,

the optimal revenue [€] of the day that starts at SoC :math:`s_i` and ends at :math:`s_j`
(:math:`\lambda` = cycle cost). :func:`daily_values` evaluates F for all grid pairs and
price vectors through the vertex library (certificate, simplex fallback), so the
backward induction itself never solves an LP.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from bess_valuation.config import BatteryConfig
from bess_valuation.library import VertexLibrary

logger = logging.getLogger(__name__)


def soc_grid(config: BatteryConfig) -> np.ndarray:
    """SoC grid points [MWh]: ``soc_grid_size`` equidistant points on [e_min, e_max]."""
    return np.linspace(config.e_min, config.e_max, config.soc_grid_size)


def daily_values(library: VertexLibrary, prices: np.ndarray, grid: np.ndarray,
                 cycle_cost: float = 0.0, desc: str = "daily values", seed: int = 0,
                 progress: Callable[[float], None] | None = None) -> np.ndarray:
    """F(s_i, s_j, p) for every price vector and SoC grid pair.

    Args:
        library: Vertex library of the battery (full-day horizon).
        prices: Prices [€/MWh], shape (..., T), e.g. (N, D, T).
        grid: SoC grid [MWh], shape (G,).
        cycle_cost: Cycle cost lambda [€/MWh discharged].
        desc: Progress-bar label.
        seed: Seed of the library processing order.
        progress: Optional callback with the finished fraction (0..1) after each SoC pair.

    Returns:
        Revenues [€], shape (..., G, G) with axes (s_start, s_end); -inf for SoC pairs
        that cannot be connected within a day.
    """
    lead = prices.shape[:-1]
    flat = np.ascontiguousarray(prices.reshape(-1, prices.shape[-1]), dtype=np.float64)
    if np.isnan(flat).any():
        raise ValueError("prices contain NaN")
    g = grid.size
    out = np.empty((flat.shape[0], g, g))
    logger.info("%s: %d price vectors x %d SoC pairs", desc, flat.shape[0], g * g)
    with tqdm(total=g * g, desc=desc, unit="pair", leave=False) as bar:
        for i in range(g):
            for j in range(g):
                _, out[:, i, j] = library.lookup(flat, grid[i], grid[j], cycle_cost, seed=seed)
                bar.update()
                if progress is not None:
                    progress((i * g + j + 1) / (g * g))
    return out.reshape(lead + (g, g))


def daily_schedules(library: VertexLibrary, prices: np.ndarray, s_start: np.ndarray,
                    s_end: np.ndarray, cycle_cost: float = 0.0, seed: int = 0,
                    progress: Callable[[float], None] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal schedules for individual (price vector, s_start, s_end) triples.

    Args:
        library: Vertex library of the battery.
        prices: Prices [€/MWh], shape (Q, T).
        s_start: Start SoC per row [MWh], shape (Q,), values on the grid.
        s_end: End SoC per row [MWh], shape (Q,), values on the grid.
        cycle_cost: Cycle cost lambda [€/MWh discharged].
        seed: Seed of the library processing order.
        progress: Optional callback with the finished fraction (0..1) after each SoC pair.

    Returns:
        (revenue [€] (Q,), charge [MWh] (Q, T), discharge [MWh] (Q, T)).
    """
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    n_q, n_t = prices.shape
    revenue = np.empty(n_q)
    charge = np.empty((n_q, n_t))
    discharge = np.empty((n_q, n_t))
    pairs = np.stack([np.round(s_start, 9), np.round(s_end, 9)], axis=1)
    keys, inverse = np.unique(pairs, axis=0, return_inverse=True)
    for k, (s0, s1) in enumerate(tqdm(keys, desc="schedules", unit="pair", leave=False)):
        rows = np.flatnonzero(inverse.ravel() == k)
        _, revenue[rows], charge[rows], discharge[rows] = library.lookup(
            prices[rows], s0, s1, cycle_cost, seed=seed, return_schedule=True)
        if progress is not None:
            progress((k + 1) / len(keys))
    return revenue, charge, discharge


# ------------------------------------------------------------------ solver choice

DAILY_SOLVERS = ("dp", "library", "cold", "cluster")


def _simplex_values(config: BatteryConfig, prices: np.ndarray, grid: np.ndarray, cycle_cost: float,
                    warm: Callable[[np.ndarray, object], tuple[np.ndarray, np.ndarray] | None],
                    desc: str, progress: Callable[[float], None] | None) -> np.ndarray:
    """F for all pairs with the own simplex, warm start per pair given by ``warm(prices, lp)``."""
    from bess_valuation.polytope import build_daily_lp
    from bess_valuation.simplex import solve_daily

    g = grid.size
    out = np.empty((prices.shape[0], g, g))
    with tqdm(total=g * g, desc=desc, unit="pair", leave=False) as bar:
        for i in range(g):
            for j in range(g):
                lp = build_daily_lp(config, grid[i], grid[j])
                start = warm(prices, lp)
                basis, status = (None, None) if start is None else start
                sol = solve_daily(lp, prices, cycle_cost,
                                  allow_simultaneous=config.allow_simultaneous_charge_discharge,
                                  basis=basis, var_status=status)
                out[:, i, j] = np.where(sol.status == 0, sol.revenue, -np.inf)
                bar.update()
                if progress is not None:
                    progress((i * g + j + 1) / (g * g))
    return out


def kmeans(Z: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means on the rows of Z; returns (centroids (k, T), labels (N,))."""
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), size=k, replace=False)].copy()
    for _ in range(iters + 1):
        d = (Z ** 2).sum(1)[:, None] - 2.0 * Z @ C.T + (C ** 2).sum(1)[None, :]
        labels = np.argmin(d, axis=1)
        for c in range(k):
            members = Z[labels == c]
            if len(members):
                C[c] = members.mean(axis=0)
    return C, labels


def daily_value_table(solver: str, config: BatteryConfig, prices: np.ndarray, grid: np.ndarray,
                      library: VertexLibrary | None = None, cycle_cost: float = 0.0,
                      desc: str = "daily values", seed: int = 0, n_clusters: int = 256,
                      progress: Callable[[float], None] | None = None) -> np.ndarray:
    """F(s_i, s_j, p) for all price vectors and SoC grid pairs with the chosen solver.

    All solvers give the same values (exact optima); they differ in speed:

    * ``"dp"``: dynamic programming over the SoC (:mod:`bess_valuation.dp`), one backward
      pass per end SoC for all start SoCs; no LP. Requires no simultaneous charge/discharge.
    * ``"library"``: vertex library (certificate, simplex warm-started from the argmax vertex).
    * ``"cold"``: own simplex from the closed-form monotone start.
    * ``"cluster"``: own simplex warm-started from the optimal basis of the k-means centroid
      of the normalised price curve (``n_clusters`` centroids, solved once per pair).

    Args:
        solver: One of :data:`DAILY_SOLVERS`.
        config: Battery parameters.
        prices: Prices [€/MWh], shape (..., T).
        grid: SoC grid [MWh], shape (G,).
        library: Vertex library (``"library"`` only).
        cycle_cost: Cycle cost lambda [€/MWh discharged].
        desc: Progress-bar label.
        seed: Library processing order / k-means initialisation.
        n_clusters: Number of centroids (``"cluster"`` only).
        progress: Optional callback with the finished fraction (0..1).

    Returns:
        Revenues [€], shape (..., G, G).
    """
    if solver not in DAILY_SOLVERS:
        raise ValueError(f"unknown daily solver {solver!r}, choose one of {DAILY_SOLVERS}")
    if solver == "dp":
        from bess_valuation.dp import dp_daily_values

        return dp_daily_values(config, prices, grid, cycle_cost, progress=progress)
    if solver == "library":
        if library is None:
            raise ValueError("solver 'library' needs a vertex library")
        return daily_values(library, prices, grid, cycle_cost, desc=desc, seed=seed, progress=progress)
    lead = prices.shape[:-1]
    flat = np.ascontiguousarray(prices.reshape(-1, prices.shape[-1]), dtype=np.float64)
    if solver == "cold":
        out = _simplex_values(config, flat, grid, cycle_cost, lambda p, lp: None, desc, progress)
    else:
        from bess_valuation.polytope import lp_cost
        from bess_valuation.simplex import solve_batch

        Z = flat - flat.mean(axis=1, keepdims=True)
        Z /= np.maximum(Z.std(axis=1, keepdims=True), 1e-9)
        centroids, labels = kmeans(Z, min(n_clusters, len(Z)), seed=seed)

        def warm(p: np.ndarray, lp: object) -> tuple[np.ndarray, np.ndarray]:
            res = solve_batch(lp, lp_cost(centroids, cycle_cost))  # any vertex basis is a valid start
            return res.basis[labels], res.var_status[labels]

        out = _simplex_values(config, flat, grid, cycle_cost, warm, desc, progress)
    return out.reshape(lead + (grid.size, grid.size))


def schedules_for(solver: str, config: BatteryConfig, library: VertexLibrary, prices: np.ndarray,
                  s_start: np.ndarray, s_end: np.ndarray, cycle_cost: float = 0.0, seed: int = 0,
                  progress: Callable[[float], None] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schedules for (prices, s_start, s_end) triples: DP for ``"dp"``, else the library."""
    if solver == "dp" and not config.allow_simultaneous_charge_discharge:
        from bess_valuation.dp import dp_daily_schedules

        out = dp_daily_schedules(config, prices, s_start, s_end, cycle_cost)
        if progress is not None:
            progress(1.0)
        return out
    return daily_schedules(library, prices, s_start, s_end, cycle_cost, seed=seed, progress=progress)
