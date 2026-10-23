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
