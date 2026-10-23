r"""Convexity check of the daily revenue along the main price factors.

For a fixed SoC pair the daily revenue :math:`F(p) = \max_k y_k^{\mathsf T} p` (maximum of
linear functions over the vertices, also without simultaneous charge/discharge) is convex
in the price vector p. This module makes that visible for one day:

1. principal components of that day's out-of-sample price curves: mean curve
   :math:`\bar p`, unit directions :math:`v_1, v_2, v_3` and standard deviations
   :math:`\sigma_k` of the scores :math:`v_k^{\mathsf T}(p - \bar p)`;
2. :math:`F(\bar p + t\,\sigma_k v_k)` for :math:`t \in [-3, 3]` and a few SoC pairs, with
   the daily solver of the run (exact);
3. second differences :math:`F(t-h) - 2F(t) + F(t+h) \ge 0` up to a tolerance.

The three directions typically resemble the level, the peak shape and the solar dip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bess_valuation.config import BatteryConfig
from bess_valuation.daily import daily_value_table


@dataclass
class ConvexityResult:
    """Revenue along the principal price directions of one day.

    Attributes:
        day: Day index.
        date: Delivery date.
        mean_price: Mean price curve [€/MWh], shape (T,).
        directions: Unit principal directions, shape (K, T).
        sigma: Standard deviation of the scores [€/MWh], shape (K,).
        explained: Share of the price variance explained by each direction, shape (K,).
        t: Multiples of sigma, shape (M,).
        pairs: SoC pairs as (start, end) in MWh, shape (P, 2).
        values: Revenue [€] F(mean + t sigma_k v_k), shape (K, M, P).
        scores: Scores of the out-of-sample curves in units of sigma, shape (K, N).
        min_second_difference: Smallest second difference per direction and pair [€],
            shape (K, P); >= -tolerance means convex.
        tolerance: Tolerance of the check [€].
    """

    day: int
    date: np.datetime64
    mean_price: np.ndarray
    directions: np.ndarray
    sigma: np.ndarray
    explained: np.ndarray
    t: np.ndarray
    pairs: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    min_second_difference: np.ndarray
    tolerance: float

    @property
    def convex(self) -> bool:
        """True if every curve passes the second-difference check."""
        return bool(np.all(self.min_second_difference >= -self.tolerance))


def convexity_analysis(prices_day: np.ndarray, config: BatteryConfig, grid: np.ndarray, solver: str,
                       day: int, date: np.datetime64, cycle_cost: float = 0.0, n_factors: int = 3,
                       n_points: int = 41, span: float = 3.0) -> ConvexityResult:
    """Revenue of one day along its first principal price directions.

    Args:
        prices_day: Price curves of the day across paths [€/MWh], shape (N, T).
        config: Battery parameters of that day (capacity already applied).
        grid: SoC grid of that day [MWh], shape (G,).
        solver: Daily solver (see :func:`bess_valuation.daily.daily_value_table`); the
            vertex library is not needed ("library" uses a fresh one).
        day: Day index (for reporting).
        date: Delivery date (for reporting).
        cycle_cost: Cost per discharged MWh [€/MWh].
        n_factors: Number of principal directions.
        n_points: Points per direction.
        span: Range of t in units of sigma: [-span, span].

    Returns:
        The analysis.
    """
    from bess_valuation.library import VertexLibrary

    P = np.asarray(prices_day, dtype=np.float64)
    mean = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - mean, full_matrices=False)
    k = min(n_factors, vt.shape[0])
    directions = vt[:k].copy()
    for i in range(k):  # orientation: largest component positive
        if directions[i, np.argmax(np.abs(directions[i]))] < 0:
            directions[i] *= -1.0
    sigma = s[:k] / np.sqrt(max(len(P) - 1, 1))
    explained = s[:k] ** 2 / max(np.sum(s ** 2), 1e-300)
    t = np.linspace(-span, span, n_points)
    curves = mean[None, None, :] + (t[None, :, None] * sigma[:, None, None]) * directions[:, None, :]
    library = VertexLibrary.empty(config) if solver == "library" else None
    F = daily_value_table(solver, config, curves.reshape(-1, P.shape[1]), grid, library=library,
                          cycle_cost=cycle_cost).reshape(k, n_points, grid.size, grid.size)
    g = grid.size
    idx = [(g // 2, g // 2), (0, g - 1), (g - 1, 0)]
    pairs = np.array([(grid[i], grid[j]) for i, j in idx])
    values = np.stack([F[:, :, i, j] for i, j in idx], axis=-1)
    second = values[:, :-2] - 2.0 * values[:, 1:-1] + values[:, 2:]
    scores = ((P - mean) @ directions.T / np.where(sigma > 0, sigma, 1.0)).T
    tolerance = 1e-6 * max(1.0, float(np.abs(values).max()))
    return ConvexityResult(day=day, date=date, mean_price=mean, directions=directions, sigma=sigma,
                           explained=explained, t=t, pairs=pairs, values=values, scores=scores,
                           min_second_difference=second.min(axis=1), tolerance=tolerance)
