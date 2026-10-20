r"""Constraint data of the daily dispatch polytope for full-day and half-day horizons.

The daily problem (SPEC 3.1) for T periods is written in bounded standard form

.. math::

    \min\ g^{\mathsf T}x \quad\text{s.t.}\quad Ax = b,\quad l \le x \le u,

with variables :math:`x = (c_0..c_{T-1},\ d_0..d_{T-1},\ s_1..s_{T-1})`: charge and
discharge energy per period [MWh] and the intermediate states of charge [MWh]. Power
and SoC limits are variable bounds, :math:`0 \le c_t \le \bar c = P_{ch}\,dt`,
:math:`0 \le d_t \le \bar d = P_{dis}\,dt`, :math:`e_{min} \le s_t \le e_{max}`, and the
T rows are the SoC dynamics

.. math::

    s_{t+1} - s_t - \eta_c c_t + d_t/\eta_d = 0, \qquad t = 0..T-1,

with the fixed end points :math:`s_0 = s_{start}` and :math:`s_T = s_{end}` moved to the
right-hand side, :math:`b = (s_{start}, 0, \dots, 0, -s_{end})`. Only :math:`b` depends on
the SoC pair, so A, l and u (and hence every normal cone) are shared across the SoC grid.

The objective of the battery is :math:`\max\ p^{\mathsf T}(d - c) - \lambda\,\mathbf 1^{\mathsf T}d`
(price :math:`p` [€/MWh], optional cycle cost :math:`\lambda` [€/MWh] per discharged MWh),
i.e. the cost vector :math:`g = (p,\ -(p - \lambda),\ 0)` of :func:`lp_cost`.

The half-day horizon (IDA3, 12-24h) is the same problem with T/2 periods and the SoC at
12:00 as :math:`s_{start}`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bess_valuation.config import BatteryConfig

Horizon = Literal["full", "half"]


@dataclass(frozen=True)
class DailyLP:
    """Bounded standard-form LP of one daily (or half-day) dispatch problem.

    Attributes:
        A: Constraint matrix, shape (T, 3T-1).
        b: Right-hand side, shape (T,).
        lo: Lower variable bounds, shape (3T-1,).
        up: Upper variable bounds, shape (3T-1,).
        n_periods: Number of periods T of the horizon.
        horizon: "full" (T periods per day) or "half" (second half of the day).
        s_start: SoC at the start of the horizon [MWh].
        s_end: SoC at the end of the horizon [MWh].
        eta_charge: One-way charging efficiency.
        eta_discharge: One-way discharging efficiency.
    """

    A: np.ndarray
    b: np.ndarray
    lo: np.ndarray
    up: np.ndarray
    n_periods: int
    horizon: Horizon
    s_start: float
    s_end: float
    eta_charge: float
    eta_discharge: float

    @property
    def n_vars(self) -> int:
        """Number of variables 3T - 1."""
        return self.A.shape[1]

    def with_soc(self, s_start: float, s_end: float) -> DailyLP:
        """Same polytope shape with a different SoC pair (only b changes)."""
        return dataclasses.replace(self, b=soc_rhs(self.n_periods, s_start, s_end),
                                   s_start=s_start, s_end=s_end)


def idx_charge(n_periods: int) -> slice:
    """Index range of the charge variables c_t."""
    return slice(0, n_periods)


def idx_discharge(n_periods: int) -> slice:
    """Index range of the discharge variables d_t."""
    return slice(n_periods, 2 * n_periods)


def idx_soc(n_periods: int) -> slice:
    """Index range of the intermediate SoC variables s_1..s_{T-1}."""
    return slice(2 * n_periods, 3 * n_periods - 1)


def soc_rhs(n_periods: int, s_start: float, s_end: float) -> np.ndarray:
    """Right-hand side b = (s_start, 0, ..., 0, -s_end)."""
    b = np.zeros(n_periods)
    b[0] += s_start
    b[-1] -= s_end
    return b


def build_daily_lp(config: BatteryConfig, s_start: float, s_end: float,
                   horizon: Horizon = "full") -> DailyLP:
    """Build the bounded standard-form LP of the daily problem.

    Args:
        config: Battery parameters.
        s_start: SoC at the start of the horizon [MWh], in [e_min, e_max].
        s_end: SoC at the end of the horizon [MWh], in [e_min, e_max].
        horizon: "full" day or "half" day (12-24h, IDA3).

    Returns:
        The LP data.
    """
    tol = 1e-9
    for name, s in (("s_start", s_start), ("s_end", s_end)):
        if not config.e_min - tol <= s <= config.e_max + tol:
            raise ValueError(f"{name} = {s} outside [{config.e_min}, {config.e_max}]")
    n_t = config.periods_per_day if horizon == "full" else config.periods_per_day // 2
    n = 3 * n_t - 1
    A = np.zeros((n_t, n))
    rows = np.arange(n_t)
    A[rows, rows] = -config.eta_charge
    A[rows, n_t + rows] = 1.0 / config.eta_discharge
    for t in range(1, n_t):  # s_t enters row t-1 with +1 and row t with -1
        col = 2 * n_t + t - 1
        A[t - 1, col] = 1.0
        A[t, col] = -1.0
    lo = np.zeros(n)
    up = np.empty(n)
    up[idx_charge(n_t)] = config.p_charge_max * config.dt
    up[idx_discharge(n_t)] = config.p_discharge_max * config.dt
    lo[idx_soc(n_t)] = config.e_min
    up[idx_soc(n_t)] = config.e_max
    return DailyLP(A=A, b=soc_rhs(n_t, s_start, s_end), lo=lo, up=up, n_periods=n_t,
                   horizon=horizon, s_start=float(s_start), s_end=float(s_end),
                   eta_charge=config.eta_charge, eta_discharge=config.eta_discharge)


def lp_cost(prices: np.ndarray, cycle_cost: float = 0.0) -> np.ndarray:
    """Minimisation cost g = (p, -(p - lambda), 0) for prices of shape (..., T).

    Args:
        prices: Prices per period [€/MWh], shape (..., T).
        cycle_cost: Cost per discharged MWh [€/MWh] (Lagrange multiplier of the cycle limit).

    Returns:
        Cost vectors, shape (..., 3T-1).
    """
    prices = np.asarray(prices, dtype=np.float64)
    n_t = prices.shape[-1]
    g = np.zeros(prices.shape[:-1] + (3 * n_t - 1,))
    g[..., :n_t] = prices
    g[..., n_t:2 * n_t] = -(prices - cycle_cost)
    return g


def split_schedule(x: np.ndarray, n_periods: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (charge c, discharge d) [MWh] from LP solutions of shape (..., 3T-1)."""
    return x[..., idx_charge(n_periods)], x[..., idx_discharge(n_periods)]


def soc_path(c: np.ndarray, d: np.ndarray, s_start: float, eta_charge: float,
             eta_discharge: float) -> np.ndarray:
    """SoC s_0..s_T [MWh] from a schedule, shape (..., T+1)."""
    delta = eta_charge * c - d / eta_discharge
    start = np.full(c.shape[:-1] + (1,), float(s_start))
    return np.concatenate([start, s_start + np.cumsum(delta, axis=-1)], axis=-1)
