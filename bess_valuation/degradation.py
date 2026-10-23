r"""Battery degradation: day-dependent usable capacity and wear cost.

**Capacity.** The usable capacity on day d is :math:`\kappa_d\,(e_{max} - e_{min})` [MWh]
with a capacity factor :math:`\kappa_d \in (0, 1]`, either

* given by the user (a function of the dates or one value per day), or
* from ``BatteryConfig.capacity_fade_per_year`` (:math:`\phi`): linear fade in monthly
  steps, :math:`\kappa_d = 1 - \phi\, m_d / 12` with :math:`m_d` the number of whole months
  since the first day of the horizon (constant within a month, so the daily problem
  only changes between months).

Within a day the capacity is fixed, so the daily problem keeps its form; its SoC bounds
are :math:`[e_{min}, e_{min} + \kappa_d (e_{max} - e_{min})]`. The SoC grid of the Bellman
recursion is *relative*: grid point j on day d is
:math:`e_{min} + f_j \kappa_d (e_{max} - e_{min})` with fixed fractions
:math:`f_j = j/(G-1)`. Between two days the fraction is kept; the energy
:math:`f_j (\kappa_d - \kappa_{d+1})(e_{max} - e_{min})` lost by the capacity step
disappears with the degradation.

**Wear cost.** ``BatteryConfig.wear_cost_per_mwh`` (:math:`k` [€ per discharged MWh])
is charged on every discharged MWh in the daily optimisation (the same mechanism as the
cycle cost :math:`\lambda`), so the battery only cycles when the spread pays for the wear.
Reported values are net of the wear cost.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import numpy as np

from bess_valuation.config import BatteryConfig

logger = logging.getLogger(__name__)

CapacityFactor = Callable[[np.ndarray], np.ndarray] | np.ndarray | None


def capacity_factors(config: BatteryConfig, dates: np.ndarray,
                     capacity_factor: CapacityFactor = None) -> np.ndarray:
    """Capacity factor kappa_d per day.

    Args:
        config: Battery parameters (``capacity_fade_per_year`` if no factor is given).
        dates: Delivery dates, shape (D,).
        capacity_factor: Optional user input: a function ``dates -> factors`` or an array
            with one factor per day; overrides ``capacity_fade_per_year``.

    Returns:
        Factors in (0, 1.5], shape (D,).
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    if capacity_factor is None:
        months = (dates.astype("datetime64[M]") - dates[0].astype("datetime64[M]")).astype(int)
        kappa = 1.0 - config.capacity_fade_per_year * months / 12.0
    elif callable(capacity_factor):
        kappa = np.asarray(capacity_factor(dates), dtype=np.float64)
    else:
        kappa = np.asarray(capacity_factor, dtype=np.float64)
    kappa = np.broadcast_to(kappa, dates.shape).astype(np.float64)
    if np.any(~np.isfinite(kappa)) or np.any(kappa <= 0.0) or np.any(kappa > 1.5):
        raise ValueError("capacity factors must be in (0, 1.5]")
    return kappa


def grid_fractions(config: BatteryConfig) -> np.ndarray:
    """Relative SoC grid f_j = j / (G - 1), shape (G,)."""
    return np.linspace(0.0, 1.0, config.soc_grid_size)


def soc_grid_days(config: BatteryConfig, kappa: np.ndarray) -> np.ndarray:
    """SoC grid in MWh per day, shape (D, G)."""
    usable = config.e_max - config.e_min
    return config.e_min + kappa[:, None] * usable * grid_fractions(config)[None, :]


def day_config(config: BatteryConfig, kappa: float) -> BatteryConfig:
    """The battery with capacity factor ``kappa`` applied (upper SoC bound scaled)."""
    if kappa == 1.0:
        return config
    e_max = config.e_min + kappa * (config.e_max - config.e_min)
    return dataclasses.replace(config, e_max=e_max, soc_init=min(config.soc_init, e_max))


def capacity_groups(kappa: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Days grouped by equal capacity factor: [(kappa, day indices), ...]."""
    rounded = np.round(kappa, 12)
    return [(float(k), np.flatnonzero(rounded == k)) for k in np.unique(rounded)]
