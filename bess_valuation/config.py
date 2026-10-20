"""Battery configuration, YAML loading and config hash."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

# Parameters that define the feasible dispatch polytope. Only these enter the config hash,
# so vertex libraries stay valid when e.g. the name or the cycle limit changes.
PHYSICAL_FIELDS: tuple[str, ...] = (
    "e_max",
    "e_min",
    "p_charge_max",
    "p_discharge_max",
    "eta_charge",
    "eta_discharge",
    "dt",
    "allow_simultaneous_charge_discharge",
)


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and economic parameters of the battery.

    Attributes:
        name: Identifier, used as output folder name.
        e_max: Maximum state of charge [MWh].
        e_min: Minimum state of charge [MWh].
        p_charge_max: Maximum charging power [MW].
        p_discharge_max: Maximum discharging power [MW].
        eta_charge: One-way charging efficiency, in (0, 1].
        eta_discharge: One-way discharging efficiency, in (0, 1].
        dt: Period length [h], 1.0 or 0.25.
        soc_init: Initial state of charge [MWh].
        soc_grid_size: Number of SoC grid points for the Bellman state.
        terminal_soc_value: Value per MWh of SoC left at the end of the horizon [€/MWh].
        max_cycles_per_day_avg: Average full cycles per day allowed; None means no limit.
        cycle_period: Period over which the cycle limit is averaged.
        cycle_method: "lagrange" (cycle cost lambda) or "counter" (extra state).
        allow_simultaneous_charge_discharge: Whether c_t > 0 and d_t > 0 may occur together.
        transaction_cost: Cost per traded MWh per auction [€/MWh].
    """

    name: str
    e_max: float
    p_charge_max: float
    p_discharge_max: float
    e_min: float = 0.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    dt: float = 0.25
    soc_init: float = 0.0
    soc_grid_size: int = 11
    terminal_soc_value: float = 0.0
    max_cycles_per_day_avg: float | None = None
    cycle_period: Literal["month", "year"] = "year"
    cycle_method: Literal["lagrange", "counter"] = "lagrange"
    allow_simultaneous_charge_discharge: bool = False
    transaction_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.e_min < self.e_max:
            raise ValueError(f"e_min ({self.e_min}) must be < e_max ({self.e_max})")
        if self.p_charge_max <= 0 or self.p_discharge_max <= 0:
            raise ValueError("power limits must be positive")
        for eta in (self.eta_charge, self.eta_discharge):
            if not 0.0 < eta <= 1.0:
                raise ValueError(f"efficiency {eta} not in (0, 1]")
        if self.dt not in (1.0, 0.25):
            raise ValueError(f"dt must be 1.0 or 0.25, got {self.dt}")
        if not self.e_min <= self.soc_init <= self.e_max:
            raise ValueError(f"soc_init ({self.soc_init}) outside [e_min, e_max]")
        if self.soc_grid_size < 2:
            raise ValueError("soc_grid_size must be >= 2")
        if self.cycle_period not in ("month", "year"):
            raise ValueError(f"unknown cycle_period {self.cycle_period!r}")
        if self.cycle_method not in ("lagrange", "counter"):
            raise ValueError(f"unknown cycle_method {self.cycle_method!r}")

    @property
    def periods_per_day(self) -> int:
        """Number of periods T per day."""
        return round(24.0 / self.dt)

    @property
    def config_hash(self) -> str:
        """Short SHA-256 hash of the physical parameters (vertex-library key)."""
        physical = {f: getattr(self, f) for f in PHYSICAL_FIELDS}
        payload = json.dumps(physical, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatteryConfig:
        """Build a config from a dict, rejecting unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BatteryConfig:
        """Load a config from a YAML file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
