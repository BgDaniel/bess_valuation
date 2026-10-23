"""Saved battery configurations and simulation sets (used by the app).

Each entry is one YAML file:

* ``<root>/batteries/<name>.yaml``: a :class:`BatteryConfig`;
* ``<root>/simulations/<name>.yaml``: a :class:`SimulationSpec` (where the price paths come
  from and how many);
* ``<root>/settings.yaml``: app settings, currently the display units (:class:`Units`).

A run combines one battery, one simulation set and the numerical settings.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from bess_valuation.config import BatteryConfig
from bess_valuation.units import Units

DEFAULT_ROOT = Path("app_data")
DEFAULT_GENERATOR_CONFIG = "configs/prices_synthetic.yaml"


@dataclass(frozen=True)
class SimulationSpec:
    """A set of price scenarios.

    Attributes:
        name: Identifier (file name).
        source: "synthetic" (price generator) or "folder" (scenarios written by
            ``PriceScenarios.save``).
        n_paths: Number of paths N (synthetic).
        n_days: Number of days D (synthetic).
        start: First delivery date, ISO format (synthetic).
        seed: Random seed (synthetic); the same seed gives the same paths.
        dt: Period length [h], must match the battery's dt.
        regression_share: Share of the paths used for the regression, rest out-of-sample.
        generator_config: Price generator parameters (YAML).
        folder: Scenario folder (source "folder").
        description: Free text.
    """

    name: str
    source: Literal["synthetic", "folder"] = "synthetic"
    n_paths: int = 100
    n_days: int = 365
    start: str = "2027-01-01"
    seed: int = 42
    dt: float = 0.25
    regression_share: float = 0.5
    generator_config: str = DEFAULT_GENERATOR_CONFIG
    folder: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.source not in ("synthetic", "folder"):
            raise ValueError(f"unknown source {self.source!r}")
        if self.source == "synthetic" and (self.n_paths < 4 or self.n_days < 2):
            raise ValueError("need at least 4 paths and 2 days")
        if self.source == "folder" and not self.folder:
            raise ValueError("source 'folder' needs a folder")
        if not 0.0 < self.regression_share < 1.0:
            raise ValueError("regression_share must be in (0, 1)")
        if self.dt not in (0.25, 1.0):
            raise ValueError("dt must be 0.25 or 1.0")

    def prices_spec(self) -> dict[str, Any]:
        """The ``prices`` part of a job specification (:mod:`bess_valuation.jobs`)."""
        if self.source == "synthetic":
            return {"source": "synthetic", "generator_config": self.generator_config,
                    "n_paths": self.n_paths, "n_days": self.n_days, "start": self.start,
                    "seed": self.seed}
        return {"source": "folder", "folder": self.folder}

    def n_regression_paths(self, n_paths: int | None = None) -> int:
        """Regression paths for ``n_paths`` paths (default: this set's number)."""
        n = n_paths or self.n_paths
        return max(1, min(n - 1, round(self.regression_share * n)))


def slug(name: str) -> str:
    """File-system friendly version of a name."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    if not s:
        raise ValueError("name must not be empty")
    return s


def _folder(root: str | Path, kind: str) -> Path:
    path = Path(root) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_battery(config: BatteryConfig, root: str | Path = DEFAULT_ROOT) -> Path:
    """Store (or overwrite) a battery configuration under its name."""
    path = _folder(root, "batteries") / f"{slug(config.name)}.yaml"
    config.to_yaml(path)
    return path


def list_batteries(root: str | Path = DEFAULT_ROOT) -> list[BatteryConfig]:
    """All stored battery configurations, sorted by name."""
    return sorted((BatteryConfig.from_yaml(p) for p in _folder(root, "batteries").glob("*.yaml")),
                  key=lambda c: c.name.lower())


def save_simulation(spec: SimulationSpec, root: str | Path = DEFAULT_ROOT) -> Path:
    """Store (or overwrite) a simulation set under its name."""
    path = _folder(root, "simulations") / f"{slug(spec.name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(spec), f, sort_keys=False)
    return path


def list_simulations(root: str | Path = DEFAULT_ROOT) -> list[SimulationSpec]:
    """All stored simulation sets, sorted by name."""
    specs = []
    for p in _folder(root, "simulations").glob("*.yaml"):
        with open(p, encoding="utf-8") as f:
            specs.append(SimulationSpec(**yaml.safe_load(f)))
    return sorted(specs, key=lambda s: s.name.lower())


def delete_entry(kind: Literal["batteries", "simulations"], name: str,
                 root: str | Path = DEFAULT_ROOT) -> None:
    """Remove a stored battery or simulation set (no error if it does not exist)."""
    (Path(root) / kind / f"{slug(name)}.yaml").unlink(missing_ok=True)


def load_units(root: str | Path = DEFAULT_ROOT) -> Units:
    """Display units stored in ``<root>/settings.yaml`` (defaults if none are stored)."""
    path = Path(root) / "settings.yaml"
    if not path.exists():
        return Units()
    with open(path, encoding="utf-8") as f:
        return Units.from_dict((yaml.safe_load(f) or {}).get("units"))


def save_units(units: Units, root: str | Path = DEFAULT_ROOT) -> Path:
    """Store the display units in ``<root>/settings.yaml``."""
    path = Path(root) / "settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"units": units.to_dict()}, f, sort_keys=False)
    return path


def seed_defaults(root: str | Path = DEFAULT_ROOT, configs: str | Path = "configs") -> None:
    """Fill an empty catalog with the example batteries and two simulation sets."""
    if not list_batteries(root):
        for path in sorted(Path(configs).glob("bess_*.yaml")):
            save_battery(BatteryConfig.from_yaml(path), root)
    if not list_simulations(root):
        save_simulation(SimulationSpec(name="synthetic_2027_100", description="quick check"), root)
        save_simulation(SimulationSpec(name="synthetic_2027_1000", n_paths=1000,
                                       description="production size"), root)
