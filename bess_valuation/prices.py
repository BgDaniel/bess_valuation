"""Price scenarios (DA, IDA1-3, state variables X), loaders for NPZ / Parquet / CSV, saving.

Arrays have shape (N, D, T): N paths, D days, T periods per day. ``ida3`` only covers
the second half of the day (periods T/2..T-1); the first half is NaN. Prices are
stored as float32 [€/MWh] to keep large scenario sets in memory.

On-disk formats (one folder per scenario set):

* NPZ (default): ``prices.npz`` with the arrays plus ``meta.json``.
* Parquet / CSV: ``prices.parquet`` / ``prices.csv`` in long format with columns
  ``path, day, period, da, ida1, ida2, ida3``, optionally ``state.parquet`` /
  ``state.csv`` with ``path, day, x_0, ..., x_{m-1}``, plus ``meta.json``.
  Long format is meant for exchange with other tools and small sets; use NPZ for
  large ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAGES: tuple[str, ...] = ("da", "ida1", "ida2", "ida3")


@dataclass
class PriceScenarios:
    """Stochastic DA and IDA1-3 price paths.

    Attributes:
        da: Day-ahead prices, shape (N, D, T) [€/MWh].
        ida1: IDA1 prices, shape (N, D, T) [€/MWh].
        ida2: IDA2 prices, shape (N, D, T) [€/MWh].
        ida3: IDA3 prices, shape (N, D, T) [€/MWh], NaN for the first T/2 periods.
        dates: Delivery dates, shape (D,), dtype datetime64[D].
        dt: Period length [h].
        X: Optional regression state variables, shape (N, D, m).
        conditional_mean: Optional E[p^DA_{d+1} | X_d], shape (N, D, T) [€/MWh].
        meta: Free-form metadata (source, seed, generator config, ...).
    """

    da: np.ndarray
    ida1: np.ndarray
    ida2: np.ndarray
    ida3: np.ndarray
    dates: np.ndarray
    dt: float
    X: np.ndarray | None = None
    conditional_mean: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.da.shape
        if len(shape) != 3:
            raise ValueError(f"price arrays must have shape (N, D, T), got {shape}")
        for name in STAGES[1:]:
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if self.dates.shape != (shape[1],):
            raise ValueError(f"dates must have shape ({shape[1]},), got {self.dates.shape}")
        if round(24.0 / self.dt) != shape[2]:
            raise ValueError(f"T = {shape[2]} does not match dt = {self.dt}")
        if self.X is not None and self.X.shape[:2] != shape[:2]:
            raise ValueError(f"X has shape {self.X.shape}, expected (N, D, m) with (N, D) = {shape[:2]}")
        if self.conditional_mean is not None and self.conditional_mean.shape != shape:
            raise ValueError("conditional_mean must have the shape of the price arrays")
        self.dates = self.dates.astype("datetime64[D]")

    @property
    def n_paths(self) -> int:
        """Number of paths N."""
        return self.da.shape[0]

    @property
    def n_days(self) -> int:
        """Number of days D."""
        return self.da.shape[1]

    @property
    def periods_per_day(self) -> int:
        """Number of periods T per day."""
        return self.da.shape[2]

    def stage(self, name: str) -> np.ndarray:
        """Return the price array of stage ``name`` ("da", "ida1", "ida2", "ida3")."""
        if name not in STAGES:
            raise KeyError(f"unknown stage {name!r}")
        return getattr(self, name)

    def subset(self, paths: slice | np.ndarray) -> PriceScenarios:
        """Return the scenarios restricted to the given paths (arrays are copied)."""
        def take(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[paths].copy()

        return PriceScenarios(
            da=take(self.da), ida1=take(self.ida1), ida2=take(self.ida2), ida3=take(self.ida3),
            dates=self.dates.copy(), dt=self.dt, X=take(self.X),
            conditional_mean=take(self.conditional_mean), meta=dict(self.meta),
        )

    def split(self, n_first: int) -> tuple[PriceScenarios, PriceScenarios]:
        """Split into two disjoint path sets, e.g. regression and out-of-sample valuation.

        Args:
            n_first: Number of paths in the first set.

        Returns:
            (first n_first paths, remaining paths).
        """
        if not 0 < n_first < self.n_paths:
            raise ValueError(f"n_first must be in (0, {self.n_paths})")
        return self.subset(slice(0, n_first)), self.subset(slice(n_first, None))

    @classmethod
    def synthetic(cls, config: Any, n_paths: int, start: str, n_days: int, dt: float,
                  seed: int) -> PriceScenarios:
        """Generate synthetic scenarios with the price generator (section 2.3).

        Args:
            config: A ``PriceGeneratorConfig``.
            n_paths: Number of paths N.
            start: First delivery date, ISO format.
            n_days: Number of days D.
            dt: Period length [h], 1.0 or 0.25.
            seed: Random seed.
        """
        from bess_valuation.price_generator import generate_prices

        return generate_prices(config, n_paths=n_paths, start=start, n_days=n_days, dt=dt, seed=seed)

    # ------------------------------------------------------------------ saving / loading

    def _meta_json(self) -> dict[str, Any]:
        meta = dict(self.meta)
        meta.update({
            "n_paths": self.n_paths, "n_days": self.n_days, "periods_per_day": self.periods_per_day,
            "dt": self.dt, "start": str(self.dates[0]),
        })
        return meta

    def save(self, folder: str | Path, fmt: str = "npz") -> Path:
        """Write the scenarios to ``folder`` in the format the loaders read.

        Args:
            folder: Target folder (created if missing).
            fmt: "npz", "parquet" or "csv".

        Returns:
            The folder.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if fmt == "npz":
            arrays = {name: getattr(self, name) for name in STAGES}
            arrays["dates"] = self.dates.astype("datetime64[D]").astype(np.int64)
            if self.X is not None:
                arrays["X"] = self.X
            if self.conditional_mean is not None:
                arrays["conditional_mean"] = self.conditional_mean
            np.savez(folder / "prices.npz", **arrays)
        elif fmt in ("parquet", "csv"):
            prices = _to_long(self)
            state = None if self.X is None else _state_to_long(self.X)
            if fmt == "parquet":
                prices.to_parquet(folder / "prices.parquet", index=False)
                if state is not None:
                    state.to_parquet(folder / "state.parquet", index=False)
            else:
                prices.to_csv(folder / "prices.csv", index=False)
                if state is not None:
                    state.to_csv(folder / "state.csv", index=False)
        else:
            raise ValueError(f"unknown format {fmt!r}")
        with open(folder / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta_json(), f, indent=2, default=str)
        return folder

    @classmethod
    def from_files(cls, folder: str | Path) -> PriceScenarios:
        """Load scenarios from a folder written by :meth:`save` (NPZ, Parquet or CSV).

        The format is detected from the file present, in the order NPZ, Parquet, CSV.
        """
        folder = Path(folder)
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        dt = float(meta["dt"])
        start = np.datetime64(meta["start"], "D")
        stored = {k: meta.pop(k) for k in ("n_paths", "n_days", "periods_per_day", "dt", "start")}

        if (folder / "prices.npz").exists():
            with np.load(folder / "prices.npz") as z:
                arrays = {k: z[k] for k in z.files}
            dates = arrays.pop("dates").astype("datetime64[D]")
            return cls(dt=dt, dates=dates, meta=meta, **arrays)

        for ext, reader in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
            path = folder / f"prices.{ext}"
            if path.exists():
                shape = (stored["n_paths"], stored["n_days"], stored["periods_per_day"])
                arrays = _from_long(reader(path), shape)
                state_path = folder / f"state.{ext}"
                X = _state_from_long(reader(state_path), shape[:2]) if state_path.exists() else None
                dates = start + np.arange(shape[1])
                return cls(dt=dt, dates=dates, X=X, meta=meta, **arrays)
        raise FileNotFoundError(f"no prices.npz / prices.parquet / prices.csv in {folder}")


def _to_long(p: PriceScenarios) -> pd.DataFrame:
    n, d, t = p.da.shape
    idx = np.indices((n, d, t)).reshape(3, -1)
    data = {"path": idx[0], "day": idx[1], "period": idx[2]}
    for name in STAGES:
        data[name] = getattr(p, name).reshape(-1)
    return pd.DataFrame(data)


def _from_long(df: pd.DataFrame, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    df = df.sort_values(["path", "day", "period"])
    if len(df) != shape[0] * shape[1] * shape[2]:
        raise ValueError(f"expected {np.prod(shape)} rows, got {len(df)}")
    return {name: df[name].to_numpy(np.float32).reshape(shape) for name in STAGES}


def _state_to_long(X: np.ndarray) -> pd.DataFrame:
    n, d, m = X.shape
    idx = np.indices((n, d)).reshape(2, -1)
    data: dict[str, np.ndarray] = {"path": idx[0], "day": idx[1]}
    flat = X.reshape(n * d, m)
    for k in range(m):
        data[f"x_{k}"] = flat[:, k]
    return pd.DataFrame(data)


def _state_from_long(df: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    df = df.sort_values(["path", "day"])
    cols = sorted((c for c in df.columns if c.startswith("x_")), key=lambda c: int(c[2:]))
    return df[cols].to_numpy(np.float32).reshape(shape[0], shape[1], len(cols))
