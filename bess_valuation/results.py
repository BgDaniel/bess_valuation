"""Result dataclass, summary tables, output folder and the engine entry point ``run_engine``."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bess_valuation.config import BatteryConfig
from bess_valuation.daily import daily_value_table, soc_grid
from bess_valuation.forward import ForwardResult, attach_schedules, forward_simulation, grid_bellman
from bess_valuation.io import new_run_dir
from bess_valuation.library import (VertexLibrary, library_path, library_stats, load_or_create,
                                    save_library)
from bess_valuation.lsmc import BackwardResult, EngineSettings, _features, backward_induction
from bess_valuation.prices import PriceScenarios

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """Everything a valuation run produces (DA only so far).

    Attributes:
        config: Battery parameters.
        settings: Numerical settings.
        prices_meta: Metadata of the scenarios (source, seed, sizes, split).
        dates: Delivery dates, shape (D,).
        dt: Period length [h].
        grid: SoC grid [MWh], shape (G,).
        i0: Grid index of the initial SoC.
        backward: Backward-induction output on the regression paths.
        forward: Realised dispatch on the out-of-sample paths.
        da_oos: Out-of-sample DA prices [€/MWh], shape (N, D, T), float32.
        intrinsic_value: Value on the mean price path [€].
        intrinsic_cash: Daily cashflow of the intrinsic schedule [€], shape (D,).
        intrinsic_soc_index: Grid index of the intrinsic SoC per day boundary, shape (D+1,).
        pf_value: Perfect-foresight value per out-of-sample path [€], shape (N,).
        pf_cash: Perfect-foresight daily cashflow [€], shape (N, D).
        library: The vertex library used (None with the DP solver; not written to disk).
        library_stats: Output of :func:`bess_valuation.library.library_stats`.
        runtimes: Wall-clock seconds per step.
        run_dir: Output folder after :meth:`save`.
        daily_solver: Daily solver actually used.
    """

    config: BatteryConfig
    settings: EngineSettings
    prices_meta: dict[str, Any]
    dates: np.ndarray
    dt: float
    grid: np.ndarray
    i0: int
    backward: BackwardResult
    forward: ForwardResult
    da_oos: np.ndarray
    intrinsic_value: float
    intrinsic_cash: np.ndarray
    intrinsic_soc_index: np.ndarray
    pf_value: np.ndarray
    pf_cash: np.ndarray
    library: VertexLibrary | None
    library_stats: dict[str, Any]
    runtimes: dict[str, float] = field(default_factory=dict)
    run_dir: Path | None = None
    daily_solver: str = "dp"

    @property
    def value_in_sample(self) -> float:
        """Mean realised value V_0(s_init) on the regression paths [€]."""
        return float(self.backward.value[:, self.i0].mean())

    @property
    def value_oos(self) -> float:
        """Mean realised value of the forward simulation (out-of-sample) [€]."""
        return float(self.forward.value.mean())

    def summary(self) -> dict[str, Any]:
        """Key figures: values and decomposition, cycles, regression and library diagnostics, runtimes."""
        fwd = self.forward
        n_oos = fwd.cash.shape[0]
        lib = self.library_stats
        bw = self.backward
        value_oos = self.value_oos
        return {
            "name": self.config.name,
            "config_hash": self.config.config_hash,
            "n_paths_regression": int(bw.value.shape[0]),
            "n_paths_oos": int(n_oos),
            "n_days": int(fwd.cash.shape[1]),
            "periods_per_day": int(self.da_oos.shape[2]),
            "soc_grid_size": int(self.grid.size),
            "start": str(self.dates[0]),
            "value_in_sample_eur": self.value_in_sample,
            "value_oos_eur": value_oos,
            "value_oos_stderr_eur": float(fwd.value.std(ddof=1) / np.sqrt(n_oos)) if n_oos > 1 else float("nan"),
            "value_intrinsic_eur": float(self.intrinsic_value),
            "value_extrinsic_eur": value_oos - float(self.intrinsic_value),
            "extrinsic_share": (value_oos - float(self.intrinsic_value)) / value_oos if value_oos else float("nan"),
            "value_perfect_foresight_eur": float(self.pf_value.mean()),
            "value_da_eur": float(fwd.cash.sum(axis=1).mean()),
            "value_ida1_eur": 0.0,
            "value_ida2_eur": 0.0,
            "value_ida3_eur": 0.0,
            "value_terminal_eur": float(fwd.terminal.mean()),
            "cycles_per_day_mean": float(np.mean(fwd.cycles)) if fwd.cycles is not None else float("nan"),
            "cycles_per_day_max": float(np.max(fwd.cycles)) if fwd.cycles is not None else float("nan"),
            "regression_r2_mean": float(np.nanmean(bw.r2)) if np.isfinite(bw.r2).any() else float("nan"),
            "regression_r2_median": float(np.nanmedian(bw.r2)) if np.isfinite(bw.r2).any() else float("nan"),
            "control_variate_variance_ratio_mean": float(np.mean(bw.var_ratio[:-1])) if bw.var_ratio.shape[0] > 1 else 1.0,
            "concavity_violation_share_mean": float(bw.violation_share.mean()),
            "daily_solver": self.daily_solver,
            "library_vertices": lib.get("vertices", 0),
            "library_queries": lib.get("queries", 0),
            "library_certified_share": lib.get("certified_share", float("nan")),
            "library_good_turing": lib.get("good_turing", float("nan")),
            "library_simplex_solves": lib.get("simplex", 0),
            "library_case_splits": lib.get("case_split", 0),
            **{f"runtime_{k}_s": v for k, v in self.runtimes.items()},
        }

    def save(self, root: str | Path = "results", plots: bool = True, pdf: bool = False,
             dark_figures: bool = False) -> Path:
        """Write the output folder ``root/<config.name>/<YYYYMMDD_HHMMSS>/`` (SPEC section 6).

        Contents: ``config.yaml``, ``prices_meta.json``, ``summary.json``, ``summary.csv``,
        ``paths.parquet`` (per path and day), ``daily.parquet`` (per day: intrinsic,
        LSMC and perfect-foresight cashflows), ``regression.parquet`` (per day and SoC grid
        point: R^2, variance ratio), ``library_stats.json`` (library solver) and
        ``figures/*.png``;
        with ``dark_figures`` also ``figures_dark/*.png`` (dark theme, used by the app).

        Returns:
            The run folder.
        """
        run_dir = new_run_dir(self.config.name, root)
        self.config.to_yaml(run_dir / "config.yaml")
        meta = dict(self.prices_meta)
        meta["settings"] = asdict(self.settings)
        _write_json(run_dir / "prices_meta.json", meta)
        summary = self.summary()
        _write_json(run_dir / "summary.json", summary)
        with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(summary.items())
        self.paths_frame().to_parquet(run_dir / "paths.parquet", index=False)
        self.daily_frame().to_parquet(run_dir / "daily.parquet", index=False)
        self.regression_frame().to_parquet(run_dir / "regression.parquet", index=False)
        if self.library_stats:
            _write_json(run_dir / "library_stats.json", self.library_stats)
        if plots:
            from bess_valuation.plots import plot_results

            figures = plot_results(self, run_dir / "figures", pdf=pdf)
            logger.info("%d figures written to %s", len(figures), run_dir / "figures")
            if dark_figures:
                plot_results(self, run_dir / "figures_dark", theme="dark")
        self.run_dir = run_dir
        logger.info("results saved to %s", run_dir)
        return run_dir

    def paths_frame(self) -> pd.DataFrame:
        """Per path and day: SoC at start / end, DA cashflow, cycles, perfect-foresight cashflow."""
        fwd = self.forward
        n, d = fwd.cash.shape
        path, day = np.indices((n, d)).reshape(2, -1)
        frame = {
            "path": path, "day": day, "date": np.tile(self.dates, n),
            "soc_start_mwh": self.grid[fwd.soc_index[:, :-1]].ravel(),
            "soc_end_mwh": self.grid[fwd.soc_index[:, 1:]].ravel(),
            "cashflow_da_eur": fwd.cash.ravel(),
            "cashflow_pf_eur": self.pf_cash.ravel(),
        }
        if fwd.cycles is not None:
            frame["cycles"] = fwd.cycles.ravel()
        return pd.DataFrame(frame)


    def daily_frame(self) -> pd.DataFrame:
        """Per day: intrinsic cashflow, LSMC cashflow (mean and quantiles over the out-of-sample
        paths), perfect-foresight mean, SoC grid indices of the intrinsic schedule, share of
        paths changed by the concavity projection."""
        cash = self.forward.cash
        q05, q25, q50, q75, q95 = np.quantile(cash, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        return pd.DataFrame({
            "day": np.arange(cash.shape[1]), "date": self.dates,
            "cashflow_intrinsic_eur": self.intrinsic_cash,
            "cashflow_lsmc_mean_eur": cash.mean(axis=0),
            "cashflow_lsmc_q05_eur": q05, "cashflow_lsmc_q25_eur": q25, "cashflow_lsmc_median_eur": q50,
            "cashflow_lsmc_q75_eur": q75, "cashflow_lsmc_q95_eur": q95,
            "cashflow_pf_mean_eur": self.pf_cash.mean(axis=0),
            "soc_end_intrinsic_mwh": self.grid[self.intrinsic_soc_index[1:]],
            "projection_share": self.backward.violation_share,
        })

    def regression_frame(self) -> pd.DataFrame:
        """Per day and SoC grid point: R^2 of the continuation regression and the variance
        ratio of the control variate (NaN / 1 on the last day, where C is the terminal value)."""
        bw = self.backward
        d, g = bw.r2.shape
        day, j = np.indices((d, g)).reshape(2, -1)
        return pd.DataFrame({"day": day, "date": self.dates[day], "soc_mwh": self.grid[j],
                             "r2": bw.r2.ravel(), "variance_ratio": bw.var_ratio.ravel()})


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def cached_daily_values(cache_dir: str | Path | None, library: VertexLibrary | None,
                        config: BatteryConfig, prices: np.ndarray, grid: np.ndarray, desc: str,
                        seed: int = 0, progress: Callable[[float], None] | None = None,
                        solver: str = "library") -> np.ndarray:
    """Daily values (:func:`daily_value_table`) with a file cache keyed by battery physics,
    SoC grid and prices (not by the solver: all solvers give the same values).

    The key is a SHA-256 over the config hash (physical parameters only), the grid, the
    zero cycle cost and the raw price array, so a cached table is only reused for
    exactly the same daily problems.

    Args:
        cache_dir: Cache folder (None: always compute).
        library: Vertex library (solver "library" only).
        config: Battery parameters.
        prices: Prices [€/MWh], shape (N, D, T).
        grid: SoC grid [MWh].
        desc: Progress-bar label.
        seed: Library processing order.
        progress: Optional callback with the finished fraction (0..1).
        solver: Daily solver (see :func:`bess_valuation.daily.daily_value_table`).

    Returns:
        Daily values [€], shape (N, D, G, G).
    """
    def compute() -> np.ndarray:
        return daily_value_table(solver, config, prices, grid, library=library, desc=desc, seed=seed,
                                 progress=progress)

    if cache_dir is None:
        return compute()
    h = hashlib.sha256()
    h.update(config.config_hash.encode())
    h.update(np.ascontiguousarray(grid, dtype=np.float64).tobytes())
    h.update(b"cycle_cost=0")
    h.update(str(prices.shape).encode() + str(prices.dtype).encode())
    h.update(np.ascontiguousarray(prices).tobytes())
    path = Path(cache_dir) / f"{config.config_hash}_{h.hexdigest()[:16]}.npy"
    if path.exists():
        logger.info("%s: loaded cached daily values from %s", desc, path)
        if progress is not None:
            progress(1.0)
        return np.load(path)
    F = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, F)
    tmp.replace(path)
    logger.info("%s: daily values cached in %s", desc, path)
    return F


# Rough share of the work per step (used only to turn step progress into overall progress).
STAGE_WEIGHTS = {"daily values (conditional means)": 0.4, "daily values (prices)": 1.0,
                 "backward induction": 0.004, "forward simulation": 0.004, "schedules": 0.015,
                 "intrinsic and perfect foresight": 0.002}


def _stage_progress(progress: Callable[[str, float], None] | None, stages: list[str]
                    ) -> Callable[[str], Callable[[float], None] | None]:
    """Turn per-step fractions into overall progress calls ``progress(step, overall)``."""
    total = sum(STAGE_WEIGHTS[s] for s in stages)
    offsets = {}
    acc = 0.0
    for stage in stages:
        offsets[stage] = acc
        acc += STAGE_WEIGHTS[stage]

    def for_stage(stage: str) -> Callable[[float], None] | None:
        if progress is None:
            return None

        def report(fraction: float) -> None:
            progress(stage, min(1.0, (offsets[stage] + STAGE_WEIGHTS[stage] * fraction) / total))

        report(0.0)
        return report

    return for_stage


def run_engine(config: BatteryConfig, prices: PriceScenarios,
               settings: EngineSettings | None = None,
               progress: Callable[[str, float], None] | None = None) -> Result:
    """Value the battery on DA prices: backward induction plus forward simulation.

    The paths are split into a regression set (first ``n_regression_paths``) and an
    out-of-sample set. Daily values for all SoC grid pairs come from the vertex library
    (loaded from and saved to ``settings.library_dir`` per config hash).

    Args:
        config: Battery parameters.
        prices: Price scenarios (only DA is used so far).
        settings: Numerical settings; defaults if None.
        progress: Optional callback ``progress(step name, overall fraction 0..1)``, called
            regularly (e.g. by the app for its progress bar and remaining-time estimate).

    Returns:
        The result; call :meth:`Result.save` to write the output folder.
    """
    settings = settings or EngineSettings()
    if abs(prices.dt - config.dt) > 1e-12:
        raise ValueError(f"price dt {prices.dt} differs from config dt {config.dt}")
    if prices.n_paths < 2:
        raise ValueError("need at least 2 paths (regression and out-of-sample)")
    runtimes: dict[str, float] = {}
    grid = soc_grid(config)
    i0 = int(np.argmin(np.abs(grid - config.soc_init)))
    if abs(grid[i0] - config.soc_init) > 1e-9:
        logger.warning("soc_init %.4f is not on the SoC grid; using %.4f", config.soc_init, grid[i0])
    n_reg = settings.n_regression_paths or prices.n_paths // 2
    reg, oos = prices.split(n_reg)
    logger.info("run %s: %d regression + %d out-of-sample paths, %d days, T = %d, grid %d",
                config.name, reg.n_paths, oos.n_paths, prices.n_days, prices.periods_per_day, grid.size)
    solver = settings.daily_solver
    if solver == "dp" and config.allow_simultaneous_charge_discharge:
        logger.warning("daily solver 'dp' needs c_t d_t = 0; using 'library' instead")
        solver = "library"
    library = None
    if solver == "library":
        library = load_or_create(settings.library_dir, config, "full", settings.max_vertices)
    elif solver != "dp":
        library = VertexLibrary.empty(config, "full", settings.max_vertices)  # schedules only
    logger.info("daily solver: %s", solver)

    feats_reg = _features(reg.da, reg.X, settings.features)
    feats_oos = _features(oos.da, oos.X, settings.features)
    use_cv = settings.control_variate and prices.conditional_mean is not None
    if settings.control_variate and not use_cv:
        logger.warning("no conditional_mean in the scenarios: control variate disabled")
    stages = (["daily values (conditional means)"] if use_cv else []) + [
        "daily values (prices)", "backward induction", "forward simulation", "schedules",
        "intrinsic and perfect foresight"]
    stage = _stage_progress(progress, stages)

    t0 = time.perf_counter()
    F_mu_reg = F_mu_oos = None
    if use_cv:
        F_mu = cached_daily_values(settings.value_cache_dir, library, config, prices.conditional_mean,
                                   grid, "F(mu)", settings.seed,
                                   stage("daily values (conditional means)"), solver)
        F_mu_reg, F_mu_oos = F_mu[:n_reg], F_mu[n_reg:]
        runtimes["daily_values_conditional_mean"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    F_all = cached_daily_values(settings.value_cache_dir, library, config, prices.da, grid, "F(p)",
                                settings.seed, stage("daily values (prices)"), solver)
    F_reg, F_oos = F_all[:n_reg], F_all[n_reg:]
    runtimes["daily_values_prices"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    backward = backward_induction(F_reg, feats_reg, grid, config.terminal_soc_value, settings, F_mu_reg,
                                  progress=stage("backward induction"))
    runtimes["backward_induction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    forward = forward_simulation(backward.model, F_oos, feats_oos, F_mu_oos, i0,
                                 settings.concave_projection, progress=stage("forward simulation"))
    attach_schedules(forward, library, oos.da, grid, config, seed=settings.seed,
                     progress=stage("schedules"), solver=solver)
    runtimes["forward_simulation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    done = stage("intrinsic and perfect foresight")
    mean_path = oos.da.astype(np.float64).mean(axis=0)
    F_bar = daily_value_table(solver, config, mean_path[None], grid, library=library,
                              desc="F(mean price)", seed=settings.seed)
    terminal = config.terminal_soc_value * grid
    iv, icash, iidx = grid_bellman(F_bar, i0, terminal)
    pf_value, pf_cash, _ = grid_bellman(F_oos, i0, terminal)
    runtimes["intrinsic_and_perfect_foresight"] = time.perf_counter() - t0
    if done is not None:
        done(1.0)

    if solver == "library" and settings.library_dir is not None:
        save_library(library, library_path(settings.library_dir, config, "full"))
    meta = {k: v for k, v in prices.meta.items() if k != "generator"}
    meta.update({"n_paths": prices.n_paths, "n_paths_regression": reg.n_paths,
                 "n_paths_oos": oos.n_paths, "n_days": prices.n_days,
                 "periods_per_day": prices.periods_per_day, "dt": prices.dt,
                 "start": str(prices.dates[0]), "generator": prices.meta.get("generator")})
    result = Result(config=config, settings=settings, prices_meta=meta, dates=prices.dates,
                    dt=prices.dt, grid=grid, i0=i0, backward=backward, forward=forward,
                    da_oos=oos.da, intrinsic_value=float(iv[0]), intrinsic_cash=icash[0],
                    intrinsic_soc_index=iidx[0], pf_value=pf_value, pf_cash=pf_cash,
                    library=library, library_stats=library_stats(library) if library else {},
                    runtimes=runtimes, daily_solver=solver)
    s = result.summary()
    logger.info("values [EUR]: in-sample %.0f, out-of-sample %.0f +- %.0f, intrinsic %.0f, "
                "perfect foresight %.0f", s["value_in_sample_eur"], s["value_oos_eur"],
                s["value_oos_stderr_eur"], s["value_intrinsic_eur"], s["value_perfect_foresight_eur"])
    return result
