"""LSMC engine: ordering of values, energy balance, reproducibility, output folder."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bess_valuation import BatteryConfig, EngineSettings, PriceGeneratorConfig, generate_prices, run_engine
from bess_valuation.cli import main as cli_main
from bess_valuation.forward import grid_bellman
from bess_valuation.lsmc import concave_projection, monomial_exponents

CONFIG = BatteryConfig(name="engine_test", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                       eta_charge=0.95, eta_discharge=0.95, dt=1.0, soc_init=1.0,
                       soc_grid_size=5, terminal_soc_value=40.0)
SETTINGS = EngineSettings(library_dir=None, value_cache_dir=None)


def prices(n_paths=40, n_days=30, seed=11):
    return generate_prices(PriceGeneratorConfig(), n_paths=n_paths, start="2027-05-01",
                           n_days=n_days, dt=1.0, seed=seed)


class TestBuildingBlocks(unittest.TestCase):
    def test_concave_projection(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(200, 9)).cumsum(axis=1)
        z = concave_projection(y)
        self.assertTrue(np.all(np.diff(z, 2, axis=1) <= 1e-9))
        np.testing.assert_allclose(z.mean(axis=1), y.mean(axis=1), atol=1e-9)
        concave = -np.linspace(-1, 1, 9)[None, :] ** 2 + rng.normal(size=(5, 1))
        np.testing.assert_allclose(concave_projection(concave), concave, atol=1e-12)

    def test_monomials(self):
        self.assertEqual(monomial_exponents(3, 2).shape, (10, 3))
        self.assertEqual(monomial_exponents(2, 3).shape, (10, 2))
        self.assertEqual(monomial_exponents(3, 2)[0].sum(), 0)

    def test_grid_bellman_matches_brute_force(self):
        rng = np.random.default_rng(1)
        F = rng.normal(size=(1, 3, 3, 3))
        terminal = np.array([0.0, 1.0, 2.0])
        value, cash, idx = grid_bellman(F, 1, terminal)
        best = max(F[0, 0, 1, a] + F[0, 1, a, b] + F[0, 2, b, c] + terminal[c]
                   for a in range(3) for b in range(3) for c in range(3))
        self.assertAlmostEqual(value[0], best)
        self.assertAlmostEqual(cash.sum() + terminal[idx[0, -1]], best)


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = prices()
        cls.result = run_engine(CONFIG, cls.prices, SETTINGS)

    def test_value_ordering(self):
        r = self.result
        s = r.summary()
        stderr = s["value_oos_stderr_eur"]
        self.assertLessEqual(s["value_intrinsic_eur"], s["value_oos_eur"] + 2 * stderr)
        # perfect foresight dominates the policy path by path
        self.assertTrue(np.all(r.pf_value >= r.forward.value - 1e-6))
        self.assertGreater(s["value_oos_eur"], 0.0)

    def test_energy_balance_and_soc_bounds(self):
        r = self.result
        fwd = r.forward
        cfg = CONFIG
        tol = 1e-5
        self.assertTrue(np.all(fwd.soc >= cfg.e_min - tol) and np.all(fwd.soc <= cfg.e_max + tol))
        np.testing.assert_allclose(fwd.soc[:, :, -1], r.grid[fwd.soc_index[:, 1:]], atol=tol)
        start = r.grid[fwd.soc_index[:, :-1]]
        first = start + cfg.eta_charge * fwd.charge[:, :, 0] - fwd.discharge[:, :, 0] / cfg.eta_discharge
        np.testing.assert_allclose(fwd.soc[:, :, 0], first, atol=tol)
        self.assertTrue(np.all(np.minimum(fwd.charge, fwd.discharge) <= tol))  # no simultaneity
        self.assertTrue(np.all(fwd.charge <= cfg.p_charge_max * cfg.dt + tol))
        self.assertTrue(np.all(fwd.discharge <= cfg.p_discharge_max * cfg.dt + tol))
        oos = self.prices.da[self.prices.n_paths // 2:].astype(np.float64)
        cash = np.einsum("ndt,ndt->nd", oos, fwd.discharge - fwd.charge)
        np.testing.assert_allclose(cash, fwd.cash, atol=1e-3)
        self.assertTrue(np.all(fwd.cycles >= 0.0))
        self.assertEqual(fwd.soc_index[0, 0], r.i0)

    def test_diagnostics(self):
        bw = self.result.backward
        n_days = self.prices.n_days
        self.assertEqual(bw.r2.shape, (n_days, CONFIG.soc_grid_size))
        self.assertTrue(np.all(np.isnan(bw.r2[-1])))
        self.assertTrue(np.all(bw.violation_share >= 0.0) and np.all(bw.violation_share <= 1.0))
        self.assertTrue(np.all(np.diff(bw.curve_projected, 2, axis=1) <= 1e-6))

    def test_reproducible(self):
        again = run_engine(CONFIG, self.prices, SETTINGS)
        # runtimes and library counters depend on timing (adaptive full-argmax stage)
        skip = ("runtime", "library_")
        a = {k: v for k, v in self.result.summary().items() if not k.startswith(skip)}
        b = {k: v for k, v in again.summary().items() if not k.startswith(skip)}
        self.assertEqual(a.keys(), b.keys())
        for key in a:  # equal up to rounding (warm starts may take other pivot paths)
            if isinstance(a[key], float):
                self.assertTrue(np.isclose(a[key], b[key], rtol=1e-9, atol=1e-9, equal_nan=True), key)
            else:
                self.assertEqual(a[key], b[key], key)
        np.testing.assert_array_equal(again.forward.soc_index, self.result.forward.soc_index)

    def test_save_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.result.save(tmp, plots=True)
            self.assertEqual(run_dir.parent, Path(tmp) / CONFIG.name)
            for name in ("config.yaml", "prices_meta.json", "summary.json", "summary.csv",
                         "paths.parquet"):
                self.assertTrue((run_dir / name).exists(), name)
            # default solver "dp": no vertex library, hence no library stats / plot 12
            self.assertFalse((run_dir / "library_stats.json").exists())
            self.assertEqual(len(list((run_dir / "figures").glob("*.png"))), 8)
            daily = pd.read_parquet(run_dir / "daily.parquet")
            self.assertEqual(len(daily), self.prices.n_days)
            np.testing.assert_allclose(daily["cashflow_intrinsic_eur"].sum(), self.result.intrinsic_value
                                       - CONFIG.terminal_soc_value * daily["soc_end_intrinsic_mwh"].iloc[-1],
                                       atol=1e-6)
            reg = pd.read_parquet(run_dir / "regression.parquet")
            self.assertEqual(len(reg), self.prices.n_days * CONFIG.soc_grid_size)
            self.assertTrue(reg["r2"].iloc[:-CONFIG.soc_grid_size].notna().all())
            frame = pd.read_parquet(run_dir / "paths.parquet")
            n_oos = self.prices.n_paths - self.prices.n_paths // 2
            self.assertEqual(len(frame), n_oos * self.prices.n_days)
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertAlmostEqual(summary["value_oos_eur"], self.result.value_oos)
            self.assertEqual(BatteryConfig.from_yaml(run_dir / "config.yaml"), CONFIG)
            second = self.result.save(tmp, plots=False)
            self.assertNotEqual(second, run_dir)  # never overwrites


class TestVariants(unittest.TestCase):
    def test_without_state_and_control_variate(self):
        p = prices(n_paths=20, n_days=10, seed=3)
        p.X = None
        p.conditional_mean = None
        r = run_engine(CONFIG, p, EngineSettings(library_dir=None, value_cache_dir=None, degree=1,
                                                 control_variate=True))
        self.assertTrue(np.all(r.backward.var_ratio == 1.0))
        self.assertTrue(np.all(r.pf_value >= r.forward.value - 1e-6))

    def test_control_variate(self):
        p = prices(n_paths=20, n_days=10, seed=5)
        r = run_engine(CONFIG, p, dataclasses.replace(SETTINGS, control_variate=True))
        self.assertTrue(r.backward.model.control_variate)
        self.assertIn("daily_values_conditional_mean", r.runtimes)
        self.assertTrue(np.all(r.pf_value >= r.forward.value - 1e-6))

    def test_value_cache(self):
        p = prices(n_paths=20, n_days=10, seed=8)
        with tempfile.TemporaryDirectory() as tmp:
            settings = dataclasses.replace(SETTINGS, value_cache_dir=tmp, daily_solver="library")
            first = run_engine(CONFIG, p, settings)
            files = list(Path(tmp).glob("*.npy"))
            self.assertEqual(len(files), 1)
            second = run_engine(CONFIG, p, dataclasses.replace(settings, degree=1))
            self.assertEqual(second.library_stats["queries"] < first.library_stats["queries"], True)
            np.testing.assert_array_equal(second.pf_value, first.pf_value)
            # other prices or other battery physics do not hit the cache
            p.da[0, 0, 0] += 1.0
            run_engine(CONFIG, p, settings)
            run_engine(dataclasses.replace(CONFIG, eta_charge=0.9), p, settings)
            self.assertEqual(len(list(Path(tmp).glob("*.npy"))), 3)
            # non-physical changes (name, start SoC, terminal value) reuse it
            run_engine(dataclasses.replace(CONFIG, name="other", soc_init=0.5), p, settings)
            self.assertEqual(len(list(Path(tmp).glob("*.npy"))), 3)

    def test_all_daily_solvers_agree(self):
        p = prices(n_paths=8, n_days=6, seed=9)
        results = {solver: run_engine(CONFIG, p, dataclasses.replace(SETTINGS, daily_solver=solver))
                   for solver in ("dp", "library", "cold", "cluster")}
        ref = results["dp"]
        for solver, r in results.items():
            with self.subTest(solver=solver):
                self.assertEqual(r.daily_solver, solver)
                np.testing.assert_allclose(r.pf_value, ref.pf_value, atol=1e-6)
                np.testing.assert_allclose(r.forward.cash, ref.forward.cash, atol=1e-6)
                np.testing.assert_array_equal(r.forward.soc_index, ref.forward.soc_index)
                self.assertAlmostEqual(r.intrinsic_value, ref.intrinsic_value, delta=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = results["library"].save(tmp, plots=True)
            self.assertTrue((run_dir / "library_stats.json").exists())
            self.assertEqual(len(list((run_dir / "figures").glob("*.png"))), 9)

    def test_simultaneous_allowed_is_not_worse(self):
        p = prices(n_paths=20, n_days=10, seed=4)
        cfg = dataclasses.replace(CONFIG, allow_simultaneous_charge_discharge=True)
        relaxed = run_engine(cfg, p, SETTINGS)
        strict = run_engine(CONFIG, p, SETTINGS)
        self.assertGreaterEqual(relaxed.pf_value.mean(), strict.pf_value.mean() - 1e-6)

    def test_cli_run(self):
        p = prices(n_paths=10, n_days=5, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            p.save(Path(tmp) / "scen")
            CONFIG.to_yaml(Path(tmp) / "cfg.yaml")
            cli_main(["--log-level", "WARNING", "run", "--config", str(Path(tmp) / "cfg.yaml"),
                      "--prices", str(Path(tmp) / "scen"), "--no-plots", "--library-dir",
                      str(Path(tmp) / "lib"), "--value-cache-dir", str(Path(tmp) / "cache"),
                      "--daily-solver", "library",
                      "--results", str(Path(tmp) / "res")])
            runs = list((Path(tmp) / "res" / CONFIG.name).iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "summary.json").exists())
            self.assertEqual(len(list((Path(tmp) / "lib").glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
