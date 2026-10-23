"""Degradation: day-dependent usable capacity and wear cost per discharged MWh."""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bess_valuation import BatteryConfig, EngineSettings, PriceGeneratorConfig, generate_prices, run_engine
from bess_valuation.degradation import capacity_factors, capacity_groups, day_config, soc_grid_days

CONFIG = BatteryConfig(name="degradation_test", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                       dt=1.0, soc_init=1.0, soc_grid_size=5, terminal_soc_value=30.0)
SETTINGS = EngineSettings(library_dir=None, value_cache_dir=None)


def prices(n_paths=12, n_days=70, seed=21, start="2027-01-20"):
    return generate_prices(PriceGeneratorConfig(), n_paths=n_paths, start=start, n_days=n_days, dt=1.0,
                           seed=seed)


class TestCapacityFactors(unittest.TestCase):
    def test_monthly_fade(self):
        dates = np.arange(np.datetime64("2027-01-20"), np.datetime64("2027-04-02"))
        cfg = dataclasses.replace(CONFIG, capacity_fade_per_year=0.12)
        kappa = capacity_factors(cfg, dates)
        self.assertEqual(kappa[0], 1.0)
        self.assertEqual(kappa[dates == np.datetime64("2027-01-31")][0], 1.0)
        self.assertAlmostEqual(kappa[dates == np.datetime64("2027-02-01")][0], 0.99)
        self.assertAlmostEqual(kappa[-1], 1.0 - 0.12 * 3 / 12)
        self.assertEqual(len(capacity_groups(kappa)), 4)

    def test_user_function_and_array(self):
        dates = np.arange(np.datetime64("2027-01-01"), np.datetime64("2027-01-11"))
        f = capacity_factors(CONFIG, dates, lambda d: 1.0 - 0.01 * np.arange(len(d)))
        np.testing.assert_allclose(f, 1.0 - 0.01 * np.arange(10))
        np.testing.assert_allclose(capacity_factors(CONFIG, dates, np.full(10, 0.8)), 0.8)
        with self.assertRaises(ValueError):
            capacity_factors(CONFIG, dates, np.zeros(10))
        grid = soc_grid_days(CONFIG, f)
        np.testing.assert_allclose(grid[:, -1], 2.0 * f)
        self.assertEqual(day_config(CONFIG, 0.5).e_max, 1.0)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(CONFIG, capacity_fade_per_year=-0.1)
        with self.assertRaises(ValueError):
            dataclasses.replace(CONFIG, wear_cost_per_mwh=-1.0)


class TestEngineWithDegradation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = prices()
        cls.base = run_engine(CONFIG, cls.p, SETTINGS)

    def test_no_degradation_is_unchanged(self):
        ones = run_engine(CONFIG, self.p, SETTINGS, capacity_factor=np.ones(self.p.n_days))
        np.testing.assert_allclose(ones.pf_value, self.base.pf_value)
        np.testing.assert_allclose(ones.forward.cash, self.base.forward.cash)
        self.assertEqual(self.base.summary()["wear_cost_eur"], 0.0)

    def test_constant_factor_equals_smaller_battery(self):
        half = run_engine(CONFIG, self.p, SETTINGS, capacity_factor=0.5)
        small = run_engine(dataclasses.replace(CONFIG, e_max=1.0), self.p, SETTINGS)
        np.testing.assert_allclose(half.pf_value, small.pf_value, atol=1e-6)
        np.testing.assert_allclose(half.forward.cash, small.forward.cash, atol=1e-6)
        self.assertAlmostEqual(half.intrinsic_value, small.intrinsic_value, delta=1e-6)

    def test_fade_lowers_value_and_respects_capacity(self):
        cfg = dataclasses.replace(CONFIG, capacity_fade_per_year=0.6)
        r = run_engine(cfg, self.p, SETTINGS)
        self.assertLess(r.pf_value.mean(), self.base.pf_value.mean())
        self.assertLess(r.value_oos, self.base.value_oos)
        cap = cfg.e_min + r.kappa * (cfg.e_max - cfg.e_min)
        self.assertTrue(np.all(r.forward.soc <= cap[None, :, None] + 1e-5))
        self.assertTrue(np.all(r.forward.soc >= cfg.e_min - 1e-5))
        np.testing.assert_allclose(r.forward.soc[:, :, -1], r.soc_mwh(r.forward.soc_index[:, 1:]), atol=1e-5)
        s = r.summary()
        self.assertLess(s["usable_capacity_end_mwh"], s["usable_capacity_start_mwh"])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = r.save(tmp, plots=True)
            import pandas as pd

            daily = pd.read_parquet(Path(run_dir) / "daily.parquet")
            np.testing.assert_allclose(daily["capacity_factor"], r.kappa)

    def test_wear_cost(self):
        k = 20.0
        cfg = dataclasses.replace(CONFIG, wear_cost_per_mwh=k)
        r = run_engine(cfg, self.p, SETTINGS)
        s = r.summary()
        self.assertLess(np.mean(r.forward.cycles), np.mean(self.base.forward.cycles))
        wear = k * r.forward.discharge.sum(axis=(1, 2), dtype=np.float64).mean()
        self.assertAlmostEqual(s["wear_cost_eur"], wear, delta=1e-6 * max(1.0, wear))
        self.assertAlmostEqual(s["value_before_wear_cost_eur"], s["value_oos_eur"] + wear, delta=1e-6)
        gross = np.einsum("ndt,ndt->nd", self.p.da[self.p.n_paths // 2:].astype(np.float64),
                          r.forward.discharge - r.forward.charge)
        np.testing.assert_allclose(r.forward.cash, gross - k * r.forward.discharge.sum(axis=2), atol=1e-3)
        self.assertLessEqual(r.pf_value.mean(), self.base.pf_value.mean())

    def test_solvers_agree_with_degradation(self):
        p = prices(n_paths=6, n_days=40, seed=4, start="2027-05-15")
        cfg = dataclasses.replace(CONFIG, capacity_fade_per_year=0.3, wear_cost_per_mwh=5.0)
        ref = run_engine(cfg, p, SETTINGS)
        lib = run_engine(cfg, p, dataclasses.replace(SETTINGS, daily_solver="library"))
        np.testing.assert_allclose(lib.pf_value, ref.pf_value, atol=1e-6)
        np.testing.assert_allclose(lib.forward.cash, ref.forward.cash, atol=1e-6)

    def test_cache_key_includes_degradation(self):
        p = prices(n_paths=6, n_days=10, seed=8)
        with tempfile.TemporaryDirectory() as tmp:
            settings = dataclasses.replace(SETTINGS, value_cache_dir=tmp)
            run_engine(CONFIG, p, settings)
            run_engine(dataclasses.replace(CONFIG, wear_cost_per_mwh=3.0), p, settings)
            run_engine(CONFIG, p, settings, capacity_factor=0.9)
            self.assertEqual(len(list(Path(tmp).glob("*.npy"))), 3)
            run_engine(CONFIG, p, settings, capacity_factor=0.9)
            self.assertEqual(len(list(Path(tmp).glob("*.npy"))), 3)


if __name__ == "__main__":
    unittest.main()
