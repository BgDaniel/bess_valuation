"""Convexity of the daily revenue along principal price directions."""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bess_valuation import BatteryConfig, EngineSettings, PriceGeneratorConfig, generate_prices, run_engine
from bess_valuation.convexity import convexity_analysis
from bess_valuation.daily import soc_grid

CONFIG = BatteryConfig(name="convexity_test", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, dt=1.0,
                       soc_init=1.0, soc_grid_size=5)


class TestConvexity(unittest.TestCase):
    def test_principal_directions_and_convexity(self):
        gen = PriceGeneratorConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "prices_synthetic.yaml")
        p = generate_prices(gen, n_paths=80, start="2027-06-01", n_days=2, dt=1.0, seed=3)
        prices = p.da[:, 1].astype(np.float64)
        grid = soc_grid(CONFIG)
        res = {}
        for solver in ("dp", "library"):
            r = convexity_analysis(prices, CONFIG, grid, solver, 1, p.dates[1], n_points=21)
            res[solver] = r
            self.assertTrue(r.convex, r.min_second_difference)
            np.testing.assert_allclose(np.linalg.norm(r.directions, axis=1), 1.0)
            np.testing.assert_allclose(r.directions @ r.directions.T, np.eye(3), atol=1e-10)
            self.assertTrue(np.all(np.diff(r.explained) <= 1e-12))
            np.testing.assert_allclose(r.scores.std(axis=1, ddof=1), 1.0, rtol=1e-6)
            self.assertEqual(r.values.shape, (3, 21, 3))
        np.testing.assert_allclose(res["dp"].values, res["library"].values, atol=1e-6)

    def test_wear_cost_keeps_convexity(self):
        rng = np.random.default_rng(0)
        prices = 80.0 + rng.normal(0.0, 40.0, size=(50, 24)).cumsum(axis=1) / 3.0
        r = convexity_analysis(prices, dataclasses.replace(CONFIG, wear_cost_per_mwh=15.0), soc_grid(CONFIG),
                               "dp", 0, np.datetime64("2027-01-01"), cycle_cost=15.0)
        self.assertTrue(r.convex)

    def test_engine_reports_convexity(self):
        p = generate_prices(PriceGeneratorConfig(), n_paths=10, start="2027-03-01", n_days=6, dt=1.0, seed=2)
        r = run_engine(CONFIG, p, EngineSettings(library_dir=None, value_cache_dir=None, convexity_day=4))
        self.assertEqual(r.convexity.day, 4)
        s = r.summary()
        self.assertTrue(s["convexity_ok"])
        self.assertEqual(s["convexity_date"], "2027-03-05")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = r.save(tmp, plots=True)
            frame = pd.read_parquet(Path(run_dir) / "convexity.parquet")
            self.assertEqual(len(frame), r.convexity.values.size)
            self.assertTrue((Path(run_dir) / "figures" / "15_convexity.png").exists())


if __name__ == "__main__":
    unittest.main()
