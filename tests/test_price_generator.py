"""Tests for the synthetic price generator and the price containers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bess_valuation import PriceGeneratorConfig, PriceScenarios, generate_prices
from bess_valuation.price_generator import make_calendar, plot_generator

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "prices_synthetic.yaml"


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PriceGeneratorConfig.from_yaml(CONFIG_PATH)
        cls.prices = generate_prices(cls.cfg, n_paths=400, start="2027-01-01", n_days=365, dt=0.25, seed=42)

    def test_shapes_and_nan_pattern(self):
        p = self.prices
        for name in ("da", "ida1", "ida2", "ida3"):
            self.assertEqual(p.stage(name).shape, (400, 365, 96))
        self.assertTrue(np.all(np.isnan(p.ida3[:, :, :48])))
        self.assertFalse(np.any(np.isnan(p.ida3[:, :, 48:])))
        for name in ("da", "ida1", "ida2"):
            self.assertFalse(np.any(np.isnan(p.stage(name))))
        self.assertEqual(p.X.shape, (400, 365, 3))
        self.assertEqual(p.conditional_mean.shape, (400, 365, 96))

    def test_hourly_resolution(self):
        p = generate_prices(self.cfg, n_paths=5, start="2027-01-01", n_days=3, dt=1.0, seed=1)
        self.assertEqual(p.da.shape, (5, 3, 24))
        self.assertTrue(np.all(np.isnan(p.ida3[:, :, :12])))

    def test_reproducible(self):
        a = generate_prices(self.cfg, n_paths=20, start="2027-03-01", n_days=10, dt=0.25, seed=7)
        b = generate_prices(self.cfg, n_paths=20, start="2027-03-01", n_days=10, dt=0.25, seed=7)
        c = generate_prices(self.cfg, n_paths=20, start="2027-03-01", n_days=10, dt=0.25, seed=8)
        for name in ("da", "ida1", "ida2", "ida3"):
            np.testing.assert_array_equal(a.stage(name), b.stage(name))
        self.assertFalse(np.allclose(a.da, c.da))

    def test_conditional_mean_matches_monte_carlo(self):
        """Residuals p_{d+1} - mu(X_d) have zero mean and are uncorrelated with X_d."""
        p = self.prices
        resid = p.da[:, 1:].astype(np.float64) - p.conditional_mean[:, :-1]
        daily = resid.mean(axis=2)  # (N, D-1)
        n = daily.size
        self.assertLess(abs(daily.mean()), 4.0 * daily.std() / np.sqrt(n))
        X = p.X[:, :-1].reshape(-1, 3).astype(np.float64)
        r = daily.reshape(-1)
        for k in range(3):
            corr = np.corrcoef(X[:, k], r)[0, 1]
            self.assertLess(abs(corr), 0.02, f"residual correlated with X[{k}]")
        # per period too: no systematic intraday bias
        per_period = resid.mean(axis=(0, 1))
        self.assertLess(np.abs(per_period).max(), 0.6)

    def test_spreads_martingale_and_growing(self):
        p = self.prices
        half = 48
        variances = []
        for name in ("ida1", "ida2", "ida3"):
            spread = (p.stage(name) - p.da)[..., half:].astype(np.float64)
            self.assertLess(abs(spread.mean()), 0.1)
            variances.append(spread.var())
        self.assertLess(variances[0], variances[1])
        self.assertLess(variances[1], variances[2])

    def test_bias_shifts_spreads(self):
        cfg = PriceGeneratorConfig(ida_bias=(1.0, 1.0, 1.0))
        p = generate_prices(cfg, n_paths=50, start="2027-01-01", n_days=20, dt=0.25, seed=3)
        self.assertAlmostEqual(float(np.nanmean(p.ida3 - p.da)), 3.0, delta=0.3)

    def test_midday_dip_deeper_in_summer(self):
        p = self.prices
        month = p.dates.astype("datetime64[M]").astype(int) % 12 + 1
        def dip(months):
            sel = np.isin(month, months)
            profile = p.da[:, sel].mean(axis=(0, 1))
            return profile[[32, 33, 34, 35, 76, 77, 78, 79]].mean() - profile[48:56].mean()
        self.assertGreater(dip((6, 7, 8)), dip((12, 1, 2)) + 10.0)
        summer = np.isin(month, (6, 7, 8))
        self.assertGreater((p.da[:, summer, 48:56] < 0).mean(), (p.da[:, ~summer, 48:56] < 0).mean())

    def test_calendar(self):
        cal = make_calendar("2027-01-01", 7, self.cfg.holidays)
        self.assertTrue(cal.holiday[0])  # New Year
        np.testing.assert_array_equal(cal.weekend, [False, True, True, False, False, False, False])
        self.assertLess(cal.season[0], 0.01)

    def test_plots_written(self):
        small = generate_prices(self.cfg, n_paths=30, start="2027-01-01", n_days=60, dt=0.25, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_generator(small, Path(tmp))
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


class TestScenarioIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = generate_prices(PriceGeneratorConfig(), n_paths=6, start="2027-02-01", n_days=4,
                                     dt=1.0, seed=2)

    def roundtrip(self, fmt):
        with tempfile.TemporaryDirectory() as tmp:
            self.prices.save(tmp, fmt=fmt)
            return PriceScenarios.from_files(tmp)

    def test_roundtrips(self):
        for fmt in ("npz", "parquet", "csv"):
            with self.subTest(fmt=fmt):
                back = self.roundtrip(fmt)
                for name in ("da", "ida1", "ida2", "ida3"):
                    np.testing.assert_allclose(back.stage(name), self.prices.stage(name), rtol=1e-6)
                np.testing.assert_allclose(back.X, self.prices.X, rtol=1e-6)
                np.testing.assert_array_equal(back.dates, self.prices.dates)
                self.assertEqual(back.dt, 1.0)
                self.assertEqual(back.meta["seed"], 2)

    def test_split_is_disjoint(self):
        a, b = self.prices.split(4)
        self.assertEqual((a.n_paths, b.n_paths), (4, 2))
        np.testing.assert_array_equal(b.da, self.prices.da[4:])

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            PriceScenarios(da=np.zeros((2, 3, 24)), ida1=np.zeros((2, 3, 24)), ida2=np.zeros((2, 3, 24)),
                           ida3=np.zeros((2, 3, 23)), dates=np.arange(3).astype("datetime64[D]"), dt=1.0)


if __name__ == "__main__":
    unittest.main()
