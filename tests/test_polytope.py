"""Tests for the daily LP construction."""

import unittest

import numpy as np

from bess_valuation import BatteryConfig
from bess_valuation.polytope import (build_daily_lp, idx_soc, lp_cost, soc_path, soc_rhs,
                                     split_schedule)

CONFIG = BatteryConfig(name="t", e_max=2.0, e_min=0.2, p_charge_max=1.0, p_discharge_max=0.8,
                       eta_charge=0.9, eta_discharge=0.95, dt=0.25, soc_init=1.0)


class TestPolytope(unittest.TestCase):
    def test_dimensions_and_bounds(self):
        lp = build_daily_lp(CONFIG, 1.0, 1.5)
        self.assertEqual(lp.A.shape, (96, 3 * 96 - 1))
        np.testing.assert_allclose(lp.up[:96], 0.25)
        np.testing.assert_allclose(lp.up[96:192], 0.2)
        np.testing.assert_allclose(lp.lo[idx_soc(96)], 0.2)
        half = build_daily_lp(CONFIG, 1.0, 1.5, horizon="half")
        self.assertEqual(half.A.shape, (48, 143))

    def test_rows_reproduce_soc_dynamics(self):
        rng = np.random.default_rng(0)
        lp = build_daily_lp(CONFIG, 1.0, 1.0)
        c = rng.uniform(0, 0.25, 96)
        d = rng.uniform(0, 0.2, 96)
        soc = soc_path(c, d, 1.0, CONFIG.eta_charge, CONFIG.eta_discharge)
        x = np.r_[c, d, soc[1:-1]]
        np.testing.assert_allclose(lp.with_soc(1.0, soc[-1]).A @ x, soc_rhs(96, 1.0, soc[-1]), atol=1e-12)
        c2, d2 = split_schedule(x, 96)
        np.testing.assert_array_equal(c2, c)
        np.testing.assert_array_equal(d2, d)

    def test_cost_vector(self):
        g = lp_cost(np.array([[10.0, -5.0]]), cycle_cost=2.0)
        np.testing.assert_allclose(g, [[10.0, -5.0, -8.0, 7.0, 0.0]])

    def test_soc_outside_bounds_rejected(self):
        with self.assertRaises(ValueError):
            build_daily_lp(CONFIG, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
