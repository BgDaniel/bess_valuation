"""Dynamic programming over SoC versus the simplex and HiGHS MILP."""

import dataclasses
import unittest

import numpy as np

from bess_valuation import BatteryConfig, PriceGeneratorConfig, generate_prices
from bess_valuation.daily import daily_value_table, soc_grid
from bess_valuation.dp import dp_daily_schedules, dp_daily_values
from bess_valuation.library import VertexLibrary
from bess_valuation.polytope import build_daily_lp, lp_cost
from bess_valuation.simplex import solve_daily
from tests.test_simplex import highs_milp

CONFIG_QH = BatteryConfig(name="dp_qh", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, dt=0.25,
                          soc_init=1.0, soc_grid_size=5)
CONFIG_8H = BatteryConfig(name="dp_8h", e_max=8.0, p_charge_max=1.0, p_discharge_max=1.0, dt=0.25,
                          eta_charge=0.92, eta_discharge=0.93, soc_init=4.0, soc_grid_size=5)


def summer_prices(cfg, n_days=12, seed=3):
    p = generate_prices(PriceGeneratorConfig(), n_paths=1, start="2027-06-10", n_days=n_days,
                        dt=cfg.dt, seed=seed)
    return p.da[0].astype(np.float64)


class TestDP(unittest.TestCase):
    def test_values_match_simplex(self):
        for cfg in (CONFIG_QH, CONFIG_8H):
            prices = summer_prices(cfg)
            self.assertTrue((prices < 0).any())  # negative prices: case split exercised
            grid = soc_grid(cfg)
            F = dp_daily_values(cfg, prices, grid)
            for i, s0 in enumerate(grid):
                for j, s1 in enumerate(grid):
                    ref = solve_daily(build_daily_lp(cfg, s0, s1), prices, allow_simultaneous=False)
                    np.testing.assert_allclose(F[:, i, j], ref.revenue, atol=1e-6, err_msg=cfg.name)

    def test_values_match_milp_hourly(self):
        cfg = dataclasses.replace(CONFIG_QH, dt=1.0)
        rng = np.random.default_rng(4)
        prices = rng.normal(-10.0, 40.0, size=(6, 24))
        grid = soc_grid(cfg)
        F = dp_daily_values(cfg, prices, grid)
        for i, j in ((0, 4), (2, 2), (4, 0), (1, 3)):
            lp = build_daily_lp(cfg, grid[i], grid[j])
            for q in range(len(prices)):
                self.assertAlmostEqual(F[q, i, j], -highs_milp(lp, lp_cost(prices[q])), delta=1e-6)

    def test_cycle_cost(self):
        prices = summer_prices(CONFIG_QH, n_days=4)
        grid = soc_grid(CONFIG_QH)
        F = dp_daily_values(CONFIG_QH, prices, grid, cycle_cost=15.0)
        ref = solve_daily(build_daily_lp(CONFIG_QH, grid[2], grid[2]), prices, cycle_cost=15.0,
                          allow_simultaneous=False)
        np.testing.assert_allclose(F[:, 2, 2], ref.revenue, atol=1e-6)

    def test_schedules(self):
        cfg = CONFIG_QH
        prices = summer_prices(cfg)
        grid = soc_grid(cfg)
        s_start = grid[np.arange(len(prices)) % grid.size]
        s_end = grid[(np.arange(len(prices)) * 2) % grid.size]
        rev, c, d = dp_daily_schedules(cfg, prices, s_start, s_end)
        F = dp_daily_values(cfg, prices, grid)
        idx = np.arange(len(prices))
        np.testing.assert_allclose(rev, F[idx, idx % grid.size, (idx * 2) % grid.size], atol=1e-6)
        np.testing.assert_allclose(rev, np.einsum("qt,qt->q", prices, d - c), atol=1e-9)
        soc = s_start[:, None] + np.cumsum(cfg.eta_charge * c - d / cfg.eta_discharge, axis=1)
        np.testing.assert_allclose(soc[:, -1], s_end, atol=1e-9)
        self.assertTrue(np.all(soc >= cfg.e_min - 1e-9) and np.all(soc <= cfg.e_max + 1e-9))
        self.assertTrue(np.all(np.minimum(c, d) <= 1e-12))
        self.assertTrue(np.all(c <= 0.25 + 1e-12) and np.all(d <= 0.25 + 1e-12))

    def test_all_solvers_same_table(self):
        cfg = CONFIG_QH
        prices = summer_prices(cfg, n_days=6)
        grid = soc_grid(cfg)
        ref = daily_value_table("dp", cfg, prices, grid)
        for solver in ("library", "cold", "cluster"):
            F = daily_value_table(solver, cfg, prices, grid, library=VertexLibrary.empty(cfg), n_clusters=3)
            np.testing.assert_allclose(F, ref, atol=1e-6, err_msg=solver)

    def test_rejects_simultaneous(self):
        cfg = dataclasses.replace(CONFIG_QH, allow_simultaneous_charge_discharge=True)
        with self.assertRaises(ValueError):
            dp_daily_values(cfg, summer_prices(cfg, n_days=1), soc_grid(cfg))


if __name__ == "__main__":
    unittest.main()
