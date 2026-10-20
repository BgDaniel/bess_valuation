"""Numba simplex versus HiGHS (scipy) on the daily dispatch LP."""

import unittest

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from bess_valuation import BatteryConfig
from bess_valuation.polytope import build_daily_lp, lp_cost, soc_path, split_schedule
from bess_valuation.price_generator import PriceGeneratorConfig, generate_prices
from bess_valuation.simplex import OPTIMAL, solve_batch, solve_daily

CONFIG_QH = BatteryConfig(name="test_qh", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                          eta_charge=0.95, eta_discharge=0.95, dt=0.25, soc_init=1.0)
CONFIG_H = BatteryConfig(name="test_h", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                         eta_charge=0.95, eta_discharge=0.95, dt=1.0, soc_init=1.0)
CONFIG_LONG = BatteryConfig(name="test_8h", e_max=8.0, p_charge_max=1.0, p_discharge_max=1.0,
                            eta_charge=0.92, eta_discharge=0.93, dt=0.25, soc_init=4.0)


def highs_lp(lp, cost):
    res = linprog(cost, A_eq=lp.A, b_eq=lp.b, bounds=list(zip(lp.lo, lp.up)), method="highs")
    assert res.status == 0, res.message
    return res.fun


def highs_milp(lp, cost):
    """Same LP plus binaries z_t with c_t <= c_max z_t and d_t <= d_max (1 - z_t)."""
    n_t, n = lp.n_periods, lp.n_vars
    c_max, d_max = lp.up[0], lp.up[n_t]
    big = np.zeros((2 * n_t, n + n_t))
    for t in range(n_t):
        big[t, t], big[t, n + t] = 1.0, -c_max
        big[n_t + t, n_t + t], big[n_t + t, n + t] = 1.0, d_max
    eq = np.hstack([lp.A, np.zeros((n_t, n_t))])
    constraints = [LinearConstraint(eq, lp.b, lp.b),
                   LinearConstraint(big, -np.inf, np.r_[np.zeros(n_t), np.full(n_t, d_max)])]
    bounds = Bounds(np.r_[lp.lo, np.zeros(n_t)], np.r_[lp.up, np.ones(n_t)])
    integrality = np.r_[np.zeros(n), np.ones(n_t)]
    res = milp(np.r_[cost, np.zeros(n_t)], constraints=constraints, bounds=bounds,
               integrality=integrality, options={"mip_rel_gap": 1e-9})
    assert res.status == 0, res.message
    return res.fun


class SimplexTestCase(unittest.TestCase):
    def assert_feasible(self, lp, x, tol=1e-7):
        np.testing.assert_allclose(lp.A @ x, lp.b, atol=tol)
        self.assertTrue(np.all(x >= lp.lo - tol) and np.all(x <= lp.up + tol))
        c, d = split_schedule(x, lp.n_periods)
        soc = soc_path(c, d, lp.s_start, lp.eta_charge, lp.eta_discharge)
        self.assertAlmostEqual(soc[-1], lp.s_end, delta=tol)

    def check_against_highs(self, lp, prices, cycle_cost=0.0):
        costs = lp_cost(prices, cycle_cost)
        res = solve_batch(lp, costs)
        self.assertTrue(np.all(res.status == OPTIMAL), res.status)
        for k in range(len(costs)):
            ref = highs_lp(lp, costs[k])
            self.assertAlmostEqual(res.objective[k], ref, delta=1e-6 * max(1.0, abs(ref)))
            self.assert_feasible(lp, res.x[k])
        return res


class TestSimplexVsHighs(SimplexTestCase):
    def test_random_prices_hourly(self):
        rng = np.random.default_rng(1)
        lp = build_daily_lp(CONFIG_H, 1.0, 1.0)
        self.check_against_highs(lp, rng.normal(80.0, 30.0, size=(40, 24)))

    def test_random_prices_quarter_hourly(self):
        rng = np.random.default_rng(2)
        lp = build_daily_lp(CONFIG_QH, 1.0, 1.0)
        self.check_against_highs(lp, rng.normal(80.0, 30.0, size=(30, 96)))

    def test_soc_start_differs_from_end(self):
        rng = np.random.default_rng(3)
        prices = rng.normal(80.0, 30.0, size=(5, 96))
        for s_start, s_end in ((0.0, 2.0), (2.0, 0.0), (0.25, 1.75), (1.5, 0.5), (0.0, 0.0), (2.0, 2.0)):
            with self.subTest(s_start=s_start, s_end=s_end):
                self.check_against_highs(build_daily_lp(CONFIG_QH, s_start, s_end), prices)

    def test_negative_prices(self):
        rng = np.random.default_rng(4)
        prices = rng.normal(0.0, 40.0, size=(30, 96))
        prices[:, 40:56] -= 80.0
        self.check_against_highs(build_daily_lp(CONFIG_QH, 1.0, 1.0), prices)

    def test_degenerate_prices(self):
        rng = np.random.default_rng(5)
        cases = [np.zeros(96), np.full(96, 50.0), np.repeat([30.0, 90.0, 30.0, 90.0], 24),
                 np.round(rng.normal(80.0, 30.0, size=96), -1), np.tile([10.0, 10.0, 100.0, 100.0], 24)]
        for s_start, s_end in ((1.0, 1.0), (0.0, 2.0), (2.0, 2.0)):
            with self.subTest(s_start=s_start, s_end=s_end):
                self.check_against_highs(build_daily_lp(CONFIG_QH, s_start, s_end), np.array(cases))

    def test_cycle_cost_shifted_prices(self):
        rng = np.random.default_rng(6)
        prices = rng.normal(80.0, 30.0, size=(20, 96))
        for cycle_cost in (5.0, 25.0, 200.0):
            with self.subTest(cycle_cost=cycle_cost):
                self.check_against_highs(build_daily_lp(CONFIG_QH, 0.5, 1.5), prices, cycle_cost)

    def test_half_day_horizon(self):
        rng = np.random.default_rng(7)
        prices = rng.normal(80.0, 30.0, size=(20, 48))
        for s_start in (0.0, 0.37, 1.234, 2.0):
            with self.subTest(s_start=s_start):
                lp = build_daily_lp(CONFIG_QH, s_start, 1.0, horizon="half")
                self.assertEqual(lp.n_periods, 48)
                self.check_against_highs(lp, prices)

    def test_long_duration_storage(self):
        rng = np.random.default_rng(8)
        prices = rng.normal(80.0, 30.0, size=(15, 96))
        for s_start, s_end in ((4.0, 4.0), (0.0, 8.0), (8.0, 1.0)):
            with self.subTest(s_start=s_start, s_end=s_end):
                self.check_against_highs(build_daily_lp(CONFIG_LONG, s_start, s_end), prices)

    def test_simulated_prices(self):
        gen = generate_prices(PriceGeneratorConfig(), n_paths=4, start="2027-06-01", n_days=10,
                              dt=0.25, seed=11)
        prices = np.concatenate([gen.da.reshape(-1, 96), gen.ida2.reshape(-1, 96)]).astype(np.float64)
        self.check_against_highs(build_daily_lp(CONFIG_QH, 1.0, 1.0), prices)

    def test_infeasible_soc_pair(self):
        tiny = BatteryConfig(name="tiny", e_max=10.0, p_charge_max=0.1, p_discharge_max=0.1, dt=1.0)
        res = solve_batch(build_daily_lp(tiny, 0.0, 10.0), lp_cost(np.ones(24)))
        self.assertNotEqual(res.status[0], OPTIMAL)


class TestWarmStart(SimplexTestCase):
    def test_warm_start_on_perturbed_prices(self):
        rng = np.random.default_rng(9)
        lp = build_daily_lp(CONFIG_QH, 1.0, 1.0)
        base = rng.normal(80.0, 30.0, size=96)
        first = solve_batch(lp, lp_cost(base))
        perturbed = base + rng.normal(0.0, 2.0, size=(20, 96))
        warm = solve_batch(lp, lp_cost(perturbed), basis=first.basis[0], var_status=first.var_status[0])
        cold = solve_batch(lp, lp_cost(perturbed))
        np.testing.assert_allclose(warm.objective, cold.objective, rtol=1e-9, atol=1e-7)
        self.assertTrue(np.all(warm.warm))
        self.assertLess(warm.iterations.mean(), cold.iterations.mean())

    def test_warm_basis_infeasible_for_new_rhs_falls_back(self):
        rng = np.random.default_rng(10)
        prices = rng.normal(80.0, 30.0, size=96)
        first = solve_batch(build_daily_lp(CONFIG_QH, 0.0, 0.0), lp_cost(prices))
        other = build_daily_lp(CONFIG_QH, 2.0, 0.0)
        res = solve_batch(other, lp_cost(prices), basis=first.basis[0], var_status=first.var_status[0])
        self.assertEqual(res.status[0], OPTIMAL)
        self.assertAlmostEqual(res.objective[0], highs_lp(other, lp_cost(prices)), delta=1e-6)


class TestNoSimultaneous(SimplexTestCase):
    def test_case_split_matches_milp(self):
        rng = np.random.default_rng(12)
        prices = rng.normal(-20.0, 40.0, size=(12, 24))
        for s_start, s_end in ((2.0, 2.0), (1.0, 1.0), (0.0, 2.0)):
            lp = build_daily_lp(CONFIG_H, s_start, s_end)
            sol = solve_daily(lp, prices, allow_simultaneous=False)
            relaxed = solve_daily(lp, prices, allow_simultaneous=True)
            for k in range(len(prices)):
                with self.subTest(s_start=s_start, s_end=s_end, k=k):
                    ref = -highs_milp(lp, lp_cost(prices[k]))
                    self.assertAlmostEqual(sol.revenue[k], ref, delta=1e-6 * max(1.0, abs(ref)))
                    self.assertLessEqual(sol.revenue[k], relaxed.revenue[k] + 1e-9)
                    self.assertTrue(np.all(np.minimum(sol.charge[k], sol.discharge[k]) < 1e-7))
                    self.assert_feasible(lp, sol.x[k])
            self.assertGreater(sol.n_split.sum(), 0)  # the split was actually exercised

    def test_positive_prices_need_no_split(self):
        rng = np.random.default_rng(13)
        lp = build_daily_lp(CONFIG_QH, 1.0, 1.0)
        sol = solve_daily(lp, rng.normal(80.0, 20.0, size=(10, 96)).clip(1.0), allow_simultaneous=False)
        self.assertEqual(sol.n_split.sum(), 0)


if __name__ == "__main__":
    unittest.main()
