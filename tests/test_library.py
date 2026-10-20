"""Vertex library: lookup + certificate versus direct solves, persistence, daily lookup."""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bess_valuation import BatteryConfig
from bess_valuation.daily import daily_schedules, daily_values, soc_grid
from bess_valuation.library import (VertexLibrary, certify, library_stats, load_library,
                                    load_or_create, save_library)
from bess_valuation.polytope import build_daily_lp, lp_cost
from bess_valuation.price_generator import PriceGeneratorConfig, generate_prices
from bess_valuation.simplex import _basis_inverse, _purify, solve_batch, solve_daily
from tests.test_simplex import highs_lp, highs_milp

CONFIG_H = BatteryConfig(name="lib_h", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                         eta_charge=0.95, eta_discharge=0.95, dt=1.0, soc_init=1.0,
                         soc_grid_size=3)
CONFIG_H_SIM = dataclasses.replace(CONFIG_H, allow_simultaneous_charge_discharge=True)


def small_prices(n_paths=3, n_days=20, dt=1.0, seed=5):
    gen = PriceGeneratorConfig()
    p = generate_prices(gen, n_paths=n_paths, start="2027-06-01", n_days=n_days, dt=dt, seed=seed)
    return p.da.reshape(-1, p.da.shape[-1]).astype(np.float64)


class TestBasisTools(unittest.TestCase):
    def test_structured_inverse_matches_dense(self):
        rng = np.random.default_rng(0)
        lp = build_daily_lp(CONFIG_H, 0.5, 1.5)
        res = solve_batch(lp, lp_cost(rng.normal(60.0, 40.0, size=(20, 24))))
        for basis in res.basis:
            Binv = _basis_inverse(lp.A, basis)
            np.testing.assert_allclose(Binv @ lp.A[:, basis], np.eye(24), atol=1e-12)

    def test_purify_gives_basic_feasible_solution(self):
        rng = np.random.default_rng(1)
        lp = build_daily_lp(CONFIG_H, 1.0, 1.0)
        for _ in range(20):
            # random feasible point: average of two vertices (interior on an edge or face)
            res = solve_batch(lp, lp_cost(rng.normal(60.0, 40.0, size=(2, 24))))
            x = 0.5 * (res.x[0] + res.x[1])
            ok, basis, status = _purify(lp.A, lp.lo, lp.up, x, 24, 1e-9)
            self.assertTrue(ok)
            np.testing.assert_allclose(lp.A @ x, lp.b, atol=1e-9)
            self.assertTrue(np.all(x >= lp.lo - 1e-9) and np.all(x <= lp.up + 1e-9))
            self.assertEqual(len(set(basis.tolist())), 24)
            self.assertGreater(abs(np.linalg.det(lp.A[:, basis])), 1e-12)
            nonbasic = np.setdiff1d(np.arange(lp.n_vars), basis)
            at_bound = np.minimum(np.abs(x - lp.lo), np.abs(x - lp.up))[nonbasic]
            self.assertLess(at_bound.max(), 1e-9)


class TestLookup(unittest.TestCase):
    def check_lookup(self, config, reference):
        prices = small_prices()
        library = VertexLibrary.empty(config)
        for s_start, s_end in ((1.0, 1.0), (0.0, 2.0), (2.0, 0.5)):
            lp = build_daily_lp(config, s_start, s_end)
            ref = np.array([-reference(lp, lp_cost(p)) for p in prices])
            for rnd in range(2):  # the second round is answered from the library
                with self.subTest(s_start=s_start, s_end=s_end, round=rnd):
                    idx, rev, c, d = library.lookup(prices, s_start, s_end, seed=rnd,
                                                    return_schedule=True)
                    np.testing.assert_allclose(rev, ref, atol=1e-6, rtol=1e-9)
                    np.testing.assert_allclose(rev, np.einsum("qt,qt->q", prices, d - c), atol=1e-9)
            pair = library.pair(s_start, s_end)
            self.assertEqual(pair.stats.queries, 2 * len(prices))
            # second round: every price vector is certified, except - without simultaneous
            # charge/discharge - days with negative prices (Lagrangian duality gap)
            expected = len(prices)
            if not config.allow_simultaneous_charge_discharge:
                expected = int(np.sum(np.all(prices >= 0.0, axis=1)))
                self.assertGreater(expected, 0)
            self.assertGreaterEqual(pair.stats.hot + pair.stats.full, expected)
        return library

    def test_lp_mode_matches_highs(self):
        self.check_lookup(CONFIG_H_SIM, highs_lp)

    def test_restricted_mode_matches_milp(self):
        library = self.check_lookup(CONFIG_H, highs_milp)
        stats = library_stats(library)
        self.assertTrue(stats["restricted"])
        self.assertGreater(stats["case_split"], 0)

    def test_degenerate_prices(self):
        # flat and two-level prices: many optimal vertices, all with the same value
        prices = np.vstack([np.full(24, 50.0), np.r_[np.full(12, 20.0), np.full(12, 80.0)],
                            np.r_[np.full(12, -10.0), np.full(12, 80.0)]])
        library = VertexLibrary.empty(CONFIG_H)
        lp = build_daily_lp(CONFIG_H, 1.0, 1.0)
        ref = np.array([-highs_milp(lp, lp_cost(p)) for p in prices])
        for rnd in range(2):
            _, rev = library.lookup(prices, 1.0, 1.0, seed=rnd)
            np.testing.assert_allclose(rev, ref, atol=1e-6)

    def test_certificate_rejects_suboptimal_vertex(self):
        rng = np.random.default_rng(3)
        prices = rng.normal(60.0, 30.0, size=(2, 24))
        library = VertexLibrary.empty(CONFIG_H_SIM)
        idx, rev = library.lookup(prices, 1.0, 1.0)
        pair = library.pair(1.0, 1.0)
        lp = pair.lp
        n_t = lp.n_periods
        for q in range(2):
            own = certify(pair.codes[idx[q]], prices[q], 0.0, lp.eta_charge, lp.eta_discharge,
                          lp.up[0], lp.up[n_t], False, 1e-7)
            self.assertTrue(own)
        if idx[0] != idx[1]:
            other = certify(pair.codes[idx[1]], prices[0], 0.0, lp.eta_charge, lp.eta_discharge,
                            lp.up[0], lp.up[n_t], False, 1e-7)
            better = rev[0] > np.dot(prices[0], pair.net[idx[1]]) + 1e-6
            self.assertEqual(other, not better)

    def test_capped_library_stays_exact(self):
        prices = small_prices(n_paths=2, n_days=15)
        library = VertexLibrary.empty(CONFIG_H, max_vertices=5)
        _, rev = library.lookup(prices, 1.0, 1.0)
        lp = build_daily_lp(CONFIG_H, 1.0, 1.0)
        ref = solve_daily(lp, prices, allow_simultaneous=False).revenue
        np.testing.assert_allclose(rev, ref, atol=1e-6)
        pair = library.pair(1.0, 1.0)
        self.assertEqual(pair.size, 5)
        self.assertGreater(pair.stats.uncached, 0)

    def test_neighbours_and_edges(self):
        prices = small_prices(n_paths=2, n_days=10)
        library = VertexLibrary.empty(CONFIG_H_SIM)
        library.lookup(prices, 1.0, 1.0)
        pair = library.pair(1.0, 1.0)
        rays = pair.edge_directions(0)
        self.assertEqual(rays.shape, (2 * 24 - 1, 3 * 24 - 1))
        np.testing.assert_allclose(pair.lp.A @ rays.T, 0.0, atol=1e-12)
        nb = pair.neighbour_indices(0)
        self.assertTrue(np.all((nb >= 0) & (nb < pair.size)))
        self.assertNotIn(0, nb.tolist())


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        prices = small_prices(n_paths=2, n_days=10)
        library = VertexLibrary.empty(CONFIG_H_SIM)
        _, rev = library.lookup(prices, 1.0, 1.0)
        library.lookup(prices, 0.0, 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_library(library, Path(tmp) / "lib.npz")
            loaded = load_library(path, CONFIG_H_SIM)
            self.assertEqual(sorted(loaded.pairs), sorted(library.pairs))
            for key, pair in library.pairs.items():
                other = loaded.pairs[key]
                k = pair.size
                self.assertEqual(other.size, k)
                np.testing.assert_array_equal(other.codes, pair.codes[:k])
                np.testing.assert_array_equal(other.charge, pair.charge[:k])
                np.testing.assert_array_equal(other.basis, pair.basis[:k])
                np.testing.assert_array_equal(other.hits, pair.hits[:k])
                self.assertEqual(other.stats.queries, pair.stats.queries)
                self.assertEqual(other.stats.discovery, pair.stats.discovery)
            simplex_before = loaded.pair(1.0, 1.0).stats.simplex
            _, rev2 = loaded.lookup(prices, 1.0, 1.0)
            np.testing.assert_allclose(rev2, rev, atol=1e-9)
            self.assertEqual(loaded.pair(1.0, 1.0).stats.simplex, simplex_before)
            # wrong config is rejected, load_or_create picks the file by config hash
            other_cfg = dataclasses.replace(CONFIG_H_SIM, e_max=3.0)
            with self.assertRaises(ValueError):
                load_library(path, other_cfg)
            path.rename(Path(tmp) / f"{CONFIG_H_SIM.config_hash}_full.npz")
            self.assertEqual(load_or_create(tmp, CONFIG_H_SIM).pair(1.0, 1.0).size,
                             library.pair(1.0, 1.0).size)
            self.assertEqual(len(load_or_create(tmp, other_cfg).pairs), 0)

    def test_stats(self):
        prices = small_prices(n_paths=2, n_days=10)
        library = VertexLibrary.empty(CONFIG_H_SIM)
        library.lookup(prices, 1.0, 1.0)
        library.lookup(prices, 1.0, 1.0, seed=1)
        stats = library_stats(library)
        self.assertEqual(stats["queries"], 2 * len(prices))
        self.assertGreaterEqual(stats["certified_share"], 0.5)
        self.assertTrue(0.0 <= stats["good_turing"] <= 1.0)
        pair = stats["pairs"][0]
        self.assertEqual(pair["certified_hot"] + pair["certified_full"] + pair["simplex"],
                         pair["queries"])


class TestDaily(unittest.TestCase):
    def test_values_and_schedules(self):
        gen = PriceGeneratorConfig()
        p = generate_prices(gen, n_paths=2, start="2027-03-01", n_days=4, dt=1.0, seed=2)
        prices = p.da.astype(np.float64)
        grid = soc_grid(CONFIG_H)
        library = VertexLibrary.empty(CONFIG_H)
        F = daily_values(library, prices, grid)
        self.assertEqual(F.shape, (2, 4, 3, 3))
        for i, s0 in enumerate(grid):
            for j, s1 in enumerate(grid):
                lp = build_daily_lp(CONFIG_H, s0, s1)
                ref = solve_daily(lp, prices.reshape(-1, 24), allow_simultaneous=False).revenue
                np.testing.assert_allclose(F[:, :, i, j].ravel(), ref, atol=1e-6)
        flat = prices.reshape(-1, 24)
        s_start = np.full(len(flat), 1.0)
        s_end = np.where(np.arange(len(flat)) % 2 == 0, 0.0, 2.0)
        rev, c, d = daily_schedules(library, flat, s_start, s_end)
        np.testing.assert_allclose(rev, np.einsum("qt,qt->q", flat, d - c), atol=1e-9)
        soc_end = s_start + (0.95 * c - d / 0.95).sum(axis=1)
        np.testing.assert_allclose(soc_end, s_end, atol=1e-9)
        expect = np.where(s_end == 0.0, F.reshape(-1, 3, 3)[:, 1, 0], F.reshape(-1, 3, 3)[:, 1, 2])
        np.testing.assert_allclose(rev, expect, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
