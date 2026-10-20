r"""Vertex library: argmax lookup, optimality certificate, persistence (NPZ), Good-Turing diagnostic.

One :class:`PairLibrary` per SoC pair :math:`(s_{start}, s_{end})` and horizon caches the
vertices of the daily polytope P found so far (SPEC 4.1). A query with price vector
:math:`p` [€/MWh] proceeds as

1. **argmax** over the cached vertices of the revenue :math:`p^{\mathsf T}y_k - \lambda
   \mathbf 1^{\mathsf T}d_k` (:math:`y_k = d_k - c_k`; :math:`\lambda` the cycle cost) - first
   over the most frequently hit vertices, then over all of them via ``P @ Y.T``;
2. **certificate** that the argmax vertex is optimal for :math:`p` (below);
3. if the certificate fails: **pivot** with the own simplex, warm-started from the basis
   stored with the argmax vertex, and insert the new vertex.

The result is exact: a cached vertex is only returned with a certificate.

Certificate
-----------
With the duals :math:`y_t` of the SoC rows :math:`s_{t+1} - s_t - \eta_c c_t + d_t/\eta_d = 0`
the reduced costs of the minimisation form :math:`g = (p, -(p-\lambda), 0)` are

.. math::

    r_{c_t} = g_{c_t} + \eta_c y_t, \qquad r_{d_t} = g_{d_t} - y_t/\eta_d,
    \qquad r_{s_t} = y_t - y_{t-1}.

A vertex v (identified by its full active set: every variable at its lower bound, at
its upper bound or strictly between) is optimal iff some :math:`y` gives every
reduced cost the sign its bound requires (:math:`\ge 0` at the lower bound, :math:`\le 0`
at the upper bound, :math:`= 0` in between), i.e. iff :math:`-g` lies in the normal cone
of P at v. The conditions on :math:`c_t, d_t` restrict :math:`y_t` to an interval
:math:`[L_t, U_t]`; the condition on :math:`s_t` links :math:`y_t` to :math:`y_{t-1}` by
``=``, ``>=`` or ``<=``. Propagating the feasible interval of :math:`y_t` forward over
the day decides feasibility exactly in O(T). This does not depend on a particular
basis, so degenerate vertices (several bases, SPEC: "check all tangent-cone rays") are
handled exactly; a basis is stored only as warm start for the simplex.

Without simultaneous charge and discharge (:math:`c_t d_t = 0`) the feasible set is a
union of faces of P. A vertex v with :math:`c_t d_t = 0` is then certified by
Lagrangian duality: if some :math:`y` makes v minimise the Lagrangian
:math:`\sum_t r_{c_t} c_t + r_{d_t} d_t + \dots` over the non-convex per-period set
:math:`\{(c,0)\} \cup \{(0,d)\}` (whose minimum lies at :math:`(0,0)`,
:math:`(\bar c,0)` or :math:`(0,\bar d)`), v is optimal. For v at :math:`(\bar c, 0)`
this replaces :math:`r_d \ge 0` by :math:`r_c\bar c \le r_d\bar d`, likewise for
:math:`(0,\bar d)`; the other cases equal the LP conditions. The test is sufficient,
not necessary (duality gap); a failed test falls back to the simplex plus case split
of :func:`bess_valuation.simplex.solve_daily`, which is exact.

The number of cached vertices per pair is capped (``max_vertices``): on quarter-hourly
prices almost every price vector has its own optimal vertex, so the library mainly
supplies warm starts there; queries beyond the cap are still solved exactly.

Good-Turing: the probability that the next query needs an unseen vertex is estimated
by :math:`n_1/N` (:math:`n_1` vertices hit exactly once, :math:`N` queries).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit, prange

from bess_valuation.config import BatteryConfig
from bess_valuation.polytope import DailyLP, Horizon, build_daily_lp
from bess_valuation.simplex import AT_LOWER, BASIC, OPTIMAL, battery_start, solve_daily

logger = logging.getLogger(__name__)

AT_LOWER_BOUND = 0
AT_UPPER_BOUND = 1
BETWEEN = 2

ACTIVE_TOL = 1e-9
CERT_TOL = 1e-7
HOT_CANDIDATES = 32
FIRST_CHUNK = 64
MAX_CHUNK = 32768
FULL_CHUNK = 4096
FULL_PROBE = 64
MAX_VERTICES = 1000


# ------------------------------------------------------------------ Numba kernels


@njit(cache=True)
def active_codes(x, lo, up, tol):
    """Active set of a vertex: AT_LOWER_BOUND / AT_UPPER_BOUND / BETWEEN per variable."""
    n = x.shape[0]
    codes = np.empty(n, dtype=np.int8)
    for j in range(n):
        if x[j] <= lo[j] + tol * max(1.0, abs(lo[j])):
            codes[j] = AT_LOWER_BOUND
        elif x[j] >= up[j] - tol * max(1.0, abs(up[j])):
            codes[j] = AT_UPPER_BOUND
        else:
            codes[j] = BETWEEN
    return codes


@njit(cache=True)
def certify(codes, prices, cycle_cost, eta_c, eta_d, c_max, d_max, restricted, tol):
    """Exact optimality test of a vertex for one price vector (module docstring).

    Args:
        codes: Active set of the vertex, shape (3T-1,).
        prices: Prices p, shape (T,).
        cycle_cost: Cycle cost lambda per discharged MWh.
        eta_c, eta_d: Efficiencies.
        c_max, d_max: Upper bounds of c_t and d_t (energy per period).
        restricted: True if c_t d_t = 0 is required (Lagrangian test).
        tol: Tolerance on the dual intervals [€/MWh].

    Returns:
        True if the vertex is certified optimal.
    """
    n_t = prices.shape[0]
    k = eta_c * c_max + d_max / eta_d
    lo_prev = -np.inf
    hi_prev = np.inf
    for t in range(n_t):
        if t == 0:
            lo, hi = -np.inf, np.inf
        else:
            code_s = codes[2 * n_t + t - 1]
            if code_s == BETWEEN:
                lo, hi = lo_prev, hi_prev
            elif code_s == AT_LOWER_BOUND:  # r_s = y_t - y_{t-1} >= 0
                lo, hi = lo_prev, np.inf
            else:
                lo, hi = -np.inf, hi_prev
        gc = prices[t]
        gd = cycle_cost - prices[t]
        a = -gc / eta_c  # r_c = 0, r_c increasing in y
        b = eta_d * gd   # r_d = 0, r_d decreasing in y
        cc = codes[t]
        cd = codes[n_t + t]
        if restricted and cc != AT_LOWER_BOUND and cd != AT_LOWER_BOUND:
            return False
        if restricted and cc == AT_UPPER_BOUND and cd == AT_LOWER_BOUND:
            hi = min(hi, a, (gd * d_max - gc * c_max) / k)
        elif restricted and cc == AT_LOWER_BOUND and cd == AT_UPPER_BOUND:
            lo = max(lo, b, (gd * d_max - gc * c_max) / k)
        else:
            if cc == AT_LOWER_BOUND:
                lo = max(lo, a)
            elif cc == AT_UPPER_BOUND:
                hi = min(hi, a)
            else:
                lo = max(lo, a)
                hi = min(hi, a)
            if cd == AT_LOWER_BOUND:
                hi = min(hi, b)
            elif cd == AT_UPPER_BOUND:
                lo = max(lo, b)
            else:
                lo = max(lo, b)
                hi = min(hi, b)
        if lo > hi + tol * max(1.0, abs(lo), abs(hi)):
            return False
        lo_prev, hi_prev = lo, hi
    return True


@njit(parallel=True, cache=True)
def _lookup_candidates(prices, cycle_cost, Y, dsum, codes, cand, eta_c, eta_d, c_max, d_max,
                       restricted, tol):
    """Argmax over the candidate vertices ``cand`` plus certificate, per price vector."""
    n_q, n_t = prices.shape
    best_idx = np.empty(n_q, dtype=np.int64)
    ok = np.empty(n_q, dtype=np.bool_)
    for q in prange(n_q):
        best = -np.inf
        arg = -1
        for i in range(cand.shape[0]):
            k = cand[i]
            v = -cycle_cost * dsum[k]
            for t in range(n_t):
                v += Y[k, t] * prices[q, t]
            if v > best:
                best = v
                arg = k
        best_idx[q] = arg
        ok[q] = certify(codes[arg], prices[q], cycle_cost, eta_c, eta_d, c_max, d_max,
                        restricted, tol)
    return best_idx, ok


@njit(parallel=True, cache=True)
def _certify_rows(prices, cycle_cost, idx, codes, eta_c, eta_d, c_max, d_max, restricted, tol):
    """Certificate of vertex ``idx[q]`` for price vector q."""
    n_q = prices.shape[0]
    ok = np.empty(n_q, dtype=np.bool_)
    for q in prange(n_q):
        ok[q] = certify(codes[idx[q]], prices[q], cycle_cost, eta_c, eta_d, c_max, d_max,
                        restricted, tol)
    return ok


# ------------------------------------------------------------------ library


@dataclass
class PairStats:
    """Query counters of one pair library.

    Attributes:
        queries: Price vectors looked up.
        hot: Solved by the certificate of the argmax over the most frequent vertices.
        full: Solved by the certificate of the argmax over all cached vertices.
        simplex: Needed simplex pivots (certificate failed).
        split: Of those, needed the case split (c_t d_t = 0).
        new: Simplex solutions that were new vertices (cached or not).
        uncached: New vertices not inserted because the library was full.
        discovery: (queries so far, distinct vertices so far) after each processed chunk;
            beyond the cap the vertex count is cached + uncached new solutions (an upper
            bound, duplicates among uncached solutions cannot be detected).
        full_success: Running estimate of the share of queries the full argmax solves.
    """

    queries: int = 0
    hot: int = 0
    full: int = 0
    simplex: int = 0
    split: int = 0
    new: int = 0
    uncached: int = 0
    discovery: list[tuple[int, int]] = field(default_factory=list)
    full_success: float = 1.0


@dataclass
class PairLibrary:
    """Cached vertices of the daily polytope for one SoC pair.

    Attributes:
        lp: The LP of this pair (validity region: this (s_start, s_end)).
        restricted: Whether c_t d_t = 0 is enforced.
        charge: Charge schedules c [MWh], shape (K, T).
        discharge: Discharge schedules d [MWh], shape (K, T).
        codes: Active sets (AT_LOWER_BOUND / AT_UPPER_BOUND / BETWEEN), shape (K, 3T-1).
        basis: A basis realising each vertex (simplex warm start), shape (K, T).
        var_status: simplex BASIC / AT_LOWER / AT_UPPER belonging to ``basis``, shape (K, 3T-1).
        hits: Number of queries answered by each vertex, shape (K,).
        keys: Active-set bytes -> vertex index.
        neighbours: Lazily filled adjacency (vertex -> adjacent cached vertices).
        stats: Query counters.
        feasible: False if (s_start, s_end) cannot be connected within one horizon.
        max_vertices: Cap on the cached vertices; on quarter-hourly prices almost every
            price vector has its own optimal vertex, so an unbounded library would grow
            with the number of queries.
    """

    lp: DailyLP
    restricted: bool
    charge: np.ndarray
    discharge: np.ndarray
    codes: np.ndarray
    basis: np.ndarray
    var_status: np.ndarray
    hits: np.ndarray
    keys: dict[bytes, int] = field(default_factory=dict)
    neighbours: dict[int, np.ndarray] = field(default_factory=dict)
    stats: PairStats = field(default_factory=PairStats)
    feasible: bool = True
    max_vertices: int = MAX_VERTICES

    @property
    def size(self) -> int:
        """Number of cached vertices K."""
        return len(self.keys)

    @property
    def net(self) -> np.ndarray:
        """Net dispatch y = d - c of the cached vertices, shape (K, T)."""
        k = self.size
        return self.discharge[:k] - self.charge[:k]

    def insert(self, x: np.ndarray, basis: np.ndarray, var_status: np.ndarray) -> int:
        """Insert the vertex with LP solution ``x`` (if new) and return its index.

        Returns -1 (and leaves the library unchanged) if the vertex is new but the
        library holds ``max_vertices`` vertices already.
        """
        lp = self.lp
        codes = active_codes(x, lp.lo, lp.up, ACTIVE_TOL)
        key = codes.tobytes()
        idx = self.keys.get(key)
        if idx is not None:
            return idx
        if self.size >= self.max_vertices:
            return -1
        idx = self.size
        if idx == self.charge.shape[0]:
            self._grow()
        n_t = lp.n_periods
        self.charge[idx] = x[:n_t]
        self.discharge[idx] = x[n_t:2 * n_t]
        self.codes[idx] = codes
        self.basis[idx] = basis
        self.var_status[idx] = var_status
        self.hits[idx] = 0
        self.keys[key] = idx
        return idx

    def _grow(self) -> None:
        cap = max(16, 2 * self.charge.shape[0])
        for name in ("charge", "discharge", "codes", "basis", "var_status", "hits"):
            old = getattr(self, name)
            new = np.zeros((cap,) + old.shape[1:], dtype=old.dtype)
            new[:old.shape[0]] = old
            setattr(self, name, new)

    def lookup(self, prices: np.ndarray, cycle_cost: float = 0.0, seed: int = 0,
               return_schedule: bool = False):
        """Optimal vertex and revenue for each price vector.

        Queries are processed in a fixed pseudo-random order in chunks of growing size,
        so that early chunks fill the library and later ones are answered by it.

        Args:
            prices: Prices [€/MWh], shape (Q, T).
            cycle_cost: Cycle cost lambda [€/MWh discharged].
            seed: Seed of the processing order (reproducibility).
            return_schedule: Also return the optimal charge / discharge schedules.

        Returns:
            (vertex index (Q,), revenue p^T(d - c) - lambda 1^T d [€] (Q,)) and, with
            ``return_schedule``, charge and discharge [MWh] of shape (Q, T). The index is
            -1 for optimal vertices that were not cached (library full).
        """
        prices = np.ascontiguousarray(prices, dtype=np.float64)
        n_q, n_t = prices.shape
        idx = np.full(n_q, -1, dtype=np.int64)
        charge = np.zeros((n_q, n_t))
        discharge = np.zeros((n_q, n_t))
        if not self.feasible:
            revenue = np.full(n_q, -np.inf)
            return (idx, revenue, charge, discharge) if return_schedule else (idx, revenue)
        order = np.random.default_rng(seed).permutation(n_q)
        lo, chunk = 0, FIRST_CHUNK if self.size < 64 else MAX_CHUNK
        while lo < n_q:
            sel = order[lo:lo + chunk]
            idx[sel], charge[sel], discharge[sel] = self._lookup_chunk(prices[sel], cycle_cost)
            lo += chunk
            chunk = min(2 * chunk, MAX_CHUNK)
            st = self.stats
            st.discovery.append((st.queries, self.size + st.uncached))
        revenue = np.einsum("qt,qt->q", prices, discharge - charge)
        if cycle_cost:
            revenue -= cycle_cost * discharge.sum(axis=1)
        return (idx, revenue, charge, discharge) if return_schedule else (idx, revenue)

    def _lookup_chunk(self, prices: np.ndarray, cycle_cost: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lp = self.lp
        n_t = lp.n_periods
        c_max, d_max = lp.up[0], lp.up[n_t]
        n_q = prices.shape[0]
        k = self.size
        idx = np.full(n_q, -1, dtype=np.int64)
        charge = np.zeros((n_q, n_t))
        discharge = np.zeros((n_q, n_t))
        todo = np.arange(n_q)
        warm = np.full(n_q, -1, dtype=np.int64)
        st = self.stats
        st.queries += n_q
        t_full = 0.0
        if k:
            Y = np.ascontiguousarray(self.net)
            dsum = self.discharge[:k].sum(axis=1)
            codes = self.codes[:k]
            # 1) most frequently hit vertices
            cand = np.argsort(-self.hits[:k], kind="stable")[:HOT_CANDIDATES].astype(np.int64)
            best, ok = _lookup_candidates(prices, cycle_cost, Y, dsum, codes, cand,
                                          lp.eta_charge, lp.eta_discharge, c_max, d_max,
                                          self.restricted, CERT_TOL)
            idx[ok] = best[ok]
            warm[:] = best
            st.hot += int(ok.sum())
            todo = np.flatnonzero(~ok)
            # 2) all cached vertices; only probed while it rarely succeeds
            if todo.size and k > cand.size:
                probe = todo if st.full_success >= 0.05 else todo[:FULL_PROBE]
                t0 = time.perf_counter()
                full = np.empty(probe.size, dtype=np.int64)
                for s in range(0, probe.size, FULL_CHUNK):
                    rows = probe[s:s + FULL_CHUNK]
                    scores = prices[rows] @ Y.T - cycle_cost * dsum[None, :]
                    full[s:s + FULL_CHUNK] = np.argmax(scores, axis=1)
                ok = _certify_rows(prices[probe], cycle_cost, full, codes, lp.eta_charge,
                                   lp.eta_discharge, c_max, d_max, self.restricted, CERT_TOL)
                t_full = (time.perf_counter() - t0) / probe.size
                idx[probe[ok]] = full[ok]
                warm[probe] = full
                st.full += int(ok.sum())
                st.full_success = 0.5 * st.full_success + 0.5 * float(ok.mean())
                todo = np.setdiff1d(todo, probe[ok], assume_unique=True)
        cached = idx >= 0
        charge[cached] = self.charge[idx[cached]]
        discharge[cached] = self.discharge[idx[cached]]
        # 3) simplex from the basis of the argmax vertex (case split if needed), then insert
        if todo.size:
            if k:
                bases = self.basis[warm[todo]].astype(np.int64)
                var_status = self.var_status[warm[todo]].astype(np.int64)
            else:
                bases = np.full((todo.size, n_t), -1, dtype=np.int64)
                var_status = np.zeros((todo.size, lp.n_vars), dtype=np.int64)
            t0 = time.perf_counter()
            sol = solve_daily(lp, prices[todo], cycle_cost,
                              allow_simultaneous=not self.restricted, basis=bases,
                              var_status=var_status)
            t_simplex = (time.perf_counter() - t0) / todo.size
            bad = sol.status != OPTIMAL
            if np.any(bad):
                raise RuntimeError(f"{int(bad.sum())} daily LPs not solved to optimality "
                                   f"(s_start={lp.s_start}, s_end={lp.s_end})")
            st.simplex += todo.size
            st.split += int(np.sum(sol.n_split > 0))
            charge[todo] = sol.charge
            discharge[todo] = sol.discharge
            before = self.size
            seen: set[bytes] = set()
            for r, q in enumerate(todo):
                i = self.insert(sol.x[r], sol.basis[r], sol.var_status[r])
                idx[q] = i
                if i < 0:
                    key = active_codes(sol.x[r], lp.lo, lp.up, ACTIVE_TOL).tobytes()
                    if key not in seen:
                        seen.add(key)
                        st.uncached += 1
            st.new += self.size - before + len(seen)
            # keep the full argmax only while it is cheaper than the simplex it saves
            if t_full and st.full_success * t_simplex < t_full:
                st.full_success = min(st.full_success, 0.04)
        hit = idx[idx >= 0]
        np.add.at(self.hits, hit, 1)
        return idx, charge, discharge

    # ------------------------------------------------------------- lazy geometry

    def edge_directions(self, k: int) -> np.ndarray:
        """Edge directions of the stored basis of vertex k: one ray per nonbasic variable.

        Ray j moves nonbasic variable j away from its bound (+1 at the lower, -1 at the
        upper bound) and the basic variables by :math:`-B^{-1}a_j` times that step.

        Returns:
            Rays, shape (n - m, n).
        """
        lp = self.lp
        basis = self.basis[k]
        status = self.var_status[k]
        B = lp.A[:, basis]
        nonbasic = np.flatnonzero(status != BASIC)
        rays = np.zeros((nonbasic.size, lp.n_vars))
        sign = np.where(status[nonbasic] == AT_LOWER, 1.0, -1.0)
        rays[np.arange(nonbasic.size), nonbasic] = sign
        rays[:, basis] = -(np.linalg.solve(B, lp.A[:, nonbasic]) * sign).T
        return rays

    def neighbour_indices(self, k: int) -> np.ndarray:
        """Cached vertices adjacent to vertex k along its edge directions (filled lazily).

        Returns:
            Indices of adjacent cached vertices (unknown neighbours are left out).
        """
        if k in self.neighbours:
            return self.neighbours[k]
        lp = self.lp
        x = np.concatenate([self.charge[k], self.discharge[k], np.zeros(lp.n_periods - 1)])
        n_t = lp.n_periods
        x[2 * n_t:] = (lp.s_start + np.cumsum(lp.eta_charge * x[:n_t]
                                               - x[n_t:2 * n_t] / lp.eta_discharge))[:-1]
        found = set()
        for r in self.edge_directions(k):
            with np.errstate(divide="ignore", invalid="ignore"):
                step_up = np.where(r > 1e-12, (lp.up - x) / r, np.inf)
                step_lo = np.where(r < -1e-12, (lp.lo - x) / r, np.inf)
            theta = float(min(step_up.min(), step_lo.min()))
            if not np.isfinite(theta) or theta <= 1e-12:
                continue
            key = active_codes(x + theta * r, lp.lo, lp.up, ACTIVE_TOL).tobytes()
            j = self.keys.get(key)
            if j is not None and j != k:
                found.add(j)
        self.neighbours[k] = np.array(sorted(found), dtype=np.int64)
        return self.neighbours[k]


def _empty_pair(lp: DailyLP, restricted: bool) -> PairLibrary:
    n_t, n = lp.n_periods, lp.n_vars
    ok, _, _, _ = battery_start(n_t, lp.eta_charge, lp.eta_discharge, lp.s_start, lp.s_end,
                                lp.up, 1e-9)
    return PairLibrary(lp=lp, restricted=restricted, charge=np.zeros((0, n_t)),
                       discharge=np.zeros((0, n_t)), codes=np.zeros((0, n), dtype=np.int8),
                       basis=np.zeros((0, n_t), dtype=np.int32),
                       var_status=np.zeros((0, n), dtype=np.int8),
                       hits=np.zeros(0, dtype=np.int64), feasible=bool(ok))


def _pair_key(s_start: float, s_end: float) -> tuple[float, float]:
    return round(float(s_start), 9), round(float(s_end), 9)


@dataclass
class VertexLibrary:
    """Vertex libraries of one battery configuration and horizon, one per SoC pair.

    Libraries are kept separate per pair; vertices are keyed by their active set so they
    can be merged across pairs later (SPEC 4.1).

    Attributes:
        config_hash: :attr:`BatteryConfig.config_hash` of the physical parameters.
        horizon: "full" or "half".
        restricted: True if simultaneous charge and discharge is not allowed.
        template: LP of the horizon (A, l, u shared by all pairs).
        pairs: (s_start, s_end) -> pair library.
        max_vertices: Cap per pair library (see :class:`PairLibrary`).
    """

    config_hash: str
    horizon: Horizon
    restricted: bool
    template: DailyLP
    pairs: dict[tuple[float, float], PairLibrary] = field(default_factory=dict)
    max_vertices: int = MAX_VERTICES

    @classmethod
    def empty(cls, config: BatteryConfig, horizon: Horizon = "full",
              max_vertices: int = MAX_VERTICES) -> VertexLibrary:
        """New empty library for ``config``."""
        template = build_daily_lp(config, config.e_min, config.e_min, horizon)
        return cls(config_hash=config.config_hash, horizon=horizon,
                   restricted=not config.allow_simultaneous_charge_discharge, template=template,
                   max_vertices=max_vertices)

    def pair(self, s_start: float, s_end: float) -> PairLibrary:
        """Pair library for (s_start, s_end), created on first use."""
        key = _pair_key(s_start, s_end)
        lib = self.pairs.get(key)
        if lib is None:
            lib = _empty_pair(self.template.with_soc(*key), self.restricted)
            lib.max_vertices = self.max_vertices
            self.pairs[key] = lib
        return lib

    def lookup(self, prices: np.ndarray, s_start: float, s_end: float, cycle_cost: float = 0.0,
               seed: int = 0, return_schedule: bool = False):
        """Optimal vertices and revenues for prices of shape (Q, T); see :meth:`PairLibrary.lookup`."""
        return self.pair(s_start, s_end).lookup(prices, cycle_cost, seed, return_schedule)


# ------------------------------------------------------------------ stats


def library_stats(library: VertexLibrary) -> dict[str, Any]:
    """Summary of the library: sizes, certificate shares, Good-Turing estimate.

    Returns:
        Dict with totals and a list ``pairs`` with one entry per SoC pair.
    """
    pairs = []
    for (s0, s1), lib in sorted(library.pairs.items()):
        st = lib.stats
        hits = lib.hits[:lib.size]
        n1 = int(np.sum(hits == 1))
        n_hits = int(hits.sum())
        pairs.append({
            "s_start": s0, "s_end": s1, "feasible": lib.feasible, "vertices": lib.size,
            "queries": st.queries, "hit_queries": n_hits, "certified_hot": st.hot,
            "certified_full": st.full, "simplex": st.simplex, "case_split": st.split,
            "new_vertices": st.new, "uncached": st.uncached, "singletons": n1,
            "good_turing": n1 / n_hits if n_hits else float("nan"),
        })
    queries = sum(p["queries"] for p in pairs)
    hits_total = sum(p["hit_queries"] for p in pairs)
    certified = sum(p["certified_hot"] + p["certified_full"] for p in pairs)
    singletons = sum(p["singletons"] for p in pairs)
    return {
        "config_hash": library.config_hash, "horizon": library.horizon,
        "restricted": library.restricted, "n_pairs": len(pairs),
        "vertices": sum(p["vertices"] for p in pairs),
        "max_vertices_per_pair": max((p["vertices"] for p in pairs), default=0),
        "queries": queries,
        "certified_share": certified / queries if queries else float("nan"),
        "simplex": sum(p["simplex"] for p in pairs),
        "case_split": sum(p["case_split"] for p in pairs),
        "uncached": sum(p["uncached"] for p in pairs),
        "good_turing": singletons / hits_total if hits_total else float("nan"),
        "pairs": pairs,
    }


# ------------------------------------------------------------------ persistence


def library_path(root: str | Path, config: BatteryConfig, horizon: Horizon = "full") -> Path:
    """File of the persisted library: ``root/<config hash>_<horizon>.npz``."""
    return Path(root) / f"{config.config_hash}_{horizon}.npz"


def save_library(library: VertexLibrary, path: str | Path) -> Path:
    """Write the library (vertices, bases, hits, counters) to one NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(library.pairs)
    arrays: dict[str, np.ndarray] = {
        "config_hash": np.array(library.config_hash), "horizon": np.array(library.horizon),
        "restricted": np.array(library.restricted), "pair_soc": np.array(keys, dtype=np.float64).reshape(-1, 2),
    }
    for i, key in enumerate(keys):
        lib = library.pairs[key]
        k = lib.size
        st = lib.stats
        arrays[f"p{i}_charge"] = lib.charge[:k]
        arrays[f"p{i}_discharge"] = lib.discharge[:k]
        arrays[f"p{i}_codes"] = lib.codes[:k]
        arrays[f"p{i}_basis"] = lib.basis[:k]
        arrays[f"p{i}_var_status"] = lib.var_status[:k]
        arrays[f"p{i}_hits"] = lib.hits[:k]
        arrays[f"p{i}_counters"] = np.array([st.queries, st.hot, st.full, st.simplex, st.split, st.new,
                                             int(lib.feasible), st.uncached], dtype=np.int64)
        arrays[f"p{i}_discovery"] = np.array(st.discovery, dtype=np.int64).reshape(-1, 2)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    logger.info("vertex library saved to %s (%d pairs, %d vertices)", path, len(keys),
                sum(library.pairs[k].size for k in keys))
    return path


def load_library(path: str | Path, config: BatteryConfig, horizon: Horizon = "full",
                 max_vertices: int = MAX_VERTICES) -> VertexLibrary:
    """Load a library written by :func:`save_library` and check that it belongs to ``config``."""
    library = VertexLibrary.empty(config, horizon, max_vertices)
    with np.load(path) as z:
        if str(z["config_hash"]) != config.config_hash or str(z["horizon"]) != horizon:
            raise ValueError(f"library {path} was built for config {z['config_hash']} / "
                             f"{z['horizon']}, not {config.config_hash} / {horizon}")
        for i, (s0, s1) in enumerate(z["pair_soc"]):
            lib = library.pair(s0, s1)
            lib.charge = z[f"p{i}_charge"].copy()
            lib.discharge = z[f"p{i}_discharge"].copy()
            lib.codes = z[f"p{i}_codes"].copy()
            lib.basis = z[f"p{i}_basis"].copy()
            lib.var_status = z[f"p{i}_var_status"].copy()
            lib.hits = z[f"p{i}_hits"].copy()
            lib.keys = {lib.codes[k].tobytes(): k for k in range(lib.codes.shape[0])}
            c = z[f"p{i}_counters"]
            lib.stats = PairStats(queries=int(c[0]), hot=int(c[1]), full=int(c[2]),
                                  simplex=int(c[3]), split=int(c[4]), new=int(c[5]),
                                  uncached=int(c[7]),
                                  discovery=[tuple(r) for r in z[f"p{i}_discovery"].tolist()])
            lib.feasible = bool(c[6])
    logger.info("vertex library loaded from %s (%d pairs)", path, len(library.pairs))
    return library


def load_or_create(root: str | Path | None, config: BatteryConfig, horizon: Horizon = "full",
                   max_vertices: int = MAX_VERTICES) -> VertexLibrary:
    """Load the persisted library of ``config`` from ``root`` if present, else an empty one."""
    if root is not None:
        path = library_path(root, config, horizon)
        if path.exists():
            return load_library(path, config, horizon, max_vertices)
    return VertexLibrary.empty(config, horizon, max_vertices)


__all__ = [
    "AT_LOWER_BOUND", "AT_UPPER_BOUND", "BETWEEN", "PairLibrary", "PairStats", "VertexLibrary",
    "active_codes", "certify", "library_path", "library_stats", "load_library", "load_or_create",
    "save_library",
]

