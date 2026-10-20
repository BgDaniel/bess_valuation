r"""Numba bounded revised primal simplex for the daily dispatch LP (SPEC 4.2).

Solves :math:`\min g^{\mathsf T}x` s.t. :math:`Ax = b`, :math:`l \le x \le u` (see
:mod:`bess_valuation.polytope`). Design:

* **Bounded variables.** Nonbasic variables sit at a bound (status ``AT_LOWER`` /
  ``AT_UPPER``); the ratio test includes the bound flip of the entering variable.
* **Feasible start without Phase I.** :func:`battery_start` builds a basic feasible
  solution in closed form: a monotone schedule that charges (or discharges) at full
  power from the first period until :math:`s_{end}` is reached, with the one partial
  period basic and all intermediate SoCs basic. Its basis matrix is non-singular since
  :math:`\mathbf 1^{\mathsf T}` annihilates every SoC column but not a charge or
  discharge column. It exists whenever the LP is feasible, so it is also the fallback
  when a warm-start basis is infeasible for a new right-hand side; a separate Phase I is
  not needed for this polytope.
* **Basis inverse** dense (m = T is small), updated by the rank-1 product-form update
  :math:`B^{-1} \leftarrow E\,B^{-1}` and refactorised every ``refactor_every`` pivots.
* **Pricing** by exact steepest edge, :math:`\max_j d_j^2/\gamma_j` with
  :math:`\gamma_j = 1 + \lVert B^{-1}a_j\rVert^2` kept up to date by the
  Goldfarb-Reid recurrences; after ``bland_after`` consecutive degenerate pivots it
  switches to Bland's rule (smallest eligible index, smallest leaving index on ties)
  until a non-degenerate step is made.
* **Tolerances** ``feas_tol`` (primal bounds), ``opt_tol`` (reduced costs) and
  ``piv_tol`` (pivot elements).

:func:`solve_batch` solves one LP for many cost vectors in parallel (``prange``).
:func:`solve_daily` adds the no-simultaneous-charge/discharge rule by a per-period case
split (branch and bound on :math:`c_t = 0` or :math:`d_t = 0`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from bess_valuation.polytope import DailyLP, lp_cost, split_schedule

logger = logging.getLogger(__name__)

OPTIMAL = 0
INFEASIBLE = 1
ITERATION_LIMIT = 2
UNBOUNDED = 3

BASIC = 0
AT_LOWER = 1
AT_UPPER = 2

FEAS_TOL = 1e-9
OPT_TOL = 1e-9
PIV_TOL = 1e-9
REFACTOR_EVERY = 50
BLAND_AFTER = 50


@njit(cache=True)
def battery_start(n_periods, eta_charge, eta_discharge, s_start, s_end, up, feas_tol):
    """Closed-form basic feasible solution of the daily LP.

    Charges (if s_end >= s_start) or discharges at the upper bound of each period from
    the first one on until the SoC difference is covered; that last, partial period is
    basic together with all intermediate SoCs.

    Returns:
        (ok, basis (T,), status (3T-1,), x (3T-1,)); ok is False if s_end is unreachable
        within the bounds ``up``.
    """
    n = 3 * n_periods - 1
    status = np.full(n, AT_LOWER, dtype=np.int64)
    x = np.zeros(n)
    basis = np.empty(n_periods, dtype=np.int64)
    for t in range(1, n_periods):
        j = 2 * n_periods + t - 1
        basis[t - 1] = j
        status[j] = BASIC
    delta = s_end - s_start
    if delta >= 0.0:
        first, gain = 0, eta_charge
    else:
        first, gain, delta = n_periods, 1.0 / eta_discharge, -delta
    remaining = delta
    extra = -1
    for t in range(n_periods):
        j = first + t
        cap = up[j] * gain
        if remaining <= cap + feas_tol:
            x[j] = min(remaining / gain, up[j])
            extra = j
            break
        x[j] = up[j]
        status[j] = AT_UPPER
        remaining -= cap
    if extra < 0:
        return False, basis, status, x
    basis[n_periods - 1] = extra
    status[extra] = BASIC
    return True, basis, status, x


@njit(cache=True)
def _basis_inverse(A, basis):
    m = A.shape[0]
    B = np.empty((m, m))
    for i in range(m):
        B[:, i] = A[:, basis[i]]
    return np.ascontiguousarray(np.linalg.inv(B))


@njit(cache=True)
def _basic_values(A, b, Binv, basis, status, x):
    rhs = b.copy()
    for j in range(A.shape[1]):
        if status[j] != BASIC and x[j] != 0.0:
            rhs -= A[:, j] * x[j]
    xb = Binv @ rhs
    for i in range(basis.shape[0]):
        x[basis[i]] = xb[i]


@njit(cache=True)
def solve_lp_core(A, b, cost, lo, up, basis, status, x, max_iter, feas_tol, opt_tol, piv_tol,
                  refactor_every, bland_after):
    """Bounded revised primal simplex from a given basis (arrays updated in place).

    Args:
        A, b, cost, lo, up: LP data (minimisation).
        basis: Basic column indices, shape (m,), modified in place.
        status: BASIC / AT_LOWER / AT_UPPER per variable, modified in place.
        x: Solution vector, modified in place (nonbasic values are reset from status).

    Returns:
        (code, iterations, degenerate pivots); code is OPTIMAL, INFEASIBLE (start basis
        infeasible), ITERATION_LIMIT or UNBOUNDED.
    """
    m, n = A.shape
    AT = np.ascontiguousarray(A.T)
    for j in range(n):
        if status[j] == AT_LOWER:
            x[j] = lo[j]
        elif status[j] == AT_UPPER:
            x[j] = up[j]
    Binv = _basis_inverse(A, basis)
    _basic_values(A, b, Binv, basis, status, x)
    for i in range(m):
        j = basis[i]
        if x[j] < lo[j] - feas_tol or x[j] > up[j] + feas_tol:
            return INFEASIBLE, 0, 0

    gamma = np.ones(n)
    for j in range(n):
        if status[j] != BASIC:
            col = Binv @ A[:, j]
            gamma[j] = 1.0 + col @ col

    n_iter = 0
    n_degenerate = 0
    degenerate_run = 0
    since_refactor = 0
    bland = False
    while n_iter < max_iter:
        cb = np.empty(m)
        for i in range(m):
            cb[i] = cost[basis[i]]
        y = cb @ Binv
        reduced = cost - AT @ y

        q = -1
        best = 0.0
        for j in range(n):
            if status[j] == BASIC or lo[j] == up[j]:
                continue
            dj = reduced[j]
            if (status[j] == AT_LOWER and dj < -opt_tol) or (status[j] == AT_UPPER and dj > opt_tol):
                if bland:
                    q = j
                    break
                score = dj * dj / gamma[j]
                if score > best:
                    best = score
                    q = j
        if q < 0:
            return OPTIMAL, n_iter, n_degenerate

        direction = 1.0 if status[q] == AT_LOWER else -1.0
        alpha = Binv @ A[:, q]
        theta = up[q] - lo[q]
        r = -1
        tie = 1e-12
        for i in range(m):
            a = direction * alpha[i]
            j = basis[i]
            if a > piv_tol:
                t = (x[j] - lo[j]) / a
            elif a < -piv_tol:
                t = (up[j] - x[j]) / (-a)
            else:
                continue
            if t < 0.0:
                t = 0.0
            if t < theta - tie:
                theta, r = t, i
            elif r >= 0 and t <= theta + tie:
                if bland:
                    if j < basis[r]:
                        theta, r = t, i
                elif abs(alpha[i]) > abs(alpha[r]):
                    theta, r = t, i
        if r < 0 and not np.isfinite(theta):
            return UNBOUNDED, n_iter, n_degenerate

        if theta <= feas_tol:
            n_degenerate += 1
            degenerate_run += 1
            if degenerate_run > bland_after:
                bland = True
        else:
            degenerate_run = 0
            bland = False

        x[q] += direction * theta
        for i in range(m):
            x[basis[i]] -= direction * theta * alpha[i]
        n_iter += 1

        if r < 0:  # bound flip, basis unchanged
            if direction > 0:
                status[q] = AT_UPPER
                x[q] = up[q]
            else:
                status[q] = AT_LOWER
                x[q] = lo[q]
            continue

        p = basis[r]
        if direction * alpha[r] > 0.0:
            status[p] = AT_LOWER
            x[p] = lo[p]
        else:
            status[p] = AT_UPPER
            x[p] = up[p]

        # steepest-edge weights (Goldfarb-Reid), using the old basis inverse
        ar = alpha[r]
        rho = Binv[r, :].copy()
        w = alpha @ Binv
        gq = 1.0 + alpha @ alpha
        row = AT @ rho
        aw = AT @ w
        for j in range(n):
            if status[j] == BASIC or j == q or row[j] == 0.0:
                continue
            ratio = row[j] / ar
            g = gamma[j] - 2.0 * ratio * aw[j] + ratio * ratio * gq
            lower = 1.0 + ratio * ratio
            gamma[j] = g if g > lower else lower
        gamma[p] = max(gq / (ar * ar), 1.0 + 1.0 / (ar * ar))

        # product-form update of the basis inverse
        Binv[r, :] /= ar
        for i in range(m):
            if i != r and alpha[i] != 0.0:
                Binv[i, :] -= alpha[i] * Binv[r, :]
        basis[r] = q
        status[q] = BASIC

        since_refactor += 1
        if since_refactor >= refactor_every:
            Binv = _basis_inverse(A, basis)
            _basic_values(A, b, Binv, basis, status, x)
            since_refactor = 0
    return ITERATION_LIMIT, n_iter, n_degenerate


@njit(cache=True)
def _solve_one(A, b, cost, lo, up, n_periods, eta_c, eta_d, s_start, s_end, basis0, status0,
               max_iter, feas_tol, opt_tol, piv_tol, refactor_every, bland_after):
    """Warm start from (basis0, status0) if given (basis0[0] >= 0), else or on failure cold."""
    n = A.shape[1]
    x = np.zeros(n)
    if basis0[0] >= 0:
        basis = basis0.copy()
        status = status0.copy()
        code, it, dg = solve_lp_core(A, b, cost, lo, up, basis, status, x, max_iter, feas_tol,
                                     opt_tol, piv_tol, refactor_every, bland_after)
        if code != INFEASIBLE:
            return code, it, dg, x, basis, status, True
    ok, basis, status, x = battery_start(n_periods, eta_c, eta_d, s_start, s_end, up, feas_tol)
    if not ok:
        return INFEASIBLE, 0, 0, x, basis, status, False
    code, it, dg = solve_lp_core(A, b, cost, lo, up, basis, status, x, max_iter, feas_tol,
                                 opt_tol, piv_tol, refactor_every, bland_after)
    return code, it, dg, x, basis, status, False


@njit(parallel=True, cache=True)
def _solve_batch(A, b, costs, lo, up, n_periods, eta_c, eta_d, s_start, s_end, basis0, status0,
                 max_iter, feas_tol, opt_tol, piv_tol, refactor_every, bland_after):
    n_lp = costs.shape[0]
    m, n = A.shape
    X = np.empty((n_lp, n))
    bases = np.empty((n_lp, m), dtype=np.int64)
    statuses = np.empty((n_lp, n), dtype=np.int64)
    codes = np.empty(n_lp, dtype=np.int64)
    iters = np.empty(n_lp, dtype=np.int64)
    degens = np.empty(n_lp, dtype=np.int64)
    warm = np.empty(n_lp, dtype=np.bool_)
    for k in prange(n_lp):
        code, it, dg, x, basis, status, used_warm = _solve_one(
            A, b, costs[k], lo, up, n_periods, eta_c, eta_d, s_start, s_end, basis0, status0,
            max_iter, feas_tol, opt_tol, piv_tol, refactor_every, bland_after)
        X[k] = x
        bases[k] = basis
        statuses[k] = status
        codes[k] = code
        iters[k] = it
        degens[k] = dg
        warm[k] = used_warm
    return codes, X, bases, statuses, iters, degens, warm


@dataclass
class BatchResult:
    """Solutions of one LP for many cost vectors.

    Attributes:
        status: OPTIMAL / INFEASIBLE / ITERATION_LIMIT / UNBOUNDED per LP, shape (P,).
        x: Solutions, shape (P, n).
        objective: Minimised cost g^T x, shape (P,).
        basis: Final basis per LP, shape (P, m).
        var_status: Final BASIC / AT_LOWER / AT_UPPER per variable, shape (P, n).
        iterations: Simplex iterations (pivots and bound flips), shape (P,).
        degenerate: Degenerate steps, shape (P,).
        warm: Whether the warm-start basis was used (not the cold fallback), shape (P,).
    """

    status: np.ndarray
    x: np.ndarray
    objective: np.ndarray
    basis: np.ndarray
    var_status: np.ndarray
    iterations: np.ndarray
    degenerate: np.ndarray
    warm: np.ndarray


def solve_batch(lp: DailyLP, costs: np.ndarray, basis: np.ndarray | None = None,
                var_status: np.ndarray | None = None, up: np.ndarray | None = None,
                max_iter: int | None = None) -> BatchResult:
    """Solve ``lp`` for each row of ``costs`` (shape (P, n) or (n,)) in parallel.

    Args:
        lp: LP data.
        costs: Minimisation cost vectors, e.g. from :func:`bess_valuation.polytope.lp_cost`.
        basis: Optional warm-start basis (m,), shared by all rows.
        var_status: Variable status belonging to ``basis``.
        up: Optional replacement of the upper bounds (used by the case split).
        max_iter: Iteration limit per LP; default 20 (m + n).
    """
    costs = np.atleast_2d(np.asarray(costs, dtype=np.float64))
    m, n = lp.A.shape
    if basis is None:
        basis0 = np.full(m, -1, dtype=np.int64)
        status0 = np.zeros(n, dtype=np.int64)
    else:
        basis0 = np.asarray(basis, dtype=np.int64)
        status0 = np.asarray(var_status, dtype=np.int64)
    upper = lp.up if up is None else np.asarray(up, dtype=np.float64)
    limit = 20 * (m + n) if max_iter is None else max_iter
    codes, X, bases, statuses, iters, degens, warm = _solve_batch(
        lp.A, lp.b, costs, lp.lo, upper, lp.n_periods, lp.eta_charge, lp.eta_discharge,
        lp.s_start, lp.s_end, basis0, status0, limit, FEAS_TOL, OPT_TOL, PIV_TOL,
        REFACTOR_EVERY, BLAND_AFTER)
    objective = np.einsum("pn,pn->p", costs, X)
    return BatchResult(status=codes, x=X, objective=objective, basis=bases, var_status=statuses,
                       iterations=iters, degenerate=degens, warm=warm)


@dataclass
class DailySolution:
    """Optimal daily schedules for P price vectors.

    Attributes:
        charge: Charge energy c [MWh], shape (P, T).
        discharge: Discharge energy d [MWh], shape (P, T).
        revenue: Objective p^T(d - c) - lambda 1^T d [€], shape (P,).
        status: Solver status per price vector, shape (P,).
        iterations: Simplex iterations, shape (P,) (summed over case-split nodes).
        n_split: Case-split LPs solved beyond the root, shape (P,).
        x: Full LP solutions, shape (P, 3T-1).
    """

    charge: np.ndarray
    discharge: np.ndarray
    revenue: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    n_split: np.ndarray
    x: np.ndarray

    @property
    def net(self) -> np.ndarray:
        """Net dispatch y = d - c [MWh], shape (P, T)."""
        return self.discharge - self.charge


def solve_daily(lp: DailyLP, prices: np.ndarray, cycle_cost: float = 0.0,
                allow_simultaneous: bool = True, tol: float = 1e-7) -> DailySolution:
    """Optimal schedules for prices of shape (P, T) or (T,).

    If simultaneous charge and discharge is not allowed, every solution with
    :math:`c_t d_t > 0` is resolved by a per-period case split: branch on
    :math:`d_t = 0` versus :math:`c_t = 0` for the first offending period and recurse,
    pruning with the LP bound (depth-first branch and bound). With
    :math:`\\eta_c\\eta_d < 1` an offending period can only have a negative price, so
    the split is rarely needed. Each branch fixes variables at their lower bound 0,
    i.e. restricts to a face of the polytope, so the result is still a vertex of the
    original polytope.
    """
    prices = np.atleast_2d(np.asarray(prices, dtype=np.float64))
    costs = lp_cost(prices, cycle_cost)
    res = solve_batch(lp, costs)
    if np.any(res.status != OPTIMAL):
        logger.warning("%d of %d daily LPs not optimal (status codes %s)",
                       int(np.sum(res.status != OPTIMAL)), len(costs), np.unique(res.status).tolist())
    X = res.x.copy()
    objective = res.objective.copy()
    status = res.status.copy()
    iterations = res.iterations.copy()
    n_split = np.zeros(len(prices), dtype=np.int64)
    if not allow_simultaneous:
        c, d = split_schedule(X, lp.n_periods)
        offending = np.flatnonzero(np.any((c > tol) & (d > tol), axis=1))
        if offending.size:
            logger.debug("case split for %d of %d price vectors (simultaneous charge/discharge)",
                         offending.size, len(prices))
        for k in offending:
            x, obj, code, it, nodes = _case_split(lp, costs[k], tol)
            X[k], objective[k], status[k] = x, obj, code
            iterations[k] += it
            n_split[k] = nodes
    c, d = split_schedule(X, lp.n_periods)
    return DailySolution(charge=c, discharge=d, revenue=-objective, status=status,
                         iterations=iterations, n_split=n_split, x=X)


def _case_split(lp: DailyLP, cost: np.ndarray, tol: float) -> tuple[np.ndarray, float, int, int, int]:
    """Depth-first branch and bound on c_t = 0 / d_t = 0 for one cost vector."""
    n_t = lp.n_periods
    best_x = np.zeros(lp.n_vars)
    best = np.inf
    iterations = 0
    nodes = 0
    stack = [lp.up.copy()]
    while stack:
        up = stack.pop()
        res = solve_batch(lp, cost, up=up)
        nodes += 1
        iterations += int(res.iterations[0])
        if res.status[0] != OPTIMAL or res.objective[0] >= best - 1e-12:
            continue
        x = res.x[0]
        both = np.flatnonzero((x[:n_t] > tol) & (x[n_t:2 * n_t] > tol))
        if both.size == 0:
            best, best_x = res.objective[0], x
            continue
        t = both[0]
        no_charge = up.copy()
        no_charge[t] = 0.0
        no_discharge = up.copy()
        no_discharge[n_t + t] = 0.0
        stack.extend([no_charge, no_discharge])
    code = OPTIMAL if np.isfinite(best) else INFEASIBLE
    return best_x, best, code, iterations, nodes - 1
