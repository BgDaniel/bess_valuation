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
  The (re)factorisation uses the chain structure of the battery basis (closed form in
  O(m^2), :func:`_basis_inverse`); products with A use its column sparsity, and the
  reduced costs are updated from the pivot row, so no BLAS / LAPACK call runs inside the
  parallel loops (OpenBLAS threads would oversubscribe the ``prange`` workers).
* **Pricing** by exact steepest edge, :math:`\max_j d_j^2/\gamma_j` with
  :math:`\gamma_j = 1 + \lVert B^{-1}a_j\rVert^2` kept up to date by the
  Goldfarb-Reid recurrences; after ``bland_after`` consecutive degenerate pivots it
  switches to Bland's rule (smallest eligible index, smallest leaving index on ties)
  until a non-degenerate step is made.
* **Tolerances** ``feas_tol`` (primal bounds), ``opt_tol`` (reduced costs) and
  ``piv_tol`` (pivot elements).

:func:`solve_batch` solves one LP for many cost vectors in parallel (``prange``).
:func:`solve_daily` adds the no-simultaneous-charge/discharge rule by a per-period case
split (charge only / discharge only), resolved exactly by a dynamic program over the SoC.
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
def _sparse_columns(A):
    """Column-wise sparse copy of A: row indices, values and count of the nonzeros."""
    m, n = A.shape
    nnz = np.zeros(n, dtype=np.int64)
    for j in range(n):
        for i in range(m):
            if A[i, j] != 0.0:
                nnz[j] += 1
    width = max(1, nnz.max())
    rows = np.zeros((n, width), dtype=np.int64)
    vals = np.zeros((n, width))
    for j in range(n):
        k = 0
        for i in range(m):
            if A[i, j] != 0.0:
                rows[j, k] = i
                vals[j, k] = A[i, j]
                k += 1
    return rows, vals, nnz


@njit(cache=True)
def _col_dot(rows, vals, nnz, j, v):
    """a_j^T v for column j of the sparse A."""
    s = 0.0
    for k in range(nnz[j]):
        s += vals[j, k] * v[rows[j, k]]
    return s


@njit(cache=True)
def _ftran(Binv, rows, vals, nnz, j):
    """B^{-1} a_j for column j of the sparse A."""
    m = Binv.shape[0]
    z = np.zeros(m)
    for k in range(nnz[j]):
        r = rows[j, k]
        v = vals[j, k]
        for i in range(m):
            z[i] += v * Binv[i, r]
    return z


@njit(cache=True)
def _vec_mat(v, M):
    """v^T M for a square matrix M."""
    m = M.shape[0]
    out = np.zeros(M.shape[1])
    for i in range(m):
        vi = v[i]
        if vi != 0.0:
            for k in range(M.shape[1]):
                out[k] += vi * M[i, k]
    return out


@njit(cache=True)
def _dense_inverse(B):
    """Gauss-Jordan inverse with partial pivoting (fallback, no LAPACK in parallel loops)."""
    m = B.shape[0]
    M = B.copy()
    inv = np.eye(m)
    for c in range(m):
        p = c
        for i in range(c + 1, m):
            if abs(M[i, c]) > abs(M[p, c]):
                p = i
        if p != c:
            for k in range(m):
                M[c, k], M[p, k] = M[p, k], M[c, k]
                inv[c, k], inv[p, k] = inv[p, k], inv[c, k]
        piv = M[c, c]
        for k in range(m):
            M[c, k] /= piv
            inv[c, k] /= piv
        for i in range(m):
            if i != c and M[i, c] != 0.0:
                f = M[i, c]
                for k in range(m):
                    M[i, k] -= f * M[c, k]
                    inv[i, k] -= f * inv[c, k]
    return inv


@njit(cache=True)
def _basis_inverse(A, basis):
    r"""Inverse of the basis matrix B = A[:, basis].

    For the battery LP every column of A is either a *root* (one nonzero, c_t or d_t in
    row t) or an *edge* (+1 in row t-1, -1 in row t, the SoC s_t). A basis splits the rows
    into segments [a, b] joined by basic edges, each with exactly one basic root
    (coefficient kappa in row tau). Then, for rows i in [a, b],
    :math:`(B^{-1})_{root,i} = 1/\kappa` and
    :math:`(B^{-1})_{s_{t+1},i} = [i \le t] - [\tau \le t]` for t = a..b-1, and 0 for
    rows outside the segment, which costs O(m^2) instead of O(m^3). Any other
    structure falls back to a dense Gauss-Jordan inverse.
    """
    m = A.shape[0]
    root_pos = np.full(m, -1, dtype=np.int64)   # basis position of the root of row t
    root_val = np.zeros(m)
    edge_pos = np.full(m, -1, dtype=np.int64)   # basis position of the edge joining t-1, t
    structured = True
    for i in range(m):
        j = basis[i]
        first = -1
        second = -1
        count = 0
        for r in range(m):
            if A[r, j] != 0.0:
                if count == 0:
                    first = r
                else:
                    second = r
                count += 1
        if count == 1 and root_pos[first] < 0:
            root_pos[first] = i
            root_val[first] = A[first, j]
        elif (count == 2 and second == first + 1 and A[first, j] == 1.0
              and A[second, j] == -1.0 and edge_pos[second] < 0):
            edge_pos[second] = i
        else:
            structured = False
            break
    if structured:
        Binv = np.zeros((m, m))
        a = 0
        while a < m:
            b = a
            while b + 1 < m and edge_pos[b + 1] >= 0:
                b += 1
            tau = -1
            for t in range(a, b + 1):
                if root_pos[t] >= 0:
                    if tau >= 0:
                        structured = False
                    tau = t
            if tau < 0 or not structured:
                structured = False
                break
            kappa = root_val[tau]
            for i in range(a, b + 1):
                Binv[root_pos[tau], i] = 1.0 / kappa
            for t in range(a, b):
                pos = edge_pos[t + 1]
                shift = 1.0 if tau <= t else 0.0
                for i in range(a, b + 1):
                    Binv[pos, i] = (1.0 if i <= t else 0.0) - shift
            a = b + 1
        if structured:
            return Binv
    B = np.empty((m, m))
    for i in range(m):
        B[:, i] = A[:, basis[i]]
    return _dense_inverse(B)


@njit(cache=True)
def _basic_values(b, Binv, basis, status, x, rows, vals, nnz):
    m = Binv.shape[0]
    rhs = b.copy()
    for j in range(x.shape[0]):
        if status[j] != BASIC and x[j] != 0.0:
            for k in range(nnz[j]):
                rhs[rows[j, k]] -= vals[j, k] * x[j]
    for i in range(m):
        s = 0.0
        for k in range(m):
            s += Binv[i, k] * rhs[k]
        x[basis[i]] = s


@njit(cache=True)
def solve_lp_core(A, b, cost, lo, up, basis, status, x, max_iter, feas_tol, opt_tol, piv_tol,
                  refactor_every, bland_after):
    """Bounded revised primal simplex from a given basis (arrays updated in place).

    Products with A use its column-wise sparsity (at most two nonzeros per column for
    the battery LP); the basis inverse is dense.

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
    rows, vals, nnz = _sparse_columns(A)
    for j in range(n):
        if status[j] == AT_LOWER:
            x[j] = lo[j]
        elif status[j] == AT_UPPER:
            x[j] = up[j]
    Binv = _basis_inverse(A, basis)
    _basic_values(b, Binv, basis, status, x, rows, vals, nnz)
    for i in range(m):
        j = basis[i]
        if x[j] < lo[j] - feas_tol or x[j] > up[j] + feas_tol:
            return INFEASIBLE, 0, 0

    gamma = np.ones(n)
    for j in range(n):
        if status[j] != BASIC:
            col = _ftran(Binv, rows, vals, nnz, j)
            gamma[j] = 1.0 + col @ col

    n_iter = 0
    n_degenerate = 0
    degenerate_run = 0
    since_refactor = 0
    bland = False
    cb = np.empty(m)
    reduced = np.empty(n)
    row = np.zeros(n)
    aw = np.zeros(n)
    fresh = True
    while n_iter < max_iter:
        if fresh:  # exact reduced costs after (re)factorisation, updated in O(n) otherwise
            for i in range(m):
                cb[i] = cost[basis[i]]
            y = _vec_mat(cb, Binv)
            for j in range(n):
                reduced[j] = cost[j] - _col_dot(rows, vals, nnz, j, y)
            fresh = False

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
        alpha = _ftran(Binv, rows, vals, nnz, q)
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
        w = _vec_mat(alpha, Binv)
        gq = 1.0 + alpha @ alpha
        for j in range(n):
            row[j] = _col_dot(rows, vals, nnz, j, rho)
            aw[j] = _col_dot(rows, vals, nnz, j, w)
        for j in range(n):
            if status[j] == BASIC or j == q or row[j] == 0.0:
                continue
            ratio = row[j] / ar
            g = gamma[j] - 2.0 * ratio * aw[j] + ratio * ratio * gq
            lower = 1.0 + ratio * ratio
            gamma[j] = g if g > lower else lower
        gamma[p] = max(gq / (ar * ar), 1.0 + 1.0 / (ar * ar))

        # reduced costs: d_j -= (d_q / alpha_r) * (row r of B^{-1} A)_j
        theta_d = reduced[q] / ar
        for j in range(n):
            if row[j] != 0.0:
                reduced[j] -= theta_d * row[j]
        reduced[q] = 0.0
        reduced[p] = -theta_d

        # product-form update of the basis inverse
        for k in range(m):
            Binv[r, k] /= ar
        for i in range(m):
            if i != r and alpha[i] != 0.0:
                ai = alpha[i]
                for k in range(m):
                    Binv[i, k] -= ai * Binv[r, k]
        basis[r] = q
        status[q] = BASIC

        since_refactor += 1
        if since_refactor >= refactor_every:
            Binv = _basis_inverse(A, basis)
            _basic_values(b, Binv, basis, status, x, rows, vals, nnz)
            since_refactor = 0
            fresh = True
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


@njit(parallel=True, cache=True)
def _solve_batch_rows(A, b, costs, lo, up, n_periods, eta_c, eta_d, s_start, s_end, bases0,
                      statuses0, max_iter, feas_tol, opt_tol, piv_tol, refactor_every, bland_after):
    """Like :func:`_solve_batch` but with one warm-start basis per cost vector."""
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
            A, b, costs[k], lo, up, n_periods, eta_c, eta_d, s_start, s_end, bases0[k],
            statuses0[k], max_iter, feas_tol, opt_tol, piv_tol, refactor_every, bland_after)
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
        basis: Optional warm-start basis, shape (m,) shared by all rows or (P, m) per row.
        var_status: Variable status belonging to ``basis``, shape (n,) or (P, n).
        up: Optional replacement of the upper bounds (used by the case split).
        max_iter: Iteration limit per LP; default 20 (m + n).

    A basis of shape (P, m) (with ``var_status`` of shape (P, n)) gives one warm start per
    cost vector; rows whose basis starts with -1 are solved cold.
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
    solver = _solve_batch_rows if basis0.ndim == 2 else _solve_batch
    codes, X, bases, statuses, iters, degens, warm = solver(
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
        n_split: 1 if the case split (DP) was needed, else 0, shape (P,).
        x: Full LP solutions, shape (P, 3T-1).
        basis: Final basis per price vector (of the case-split leaf if split), shape (P, T).
        var_status: Variable status belonging to ``basis``, shape (P, 3T-1).
    """

    charge: np.ndarray
    discharge: np.ndarray
    revenue: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    n_split: np.ndarray
    x: np.ndarray
    basis: np.ndarray
    var_status: np.ndarray

    @property
    def net(self) -> np.ndarray:
        """Net dispatch y = d - c [MWh], shape (P, T)."""
        return self.discharge - self.charge


def solve_daily(lp: DailyLP, prices: np.ndarray, cycle_cost: float = 0.0,
                allow_simultaneous: bool = True, tol: float = 1e-7,
                basis: np.ndarray | None = None,
                var_status: np.ndarray | None = None) -> DailySolution:
    r"""Optimal schedules for prices of shape (P, T) or (T,).

    If simultaneous charge and discharge is not allowed (:math:`c_t d_t = 0`), every
    period is split into the two cases "charge only" (:math:`d_t = 0`) and "discharge
    only" (:math:`c_t = 0`). In terms of the SoC change :math:`e_t` the gain of each case
    is linear; their maximum is concave, and then equal to the LP relaxation, unless
    :math:`g_{d_t}\eta_d < -g_{c_t}/\eta_c` (:func:`nonconcave_periods`; without cycle
    cost: :math:`p_t < 0`). Price vectors without such periods are solved as plain LPs
    (``basis`` / ``var_status`` are optional warm starts, see :func:`solve_batch`).

    For the others the case split is resolved exactly by dynamic programming over the
    SoC (:func:`_restricted_dp`): the value function of the remaining day is a
    piecewise-linear function of the SoC, and each period takes the maximum over the two
    cases. A branch and bound over the cases with LP bounds grows like 2^k in the number
    k of negative-price periods (when the battery is full, the relaxation gains a little
    from mixing charge and discharge in every idle period and both branches reduce to
    idle), while the DP costs O(T K) for K breakpoints. The optimal schedule is a vertex
    of the daily polytope (a face of it, in fact); its basis is recovered by
    :func:`_purify`, so the result can go into the vertex library like an LP solution.
    A plain LP solution with :math:`c_t d_t > 0` (possible only at ties) is re-solved by
    the DP as well.
    """
    prices = np.atleast_2d(np.asarray(prices, dtype=np.float64))
    costs = lp_cost(prices, cycle_cost)
    n_p = len(prices)
    n_t = lp.n_periods
    X = np.zeros((n_p, lp.n_vars))
    objective = np.zeros(n_p)
    status = np.zeros(n_p, dtype=np.int64)
    iterations = np.zeros(n_p, dtype=np.int64)
    bases = np.zeros((n_p, n_t), dtype=np.int64)
    var_status_out = np.zeros((n_p, lp.n_vars), dtype=np.int64)
    n_split = np.zeros(n_p, dtype=np.int64)

    split = np.zeros(n_p, dtype=bool)
    if not allow_simultaneous:
        split = nonconcave_periods(prices, cycle_cost, lp.eta_charge, lp.eta_discharge).any(axis=1)
    plain = np.flatnonzero(~split)
    if plain.size:
        warm_b = warm_s = None
        if basis is not None:
            warm_b = basis[plain] if np.ndim(basis) == 2 else basis
            warm_s = var_status[plain] if np.ndim(var_status) == 2 else var_status
        res = solve_batch(lp, costs[plain], basis=warm_b, var_status=warm_s)
        X[plain], objective[plain], status[plain] = res.x, res.objective, res.status
        iterations[plain] = res.iterations
        bases[plain], var_status_out[plain] = res.basis, res.var_status
        if not allow_simultaneous:
            c, d = split_schedule(res.x, n_t)
            split[plain[np.any((c > tol) & (d > tol), axis=1)]] = True
    rows = np.flatnonzero(split)
    if rows.size:
        logger.debug("case split (DP) for %d of %d price vectors", rows.size, n_p)
        codes, Xs, bs, ss, objs = _restricted_batch(
            lp.A, lp.lo, lp.up, prices[rows], float(cycle_cost), lp.eta_charge, lp.eta_discharge,
            lp.s_start, lp.s_end)
        # the purified vertex has the same value; recompute it from the final schedule
        X[rows], status[rows] = Xs, codes
        objective[rows] = np.einsum("pn,pn->p", costs[rows], Xs)
        bases[rows], var_status_out[rows] = bs, ss
        n_split[rows] = 1
    if np.any(status != OPTIMAL):
        logger.warning("%d of %d daily problems not optimal (status codes %s)",
                       int(np.sum(status != OPTIMAL)), n_p, np.unique(status).tolist())
    c, d = split_schedule(X, n_t)
    return DailySolution(charge=c, discharge=d, revenue=-objective, status=status,
                         iterations=iterations, n_split=n_split, x=X, basis=bases,
                         var_status=var_status_out)


def nonconcave_periods(prices: np.ndarray, cycle_cost: float, eta_charge: float,
                       eta_discharge: float) -> np.ndarray:
    r"""Periods whose gain without simultaneous charge/discharge is not concave.

    True where :math:`g_d\eta_d < -g_c/\eta_c` with :math:`g_c = p`, :math:`g_d = \lambda - p`
    (for :math:`\lambda = 0`: :math:`p < 0`). Only there can the LP relaxation profit from
    simultaneous charge and discharge.
    """
    gc = np.asarray(prices, dtype=np.float64)
    gd = cycle_cost - gc
    return gd * eta_discharge < -gc / eta_charge - 1e-12 * (1.0 + np.abs(gc))


# --------------------------------------------------------------- case split internals

@njit(cache=True)
def _interior(x, lo, up, j, tol):
    return (lo[j] < up[j] and x[j] > lo[j] + tol * max(1.0, abs(lo[j]))
            and x[j] < up[j] - tol * max(1.0, abs(up[j])))


@njit(cache=True)
def _purify(A, lo, up, x, n_t, tol):
    """Turn a feasible point of a (node) battery LP into a basic feasible solution.

    Within a segment of periods joined by SoC values strictly inside their bounds, two
    interior charge/discharge variables are shifted against each other (energy moves
    from one period to the other, the SoC in between follows) until a variable reaches
    a bound or the SoC in between does, which splits the segment. At the end each
    segment holds at most one interior root, which becomes basic together with the
    interior SoCs; a segment without one gets a root at its bound as degenerate basic.

    Returns:
        (ok, basis (T,), var_status (3T-1,)); x is modified in place.
    """
    n = 3 * n_t - 1
    eps = np.empty(2 * n_t)  # SoC change per unit of each root variable
    for j in range(2 * n_t):
        eps[j] = -A[j if j < n_t else j - n_t, j]
    for _ in range(4 * n):
        changed = False
        a = 0
        while a < n_t and not changed:
            b = a
            while b + 1 < n_t and _interior(x, lo, up, 2 * n_t + b, tol):
                b += 1
            first = -1
            second = -1
            for t in range(a, b + 1):
                for j in (t, n_t + t):
                    if _interior(x, lo, up, j, tol):
                        if first < 0:
                            first = j
                        elif second < 0:
                            second = j
            if second >= 0:
                tf = first if first < n_t else first - n_t
                ts = second if second < n_t else second - n_t
                ef = eps[first]
                es = eps[second]
                delta = (up[first] - x[first]) * ef if ef > 0 else (lo[first] - x[first]) * ef
                lim = first
                bound = up[first] if ef > 0 else lo[first]
                cand = (x[second] - lo[second]) * es if es > 0 else (x[second] - up[second]) * es
                if cand < delta:
                    delta, lim, bound = cand, second, (lo[second] if es > 0 else up[second])
                for t in range(tf + 1, ts + 1):
                    j = 2 * n_t + t - 1
                    if up[j] - x[j] < delta:
                        delta, lim, bound = up[j] - x[j], j, up[j]
                x[first] += delta / ef
                x[second] -= delta / es
                for t in range(tf + 1, ts + 1):
                    x[2 * n_t + t - 1] += delta
                x[lim] = bound
                changed = True
            a = b + 1
        if not changed:
            break

    status = np.empty(n, dtype=np.int64)
    for j in range(n):
        if _interior(x, lo, up, j, tol):
            status[j] = BASIC
        elif lo[j] < up[j] and abs(x[j] - up[j]) < abs(x[j] - lo[j]):
            status[j] = AT_UPPER
            x[j] = up[j]
        else:
            status[j] = AT_LOWER
            x[j] = lo[j]
    a = 0
    while a < n_t:
        b = a
        while b + 1 < n_t and status[2 * n_t + b] == BASIC:
            b += 1
        n_basic = 0
        pick = -1
        for t in range(a, b + 1):
            for j in (t, n_t + t):
                if status[j] == BASIC:
                    n_basic += 1
                elif pick < 0 or (lo[pick] == up[pick] and lo[j] < up[j]):
                    pick = j
        if n_basic == 0:
            status[pick] = BASIC
        elif n_basic > 1:
            return False, np.zeros(n_t, dtype=np.int64), status
        a = b + 1
    basis = np.empty(n_t, dtype=np.int64)
    k = 0
    for j in range(n):
        if status[j] == BASIC:
            if k == n_t:
                return False, basis, status
            basis[k] = j
            k += 1
    return k == n_t, basis, status


PWL_TOL = 1e-11


@njit(cache=True)
def _pwl_eval(xs, vs, u):
    """Value of the piecewise-linear function (xs, vs) at u (clamped to its domain)."""
    k = xs.shape[0]
    if k == 1 or u <= xs[0]:
        return vs[0]
    if u >= xs[k - 1]:
        return vs[k - 1]
    lo = 0
    hi = k - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= u:
            lo = mid
        else:
            hi = mid
    w = (u - xs[lo]) / (xs[hi] - xs[lo])
    return vs[lo] + w * (vs[hi] - vs[lo])


@njit(cache=True)
def _sorted_unique(points, n, tol):
    pts = np.sort(points[:n])
    out = np.empty(n)
    m = 0
    for i in range(n):
        if m == 0 or pts[i] - out[m - 1] > tol:
            out[m] = pts[i]
            m += 1
    return out[:m]


@njit(cache=True)
def _compress(xs, vs, n, tol):
    """Drop collinear interior breakpoints."""
    ox = np.empty(n)
    ov = np.empty(n)
    m = 0
    for i in range(n):
        if m >= 2:
            x0, v0, x1, v1 = ox[m - 2], ov[m - 2], ox[m - 1], ov[m - 1]
            # value of the line through points m-2 and i at x1
            line = v0 + (vs[i] - v0) * (x1 - x0) / (xs[i] - x0)
            if abs(line - v1) <= tol * (1.0 + abs(v1)):
                m -= 1
        ox[m] = xs[i]
        ov[m] = vs[i]
        m += 1
    return ox[:m].copy(), ov[:m].copy()


@njit(cache=True)
def _envelope_on(sa, sb, fa, fb, n_f, out_x, out_v, m):
    """Append the upper envelope of n_f lines on [sa, sb] (given by end values) to out."""
    pts = np.empty(2 + n_f * n_f)
    pts[0] = sa
    pts[1] = sb
    k = 2
    if sb > sa:
        for i in range(n_f):
            for j in range(i + 1, n_f):
                da = fa[i] - fa[j]
                db = fb[i] - fb[j]
                if (da > 0.0 and db < 0.0) or (da < 0.0 and db > 0.0):
                    pts[k] = sa + (sb - sa) * da / (da - db)
                    k += 1
    pts = np.sort(pts[:k])
    for i in range(k):
        s = pts[i]
        if m > 0 and s - out_x[m - 1] <= PWL_TOL:
            continue
        w = 0.0 if sb <= sa else (s - sa) / (sb - sa)
        best = -np.inf
        for j in range(n_f):
            v = fa[j] + w * (fb[j] - fa[j])
            if v > best:
                best = v
        out_x[m] = s
        out_v[m] = best
        m += 1
    return m


@njit(cache=True)
def _case_step(xs, vs, a, lo, hi, smin, smax):
    """h(s) = max over e in [lo, hi] with s + e in dom(V) of a e + V(s + e), as PWL.

    With W(u) = V(u) + a u the inner maximum over the window [l(s), r(s)] of a
    piecewise-linear W is attained at a window end or a breakpoint inside, so between
    the candidate points u_k - lo, u_k - hi it is the upper envelope of two lines and a
    constant.
    """
    L = xs[0]
    U = xs[-1]
    s_lo = max(smin, L - hi)
    s_hi = min(smax, U - lo)
    if s_lo > s_hi + PWL_TOL:
        return np.empty(0), np.empty(0)
    if s_hi < s_lo:
        s_hi = s_lo
    k = xs.shape[0]
    W = vs + a * xs
    cand = np.empty(2 * k + 2)
    cand[0] = s_lo
    cand[1] = s_hi
    n = 2
    for i in range(k):
        for c in (xs[i] - lo, xs[i] - hi):
            if s_lo < c < s_hi:
                cand[n] = c
                n += 1
    grid = _sorted_unique(cand, n, PWL_TOL)
    g = grid.shape[0]
    out_x = np.empty(4 * g + 4)
    out_v = np.empty(4 * g + 4)
    m = 0
    fa = np.empty(3)
    fb = np.empty(3)
    if g == 1:
        s = grid[0]
        l = max(s + lo, L)
        r = min(s + hi, U)
        best = max(_pwl_eval(xs, W, l), _pwl_eval(xs, W, r))
        for i in range(k):
            if l < xs[i] < r and W[i] > best:
                best = W[i]
        out_x[0] = s
        out_v[0] = best - a * s
        return out_x[:1].copy(), out_v[:1].copy()
    for t in range(g - 1):
        sa = grid[t]
        sb = grid[t + 1]
        mid = 0.5 * (sa + sb)
        lm = max(mid + lo, L)
        rm = min(mid + hi, U)
        const = -np.inf
        for i in range(k):
            if lm < xs[i] < rm and W[i] > const:
                const = W[i]
        fa[0] = _pwl_eval(xs, W, max(sa + lo, L))
        fb[0] = _pwl_eval(xs, W, max(sb + lo, L))
        fa[1] = _pwl_eval(xs, W, min(sa + hi, U))
        fb[1] = _pwl_eval(xs, W, min(sb + hi, U))
        n_f = 2
        if const > -np.inf:
            fa[2] = const
            fb[2] = const
            n_f = 3
        m = _envelope_on(sa, sb, fa, fb, n_f, out_x, out_v, m)
    for i in range(m):
        out_v[i] -= a * out_x[i]
    return _compress(out_x, out_v, m, PWL_TOL)


@njit(cache=True)
def _pwl_max(x1, v1, x2, v2):
    """Pointwise maximum of two PWL functions on the union of their (overlapping) domains."""
    if x1.shape[0] == 0:
        return x2, v2
    if x2.shape[0] == 0:
        return x1, v1
    cand = np.concatenate((x1, x2))
    grid = _sorted_unique(cand, cand.shape[0], PWL_TOL)
    g = grid.shape[0]
    out_x = np.empty(3 * g + 2)
    out_v = np.empty(3 * g + 2)
    fa = np.empty(2)
    fb = np.empty(2)
    m = 0
    if g == 1:
        out_x[0] = grid[0]
        out_v[0] = max(v1[0], v2[0])
        return out_x[:1].copy(), out_v[:1].copy()
    for t in range(g - 1):
        sa = grid[t]
        sb = grid[t + 1]
        mid = 0.5 * (sa + sb)
        n_f = 0
        if x1[0] - PWL_TOL <= mid <= x1[-1] + PWL_TOL:
            fa[n_f] = _pwl_eval(x1, v1, sa)
            fb[n_f] = _pwl_eval(x1, v1, sb)
            n_f += 1
        if x2[0] - PWL_TOL <= mid <= x2[-1] + PWL_TOL:
            fa[n_f] = _pwl_eval(x2, v2, sa)
            fb[n_f] = _pwl_eval(x2, v2, sb)
            n_f += 1
        m = _envelope_on(sa, sb, fa, fb, n_f, out_x, out_v, m)
    return _compress(out_x, out_v, m, PWL_TOL)


@njit(cache=True)
def _restricted_dp(prices, cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max, s_start, s_end):
    r"""Exact daily optimum without simultaneous charge/discharge by DP over the SoC.

    :math:`V_T` is the point :math:`s_{end}` with value 0 and

    .. math::

        V_t(s) = \max\Big(\max_{0 \le e \le \eta_c\bar c_t} a^+_t e + V_{t+1}(s+e),\;
                 \max_{-\bar d_t/\eta_d \le e \le 0} a^-_t e + V_{t+1}(s+e)\Big)

    with :math:`a^+_t = -g_{c_t}/\eta_c` (charge only) and :math:`a^-_t = g_{d_t}\eta_d`
    (discharge only). All :math:`V_t` are continuous piecewise-linear functions of the
    SoC and are computed exactly (breakpoints and crossings); the schedule is recovered
    by a forward pass that picks the maximising case and SoC change per period.

    Returns:
        (ok, x in the original variables (c, d, s), revenue).
    """
    n_t = prices.shape[0]
    xs_list = [np.array([s_end])]
    vs_list = [np.array([0.0])]
    xs = xs_list[0]
    vs = vs_list[0]
    for t in range(n_t - 1, -1, -1):
        gc = prices[t]
        gd = cycle_cost - prices[t]
        hx, hv = _case_step(xs, vs, -gc / eta_c, 0.0, eta_c * c_up[t], e_min, e_max)
        dx, dv = _case_step(xs, vs, gd * eta_d, -d_up[t] / eta_d, 0.0, e_min, e_max)
        xs, vs = _pwl_max(hx, hv, dx, dv)
        if xs.shape[0] == 0:
            return False, np.zeros(3 * n_t - 1), -np.inf
        xs_list.append(xs)
        vs_list.append(vs)
    if s_start < xs[0] - 1e-9 or s_start > xs[-1] + 1e-9:
        return False, np.zeros(3 * n_t - 1), -np.inf

    x = np.zeros(3 * n_t - 1)
    s = s_start
    revenue = 0.0
    for t in range(n_t):
        nx = xs_list[n_t - 1 - t]
        nv = vs_list[n_t - 1 - t]
        gc = prices[t]
        gd = cycle_cost - prices[t]
        best = -np.inf
        best_u = s
        for case in range(2):
            if case == 0:
                a, lo, hi = -gc / eta_c, 0.0, eta_c * c_up[t]
            else:
                a, lo, hi = gd * eta_d, -d_up[t] / eta_d, 0.0
            left = max(s + lo, nx[0])
            right = min(s + hi, nx[-1])
            if left > right + 1e-9:
                continue
            right = max(right, left)
            for i in range(-2, nx.shape[0]):
                if i == -2:
                    u = left
                elif i == -1:
                    u = right
                else:
                    u = nx[i]
                    if not left < u < right:
                        continue
                v = a * (u - s) + _pwl_eval(nx, nv, u)
                # ties: prefer the smaller SoC change (idle before cycling)
                if v > best + 1e-9 * (1.0 + abs(v)) or (
                        v >= best - 1e-9 * (1.0 + abs(v)) and abs(u - s) < abs(best_u - s)):
                    best = max(v, best)
                    best_u = u
        e = best_u - s
        if t == n_t - 1:
            e = s_end - s
        if e >= 0.0:
            x[t] = min(e / eta_c, c_up[t])
            revenue -= gc * x[t]
        else:
            x[n_t + t] = min(-e * eta_d, d_up[t])
            revenue -= gd * x[n_t + t]
        s += e
        if t < n_t - 1:
            x[2 * n_t + t] = min(max(s, e_min), e_max)
    return True, x, revenue


@njit(parallel=True, cache=True)
def _restricted_batch(A0, lo0, up0, prices, cycle_cost, eta_c, eta_d, s_start, s_end):
    """:func:`_restricted_dp` for each row of ``prices`` in parallel, plus the vertex basis."""
    n_p, n_t = prices.shape
    n = 3 * n_t - 1
    codes = np.empty(n_p, dtype=np.int64)
    X = np.zeros((n_p, n))
    bases = np.zeros((n_p, n_t), dtype=np.int64)
    statuses = np.zeros((n_p, n), dtype=np.int64)
    objs = np.empty(n_p)
    c_up = up0[:n_t].copy()
    d_up = up0[n_t:2 * n_t].copy()
    e_min = lo0[2 * n_t] if n_t > 1 else s_start
    e_max = up0[2 * n_t] if n_t > 1 else s_start
    for k in prange(n_p):
        ok, x, rev = _restricted_dp(prices[k], cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max,
                                    s_start, s_end)
        code = INFEASIBLE
        if ok:
            ok2, basis, status = _purify(A0, lo0, up0, x, n_t, 1e-9)
            if ok2:
                bases[k] = basis
                statuses[k] = status
                code = OPTIMAL
        codes[k] = code
        X[k] = x
        objs[k] = -rev
    return codes, X, bases, statuses, objs
