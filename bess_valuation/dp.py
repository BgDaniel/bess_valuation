r"""Exact daily optimisation by dynamic programming over the state of charge.

For one day with prices :math:`p_t` [€/MWh], cycle cost :math:`\lambda` [€/MWh
discharged] and end SoC :math:`s_{end}`, :math:`V_t(s)` is the best revenue from period
t to the end of the day when the SoC at the start of t is s [MWh]:

.. math::

    V_T(s_{end}) = 0, \qquad
    V_t(s) = \max\Big(\max_{0 \le e \le \eta_c\bar c} a^+_t e + V_{t+1}(s+e),\;
                     \max_{-\bar d/\eta_d \le e \le 0} a^-_t e + V_{t+1}(s+e)\Big)

with :math:`a^+_t = -p_t/\eta_c` (charge only), :math:`a^-_t = (\lambda - p_t)\eta_d`
(discharge only) and :math:`e_{min} \le s+e \le e_{max}`; F(s_start, s_end, p) =
:math:`V_0(s_{start})`. The two cases are the per-period case split for
:math:`c_t d_t = 0`, so this solves the daily problem without simultaneous charge and
discharge exactly, without an LP.

Every :math:`V_t` is a continuous piecewise-linear function of s (on synthetic
quarter-hour prices about 10, at most ~20 breakpoints) and is stored exactly in
preallocated buffers:

* If :math:`V_{t+1}` is concave and the period gain is concave
  (:math:`a^-_t \ge a^+_t`, i.e. no negative price), the step is a sup-convolution of
  concave functions: merge the segments by decreasing slope (:func:`_conv_concave`).
* Otherwise both cases are computed as windowed maxima (:func:`_case`: candidate points
  from two merged sorted lists, monotone pointers, a monotone deque for the window
  maximum) and combined by their pointwise maximum (:func:`_max`).

One backward pass for a given :math:`s_{end}` yields :math:`V_0` and hence F for all start
SoCs on the grid at once. Schedules are recovered by a forward pass over the stored
:math:`V_t`. A buffer overflow (never observed) falls back to the simplex.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numba import njit, prange

from bess_valuation.config import BatteryConfig

logger = logging.getLogger(__name__)

TOL = 1e-11
CAP = 512


@njit(cache=True, inline="always")
def _ev(xs, ys, k, u, p):
    """Value at u of the PWL (xs, ys) of length k, pointer p only moves forward."""
    if k == 1 or u <= xs[0]:
        return ys[0], p
    if u >= xs[k - 1]:
        return ys[k - 1], p
    while p < k - 2 and xs[p + 1] <= u:
        p += 1
    w = (u - xs[p]) / (xs[p + 1] - xs[p])
    return ys[p] + w * (ys[p + 1] - ys[p]), p


@njit(cache=True, inline="always")
def _push(ox, ov, m, x, v):
    if m > 0 and x - ox[m - 1] <= TOL:
        if v > ov[m - 1]:
            ov[m - 1] = v
        return m
    if m >= 2:
        x0 = ox[m - 2]
        v0 = ov[m - 2]
        line = v0 + (v - v0) * (ox[m - 1] - x0) / (x - x0)
        if abs(line - ov[m - 1]) <= TOL * (1.0 + abs(ov[m - 1])):
            ox[m - 1] = x
            ov[m - 1] = v
            return m
    ox[m] = x
    ov[m] = v
    return m + 1


@njit(cache=True)
def _env(sa, sb, fa, fb, nf, a, pts, ox, ov, m):
    """Append the upper envelope of nf lines on [sa, sb] (minus a*s) to (ox, ov)."""
    n = 0
    pts[n] = sa
    n += 1
    if sb > sa:
        for i in range(nf):
            for j in range(i + 1, nf):
                da = fa[i] - fa[j]
                db = fb[i] - fb[j]
                if (da > 0.0 and db < 0.0) or (da < 0.0 and db > 0.0):
                    pts[n] = sa + (sb - sa) * da / (da - db)
                    n += 1
        pts[n] = sb
        n += 1
        for i in range(1, n):  # insertion sort, n <= 5
            x = pts[i]
            j = i - 1
            while j >= 0 and pts[j] > x:
                pts[j + 1] = pts[j]
                j -= 1
            pts[j + 1] = x
    for i in range(n):
        s = pts[i]
        w = 0.0 if sb <= sa else (s - sa) / (sb - sa)
        best = -np.inf
        for j in range(nf):
            v = fa[j] + w * (fb[j] - fa[j])
            if v > best:
                best = v
        m = _push(ox, ov, m, s, best - a * s)
    return m


@njit(cache=True)
def _case(xs, vs, k, a, lo, hi, smin, smax, W, cand, dq, fa, fb, pts, ox, ov):
    """h(s) = max_{e in [lo, hi], s+e in dom V} a e + V(s+e); returns length or -1 (overflow)."""
    L = xs[0]
    U = xs[k - 1]
    s_lo = max(smin, L - hi)
    s_hi = min(smax, U - lo)
    if s_lo > s_hi + TOL:
        return 0
    if s_hi < s_lo:
        s_hi = s_lo
    for i in range(k):
        W[i] = vs[i] + a * xs[i]
    n = 1
    cand[0] = s_lo
    i = 0
    j = 0
    while i < k or j < k:
        if j >= k or (i < k and xs[i] - lo <= xs[j] - hi):
            c = xs[i] - lo
            i += 1
        else:
            c = xs[j] - hi
            j += 1
        if c > cand[n - 1] + TOL and c < s_hi - TOL:
            cand[n] = c
            n += 1
    if s_hi > cand[n - 1] + TOL:
        cand[n] = s_hi
        n += 1
    if n == 1:
        s = s_lo
        l = max(s + lo, L)
        r = min(s + hi, U)
        best, _ = _ev(xs, W, k, l, 0)
        v, _ = _ev(xs, W, k, r, 0)
        best = max(best, v)
        for q in range(k):
            if l < xs[q] < r and W[q] > best:
                best = W[q]
        ox[0] = s
        ov[0] = best - a * s
        return 1
    pl = 0
    pr = 0
    head = 0
    tail = 0
    nxt = 0
    m = 0
    for t in range(n - 1):
        sa = cand[t]
        sb = cand[t + 1]
        mid = 0.5 * (sa + sb)
        lm = max(mid + lo, L)
        rm = min(mid + hi, U)
        while nxt < k and xs[nxt] < rm:
            while tail > head and W[dq[tail - 1]] <= W[nxt]:
                tail -= 1
            dq[tail] = nxt
            tail += 1
            nxt += 1
        while head < tail and xs[dq[head]] <= lm:
            head += 1
        fa[0], pl = _ev(xs, W, k, max(sa + lo, L), pl)
        fb[0], pl = _ev(xs, W, k, max(sb + lo, L), pl)
        fa[1], pr = _ev(xs, W, k, min(sa + hi, U), pr)
        fb[1], pr = _ev(xs, W, k, min(sb + hi, U), pr)
        nf = 2
        if head < tail:
            fa[2] = W[dq[head]]
            fb[2] = fa[2]
            nf = 3
        m = _env(sa, sb, fa, fb, nf, a, pts, ox, ov, m)
        if m > ox.shape[0] - 8:
            return -1
    return m


@njit(cache=True)
def _max(x1, v1, k1, x2, v2, k2, cand, fa, fb, pts, ox, ov):
    """Pointwise maximum of two PWLs on the union of their domains."""
    if k1 == 0 or k2 == 0:
        src_x, src_v, kk = (x2, v2, k2) if k1 == 0 else (x1, v1, k1)
        for i in range(kk):
            ox[i] = src_x[i]
            ov[i] = src_v[i]
        return kk
    n = 0
    i = 0
    j = 0
    while i < k1 or j < k2:
        if j >= k2 or (i < k1 and x1[i] <= x2[j]):
            c = x1[i]
            i += 1
        else:
            c = x2[j]
            j += 1
        if n == 0 or c > cand[n - 1] + TOL:
            cand[n] = c
            n += 1
    if n == 1:
        ox[0] = cand[0]
        ov[0] = max(v1[0], v2[0])
        return 1
    p1 = 0
    p2 = 0
    m = 0
    for t in range(n - 1):
        sa = cand[t]
        sb = cand[t + 1]
        mid = 0.5 * (sa + sb)
        nf = 0
        a1, p1 = _ev(x1, v1, k1, sa, p1)
        b1, p1 = _ev(x1, v1, k1, sb, p1)
        a2, p2 = _ev(x2, v2, k2, sa, p2)
        b2, p2 = _ev(x2, v2, k2, sb, p2)
        if x1[0] - TOL <= mid <= x1[k1 - 1] + TOL:
            fa[nf] = a1
            fb[nf] = b1
            nf += 1
        if x2[0] - TOL <= mid <= x2[k2 - 1] + TOL:
            fa[nf] = a2
            fb[nf] = b2
            nf += 1
        m = _env(sa, sb, fa, fb, nf, 0.0, pts, ox, ov, m)
        if m > ox.shape[0] - 8:
            return -1
    return m


@njit(cache=True)
def _conv_concave(xs, vs, k, a_plus, e_plus, a_minus, e_minus, smin, smax, ox, ov):
    """Fast DP step when V and the period gain are both concave (a_minus >= a_plus).

    With d = s - u the step is the sup-convolution of V with f~(d) (slope -a_plus on
    [-e_plus, 0], slope -a_minus on [0, e_minus]); for concave functions it merges the
    segments in order of decreasing slope. The result is clipped to [smin, smax].
    """
    # merged segments: start point, then segments of V and f~ by decreasing slope
    x = xs[0] - e_plus
    v = vs[0] + a_plus * e_plus
    m = 0
    i = 0
    f_left = 0
    if e_plus <= 0.0:
        f_left = 1
    if e_minus <= 0.0 and f_left == 1:
        f_left = 2
    # emit points while merging, clipped later
    tx = ox
    tv = ov
    tx[0] = x
    tv[0] = v
    m = 1
    while i < k - 1 or f_left < 2:
        sv = -np.inf
        if i < k - 1:
            sv = (vs[i + 1] - vs[i]) / (xs[i + 1] - xs[i])
        sf = -np.inf
        lf = 0.0
        if f_left == 0:
            sf, lf = -a_plus, e_plus
        elif f_left == 1:
            sf, lf = -a_minus, e_minus
        if i < k - 1 and (f_left >= 2 or sv >= sf):
            ln = xs[i + 1] - xs[i]
            x += ln
            v += sv * ln
            i += 1
        else:
            x += lf
            v += sf * lf
            f_left += 1
            if f_left == 1 and e_minus <= 0.0:
                f_left = 2
        tx[m] = x
        tv[m] = v
        m += 1
    # clip to [smin, smax] in place
    lo_i = 0
    while lo_i < m - 1 and tx[lo_i + 1] <= smin:
        lo_i += 1
    hi_i = m - 1
    while hi_i > 0 and tx[hi_i - 1] >= smax:
        hi_i -= 1
    if tx[lo_i] > smax + TOL or tx[hi_i] < smin - TOL:
        return 0
    n = 0
    for j in range(lo_i, hi_i + 1):
        xj = tx[j]
        vj = tv[j]
        if xj < smin and j + 1 <= hi_i:
            vj = vj + (tv[j + 1] - vj) * (smin - xj) / (tx[j + 1] - xj)
            xj = smin
        if xj > smax and j - 1 >= lo_i:
            vj = tv[j - 1] + (vj - tv[j - 1]) * (smax - tx[j - 1]) / (xj - tx[j - 1])
            xj = smax
        xj = min(max(xj, smin), smax)
        if n > 0 and xj - ox[n - 1] <= TOL:
            ov[n - 1] = max(ov[n - 1], vj)
            continue
        ox[n] = xj
        ov[n] = vj
        n += 1
    return n


@njit(cache=True)
def _is_concave(xs, vs, k):
    prev = np.inf
    for i in range(k - 1):
        sl = (vs[i + 1] - vs[i]) / (xs[i + 1] - xs[i])
        if sl > prev + 1e-9 * (1.0 + abs(prev)):
            return False
        prev = sl
    return True


@njit(cache=True)
def _backward(prices, cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max, s_end, buf, ibuf,
              keep_x, keep_v, keep_k):
    """Backward pass; V_0 ends in buf[0], buf[1]. Returns its length, 0 if infeasible, -1 on overflow.

    If keep_x has rows, V_t is stored in keep_x[t], keep_v[t] (length keep_k[t]) for t = 0..T.
    """
    cap = buf.shape[1]
    Ax, Av, Hx, Hv, Dx, Dv, W, cand = buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]
    fa, fb, pts = buf[8, :3], buf[8, 3:6], buf[8, 8:16]
    n_t = prices.shape[0]
    keep = keep_x.shape[0] > 0
    k = 1
    Ax[0] = s_end
    Av[0] = 0.0
    if keep:
        keep_x[n_t, 0] = s_end
        keep_v[n_t, 0] = 0.0
        keep_k[n_t] = 1
    concave = True
    for t in range(n_t - 1, -1, -1):
        a_plus = -prices[t] / eta_c
        a_minus = (cycle_cost - prices[t]) * eta_d
        if concave and a_minus >= a_plus:
            k = _conv_concave(Ax, Av, k, a_plus, eta_c * c_up, a_minus, d_up / eta_d, e_min, e_max,
                              Hx, Hv)
            if k > cap - 8:
                return -1
            for i in range(k):
                Ax[i] = Hx[i]
                Av[i] = Hv[i]
        else:
            kh = _case(Ax, Av, k, a_plus, 0.0, eta_c * c_up, e_min, e_max, W, cand, ibuf, fa, fb,
                       pts, Hx, Hv)
            kd = _case(Ax, Av, k, a_minus, -d_up / eta_d, 0.0, e_min, e_max, W, cand, ibuf, fa, fb,
                       pts, Dx, Dv)
            if kh < 0 or kd < 0:
                return -1
            k = _max(Hx, Hv, kh, Dx, Dv, kd, cand, fa, fb, pts, Ax, Av)
            if k < 0:
                return -1
            if k > 0:
                concave = _is_concave(Ax, Av, k)
        if k == 0:
            return 0
        if keep:
            for i in range(k):
                keep_x[t, i] = Ax[i]
                keep_v[t, i] = Av[i]
            keep_k[t] = k
    return k


@njit(parallel=True, cache=True)
def _value_table(prices, cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max, grid):
    n_q = prices.shape[0]
    g = grid.shape[0]
    F = np.empty((n_q, g, g))
    overflow = np.zeros((n_q, g), dtype=np.bool_)
    n_chunks = min(n_q, 256)
    empty = np.empty((0, CAP))
    empty_k = np.empty(0, dtype=np.int64)
    for c in prange(n_chunks):
        buf = np.empty((9, CAP))
        ibuf = np.empty(CAP, dtype=np.int64)
        for q in range(c, n_q, n_chunks):
            for j in range(g):
                k = _backward(prices[q], cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max, grid[j],
                              buf, ibuf, empty, empty, empty_k)
                if k < 0:
                    overflow[q, j] = True
                    continue
                p = 0
                for i in range(g):
                    s0 = grid[i]
                    if k > 0 and buf[0, 0] - 1e-9 <= s0 <= buf[0, k - 1] + 1e-9:
                        F[q, i, j], p = _ev(buf[0], buf[1], k, s0, p)
                    else:
                        F[q, i, j] = -np.inf
    return F, overflow


@njit(cache=True)
def _forward(prices, cycle_cost, eta_c, eta_d, c_up, d_up, s_start, s_end, keep_x, keep_v, keep_k,
             charge, discharge):
    """Recover the optimal schedule from the stored V_t; returns the revenue."""
    n_t = prices.shape[0]
    s = s_start
    revenue = 0.0
    for t in range(n_t):
        nx = keep_x[t + 1]
        nv = keep_v[t + 1]
        nk = keep_k[t + 1]
        best = -np.inf
        best_u = s
        for case in range(2):
            if case == 0:
                a, lo, hi = -prices[t] / eta_c, 0.0, eta_c * c_up
            else:
                a, lo, hi = (cycle_cost - prices[t]) * eta_d, -d_up / eta_d, 0.0
            left = max(s + lo, nx[0])
            right = min(s + hi, nx[nk - 1])
            if left > right + 1e-9:
                continue
            right = max(right, left)
            p = 0
            for i in range(-2, nk):
                if i == -2:
                    u = left
                elif i == -1:
                    u = right
                else:
                    u = nx[i]
                    if not left < u < right:
                        continue
                v, _ = _ev(nx, nv, nk, u, 0)
                v += a * (u - s)
                if v > best + 1e-9 * (1.0 + abs(v)) or (
                        v >= best - 1e-9 * (1.0 + abs(v)) and abs(u - s) < abs(best_u - s)):
                    best = max(v, best)
                    best_u = u
            _ = p
        e = best_u - s
        if t == n_t - 1:
            e = s_end - s
        if e >= 0.0:
            charge[t] = min(e / eta_c, c_up)
            discharge[t] = 0.0
            revenue -= prices[t] * charge[t]
        else:
            discharge[t] = min(-e * eta_d, d_up)
            charge[t] = 0.0
            revenue += (prices[t] - cycle_cost) * discharge[t]
        s += e
    return revenue


@njit(parallel=True, cache=True)
def _schedule_batch(prices, cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max, s_start, s_end):
    n_q, n_t = prices.shape
    charge = np.zeros((n_q, n_t))
    discharge = np.zeros((n_q, n_t))
    revenue = np.full(n_q, -np.inf)
    ok = np.zeros(n_q, dtype=np.bool_)
    n_chunks = min(n_q, 256)
    for c in prange(n_chunks):
        buf = np.empty((9, CAP))
        ibuf = np.empty(CAP, dtype=np.int64)
        keep_x = np.empty((n_t + 1, CAP))
        keep_v = np.empty((n_t + 1, CAP))
        keep_k = np.zeros(n_t + 1, dtype=np.int64)
        for q in range(c, n_q, n_chunks):
            k = _backward(prices[q], cycle_cost, eta_c, eta_d, c_up, d_up, e_min, e_max, s_end[q],
                          buf, ibuf, keep_x, keep_v, keep_k)
            if k <= 0:
                continue
            if not (keep_x[0, 0] - 1e-9 <= s_start[q] <= keep_x[0, k - 1] + 1e-9):
                continue
            revenue[q] = _forward(prices[q], cycle_cost, eta_c, eta_d, c_up, d_up, s_start[q],
                                  s_end[q], keep_x, keep_v, keep_k, charge[q], discharge[q])
            ok[q] = True
    return revenue, charge, discharge, ok


# ------------------------------------------------------------------ Python interface


def _params(config: BatteryConfig) -> tuple[float, float, float, float, float, float]:
    if config.allow_simultaneous_charge_discharge:
        raise ValueError("the DP solves the problem without simultaneous charge and discharge; "
                         "use another daily solver for allow_simultaneous_charge_discharge=True")
    return (config.eta_charge, config.eta_discharge, config.p_charge_max * config.dt,
            config.p_discharge_max * config.dt, config.e_min, config.e_max)


def dp_daily_values(config: BatteryConfig, prices: np.ndarray, grid: np.ndarray,
                    cycle_cost: float = 0.0, progress: Callable[[float], None] | None = None,
                    n_parts: int = 50) -> np.ndarray:
    """F(s_i, s_j, p) for all price vectors and SoC grid pairs by DP.

    Args:
        config: Battery parameters (no simultaneous charge/discharge).
        prices: Prices [€/MWh], shape (..., T).
        grid: SoC grid [MWh], shape (G,).
        cycle_cost: Cycle cost lambda [€/MWh discharged].
        progress: Optional callback with the finished fraction (0..1).
        n_parts: Number of parts the price vectors are processed in (progress granularity).

    Returns:
        Revenues [€], shape (..., G, G) with axes (s_start, s_end); -inf if infeasible.
    """
    from tqdm import tqdm

    params = _params(config)
    lead = prices.shape[:-1]
    flat = np.ascontiguousarray(prices.reshape(-1, prices.shape[-1]), dtype=np.float64)
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    out = np.empty((flat.shape[0], grid.size, grid.size))
    bounds = np.linspace(0, flat.shape[0], min(n_parts, flat.shape[0]) + 1).astype(int)
    n_overflow = 0
    for k in tqdm(range(bounds.size - 1), desc="daily values (DP)", unit="part", leave=False):
        lo, hi = bounds[k], bounds[k + 1]
        F, overflow = _value_table(flat[lo:hi], float(cycle_cost), *params, grid)
        if overflow.any():
            n_overflow += int(overflow.sum())
            _fallback_values(config, flat[lo:hi], grid, cycle_cost, F, overflow)
        out[lo:hi] = F
        if progress is not None:
            progress(hi / flat.shape[0])
    if n_overflow:
        logger.warning("DP buffer overflow in %d backward passes, solved by the simplex instead", n_overflow)
    return out.reshape(lead + (grid.size, grid.size))


def _fallback_values(config, prices, grid, cycle_cost, F, overflow):
    from bess_valuation.polytope import build_daily_lp
    from bess_valuation.simplex import solve_daily

    for q, j in zip(*np.nonzero(overflow)):
        for i in range(grid.size):
            sol = solve_daily(build_daily_lp(config, grid[i], grid[j]), prices[q], cycle_cost,
                              allow_simultaneous=False)
            F[q, i, j] = sol.revenue[0] if sol.status[0] == 0 else -np.inf


def dp_daily_schedules(config: BatteryConfig, prices: np.ndarray, s_start: np.ndarray,
                       s_end: np.ndarray, cycle_cost: float = 0.0
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal schedules for (price vector, s_start, s_end) triples by DP.

    Returns:
        (revenue [€] (Q,), charge [MWh] (Q, T), discharge [MWh] (Q, T)).
    """
    params = _params(config)
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    revenue, charge, discharge, ok = _schedule_batch(
        prices, float(cycle_cost), *params, np.asarray(s_start, dtype=np.float64),
        np.asarray(s_end, dtype=np.float64))
    if not ok.all():
        from bess_valuation.polytope import build_daily_lp
        from bess_valuation.simplex import solve_daily

        for q in np.flatnonzero(~ok):
            sol = solve_daily(build_daily_lp(config, s_start[q], s_end[q]), prices[q], cycle_cost,
                              allow_simultaneous=False)
            revenue[q], charge[q], discharge[q] = sol.revenue[0], sol.charge[0], sol.discharge[0]
    return revenue, charge, discharge
