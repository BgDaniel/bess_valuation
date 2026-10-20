# Handoff prompt: continue `bess_valuation` from milestone 2

You are continuing the Python project `bess_valuation`: valuation and optimal dispatch of a
battery (BESS) on stochastic day-ahead (DA) and intraday-auction (IDA1–3) prices by backward
Bellman induction (LSMC), with an own Numba simplex and a vertex library instead of an LP solver
in the hot loop.

## Read first

1. `docs/SPEC.md` — the full specification (sections 0–9 + appendix). It is binding.
2. `CLAUDE.md` — working rules and milestone checklist. In short:
   - Talk to the user in **German, addressing them as "du"**; define every mathematical symbol on
     first use. Code, docstrings, identifiers in English.
   - Python 3.11+, Poetry, type annotations everywhere, Google-style docstrings, tests with
     `unittest` only (no pytest), plain functions and a few dataclasses, no deep hierarchies.
   - Hot loops in Numba (`@njit`, `prange`). **No external LP solver in the hot loop**; HiGHS /
     scipy only inside tests as a reference. Never reintroduce LP calls into the backward induction.
   - `logging` (`logger = logging.getLogger(__name__)`) in every long-running step and a `tqdm`
     progress bar for long loops — **explicitly requested by the user for the backward induction
     over days and SoC grid points**, also for the forward simulation and the λ calibration.
   - Change only what is asked for; work milestone by milestone.

## Setup

```bash
poetry install                       # in-project venv (.venv), Python 3.14 / numba 0.68 used so far
poetry run python -m unittest discover -s tests -v      # 34 tests, ~30 s incl. Numba compile
poetry run python -m bess_valuation generate-prices --config configs/prices_synthetic.yaml \
    --out data/scenarios/synthetic_2027 --n-paths 500   # data/ and results/ are git-ignored
```

## State: milestone 1 is done (commit "Milestone 1: …")

| Module | Content |
|---|---|
| `config.py` | `BatteryConfig` (frozen dataclass, YAML in/out, validation), `periods_per_day`, `config_hash` (SHA-256 of the physical fields only: e_min/e_max, power limits, efficiencies, dt, allow_simultaneous). |
| `prices.py` | `PriceScenarios`: `da, ida1, ida2, ida3` (N, D, T) float32, `ida3` NaN for periods < T/2; `dates`, `dt`, optional `X` (N, D, m) and `conditional_mean` (N, D, T) where entry d is E[p^DA_{d+1} \| X_d]; `meta`. `save()` / `from_files()` for NPZ (default), Parquet, CSV (long format); `split(n_first)` for regression vs out-of-sample paths; `synthetic(...)`. |
| `price_generator.py` | `PriceGeneratorConfig` (flat dataclass, `configs/prices_synthetic.yaml`), `generate_prices(...)`, closed-form `conditional_mean`, `plot_generator` (5 figures), `generate_and_plot` → `results/price_generator/<timestamp>/`. Model and formulas in the module docstring; state X_d = (L_d, A_d, B_d). |
| `polytope.py` | Daily LP in bounded standard form: min gᵀx, Ax = b, l ≤ x ≤ u, x = (c_0..c_{T-1}, d_0..d_{T-1}, s_1..s_{T-1}), T rows `s_{t+1} − s_t − η_c c_t + d_t/η_d = 0`, b = (s_start, 0, …, 0, −s_end). **Only b depends on (s_start, s_end)**; A, l, u are shared over the SoC grid. `build_daily_lp(config, s_start, s_end, horizon="full"/"half")`, `DailyLP.with_soc`, `lp_cost(prices, cycle_cost)` → g = (p, −(p − λ), 0), `split_schedule`, `soc_path`. |
| `simplex.py` | Numba bounded revised primal simplex: closed-form feasible start `battery_start` (monotone charge/discharge, replaces Phase I), dense B⁻¹ with product-form rank-1 updates and refactorisation every 50 pivots, exact steepest-edge pricing (Goldfarb–Reid updates) with Bland fallback after 50 degenerate steps, bound flips, tolerances 1e-9. `solve_lp_core` (from a given basis, in place), `solve_batch(lp, costs, basis=None, var_status=None, up=None)` → `BatchResult(status, x, objective, basis, var_status, iterations, degenerate, warm)` in parallel with `prange`, warm start with cold fallback. Variable status codes `BASIC=0, AT_LOWER=1, AT_UPPER=2`. `solve_daily(lp, prices, cycle_cost, allow_simultaneous)` → `DailySolution`; c_t·d_t = 0 is enforced by a per-period case split (depth-first B&B on c_t = 0 / d_t = 0); each branch is a face of P, so the optimum stays a vertex of P. |
| `plots.py` | Shared style (`apply_style`, validated categorical palette, `STAGE_COLORS`, blue quantile ramp, `fan`, `hour_axis`, `save_figure` at 150 dpi). Valuation plots 1–14 still to do. |
| `io.py` | `new_run_dir(name, root)` → `results/<name>/<YYYYMMDD_HHMMSS>[_k]/`, never overwrites. |
| `cli.py` | `python -m bess_valuation [--log-level] generate-prices …`; `run` is a stub raising NotImplementedError. |
| empty stubs | `library.py`, `daily.py`, `lsmc.py`, `forward.py`, `results.py`, `cycles.py` (docstring only). |

Tests (`tests/`): config; polytope; generator (shapes, NaN pattern, reproducibility, conditional
mean vs Monte Carlo, zero-mean spreads with growing variance, deeper summer midday dip, plots,
I/O round trips); simplex vs HiGHS LP (random, simulated, degenerate, negative prices,
s_start ≠ s_end, cycle-cost shift, half-day, 8 h storage, infeasible pair, warm start, fallback)
and the case split vs a HiGHS MILP.

Benchmark (5000 DA days, T = 96, 1 MW / 2 MWh): cold 516 µs/LP (69 iterations), warm 440 µs/LP
(34 iterations), HiGHS via scipy ≈ 13.7 ms/LP. Warm start halves iterations but not time: the
per-LP setup (B⁻¹ and steepest-edge weights, O(m²n)) dominates — candidate for milestone 6.

Known deviations / open points to keep in mind:
- No general Phase I: the closed-form start exists whenever the LP is feasible and is also the
  fallback for infeasible warm bases. Ask the user before adding a Phase I.
- Generator parameters are plausible defaults (mean DA ≈ 80 €/MWh, ≈ 2 % negative quarter-hours,
  ≈ 30 % negative at summer noon), not a calibration.
- Prices are float32 in memory; convert to float64 before building cost vectors.

## What remains (SPEC section 9)

**Milestone 2 — vertex library (`library.py`) and DA-only daily lookup (`daily.py`).**
- Vertex = full active set (variables at a bound, including degenerate basics at a bound); store
  per vertex: active set key, a basis + `var_status` that realises it, schedule (c, d) and
  y = d − c, hit counts, validity in (s_start, s_end), neighbour pointers (lazy), edge directions
  (columns of −A_N⁻¹ / tangent-cone rays). Key: (config hash, horizon full/half, active set).
  Because A, l, u do not depend on the SoC pair, normal cones are shared across the grid; start
  with separate libraries per grid point, keyed by active set so they can be merged later.
- Query vectorised over paths: argmax over cached vertices via `Y @ p`, then certificate
  y_N = A_N^{-T} g ≥ 0 (in this code base: reduced costs of the stored basis have the optimal
  sign for each nonbasic variable's bound); at degenerate vertices check all tangent-cone rays;
  if violated, pivot with the own simplex from the stored basis (`solve_lp_core` /
  `solve_batch(basis=…)`) and insert the new vertex.
- Good–Turing estimate P(unseen) ≈ n₁/N; persistence to NPZ per config hash; `library_stats`.
- Tests: library lookup + certificate equals the direct simplex/HiGHS optimum on all scenarios of
  a small case; persistence round trip.
- Optional: warm-start table on PCA coefficients of the demeaned price curve (start only — the
  certificate uses the full price vector, so results stay exact).

**Milestone 3 — LSMC DA-only (`lsmc.py`), forward simulation (`forward.py`), `results.py`,
`run_engine`, output folder, plots 1, 2, 4, 5, 9, 12.**
- Backward over days t = D−1..0: regress V_{t+1}(s_end, ·) on φ(X_t) per s_end (low-order
  polynomials of daily mean / spread features, configurable basis); choose
  s_end* = argmax_s [F(s_start, s, p_t) + Ĉ_t(s, X_t)]; propagate realised values
  (Longstaff–Schwartz). Intrinsic control variate: compute h_Q(μ(X)) exactly with
  `prices.conditional_mean` and regress only the extrinsic residual. Concave / isotone (PAV-type)
  projection of Ĉ over the SoC grid after each regression. Report in-sample and out-of-sample.
- **tqdm progress bar + logging over days and SoC grid points** (user request).
- Forward on the out-of-sample paths: c, d, SoC, cashflow per stage, cycles; intrinsic value
  (mean price path) and perfect-foresight bound. Output folder per SPEC section 6
  (`config.yaml`, `prices_meta.json`, `summary.json/.csv`, `paths.parquet`, `figures/*.png`,
  `library_stats.json`) via `io.new_run_dir(config.name)`. Wire up `cli run` and
  `examples/run_synthetic.py`; export `run_engine` from `__init__`.
- Tests: intrinsic ≤ LSMC (out-of-sample) ≤ perfect foresight; energy balance and SoC bounds on
  every forward path; reproducibility with fixed seeds.

**Milestone 4** — IDA1–3 rolling intrinsic (`daily.py`), half-day library for IDA3 (start SoC at
12:00, not on the grid), plots 3, 6, 8; test each IDA increment ≥ 0.
**Milestone 5** — cycle limit (`cycles.py`): Lagrange λ by bisection on cycles of the final
physical schedule (default), counter variant as validation on a small case; plot 11.
**Milestone 6** — performance tuning (incl. the per-LP setup cost above), remaining plots
(7, 10, 13, 14), runtime report.

## How to work

- One milestone at a time. After each: run the full test suite, commit, push, and give the user a
  short German summary (du) with the key numbers and the figure paths; look at every figure you
  produce before reporting it.
- Scope for this session: milestones 2 and 3, then stop and report. Ask before changing the
  specification, the public interface in SPEC section 1, or adding dependencies.
- Commit messages end with the co-author line given by your environment.
