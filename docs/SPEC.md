# Task: set up the Python project `bess_valuation`

Build a Python package that values a battery (BESS) and computes its optimal dispatch on stochastic day-ahead (DA) and intraday-auction (IDA1–3) prices, using backward Bellman induction (LSMC). Produce plots and store all results in an output folder per battery configuration.

## 0. Working rules

- Explain things to me in German, address me with "du", and define mathematical symbols on first use. Code, docstrings and identifiers in English.
- Python 3.11+, Poetry project, type annotations everywhere, Google-style docstrings, tests with `unittest` (not pytest), minimal architectural indirection (plain functions and a few dataclasses, no deep class hierarchies or plugin frameworks).
- Speed: hot loops in Numba (`@njit`, `prange`). No external LP solver in the hot loop (see section 4). HiGHS/scipy only inside tests as reference.
- Change only what I explicitly ask for when we iterate on the code; never silently reintroduce slow LP calls into the backward induction.
- Work in milestones (section 9); after each milestone run the tests and show me a short summary plus the plots.

## 1. Interface

```python
from bess_valuation import BatteryConfig, PriceScenarios, run_engine

config = BatteryConfig.from_yaml("configs/bess_1mw_2mwh.yaml")
prices = PriceScenarios.from_files("data/scenarios/")      # or PriceScenarios.synthetic(...)
result = run_engine(config, prices)                         # backward induction + forward simulation
result.save()                                               # writes results/<config.name>/<timestamp>/
```

Plus a CLI: `python -m bess_valuation run --config configs/x.yaml --prices data/scenarios/ [--plots/--no-plots]`.

## 2. Inputs

### 2.1 Battery config (`BatteryConfig`, dataclass, YAML-loadable)

- `name` (str, used as output folder name)
- `e_max`, `e_min` [MWh], `p_charge_max`, `p_discharge_max` [MW]
- `eta_charge`, `eta_discharge` (one-way efficiencies, < 1)
- `dt` [h]: 1.0 (hourly) or 0.25 (quarter-hourly; DA is 15-min since the SDAC go-live, 96 periods/day)
- `soc_init` [MWh], `soc_grid_size` (number of SoC grid points for the Bellman state)
- `terminal_soc_value`: rule for the last day (e.g. 0, or value per MWh)
- Cycle limit: `max_cycles_per_day_avg` (e.g. 1.5), `cycle_period` ("month" | "year"), `cycle_method` ("lagrange" | "counter")
- `allow_simultaneous_charge_discharge` (bool, default False; matters only for negative prices)
- `transaction_cost` [€/MWh] per auction trade (default 0; first version ignores it, later sensitivity)

A config hash (all physical parameters) is part of every vertex-library key.

### 2.2 Prices (`PriceScenarios`)

- Arrays of shape `(N, D, T)`: `da`, `ida1`, `ida2`, `ida3` (N paths, D days, T periods per day). `ida3` only covers the second half of the day (12–24h); first half NaN.
- Optional state variables `X` of shape `(N, D, m)` for the regression (e.g. forward-curve factors). If absent, derive features from the prices known at decision time.
- Loaders for CSV / Parquet / NPZ.
- `PriceScenarios.synthetic(...)` delegates to the price generator (section 2.3).
- Two independent path sets: one for estimating the regression coefficients, one for out-of-sample valuation and forward simulation.

### 2.3 Price generator (separate small engine, `price_generator.py`)

Goal: a simple stochastic model that produces plausible DA and IDA1–3 scenarios with typical intraday shapes, so the valuation engine can be run end to end. Not a sophisticated market model; parameters are plausible defaults to be tuned, not a calibration.

Interface:

```python
from bess_valuation import PriceGeneratorConfig, generate_prices

gen_cfg = PriceGeneratorConfig.from_yaml("configs/prices_synthetic.yaml")
prices = generate_prices(gen_cfg, n_paths=2000, start="2027-01-01", n_days=365, dt=0.25, seed=42)
prices.save("data/scenarios/synthetic_2027/")   # same format the loaders read
```

Model per path n, day d, period t (t = period of day):

p^DA_{d,t} = L_d + A_d · S_peak(t, d) − B_d · S_solar(t, d) + Z_{d,t} + J_{d,t} + Q(t)

- Daily level L_d: seasonal mean m(d) (annual cosine, higher in winter) plus weekday/weekend/holiday offset, plus an AR(1) deviation across days (mean reversion, OU in discrete time).
- Shape templates (deterministic, smooth, normalised):
  - S_peak: morning and evening peak (winter-weighted double hump; summer mainly evening peak after sunset).
  - S_solar: midday dip whose width and depth follow the season (wide/deep in summer, small in winter); can push midday prices below zero on strong-solar days.
  - Blend templates by day of year; damp peaks on weekends.
- Stochastic shape factors A_d (peak intensity) and B_d (solar intensity): each AR(1) across days, B_d ≥ 0 with seasonal mean; moderate correlation between L_d and A_d.
- Intraday noise Z_{d,t}: AR(1) within the day (smooth deviations), small variance.
- Spikes J_{d,t}: rare positive jumps concentrated in evening peak periods (Poisson, exponential sizes), rare negative midday jumps on high-solar weekend days.
- Quarter-hour structure Q(t) (only if dt = 0.25): small within-hour sawtooth around the hourly ramps (morning ramp-up, evening ramp-down), as observed in 15-min products.
- IDA prices: p^{IDAj}_{d,t} = p^DA_{d,t} + Σ_{i≤j} ε^{(i)}_{d,t}, j = 1, 2, 3, i.e. the spread is a sum of successive forecast-update shocks (martingale-type: zero mean by default, optional bias parameter).
  - Each shock has a common solar forecast-error component (one random factor per day and stage, multiplied by S_solar, so midday spreads are correlated) plus a common level component plus iid period noise.
  - Shock volatilities configurable per stage (e.g. growing from IDA1 to IDA3); IDA3 only defined for 12–24h (NaN before).
- State variables X_d = (L_d, A_d, B_d) are returned as `X` for the regression. Because the factors are AR(1), the conditional expectation μ(X_d) = E[p^DA_{d+1} | X_d] is available in closed form; provide it (`prices.conditional_mean`) for the intrinsic control variate.
- Calendar: real dates from `start`, weekdays and a configurable holiday list.
- Implementation: fully vectorised numpy (or Numba), fixed seeds, all parameters in `PriceGeneratorConfig` (YAML), shapes documented with formulas in the docstrings.

Generator plots (saved under `results/price_generator/<timestamp>/figures/` together with the config):
- a few sample paths (days stacked) and a full-year daily-mean fan;
- DA price fan per period of day, separately for winter / summer and weekday / weekend;
- share of negative prices per period of day;
- distribution of IDA − DA spreads per stage and per period of day;
- autocorrelation of daily means across days.

## 3. Model

### 3.1 Daily problem (per stage)

- Variables: charge `c`, discharge `d` ∈ R^T, 0 ≤ c_t ≤ p_charge_max·dt, 0 ≤ d_t ≤ p_discharge_max·dt.
- SoC dynamics s_{t+1} = s_t + η_c·c_t − d_t/η_d, e_min ≤ s_t ≤ e_max, fixed s_start and s_end (both on the SoC grid).
- Objective max p^T(d − c). Feasible set P is a polytope; the optimum is a vertex. F(p) = max_k y_k^T p with y = d − c (support function of the net-dispatch polytope Q).
- Negative prices: if simultaneous charge/discharge is not allowed, enforce c_t·d_t = 0 by a per-period case split (document the approach).

### 3.2 Stages within a day (scope: DA + IDA1–3, no continuous intraday)

- j = 0 DA (D−1 12:00), j = 1 IDA1 (D−1 15:00), j = 2 IDA2 (D−1 22:00), j = 3 IDA3 (D 10:00, only 12–24h).
- Rolling intrinsic: at stage j look up x^j = argmax_{x∈Q} p^{jT} x and trade the difference x^j − x^{j−1} at p^j. Daily cashflow F = p^{0T} x^0 + Σ_{j≥1} [h_Q(p^j) − p^{jT} x^{j−1}]; each increment is ≥ 0.
- IDA3: the first half of the day is fixed; optimise only 12–24h with the SoC at 12:00 as start (separate half-day library).
- s_end is decided per day in the Bellman recursion and stays fixed across all four stages.

### 3.3 Backward Bellman induction (LSMC)

- State: (SoC on grid, price state X). V_t(s, n) = realised value on path n; Ĉ_t(s, X) = β(s)^T φ(X) = regressed continuation value.
- For day t = D−1, …, 0: regress V_{t+1}(s_end, ·) on φ(X_t) for each s_end; for each s_start and path choose s_end* = argmax [F(s_start, s_end, prices_t) + Ĉ_t(s_end, X_t)]; propagate the realised value (Longstaff–Schwartz).
- Continuation value enters as a scalar per vertex, so the argmax-over-vertices structure is preserved.
- Control variate: compute the intrinsic part h_Q(μ(X)) exactly (μ = expected price given X) and regress only the extrinsic residual.
- Concavity in SoC: enforce by a concave/isotone projection over the SoC grid (PAV-type) after each regression; OLS on polynomials can break concavity, especially for long-duration storage.
- Basis functions: start with low-order polynomials in daily mean and spread features; keep the basis configurable.
- Report in-sample and out-of-sample values; in-sample OLS can exceed the perfect-foresight bound.

### 3.4 Cycle limit

- Cycles are counted on the final physical schedule (after IDA3), not on traded volumes.
- `lagrange` (default): cycle cost λ in every stage, objective (−p, p − λ·1) = shifted price; calibrate λ by bisection so that the average cycles per day match the target. Plot the calibration curve.
- `counter`: extra integer state n = cycles used in the period; daily budget θ ∈ {0, 0.5, 1, …} enters the RHS. Used as validation of the Lagrange variant on a small case.

### 3.5 Forward simulation

On the out-of-sample paths: apply the decisions (s_end* from Ĉ, rolling-intrinsic stages within the day), record per path and period c, d, SoC, cashflow per stage, cycles. Also compute the intrinsic value (mean price path) and the perfect-foresight upper bound.

## 4. Optimisation core (no external LP solver in the hot loop)

### 4.1 Vertex library as an incrementally built graph

- Key: (config hash, horizon type full/half-day, active set). Per vertex store: active set, schedule (c, d) and y, edge directions (rays of the tangent cone T(v) = {r : A_act r ≤ 0}, equivalently columns of −A_N^{-1}), neighbour pointers (filled lazily), validity region in (s_start, s_end), hit counts.
- Query (vectorised over all paths): argmax over cached vertices via `Y @ p` → certificate y_N = A_N^{-T} g ≥ 0 (dual feasibility / shadow prices), at degenerate vertices check all tangent-cone rays → if violated, pivot with the own simplex and add the new vertex.
- Normal cones depend only on the active set, not on the RHS, so the library is shared across SoC grid points and days. Start with separate libraries per grid point, keyed by active set so they can be merged later.
- Warm start: argmax cached vertex; optionally a table lookup on PCA coefficients of the demeaned price curve. The projection only chooses the start; certificate and pivots use the full price vector, so the result stays exact.
- Stopping diagnostic: Good–Turing estimate P(unseen vertex) ≈ n_1/N (n_1 = vertices hit exactly once).
- Persist the library to disk (NPZ) per config hash for reuse between runs.

### 4.2 Own Numba simplex

- Revised simplex with bounded variables (power and SoC bounds as variable bounds), structured SoC rows.
- Warm start from the library basis; feasible start without Phase I where possible (idle schedule if s_start = s_end, otherwise a monotone charge/discharge schedule); Phase I fallback.
- Basis inverse via rank-1 updates (product form / Sherman–Morrison) with periodic refactorisation.
- Pricing: steepest edge, Bland's rule as anti-cycling fallback at degenerate vertices; explicit feasibility/optimality/pivot tolerances.
- Batch over paths with `prange`; return the vertex ID / active set so new vertices go straight into the library.

## 5. Package layout

```
bess_valuation/
  pyproject.toml
  configs/                 example YAML configs (1 MW / 2 MWh, 1 MW / 8 MWh, prices_synthetic.yaml)
  bess_valuation/
    __init__.py
    config.py              BatteryConfig, YAML loading, config hash
    prices.py              PriceScenarios, loaders, saving
    price_generator.py     PriceGeneratorConfig, generate_prices, shape templates, generator plots
    polytope.py            constraint matrices for full and half-day horizons
    simplex.py             Numba bounded revised simplex
    library.py             vertex library, certificate, persistence, Good–Turing
    daily.py               stage lookup, rolling intrinsic over DA + IDA1–3
    cycles.py              Lagrange calibration, counter state
    lsmc.py                backward induction, regression, concavity projection
    forward.py             forward simulation, intrinsic, perfect foresight
    results.py             Result dataclass, summary tables
    plots.py               all figures (section 7)
    io.py                  output folder handling
    cli.py
  tests/                   unittest
  examples/run_synthetic.py
```

## 6. Output folder

- `results/<config.name>/<YYYYMMDD_HHMMSS>/`; create a new folder for each config name and a new timestamped subfolder per run (never overwrite).
- Contents: `config.yaml` (copy), `prices_meta.json` (source, N, D, T, seeds), `summary.json` and `summary.csv` (values, decomposition, cycles, runtimes), `paths.parquet` (per-path daily cashflows, SoC, cycles), `figures/*.png` (150 dpi) and optionally `figures/*.pdf`, `library_stats.json`.

## 7. Plots (matplotlib, one consistent style, units on all axes)

Value
1. Value fan: cumulative realised value over time — mean plus quantile bands (5/25/50/75/95 %) across paths, with intrinsic and perfect-foresight curves overlaid.
2. Value decomposition: intrinsic vs extrinsic, and value added per stage (DA, IDA1, IDA2, IDA3) as stacked bars per month.
3. Distribution of total value (histogram + quantiles) and of value added per auction stage.

Dispatch vs prices
4. SoC fan over time (daily end-of-day SoC quantiles) and SoC heatmap (period-of-day × day, path median).
5. Charge and discharge profile per period of day (mean and quantiles) overlaid on the DA price quantile fan per period of day.
6. Dispatch-vs-price scatter (net dispatch per period against price, coloured by stage), i.e. the battery's empirical bid curve.
7. Daily revenue vs daily price spread (max − min) scatter.

Prices
8. DA and IDA1–3 price fans per period of day; distribution of IDA − DA spreads.

State and Bellman
9. Continuation value Ĉ_t(s) vs SoC for selected days (concavity check, before/after projection).
10. Water value ∂V/∂s over time (quantiles across paths).

Cycles
11. Histogram of cycles per day, cumulative cycles vs limit per period, λ calibration curve.

Diagnostics
12. Vertex library: discovery curve (distinct vertices vs number of price vectors), Good–Turing estimate, hit-frequency rank plot, share of queries solved by certificate without pivot.
13. Regression: R² per day, in-sample vs out-of-sample value, variance reduction from the intrinsic control variate.
14. Runtime breakdown (library lookup, simplex pivots, regression, forward simulation).

## 8. Tests and validation (unittest)

- Simplex vs HiGHS on random and simulated prices: equal objective up to tolerance, feasible schedules; include degenerate cases, negative prices, s_start ≠ s_end, cycle-cost-shifted prices, half-day horizon.
- Library lookup + certificate vs direct LP on all scenarios of a small case.
- Ordering: intrinsic ≤ LSMC (out-of-sample) ≤ perfect foresight; each IDA increment ≥ 0.
- Energy balance and SoC bounds on every forward path; cycle limit met on average.
- Lagrange vs counter variant on a small case.
- Reproducibility with fixed seeds.
- Price generator: array shapes and NaN pattern (IDA3 only 12–24h); same seed gives identical output; Monte Carlo mean of p^DA_{d+1} given X_d matches the closed-form conditional mean; IDA − DA spreads have zero mean (if no bias) and spread variance grows from IDA1 to IDA3; seasonal shapes visible (midday dip deeper in summer).

## 9. Milestones

1. Config, prices (loaders + price generator with its plots), polytope, Numba simplex with tests against HiGHS.
2. Vertex library with certificate and persistence; DA-only daily lookup.
3. LSMC for DA only + forward simulation + output folder + plots 1, 2, 4, 5, 9, 12.
4. IDA1–3 stages (rolling intrinsic, half-day library for IDA3) + plots 3, 6, 8.
5. Cycle limit (Lagrange + counter) + plot 11.
6. Performance tuning, remaining plots, runtime report.

## Appendix: theory background

- Vertices are identified by their full active set; battery polytopes are highly degenerate, so one vertex can have several bases.
- Standard form [A | I]; basis = linearly independent columns; nonbasic slacks = active constraints (u = 0). Edge directions r_i = −A_N^{-1} e_i, reduced costs g^T r_i, ratio test over inactive constraints. Dual: y_N = A_N^{-T} g is linear in price; y_N ≥ 0 ⇔ g in the normal cone ⇔ optimal.
- F is convex in prices (Jensen ⇒ extrinsic ≥ 0) and concave in (s_start, s_end).
- Extrinsic = Σ_k π_k(X) (y_k − y_{k0})^T μ_k(X) with π_k = P(price in cone k | X): probability-weighted gain from switching away from the intrinsic vertex k0. Optional regression basis: spread options E[max(0, Δ_k^T p) | X] over neighbour cones.
- Known reference results: 2 MWh / 1 MW, η = 0.95 each way: ~630–740 distinct DA vertices found empirically (combinatorial bound 12,951); extrinsic ≈ 22.5 % over 3 years (DA only); concavity violations negligible for 2h storage but large for 8h storage.
