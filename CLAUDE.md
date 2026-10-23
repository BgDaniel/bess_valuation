# bess_valuation

Full specification: [docs/SPEC.md](docs/SPEC.md). Read it before working on a milestone.

## Working rules

- Explain things to the user in German, address them with "du", define mathematical symbols on first use.
  Code, docstrings and identifiers in English.
- Python 3.11+, Poetry, type annotations everywhere, Google-style docstrings.
- Tests with `unittest` only (no pytest): `poetry run python -m unittest discover -s tests -v`.
- Minimal indirection: plain functions and a few dataclasses; no deep class hierarchies or plugin frameworks.
- Hot loops in Numba (`@njit`, `prange`). No external LP solver in the hot loop; HiGHS/scipy only in tests as reference.
- When iterating, change only what the user explicitly asks for. Never reintroduce LP calls into the backward induction.
- Work in milestones (SPEC section 9); after each milestone run the tests and show a short summary plus the plots.
- Logging via the standard `logging` module (`logger = logging.getLogger(__name__)`) in every
  long-running step; the CLI configures the level (`--log-level`). Long loops show a `tqdm`
  progress bar — in particular the backward induction over days and SoC grid points (lsmc.py),
  the forward simulation and the lambda calibration.

## Milestone status

- [x] 1. Config, prices (loaders + generator + plots), polytope, Numba simplex vs HiGHS
- [x] 2. Vertex library, certificate, persistence; DA-only daily lookup
- [x] 3. LSMC DA-only, forward simulation, output folder, plots 1, 2, 4, 5, 9, 12
- [ ] 4. IDA1–3 stages, half-day library, plots 3, 6, 8
- [ ] 5. Cycle limit (Lagrange + counter), plot 11
- [ ] 6. Performance tuning, remaining plots, runtime report

## Decisions

- Daily solver default is the exact DP over SoC (`daily_solver="dp"`, ~200x faster than the
  simplex variants on quarter-hourly prices, same values); "library", "cold" and "cluster"
  stay selectable for comparison.

## Backlog (agreed with the user, not yet scheduled)

- Battery degradation, monthly granularity (capacity fixed within a day, so the daily
  problem stays 1-D in SoC):
  1. user-given constant degradation (e.g. capacity / efficiency path per month);
  2. cycle-dependent degradation (wear cost per discharged MWh and/or capacity loss
     driven by throughput).
- Re-run `examples/benchmark_daily_solvers.py` when the cycle limit (or a hard daily
  limit) is added: then the ranking of the daily solvers may change.
