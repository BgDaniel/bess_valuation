# bess_valuation

Valuation and optimal dispatch of a battery energy storage system (BESS) on stochastic
day-ahead (DA) and intraday-auction (IDA1–3) prices via backward Bellman induction (LSMC).
The daily dispatch problem is solved with an incrementally built vertex library and an
own Numba bounded revised simplex (no external LP solver in the hot loop).

See [docs/SPEC.md](docs/SPEC.md) for the full specification.

## Setup

```bash
poetry install
poetry run python -m unittest discover -s tests -v
```

## Usage

```python
from bess_valuation import BatteryConfig, PriceScenarios, run_engine

config = BatteryConfig.from_yaml("configs/bess_1mw_2mwh.yaml")
prices = PriceScenarios.from_files("data/scenarios/")
result = run_engine(config, prices)
result.save()   # results/<config.name>/<YYYYMMDD_HHMMSS>/
```

CLI:

```bash
poetry run python -m bess_valuation run --config configs/bess_1mw_2mwh.yaml --prices data/scenarios/
```

Options: `--n-regression-paths` (default: half of the paths), `--degree` (polynomial
basis), `--library-dir` (persisted vertex libraries, default `library_cache/`, `none`
to disable), `--results`, `--plots/--no-plots`. Numerical settings in Python:
`run_engine(config, prices, EngineSettings(...))`.

End-to-end example (DA only so far): `poetry run python examples/run_synthetic.py`.

## App

A Streamlit front end (dark theme) to configure and start runs, follow their progress
and inspect results:

```bash
poetry install --with app
poetry run streamlit run app/streamlit_app.py
```

Pages: *Runs* (pick a stored battery and simulation set, numerical settings, start; live
table with progress and remaining time), *Result* (key figures, figures, downloads),
*Batteries* and *Simulations* (stored master data, `app_data/`, see
`bess_valuation/catalog.py`), *Settings* (display units). Runs are background jobs
(`bess_valuation/jobs.py`, folders under `results/_jobs/`), so they keep running when the
browser is closed.

Display units (`bess_valuation/units.py`, stored in `app_data/settings.yaml`): a currency
label (EUR, USD, GBP, CHF, … or any code) and MWh or kWh; power and prices follow (kW,
€/kWh). The engine always computes in MW / MWh and the currency of the price data; the
currency is a label, not converted. Battery forms convert on save, tables, metrics and the
figures of a run use the units chosen when it was started (`Result.save(units=...)`).

## Degradation

* `capacity_fade_per_year` (battery config): linear loss of usable capacity, monthly steps;
  or pass your own factor per day: `run_engine(config, prices, capacity_factor=f)` with
  `f(dates) -> factors in (0, 1]` (or an array). The SoC grid is relative to the day's
  capacity.
* `wear_cost_per_mwh`: degradation cost per discharged MWh, charged in the daily
  optimisation; values are reported net of it (`wear_cost_eur` in the summary).

## Status

Milestones 1-3 done: prices and generator, polytope, Numba simplex, vertex library
(exact O(T) certificate, persistence, Good-Turing), DA-only LSMC with control variate and
concave projection, forward simulation, intrinsic and perfect-foresight values, output
folder, plots 1, 2, 4, 5, 9, 12. Next: IDA1-3 stages (milestone 4), cycle limit (5),
performance (6).

Notes:
- Without simultaneous charge/discharge the per-period case split is solved exactly by a
  dynamic program over the SoC (piecewise-linear value function); branch and bound grew
  exponentially with the number of negative-price periods.
- On quarter-hourly prices nearly every price vector has its own optimal vertex, so the
  library mostly provides warm starts there (it is capped per SoC pair); on smooth price
  vectors (conditional means) most queries are answered by the certificate.
