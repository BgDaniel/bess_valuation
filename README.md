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

## Usage (target interface)

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
