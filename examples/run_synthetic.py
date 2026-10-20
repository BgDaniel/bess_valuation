"""End-to-end example on synthetic prices (DA only so far).

Run from the repository root:

    poetry run python examples/run_synthetic.py

Generates (or reuses) 100 synthetic paths for 2027, values the 1 MW / 2 MWh battery with
50 regression and 50 out-of-sample paths and writes ``results/bess_1mw_2mwh/<timestamp>/``.
"""

import logging
from pathlib import Path

from bess_valuation import BatteryConfig, EngineSettings, PriceGeneratorConfig, PriceScenarios, run_engine
from bess_valuation.price_generator import generate_prices

SCENARIOS = Path("data/scenarios/synthetic_2027_100")


def main() -> None:
    """Generate prices if needed, run the engine and save the results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = BatteryConfig.from_yaml("configs/bess_1mw_2mwh.yaml")
    if (SCENARIOS / "meta.json").exists():
        prices = PriceScenarios.from_files(SCENARIOS)
    else:
        gen = PriceGeneratorConfig.from_yaml("configs/prices_synthetic.yaml")
        prices = generate_prices(gen, n_paths=100, start="2027-01-01", n_days=365, dt=config.dt, seed=42)
        prices.save(SCENARIOS)
    result = run_engine(config, prices, EngineSettings(n_regression_paths=50))
    run_dir = result.save()
    print(f"results in {run_dir}")
    for key, value in result.summary().items():
        print(f"{key:40s} {value}")


if __name__ == "__main__":
    main()
