"""End-to-end example on synthetic prices (runnable from milestone 3 on)."""

from bess_valuation import BatteryConfig


def main() -> None:
    """Load the example config; the engine call follows in milestone 3."""
    config = BatteryConfig.from_yaml("configs/bess_1mw_2mwh.yaml")
    print(f"{config.name}: hash {config.config_hash}, T = {config.periods_per_day}")


if __name__ == "__main__":
    main()
