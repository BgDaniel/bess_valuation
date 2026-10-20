"""Tests for BatteryConfig."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bess_valuation import BatteryConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class TestBatteryConfig(unittest.TestCase):
    def test_example_configs_load(self) -> None:
        for name in ("bess_1mw_2mwh.yaml", "bess_1mw_8mwh.yaml"):
            config = BatteryConfig.from_yaml(CONFIG_DIR / name)
            self.assertEqual(config.periods_per_day, 96)

    def test_yaml_roundtrip(self) -> None:
        config = BatteryConfig.from_yaml(CONFIG_DIR / "bess_1mw_2mwh.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            config.to_yaml(path)
            self.assertEqual(BatteryConfig.from_yaml(path), config)

    def test_hash_depends_only_on_physical_parameters(self) -> None:
        config = BatteryConfig.from_yaml(CONFIG_DIR / "bess_1mw_2mwh.yaml")
        renamed = replace(config, name="other", max_cycles_per_day_avg=2.0)
        self.assertEqual(config.config_hash, renamed.config_hash)
        bigger = replace(config, e_max=4.0)
        self.assertNotEqual(config.config_hash, bigger.config_hash)

    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BatteryConfig(name="x", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, eta_charge=1.2)
        with self.assertRaises(ValueError):
            BatteryConfig(name="x", e_max=2.0, p_charge_max=1.0, p_discharge_max=1.0, dt=0.5)
        with self.assertRaises(ValueError):
            BatteryConfig.from_dict({"name": "x", "e_max": 2.0, "p_charge_max": 1.0,
                                     "p_discharge_max": 1.0, "typo": 1})


if __name__ == "__main__":
    unittest.main()
