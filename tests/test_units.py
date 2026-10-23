"""Display units: labels and conversion."""

import unittest

import numpy as np

from bess_valuation.units import Units


class TestUnits(unittest.TestCase):
    def test_defaults(self):
        u = Units()
        self.assertEqual((u.symbol, u.energy, u.power, u.price), ("€", "MWh", "MW", "€/MWh"))
        self.assertEqual(u.energy_value(2.0), 2.0)

    def test_kwh_and_currency(self):
        u = Units(currency="usd", energy="kWh")
        self.assertEqual((u.symbol, u.power, u.price), ("$", "kW", "$/kWh"))
        np.testing.assert_allclose(u.energy_value(np.array([2.0])), [2000.0])
        self.assertEqual(u.power_value(1.5), 1500.0)
        self.assertAlmostEqual(u.price_value(80.0), 0.08)
        self.assertEqual(Units(currency="XYZ").symbol, "XYZ")

    def test_round_trip_and_validation(self):
        u = Units(currency="GBP", energy="kWh")
        self.assertEqual(Units.from_dict(u.to_dict()), u)
        self.assertEqual(Units.from_dict(None), Units())
        self.assertEqual(Units.from_dict({"energy": "kWh", "other": 1}).energy, "kWh")
        with self.assertRaises(ValueError):
            Units(energy="GWh")
        with self.assertRaises(ValueError):
            Units(currency=" ")


if __name__ == "__main__":
    unittest.main()
