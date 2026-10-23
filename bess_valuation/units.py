"""Display units: currency label and energy / power units.

All computations use MW, MWh and the currency of the price data (€ in the synthetic
scenarios). Units only change how numbers are entered and shown (app, figures):

* ``currency``: an ISO code such as "EUR", "USD", "GBP"; a label only, amounts are not
  converted (the price data are taken to be in this currency);
* ``energy``: "MWh" or "kWh"; power follows (MW / kW) and prices are per energy unit
  (e.g. €/MWh or €/kWh).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF", "JPY": "¥", "CNY": "¥",
                    "AUD": "A$", "CAD": "C$", "SEK": "kr", "NOK": "kr", "DKK": "kr", "PLN": "zł",
                    "CZK": "Kč", "HUF": "Ft", "INR": "₹"}
ENERGY_FACTORS = {"MWh": 1.0, "kWh": 1000.0}


@dataclass(frozen=True)
class Units:
    """Display units.

    Attributes:
        currency: ISO currency code (label only, no conversion).
        energy: "MWh" or "kWh".
    """

    currency: str = "EUR"
    energy: str = "MWh"

    def __post_init__(self) -> None:
        if self.energy not in ENERGY_FACTORS:
            raise ValueError(f"energy unit must be one of {list(ENERGY_FACTORS)}")
        if not self.currency.strip():
            raise ValueError("currency must not be empty")

    @property
    def symbol(self) -> str:
        """Currency symbol (the code itself if no symbol is known)."""
        return CURRENCY_SYMBOLS.get(self.currency.upper(), self.currency)

    @property
    def power(self) -> str:
        """Power unit belonging to the energy unit."""
        return self.energy[:-1]

    @property
    def price(self) -> str:
        """Price unit, e.g. "€/MWh"."""
        return f"{self.symbol}/{self.energy}"

    @property
    def energy_factor(self) -> float:
        """Display value per MWh (1 for MWh, 1000 for kWh)."""
        return ENERGY_FACTORS[self.energy]

    def energy_value(self, mwh: Any) -> Any:
        """Energy [MWh] in the display unit."""
        return mwh * self.energy_factor

    def power_value(self, mw: Any) -> Any:
        """Power [MW] in the display unit."""
        return mw * self.energy_factor

    def price_value(self, per_mwh: Any) -> Any:
        """Price [currency/MWh] per display energy unit."""
        return per_mwh / self.energy_factor

    def to_dict(self) -> dict[str, str]:
        """Plain dict (YAML / JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Units:
        """From a dict; missing keys keep the defaults."""
        return cls(**{k: v for k, v in (data or {}).items() if k in ("currency", "energy")})
