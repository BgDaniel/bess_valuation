r"""Synthetic DA and IDA1-3 price generator (SPEC section 2.3).

A deliberately simple stochastic model with typical intraday shapes, so the valuation
engine can be run end to end. Parameters are plausible defaults, not a calibration.

Model for path n, day d and period t (hour of day h_t = (t + 1/2) dt):

.. math::

    p^{DA}_{d,t} = L_d + A_d\,S_{peak}(t,d) - B_d\,S_{solar}(t,d) + Z_{d,t} + J_{d,t} + Q(t)

* Season weight :math:`u(d) = (1 - \cos(2\pi (doy_d - 355)/365.25))/2`, 0 at the winter and
  1 at the summer solstice.
* Daily level :math:`L_d = m(d) + o(d) + \ell_d` with seasonal mean
  :math:`m(d) = \bar L + a_L (1 - 2u(d))` (higher in winter), calendar offset :math:`o(d)`
  (weekend / holiday) and AR(1) deviation :math:`\ell_d = \phi_L \ell_{d-1} + \sigma_L
  \varepsilon^L_d`.
* Peak factor :math:`A_d = w(d)(\bar A + a_d)`, :math:`a_d = \phi_A a_{d-1} + \sigma_A
  \varepsilon^A_d`, :math:`\mathrm{corr}(\varepsilon^A_d, \varepsilon^L_d) = \rho`;
  :math:`w(d)` damps the peaks on weekends and holidays.
* Solar factor :math:`B_d = \bar B(d)\exp(b_d - v_B/2) \ge 0` with seasonal mean
  :math:`\bar B(d) = B_w + (B_s - B_w)u(d)`, :math:`b_d = \phi_B b_{d-1} + \sigma_B
  \varepsilon^B_d` and stationary variance :math:`v_B = \sigma_B^2/(1-\phi_B^2)`, so
  :math:`E[B_d] = \bar B(d)`.
* Shape templates (deterministic, max 1):
  :math:`S_{peak} \propto (1 - (1-\omega_s)u)\,G(h; h_m, s_m) + G(h; h_e(d), s_e)` with
  Gaussian bumps :math:`G(h; c, s) = \exp(-(h-c)^2/(2s^2))`, the evening peak moving from
  :math:`h_{e,w}` to :math:`h_{e,s}` with the season; :math:`S_{solar} = G(h; h_{noon},
  s_w + (s_s - s_w)u(d))`, wide in summer, narrow in winter.
* Intraday noise :math:`Z_{d,t} = \phi_Z Z_{d,t-1} + \sigma_Z\sqrt{1-\phi_Z^2}\,\zeta_{d,t}`
  with :math:`\phi_Z = \phi_{Z,h}^{dt}` (``noise_phi_hour`` per hour), stationary within
  each day.
* Spikes :math:`J_{d,t}`: Poisson numbers of positive one-hour jumps in the evening hours
  (rate :math:`\lambda_+`, Exp sizes with mean :math:`\mu_+`) and of negative midday jumps
  on weekends and holidays (rate :math:`\lambda_- u(d)`, mean :math:`\mu_-`).
* Quarter-hour structure (dt = 0.25 only): :math:`Q(t) = q\,R(h)\,(k - 3/2)/(3/2)` with
  quarter :math:`k \in \{0,1,2,3\}` within the hour and :math:`R(h) = +1` on ramp-up hours,
  :math:`-1` on ramp-down hours, 0 otherwise: a within-hour sawtooth.

Optionally (``negative_price_floor`` F) negative DA prices are damped by the soft floor
:math:`p \mapsto -F(1 - e^{p/F})` (curtailment); the closed-form conditional mean below
ignores the floor, so with a floor it is an approximation (used only by the optional
control variate).

IDA prices are DA plus successive forecast-update shocks,
:math:`p^{IDA_j} = p^{DA} + \sum_{i\le j}\varepsilon^{(i)}` with
:math:`\varepsilon^{(i)}_{d,t} = \beta_i + \sigma^{sol}_i\xi^{sol}_{d,i} S_{solar}(t,d) +
\sigma^{lvl}_i\xi^{lvl}_{d,i} + \sigma^{noise}_i\zeta^{(i)}_{d,t}`, so spreads are
martingale-type (zero mean for zero bias) and their variance grows from IDA1 to IDA3.
IDA3 only covers 12-24h.

State variables :math:`X_d = (L_d, A_d, B_d)`. As all factors are AR(1),
:math:`\mu(X_d) = E[p^{DA}_{d+1} \mid X_d]` is available in closed form
(:func:`conditional_mean`).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from bess_valuation import plots
from bess_valuation.io import new_run_dir
from bess_valuation.prices import PriceScenarios

logger = logging.getLogger(__name__)

# Fixed-date German public holidays (MM-DD) plus the movable ones of 2026-2028.
DEFAULT_HOLIDAYS: tuple[str, ...] = (
    "01-01", "05-01", "10-03", "12-25", "12-26",
    "2026-04-03", "2026-04-06", "2026-05-14", "2026-05-25",
    "2027-03-26", "2027-03-29", "2027-05-06", "2027-05-17",
    "2028-04-14", "2028-04-17", "2028-05-25", "2028-06-05",
)


@dataclass(frozen=True)
class PriceGeneratorConfig:
    """All parameters of the synthetic price model; prices in €/MWh, hours in h."""

    # daily level L_d
    level_mean: float = 85.0
    level_seasonal_amplitude: float = 15.0
    weekend_offset: float = -12.0
    holiday_offset: float = -15.0
    level_ar_phi: float = 0.8
    level_ar_sigma: float = 10.0
    # peak factor A_d and template S_peak
    peak_mean: float = 30.0
    peak_ar_phi: float = 0.6
    peak_ar_sigma: float = 6.0
    peak_level_corr: float = 0.5
    weekend_peak_damping: float = 0.6
    peak_morning_hour: float = 8.0
    peak_morning_width: float = 1.5
    peak_morning_summer_weight: float = 0.3
    peak_evening_hour_winter: float = 18.5
    peak_evening_hour_summer: float = 20.5
    peak_evening_width: float = 2.0
    # solar factor B_d and template S_solar
    solar_mean_winter: float = 12.0
    solar_mean_summer: float = 60.0
    solar_ar_phi: float = 0.5
    solar_log_sigma: float = 0.4
    solar_noon: float = 13.0
    solar_width_winter: float = 1.6
    solar_width_summer: float = 3.0
    # intraday noise Z
    noise_phi_hour: float = 0.7
    noise_sigma: float = 4.0
    # spikes J
    spike_pos_rate: float = 0.08
    spike_pos_mean: float = 120.0
    spike_pos_hours: tuple[int, ...] = (17, 18, 19, 20)
    spike_neg_rate: float = 0.15
    spike_neg_mean: float = 50.0
    spike_neg_hours: tuple[int, ...] = (11, 12, 13, 14)
    # quarter-hour sawtooth Q
    quarter_hour_amplitude: float = 5.0
    ramp_up_hours: tuple[int, ...] = (6, 7, 8)
    ramp_down_hours: tuple[int, ...] = (19, 20, 21, 22)
    # IDA shocks per stage (IDA1, IDA2, IDA3)
    ida_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ida_solar_sigma: tuple[float, float, float] = (4.0, 6.0, 9.0)
    ida_level_sigma: tuple[float, float, float] = (2.0, 3.0, 4.0)
    ida_noise_sigma: tuple[float, float, float] = (3.0, 4.0, 6.0)
    # calendar: "YYYY-MM-DD" (one date) or "MM-DD" (every year)
    holidays: tuple[str, ...] = field(default=DEFAULT_HOLIDAYS)
    # soft floor for negative DA prices (curtailment); None: no floor
    negative_price_floor: float | None = None

    def __post_init__(self) -> None:
        for name in ("level_ar_phi", "peak_ar_phi", "solar_ar_phi", "noise_phi_hour"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.negative_price_floor is not None and self.negative_price_floor <= 0.0:
            raise ValueError("negative_price_floor must be positive (the floor is at minus this value)")
        if not -1.0 <= self.peak_level_corr <= 1.0:
            raise ValueError("peak_level_corr must be in [-1, 1]")
        for name in ("ida_bias", "ida_solar_sigma", "ida_level_sigma", "ida_noise_sigma"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} needs one value per IDA stage")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriceGeneratorConfig:
        """Load from a flat YAML file; missing keys keep their defaults."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown generator config keys: {sorted(unknown)}")
        data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**data)

    def to_dict(self) -> dict:
        """Return all parameters as plain YAML-serialisable types."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    def to_yaml(self, path: str | Path) -> None:
        """Write the parameters to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# ---------------------------------------------------------------------------- calendar


@dataclass(frozen=True)
class Calendar:
    """Calendar features of consecutive delivery days.

    Attributes:
        dates: Dates, shape (D,), datetime64[D].
        season: Season weight u(d) in [0, 1], 0 in winter, 1 in summer.
        weekend: Saturday or Sunday.
        holiday: Public holiday (from the config list).
    """

    dates: np.ndarray
    season: np.ndarray
    weekend: np.ndarray
    holiday: np.ndarray

    @property
    def off_day(self) -> np.ndarray:
        """Weekend or holiday."""
        return self.weekend | self.holiday


def make_calendar(start: str, n_days: int, holidays: tuple[str, ...]) -> Calendar:
    """Build the calendar of ``n_days`` days from ``start`` (ISO date)."""
    dates = np.datetime64(start, "D") + np.arange(n_days)
    years = dates.astype("datetime64[Y]")
    doy = (dates - years.astype("datetime64[D]")).astype(np.int64) + 1
    season = 0.5 * (1.0 - np.cos(2.0 * np.pi * (doy - 355) / 365.25))
    weekday = (dates.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    iso = np.datetime_as_string(dates, unit="D")
    fixed = {h for h in holidays if len(h) == 5}
    single = {h for h in holidays if len(h) == 10}
    holiday = np.array([s in single or s[5:] in fixed for s in iso], dtype=bool)
    return Calendar(dates=dates, season=season, weekend=weekday >= 5, holiday=holiday)


# --------------------------------------------------------------------- shape templates


def hours_of_day(dt: float) -> np.ndarray:
    """Period mid-points h_t = (t + 1/2) dt [h], shape (T,)."""
    return (np.arange(round(24.0 / dt)) + 0.5) * dt


def _bump(h: np.ndarray, centre: np.ndarray | float, width: np.ndarray | float) -> np.ndarray:
    return np.exp(-0.5 * ((h - centre) / width) ** 2)


def peak_shape(cfg: PriceGeneratorConfig, season: np.ndarray, dt: float) -> np.ndarray:
    r"""Peak template :math:`S_{peak}(t, d)`, shape (D, T), normalised to max 1 per day.

    Double hump: morning bump at ``peak_morning_hour`` with weight 1 in winter and
    ``peak_morning_summer_weight`` in summer, evening bump moving from
    ``peak_evening_hour_winter`` to ``peak_evening_hour_summer`` (after sunset).
    """
    h = hours_of_day(dt)[None, :]
    u = season[:, None]
    morning = (1.0 - (1.0 - cfg.peak_morning_summer_weight) * u) * _bump(
        h, cfg.peak_morning_hour, cfg.peak_morning_width)
    evening_hour = cfg.peak_evening_hour_winter + (cfg.peak_evening_hour_summer - cfg.peak_evening_hour_winter) * u
    shape = morning + _bump(h, evening_hour, cfg.peak_evening_width)
    return shape / shape.max(axis=1, keepdims=True)


def solar_shape(cfg: PriceGeneratorConfig, season: np.ndarray, dt: float) -> np.ndarray:
    r"""Solar template :math:`S_{solar}(t, d)`, shape (D, T), max 1 at ``solar_noon``.

    Width :math:`s_w + (s_s - s_w)u(d)`: wide in summer, narrow in winter. The depth of the
    midday dip is carried by :math:`B_d`.
    """
    h = hours_of_day(dt)[None, :]
    width = cfg.solar_width_winter + (cfg.solar_width_summer - cfg.solar_width_winter) * season[:, None]
    return _bump(h, cfg.solar_noon, width)


def quarter_hour_profile(cfg: PriceGeneratorConfig, dt: float) -> np.ndarray:
    """Within-hour sawtooth Q(t), shape (T,); zero unless dt = 0.25."""
    n = round(24.0 / dt)
    if dt != 0.25:
        return np.zeros(n)
    hour = np.arange(n) // 4
    quarter = np.arange(n) % 4
    ramp = np.isin(hour, cfg.ramp_up_hours).astype(float) - np.isin(hour, cfg.ramp_down_hours)
    return cfg.quarter_hour_amplitude * ramp * (quarter - 1.5) / 1.5


def _hour_mask(hours: tuple[int, ...], dt: float) -> np.ndarray:
    return np.isin(np.floor(hours_of_day(dt)).astype(int), hours)


def expected_spikes(cfg: PriceGeneratorConfig, cal: Calendar, dt: float) -> np.ndarray:
    """E[J_{d,t}], shape (D, T): Poisson rate times mean size spread over the spike hours."""
    pos = _hour_mask(cfg.spike_pos_hours, dt) * cfg.spike_pos_rate * cfg.spike_pos_mean / len(cfg.spike_pos_hours)
    neg_rate = cfg.spike_neg_rate * cal.season * cal.off_day
    neg = _hour_mask(cfg.spike_neg_hours, dt) * cfg.spike_neg_mean / len(cfg.spike_neg_hours)
    return pos[None, :] - neg_rate[:, None] * neg[None, :]


# ------------------------------------------------------------------- daily factors


def _seasonal_level(cfg: PriceGeneratorConfig, cal: Calendar) -> np.ndarray:
    """m(d) + o(d): seasonal mean plus weekend / holiday offset, shape (D,)."""
    offset = np.where(cal.holiday, cfg.holiday_offset, np.where(cal.weekend, cfg.weekend_offset, 0.0))
    return cfg.level_mean + cfg.level_seasonal_amplitude * (1.0 - 2.0 * cal.season) + offset


def _peak_weight(cfg: PriceGeneratorConfig, cal: Calendar) -> np.ndarray:
    return np.where(cal.off_day, cfg.weekend_peak_damping, 1.0)


def _solar_mean(cfg: PriceGeneratorConfig, cal: Calendar) -> np.ndarray:
    return cfg.solar_mean_winter + (cfg.solar_mean_summer - cfg.solar_mean_winter) * cal.season


def _solar_innovation_sigma(cfg: PriceGeneratorConfig) -> float:
    return cfg.solar_log_sigma * np.sqrt(1.0 - cfg.solar_ar_phi ** 2)


def _simulate_factors(cfg: PriceGeneratorConfig, cal: Calendar, n_paths: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AR(1) deviations (l_d, a_d, b_d), each shape (N, D), started from stationarity."""
    n_days = len(cal.dates)
    rho = cfg.peak_level_corr
    sd_l = cfg.level_ar_sigma / np.sqrt(1.0 - cfg.level_ar_phi ** 2)
    sd_a = cfg.peak_ar_sigma / np.sqrt(1.0 - cfg.peak_ar_phi ** 2)
    sig_b = _solar_innovation_sigma(cfg)
    ell = np.empty((n_paths, n_days))
    a = np.empty((n_paths, n_days))
    b = np.empty((n_paths, n_days))
    z = rng.standard_normal((3, n_paths, n_days))
    ell[:, 0] = sd_l * z[0, :, 0]
    a[:, 0] = sd_a * (rho * z[0, :, 0] + np.sqrt(1.0 - rho ** 2) * z[1, :, 0])
    b[:, 0] = cfg.solar_log_sigma * z[2, :, 0]
    for d in range(1, n_days):
        ell[:, d] = cfg.level_ar_phi * ell[:, d - 1] + cfg.level_ar_sigma * z[0, :, d]
        eps_a = rho * z[0, :, d] + np.sqrt(1.0 - rho ** 2) * z[1, :, d]
        a[:, d] = cfg.peak_ar_phi * a[:, d - 1] + cfg.peak_ar_sigma * eps_a
        b[:, d] = cfg.solar_ar_phi * b[:, d - 1] + sig_b * z[2, :, d]
    return ell, a, b


def _factors_to_state(cfg: PriceGeneratorConfig, cal: Calendar, ell: np.ndarray, a: np.ndarray,
                      b: np.ndarray) -> np.ndarray:
    """X_d = (L_d, A_d, B_d), shape (N, D, 3)."""
    L = _seasonal_level(cfg, cal)[None, :] + ell
    A = _peak_weight(cfg, cal)[None, :] * (cfg.peak_mean + a)
    B = _solar_mean(cfg, cal)[None, :] * np.exp(b - 0.5 * cfg.solar_log_sigma ** 2)
    return np.stack([L, A, B], axis=-1)


def conditional_mean(cfg: PriceGeneratorConfig, X: np.ndarray, start: str, dt: float) -> np.ndarray:
    r"""Closed-form :math:`\mu(X_d) = E[p^{DA}_{d+1} \mid X_d]`, shape (N, D, T).

    With :math:`\ell_d = L_d - m(d) - o(d)`, :math:`a_d = A_d/w(d) - \bar A` and
    :math:`b_d = \log(B_d/\bar B(d)) + v_B/2`:

    .. math::

        \mu_{d,t} = m(d{+}1) + o(d{+}1) + \phi_L\ell_d
          + w(d{+}1)(\bar A + \phi_A a_d) S_{peak}(t, d{+}1)
          - \bar B(d{+}1) e^{\phi_B b_d + \sigma_B^2/2 - v_B/2} S_{solar}(t, d{+}1)
          + E[J_{d+1,t}] + Q(t).

    Entry d refers to the day after ``dates[d]``, so the last entry looks one day past
    the horizon.

    Args:
        cfg: Generator parameters used to simulate X.
        X: State variables (L, A, B), shape (N, D, 3).
        start: First date of X.
        dt: Period length [h].
    """
    n_days = X.shape[1]
    cal = make_calendar(start, n_days + 1, cfg.holidays)
    now = _slice_calendar(cal, 0, n_days)
    nxt = _slice_calendar(cal, 1, n_days + 1)
    X = X.astype(np.float64)
    ell = X[..., 0] - _seasonal_level(cfg, now)[None, :]
    a = X[..., 1] / _peak_weight(cfg, now)[None, :] - cfg.peak_mean
    b = np.log(X[..., 2] / _solar_mean(cfg, now)[None, :]) + 0.5 * cfg.solar_log_sigma ** 2

    level = _seasonal_level(cfg, nxt)[None, :] + cfg.level_ar_phi * ell
    peak = _peak_weight(cfg, nxt)[None, :] * (cfg.peak_mean + cfg.peak_ar_phi * a)
    sig_b = _solar_innovation_sigma(cfg)
    solar = _solar_mean(cfg, nxt)[None, :] * np.exp(
        cfg.solar_ar_phi * b + 0.5 * sig_b ** 2 - 0.5 * cfg.solar_log_sigma ** 2)
    mu = (level[..., None]
          + peak[..., None] * peak_shape(cfg, nxt.season, dt)[None]
          - solar[..., None] * solar_shape(cfg, nxt.season, dt)[None]
          + expected_spikes(cfg, nxt, dt)[None]
          + quarter_hour_profile(cfg, dt)[None, None, :])
    return mu


def _slice_calendar(cal: Calendar, lo: int, hi: int) -> Calendar:
    return Calendar(dates=cal.dates[lo:hi], season=cal.season[lo:hi],
                    weekend=cal.weekend[lo:hi], holiday=cal.holiday[lo:hi])


# ----------------------------------------------------------------------- generation


def soft_floor(prices: np.ndarray, floor: float) -> np.ndarray:
    r"""Soft floor for negative prices: :math:`p \mapsto -F(1 - e^{p/F})` for :math:`p < 0`.

    Continuous with slope 1 at 0 and bounded below by :math:`-F` [€/MWh]: deeply negative
    prices are damped as renewables curtail. Positive prices are unchanged.
    """
    out = prices.copy()
    neg = prices < 0.0
    out[neg] = -floor * (1.0 - np.exp(prices[neg] / floor))
    return out


def _add_spikes(J: np.ndarray, rate: np.ndarray, mean: float, hours: tuple[int, ...], sign: float,
                dt: float, rng: np.random.Generator) -> None:
    """Add Poisson numbers of one-hour jumps (exponential sizes) to J in place."""
    per_hour = round(1.0 / dt)
    count = rng.poisson(np.broadcast_to(rate, J.shape[:2]))
    for k in range(int(count.max(initial=0))):
        active = count > k
        hour = rng.choice(np.asarray(hours), size=J.shape[:2])
        size = sign * rng.exponential(mean, size=J.shape[:2])
        for h in hours:
            sel = active & (hour == h)
            J[sel, h * per_hour:(h + 1) * per_hour] += size[sel][:, None]


def generate_prices(cfg: PriceGeneratorConfig, n_paths: int, start: str, n_days: int, dt: float,
                    seed: int, chunk_size: int = 256) -> PriceScenarios:
    """Simulate DA and IDA1-3 price paths (model in the module docstring).

    Args:
        cfg: Model parameters.
        n_paths: Number of paths N.
        start: First delivery date, ISO format, e.g. "2027-01-01".
        n_days: Number of days D.
        dt: Period length [h], 1.0 or 0.25.
        seed: Seed of the random generator; same seed and arguments give identical output.
        chunk_size: Paths simulated at once for the (N, D, T) parts (memory bound).

    Returns:
        Scenarios with float32 prices, state X = (L, A, B) and the closed-form
        conditional mean.
    """
    if dt not in (1.0, 0.25):
        raise ValueError("dt must be 1.0 or 0.25")
    t0 = time.perf_counter()
    logger.info("generating %d paths x %d days x %d periods from %s (seed %d)",
                n_paths, n_days, round(24.0 / dt), start, seed)
    rng = np.random.default_rng(seed)
    cal = make_calendar(start, n_days, cfg.holidays)
    n_t = round(24.0 / dt)
    half = n_t // 2

    ell, a, b = _simulate_factors(cfg, cal, n_paths, rng)
    X = _factors_to_state(cfg, cal, ell, a, b)
    s_peak = peak_shape(cfg, cal.season, dt)
    s_solar = solar_shape(cfg, cal.season, dt)
    q = quarter_hour_profile(cfg, dt)
    phi_z = cfg.noise_phi_hour ** dt
    neg_rate = (cfg.spike_neg_rate * cal.season * cal.off_day)[None, :]

    shape = (n_paths, n_days, n_t)
    out = {name: np.empty(shape, dtype=np.float32) for name in ("da", "ida1", "ida2", "ida3")}
    chunks = range(0, n_paths, chunk_size)
    for lo in tqdm(chunks, desc="price paths", unit="chunk", disable=len(chunks) < 2, leave=False):
        hi = min(lo + chunk_size, n_paths)
        n = hi - lo
        L, A, B = X[lo:hi, :, 0], X[lo:hi, :, 1], X[lo:hi, :, 2]
        # intraday AR(1) noise, stationary within each day
        zeta = rng.standard_normal((n, n_days, n_t))
        Z = np.empty((n, n_days, n_t))
        Z[..., 0] = cfg.noise_sigma * zeta[..., 0]
        innov = cfg.noise_sigma * np.sqrt(1.0 - phi_z ** 2)
        for t in range(1, n_t):
            Z[..., t] = phi_z * Z[..., t - 1] + innov * zeta[..., t]
        J = np.zeros((n, n_days, n_t))
        _add_spikes(J, np.full((n, n_days), cfg.spike_pos_rate), cfg.spike_pos_mean,
                    cfg.spike_pos_hours, +1.0, dt, rng)
        _add_spikes(J, np.broadcast_to(neg_rate, (n, n_days)), cfg.spike_neg_mean,
                    cfg.spike_neg_hours, -1.0, dt, rng)
        da = (L[..., None] + A[..., None] * s_peak[None] - B[..., None] * s_solar[None]
              + Z + J + q[None, None, :])
        if cfg.negative_price_floor is not None:
            da = soft_floor(da, cfg.negative_price_floor)
        out["da"][lo:hi] = da
        spread = np.zeros_like(da)
        for j, name in enumerate(("ida1", "ida2", "ida3")):
            xi_sol = rng.standard_normal((n, n_days, 1))
            xi_lvl = rng.standard_normal((n, n_days, 1))
            noise = rng.standard_normal((n, n_days, n_t))
            spread += (cfg.ida_bias[j] + cfg.ida_solar_sigma[j] * xi_sol * s_solar[None]
                       + cfg.ida_level_sigma[j] * xi_lvl + cfg.ida_noise_sigma[j] * noise)
            out[name][lo:hi] = da + spread
    out["ida3"][:, :, :half] = np.nan

    mu = conditional_mean(cfg, X, start, dt).astype(np.float32)
    meta = {"source": "synthetic", "seed": seed, "generator": cfg.to_dict()}
    logger.info("prices generated in %.1f s: mean DA %.1f EUR/MWh, %.1f %% negative",
                time.perf_counter() - t0, float(out["da"].mean()), 100.0 * float((out["da"] < 0).mean()))
    return PriceScenarios(dates=cal.dates, dt=dt, X=X.astype(np.float32), conditional_mean=mu,
                          meta=meta, **out)


# ------------------------------------------------------------------------- figures


def _period_hours(dt: float) -> np.ndarray:
    return np.arange(round(24.0 / dt)) * dt + 0.5 * dt


def plot_generator(prices: PriceScenarios, out_dir: Path) -> list[Path]:
    """Write the generator diagnostics (SPEC 2.3) as PNGs to ``out_dir``.

    Figures: sample paths and daily-mean fan; DA fan per period of day for winter /
    summer x weekday / weekend; share of negative prices per period of day; IDA - DA
    spreads per stage; autocorrelation of daily means.
    """
    plots.apply_style()
    cfg_holidays = tuple(prices.meta.get("generator", {}).get("holidays", DEFAULT_HOLIDAYS))
    cal = make_calendar(str(prices.dates[0]), prices.n_days, cfg_holidays)
    month = prices.dates.astype("datetime64[M]").astype(int) % 12 + 1
    h = _period_hours(prices.dt)
    da = prices.da
    paths: list[Path] = []

    # 1. sample paths and daily-mean fan
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    n_show = min(14, prices.n_days)
    x = np.arange(n_show * prices.periods_per_day) * prices.dt / 24.0
    for k in range(min(3, prices.n_paths)):
        ax1.plot(x, da[k, :n_show].reshape(-1), lw=1.0, color=plots.SERIES[k], label=f"path {k}")
    ax1.axhline(0.0, color=plots.INK_SECONDARY, lw=0.6)
    ax1.set(title=f"Sample DA paths, first {n_show} days", xlabel="day", ylabel="DA price [€/MWh]")
    ax1.legend(loc="upper right", ncol=3)
    plots.fan(ax2, prices.dates, da.mean(axis=2))
    ax2.set(title="Daily mean DA price across paths", xlabel="date", ylabel="daily mean [€/MWh]")
    ax2.legend(loc="upper right", ncol=3)
    paths.append(plots.save_figure(fig, out_dir, "sample_paths"))

    # 2. DA fan per period of day by season and day type
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True, constrained_layout=True)
    seasons = {"winter (Dec-Feb)": np.isin(month, (12, 1, 2)), "summer (Jun-Aug)": np.isin(month, (6, 7, 8))}
    days = {"weekday": ~cal.off_day, "weekend / holiday": cal.off_day}
    for i, (s_name, s_mask) in enumerate(seasons.items()):
        for j, (d_name, d_mask) in enumerate(days.items()):
            ax = axes[i, j]
            sel = s_mask & d_mask
            if sel.any():
                plots.fan(ax, h, da[:, sel].reshape(-1, prices.periods_per_day))
            ax.axhline(0.0, color=plots.INK_SECONDARY, lw=0.6)
            ax.set_title(f"{s_name}, {d_name}")
            plots.hour_axis(ax)
            ax.set_ylabel("DA price [€/MWh]")
    axes[0, 0].legend(loc="upper left")
    paths.append(plots.save_figure(fig, out_dir, "da_fan_by_season"))

    # 3. share of negative prices per period of day
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    season_sets = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
    for k, (name, months) in enumerate(season_sets.items()):
        sel = np.isin(month, months)
        if sel.any():
            ax.plot(h, 100.0 * (da[:, sel] < 0).mean(axis=(0, 1)), color=plots.SERIES[k], label=name)
    ax.set(title="Share of negative DA prices", ylabel="share [%]")
    plots.hour_axis(ax)
    ax.legend(loc="upper left")
    paths.append(plots.save_figure(fig, out_dir, "negative_share"))

    # 4. IDA - DA spreads per stage
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for j, name in enumerate(("ida1", "ida2", "ida3")):
        first = prices.periods_per_day // 2 if name == "ida3" else 0  # IDA3 covers 12-24h only
        spread = (prices.stage(name) - da)[..., first:]
        ax = axes[0, j]
        plots.fan(ax, h[first:], spread.reshape(-1, spread.shape[-1]))
        ax.set(title=f"{plots.STAGE_LABELS[name]} − DA per period", ylabel="spread [€/MWh]")
        plots.hour_axis(ax)
        axes[1, 0].hist(spread.reshape(-1), bins=120, histtype="step", density=True, lw=1.5,
                        color=plots.STAGE_COLORS[name], label=plots.STAGE_LABELS[name])
        q = np.quantile(spread, [0.05, 0.95], axis=(0, 1))
        axes[1, 1].plot(h[first:], q[1] - q[0], color=plots.STAGE_COLORS[name], label=plots.STAGE_LABELS[name])
    axes[0, 0].legend(loc="upper left")
    axes[1, 0].set(title="Spread distribution", xlabel="spread [€/MWh]", ylabel="density [1/(€/MWh)]")
    axes[1, 0].legend()
    axes[1, 1].set(title="5–95 % spread width per period", ylabel="width [€/MWh]")
    plots.hour_axis(axes[1, 1])
    axes[1, 1].legend()
    axes[1, 2].axis("off")
    paths.append(plots.save_figure(fig, out_dir, "ida_spreads"))

    # 5. autocorrelation of daily means across days (deterministic calendar part removed)
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    daily = da.mean(axis=2).astype(np.float64)
    resid = daily - daily.mean(axis=0, keepdims=True)
    max_lag = min(30, prices.n_days - 2)
    var = (resid * resid).mean()
    acf = [(resid[:, :-k] * resid[:, k:]).mean() / var for k in range(1, max_lag + 1)]
    lags = np.arange(1, max_lag + 1)
    ax.vlines(lags, 0.0, acf, color=plots.SERIES[0], lw=2)
    ax.plot(lags, acf, "o", color=plots.SERIES[0], ms=4)
    ax.axhline(0.0, color=plots.INK_SECONDARY, lw=0.6)
    ax.set(title="Autocorrelation of daily mean DA price (cross-path mean removed)",
           xlabel="lag [days]", ylabel="autocorrelation [–]")
    paths.append(plots.save_figure(fig, out_dir, "daily_mean_acf"))
    return paths


def generate_and_plot(cfg: PriceGeneratorConfig, n_paths: int, start: str, n_days: int, dt: float,
                      seed: int, scenario_dir: str | Path | None = None, plot: bool = True,
                      results_root: str | Path = "results") -> tuple[PriceScenarios, Path]:
    """Generate scenarios, optionally save them, and write config plus figures.

    The figures and a copy of the config go to ``results/price_generator/<timestamp>/``.

    Returns:
        (scenarios, run folder).
    """
    prices = generate_prices(cfg, n_paths=n_paths, start=start, n_days=n_days, dt=dt, seed=seed)
    if scenario_dir is not None:
        prices.save(scenario_dir)
        logger.info("scenarios saved to %s", scenario_dir)
    run_dir = new_run_dir("price_generator", results_root)
    cfg.to_yaml(run_dir / "config.yaml")
    if plot:
        figures = plot_generator(prices, run_dir / "figures")
        logger.info("%d generator figures written to %s", len(figures), run_dir / "figures")
    return prices, run_dir
