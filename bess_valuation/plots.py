"""Figures (SPEC section 7) and the style shared by all of them.

Valuation figures implemented so far: 1 (value fan), 2 (value decomposition), daily
cashflows (intrinsic, LSMC, perfect foresight), 4 (SoC fan and heatmap), 5 (dispatch
profile vs DA prices, winter vs summer), 9 (continuation value vs SoC), 12 (vertex-library
diagnostics, library solver only), 13 (regression R^2, in- vs out-of-sample value). Two quantities with different units are shown
in stacked panels with a shared x-axis, never on a second y-axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour tokens per theme. Categorical slots in fixed order (validated default palette:
# light steps on the light surface, dark steps checked against the dark-blue surface).
THEMES = {
    "light": {
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
        # sequential blue ramp for quantile fans: [1] 5-95 band, [2] 25-75 band, [-2] median
        "blue_ramp": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"),
        "cmap": ("#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"),
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_secondary": "#52514e", "grid": "#e4e3df",
    },
    "dark": {
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
        "blue_ramp": ("#0d366b", "#184f95", "#256abf", "#3987e5", "#9ec5f4", "#cde2fb"),
        "cmap": ("#101b2c", "#184f95", "#3987e5", "#9ec5f4"),
        "surface": "#0b0f17", "ink": "#e6e9ef", "ink_secondary": "#9aa4b2", "grid": "#1e2633",
    },
}
STAGE_LABELS = {"da": "DA", "ida1": "IDA1", "ida2": "IDA2", "ida3": "IDA3"}

# Active tokens; apply_style(theme) switches them (module globals, read at plot time).
THEME = "light"
SERIES = THEMES["light"]["series"]
STAGE_COLORS = {"da": SERIES[0], "ida1": SERIES[1], "ida2": SERIES[2], "ida3": SERIES[3]}
BLUE_RAMP = THEMES["light"]["blue_ramp"]
SURFACE = THEMES["light"]["surface"]
INK = THEMES["light"]["ink"]
INK_SECONDARY = THEMES["light"]["ink_secondary"]
GRID = THEMES["light"]["grid"]

DPI = 150


def apply_style(theme: str | None = None) -> None:
    """Set the matplotlib rcParams used by every figure of the package.

    Args:
        theme: "light" (reports) or "dark" (app); None keeps the active theme.
    """
    global THEME, SERIES, STAGE_COLORS, BLUE_RAMP, SURFACE, INK, INK_SECONDARY, GRID
    if theme is not None:
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}")
        THEME = theme
        tok = THEMES[theme]
        SERIES = tok["series"]
        STAGE_COLORS = {"da": SERIES[0], "ida1": SERIES[1], "ida2": SERIES[2], "ida3": SERIES[3]}
        BLUE_RAMP = tok["blue_ramp"]
        SURFACE, INK, INK_SECONDARY, GRID = tok["surface"], tok["ink"], tok["ink_secondary"], tok["grid"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "lines.linewidth": 1.5,
        "font.size": 9,
    })


def sequential_cmap() -> matplotlib.colors.Colormap:
    """One-hue blue colormap of the active theme (low values recede into the surface)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"blue_{THEME}", THEMES[THEME]["cmap"])


def fan(ax: plt.Axes, x: np.ndarray, samples: np.ndarray, color_ramp: tuple[str, ...] | None = None,
        label: str | None = None) -> None:
    """Draw a quantile fan (5/25/50/75/95 %) of ``samples`` along axis 0.

    Args:
        ax: Target axes.
        x: Positions, shape (K,).
        samples: Values, shape (S, K); quantiles are taken over S, NaNs ignored.
        color_ramp: Ramp; [1] for the 5-95 band, [2] for 25-75, [-2] median (default: theme).
        label: Legend label of the median line.
    """
    color_ramp = color_ramp or BLUE_RAMP
    q05, q25, q50, q75, q95 = np.nanquantile(samples, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(x, q05, q95, color=color_ramp[1], lw=0, label="5–95 %")
    ax.fill_between(x, q25, q75, color=color_ramp[2], lw=0, label="25–75 %")
    ax.plot(x, q50, color=color_ramp[-2], lw=1.5, label=label or "median")


def hour_axis(ax: plt.Axes) -> None:
    """Format the x-axis as hour of day 0-24."""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("hour of day [h]")


def save_figure(fig: plt.Figure, out_dir: Path, name: str, pdf: bool = False) -> Path:
    """Save ``fig`` as ``out_dir/name.png`` (150 dpi) and optionally as PDF, then close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if pdf:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ valuation figures


def _monthly(dates: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum the last axis of ``values`` (..., D) per calendar month -> (months, (..., M))."""
    months = dates.astype("datetime64[M]")
    keys = np.unique(months)
    out = np.stack([values[..., months == m].sum(axis=-1) for m in keys], axis=-1)
    return keys, out


def plot_value_fan(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 1: cumulative realised value over time with intrinsic and perfect foresight."""
    apply_style()
    dates = result.dates.astype("datetime64[D]").astype(object)
    x = np.arange(len(dates))
    cum = np.cumsum(result.forward.cash, axis=1) / 1e3
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fan(ax, x, cum, label="LSMC median (out-of-sample)")
    ax.plot(x, cum.mean(axis=0), color=INK, lw=1.2, ls="--", label="LSMC mean")
    ax.plot(x, np.cumsum(result.intrinsic_cash) / 1e3, color=SERIES[1], lw=2, label="intrinsic (mean prices)")
    ax.plot(x, np.cumsum(result.pf_cash, axis=1).mean(axis=0) / 1e3, color=SERIES[2], lw=2,
            label="perfect foresight (mean)")
    _date_ticks(ax, result.dates)
    ax.set_ylabel("cumulative value [k€]")
    ax.set_title(f"Plot 1 - cumulative realised value, {result.config.name} (DA only)")
    ax.legend(loc="upper left")
    return save_figure(fig, out_dir, "01_value_fan", pdf)


def _date_ticks(ax: plt.Axes, dates: np.ndarray) -> None:
    months = dates.astype("datetime64[M]")
    first = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    step = max(1, len(first) // 12)
    first = first[::step]
    ax.set_xticks(first)
    ax.set_xticklabels([str(months[i]) for i in first], rotation=45, ha="right")
    ax.set_xlim(0, len(dates) - 1)
    ax.set_xlabel("delivery month")


def plot_value_decomposition(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 2: intrinsic vs extrinsic value and value per auction stage, per month."""
    apply_style()
    keys, lsmc = _monthly(result.dates, result.forward.cash.mean(axis=0))
    _, intr = _monthly(result.dates, result.intrinsic_cash)
    extr = lsmc - intr
    x = np.arange(keys.size)
    labels = [str(k)[2:] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    ax = axes[0]
    ax.bar(x, intr / 1e3, width=0.75, color=SERIES[1], label="intrinsic", edgecolor=SURFACE, lw=1)
    ax.bar(x, extr / 1e3, width=0.75, bottom=np.where(extr >= 0, intr, 0) / 1e3, color=SERIES[0],
           label="extrinsic (LSMC - intrinsic)", edgecolor=SURFACE, lw=1)
    ax.axhline(0, color=INK_SECONDARY, lw=0.8)
    ax.set_title("intrinsic vs extrinsic value per month")
    ax.set_ylabel("value per month [k€]")
    ax.legend(loc="upper left")
    ax = axes[1]
    bottom = np.zeros(keys.size)
    stage_cash = {"da": lsmc, "ida1": 0 * lsmc, "ida2": 0 * lsmc, "ida3": 0 * lsmc}
    for stage, vals in stage_cash.items():
        if not np.any(vals):
            continue
        ax.bar(x, vals / 1e3, width=0.75, bottom=bottom / 1e3, color=STAGE_COLORS[stage],
               label=STAGE_LABELS[stage], edgecolor=SURFACE, lw=1)
        bottom += vals
    ax.set_title("value added per auction stage (IDA1-3 from milestone 4)")
    ax.legend(loc="upper left")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("delivery month (YY-MM)")
    fig.suptitle(f"Plot 2 - value decomposition, mean over out-of-sample paths, {result.config.name}",
                 fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "02_value_decomposition", pdf)


def plot_soc(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 4: end-of-day SoC fan over time and SoC heatmap (period of day x day, median)."""
    apply_style()
    fwd = result.forward
    soc_end = result.soc_mwh(fwd.soc_index[:, 1:])
    x = np.arange(soc_end.shape[1])
    capacity = result.config.e_min + result.kappa * (result.config.e_max - result.config.e_min)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), gridspec_kw={"height_ratios": [1, 1.2]},
                             layout="constrained")
    ax = axes[0]
    fan(ax, x, soc_end)
    if np.any(result.kappa != 1.0):
        ax.plot(x, capacity, color=SERIES[1], lw=1.5, ls="--", label="usable capacity (degradation)")
    ax.set_ylabel("end-of-day SoC [MWh]")
    ax.set_ylim(result.config.e_min - 0.05, result.config.e_max + 0.05)
    _date_ticks(ax, result.dates)
    ax.set_title("end-of-day SoC across out-of-sample paths")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=3)
    ax = axes[1]
    med = np.median(fwd.soc, axis=0).T  # (T, D)
    im = ax.imshow(med, aspect="auto", origin="lower", cmap=sequential_cmap(),
                   extent=(-0.5, med.shape[1] - 0.5, 0, 24), vmin=result.config.e_min,
                   vmax=result.config.e_max, interpolation="nearest")
    ax.set_yticks(range(0, 25, 4))
    ax.set_ylabel("hour of day [h]")
    ax.grid(False)
    _date_ticks(ax, result.dates)
    ax.set_title("SoC at the end of each period, median over paths")
    fig.colorbar(im, ax=ax, label="SoC [MWh]", pad=0.01)
    fig.suptitle(f"Plot 4 - state of charge, {result.config.name}", fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "04_soc", pdf)


SEASONS = {"winter (Dec-Feb)": (12, 1, 2), "summer (Jun-Aug)": (6, 7, 8)}


def plot_dispatch_profile(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 5: DA price fan and mean charge / discharge per period of day, winter vs summer.

    Days of the season that are not in the horizon leave an empty column.
    """
    apply_style()
    fwd = result.forward
    n_t = fwd.charge.shape[2]
    hours = (np.arange(n_t) + 0.5) * result.dt
    months = (result.dates.astype("datetime64[M]").astype(int) % 12) + 1
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True, sharey="row", layout="constrained")
    for col, (season, season_months) in enumerate(SEASONS.items()):
        days = np.isin(months, season_months)
        ax_p, ax_d = axes[0, col], axes[1, col]
        if days.any():
            prices = result.da_oos[:, days].reshape(-1, n_t).astype(np.float64)
            charge = fwd.charge[:, days].reshape(-1, n_t) / result.dt
            discharge = fwd.discharge[:, days].reshape(-1, n_t) / result.dt
            fan(ax_p, hours, prices, label="median DA price")
            ax_d.fill_between(hours, 0, discharge.mean(axis=0), color=SERIES[0], lw=0,
                              label="discharge, mean")
            ax_d.fill_between(hours, 0, -charge.mean(axis=0), color=SERIES[1], lw=0,
                              label="charge, mean (negative)")
        else:
            ax_p.text(0.5, 0.5, "no days of this season in the horizon", transform=ax_p.transAxes,
                      ha="center", color=INK_SECONDARY)
        n_days = int(days.sum())
        ax_p.set_title(f"{season}: DA price per period of day ({n_days} days)")
        ax_d.set_title(f"{season}: charge / discharge")
        for ax in (ax_p, ax_d):
            ax.axhline(0, color=INK_SECONDARY, lw=0.8)
        hour_axis(ax_d)
    axes[0, 0].set_ylabel("DA price [€/MWh]")
    axes[1, 0].set_ylabel("power [MW]")
    filled = [col for col, m in enumerate(SEASONS.values()) if np.isin(months, m).any()]
    if filled:
        axes[0, filled[0]].legend(loc="upper left", ncol=3)
        axes[1, filled[0]].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"Plot 5 - dispatch profile vs DA prices by season, {result.config.name}",
                 fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "05_dispatch_profile", pdf)


def plot_daily_cashflow(result, out_dir: Path, pdf: bool = False) -> Path:
    """Daily cashflow: LSMC across out-of-sample paths, intrinsic schedule, perfect foresight."""
    apply_style()
    x = np.arange(len(result.dates))
    cash = result.forward.cash
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    fan(ax, x, cash, label="LSMC median")
    ax.plot(x, cash.mean(axis=0), color=INK, lw=1.0, label="LSMC mean")
    ax.plot(x, result.intrinsic_cash, color=SERIES[1], lw=1.2, label="intrinsic (mean prices)")
    ax.plot(x, result.pf_cash.mean(axis=0), color=SERIES[2], lw=1.0, label="perfect foresight (mean)")
    ax.set_ylabel("daily cashflow [€]")
    ax.set_title("daily cashflow across out-of-sample paths")
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    ax = axes[1]
    window = max(1, min(7, len(x)))
    kernel = np.ones(window) / window
    extr = np.convolve(cash.mean(axis=0) - result.intrinsic_cash, kernel, mode="same")
    gap = np.convolve(result.pf_cash.mean(axis=0) - cash.mean(axis=0), kernel, mode="same")
    ax.plot(x, extr, color=SERIES[0], lw=1.5, label="LSMC mean - intrinsic (extrinsic)")
    ax.plot(x, gap, color=SERIES[2], lw=1.5, label="perfect foresight - LSMC mean")
    ax.axhline(0, color=INK_SECONDARY, lw=0.8)
    ax.set_ylabel("€ per day (7-day mean)")
    ax.set_title("extrinsic value and gap to perfect foresight per day")
    ax.legend(loc="upper left", fontsize=7)
    _date_ticks(ax, result.dates)
    fig.suptitle(f"Daily cashflows, {result.config.name}", fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "03_daily_cashflow", pdf)


def plot_regression(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 13: R^2 of the continuation regression per day, in-sample vs out-of-sample value."""
    apply_style()
    bw = result.backward
    x = np.arange(len(result.dates))
    has_cv = bool(np.any(bw.var_ratio != 1.0))
    fig, axes = plt.subplots(1, 3 if has_cv else 2, figsize=(13 if has_cv else 10, 3.8),
                             layout="constrained", gridspec_kw={"width_ratios": [2.2, 1] + ([1.4] if has_cv else [])})
    ax = axes[0]
    r2 = np.where(np.isfinite(bw.r2), bw.r2, np.nan)
    valid = np.isfinite(r2).any(axis=1)
    lo = np.full(len(x), np.nan)
    hi = np.full(len(x), np.nan)
    mid = np.full(len(x), np.nan)
    lo[valid] = np.nanmin(r2[valid], axis=1)
    hi[valid] = np.nanmax(r2[valid], axis=1)
    mid[valid] = np.nanmean(r2[valid], axis=1)
    ax.fill_between(x, lo, hi, color=BLUE_RAMP[1], lw=0, label="min-max over end SoC")
    ax.plot(x, mid, color=BLUE_RAMP[-2], lw=1.2, label="mean over end SoC")
    ax.set_ylim(min(0.0, np.nanmin(lo) if valid.any() else 0.0), 1.0)
    ax.set_ylabel("R² of the continuation regression")
    ax.set_title("regression fit per day")
    ax.legend(loc="upper left", fontsize=7)
    _date_ticks(ax, result.dates)
    ax = axes[1]
    s = result.summary()
    labels = ["intrinsic", "LSMC\nin-sample", "LSMC\nout-of-sample", "perfect\nforesight"]
    values = [s["value_intrinsic_eur"], s["value_in_sample_eur"], s["value_oos_eur"],
              s["value_perfect_foresight_eur"]]
    colors = [SERIES[1], BLUE_RAMP[2], SERIES[0], SERIES[2]]
    ax.bar(np.arange(4), np.array(values) / 1e3, color=colors, width=0.7, edgecolor=SURFACE, lw=1)
    ax.errorbar([2], [values[2] / 1e3], yerr=[2 * s["value_oos_stderr_eur"] / 1e3], color=INK,
                capsize=4, lw=1.2)
    for k, v in enumerate(values):
        ax.text(k, v / 1e3, f"{v / 1e3:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("value [k€]")
    ax.set_ylim(0, max(values) / 1e3 * 1.12)
    ax.set_title("values (error bar: ±2 standard errors)")
    if has_cv:
        ax = axes[2]
        ax.plot(x[:-1], bw.var_ratio[:-1].mean(axis=1), color=SERIES[0], lw=1.2)
        ax.axhline(1.0, color=INK_SECONDARY, lw=0.8, ls="--")
        ax.set_ylabel("Var(V) / Var(V - h)")
        ax.set_title("variance reduction by the control variate")
        _date_ticks(ax, result.dates)
    fig.suptitle(f"Plot 13 - regression diagnostics, {result.config.name}", fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "13_regression", pdf)


def plot_continuation(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 9: continuation value vs SoC for selected days, before / after concave projection."""
    apply_style()
    bw = result.backward
    k = bw.curve_days.size
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.8), squeeze=False, layout="constrained")
    for a, ax in enumerate(axes[0]):
        day = bw.curve_days[a]
        for s in range(bw.curve_samples_raw.shape[1]):
            ax.plot(result.grid_days[day], bw.curve_samples_raw[a, s] / 1e3, color=BLUE_RAMP[1], lw=1,
                    label="sample paths, raw" if s == 0 else None)
            ax.plot(result.grid_days[day], bw.curve_samples_projected[a, s] / 1e3, color=BLUE_RAMP[3], lw=1,
                    ls="--", label="sample paths, projected" if s == 0 else None)
        ax.plot(result.grid_days[day], bw.curve_raw[a] / 1e3, color=SERIES[1], lw=2, marker="o", ms=4,
                label="mean, raw")
        ax.plot(result.grid_days[day], bw.curve_projected[a] / 1e3, color=INK, lw=1.5, ls="--", label="mean, projected")
        ax.set_title(f"{result.dates[day]}: {100 * bw.violation_share[day]:.0f} % of paths projected",
                     fontsize=9)
        ax.set_xlabel("end-of-day SoC [MWh]")
        if a == 0:
            ax.set_ylabel("continuation value [k€]")
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle(f"Plot 9 - continuation value vs SoC (concavity check), {result.config.name}",
                 fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "09_continuation_value", pdf)


def plot_library(result, out_dir: Path, pdf: bool = False) -> Path:
    """Plot 12: discovery curve, Good-Turing estimate, hit-frequency rank plot, certificate share."""
    apply_style()
    lib = result.library
    stats = result.library_stats["pairs"]
    keys = sorted(lib.pairs)
    show = [keys[i] for i in np.unique(np.linspace(0, len(keys) - 1, 4).round().astype(int))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    ax = axes[0, 0]
    for c, key in enumerate(show):
        disc = np.array(lib.pairs[key].stats.discovery).reshape(-1, 2)
        if disc.size:
            ax.plot(disc[:, 0], disc[:, 1], color=SERIES[c], marker="o", ms=3,
                    label=f"s {key[0]:g} -> {key[1]:g} MWh")
    q = max(p["queries"] for p in stats)
    ax.plot([0, q], [0, q], color=GRID, lw=1, ls=":", label="every query new")
    ax.axhline(lib.max_vertices, color=INK_SECONDARY, lw=0.8, ls="--", label="cache cap")
    ax.set_xlabel("price vectors queried (conditional means first, then prices)")
    ax.set_ylabel("distinct optimal vertices")
    ax.set_title("discovery curve")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    gt = np.array([p["good_turing"] for p in stats])
    ax.bar(np.arange(gt.size), gt, color=SERIES[0], width=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("SoC pair (index, s_start-major)")
    ax.set_ylabel("P(unseen vertex) ≈ n1 / N")
    ax.set_title("Good-Turing estimate per SoC pair")
    ax = axes[1, 0]
    for c, key in enumerate(show):
        hits = np.sort(lib.pairs[key].hits[:lib.pairs[key].size])[::-1]
        if hits.size:
            ax.loglog(np.arange(1, hits.size + 1), hits, color=SERIES[c], lw=1.5,
                      label=f"s {key[0]:g} -> {key[1]:g} MWh")
    ax.set_xlabel("vertex rank")
    ax.set_ylabel("hits")
    ax.set_title("hit frequency vs rank (cached vertices)")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    queries = np.array([max(p["queries"], 1) for p in stats], dtype=float)
    parts = [("certificate, frequent vertices", "certified_hot", SERIES[2]),
             ("certificate, all cached vertices", "certified_full", SERIES[0]),
             ("simplex / DP", "simplex", SERIES[1])]
    bottom = np.zeros(len(stats))
    for label, key, color in parts:
        share = np.array([p[key] for p in stats]) / queries
        ax.bar(np.arange(len(stats)), share, bottom=bottom, color=color, width=0.8, label=label)
        bottom += share
    ax.set_ylim(0, 1)
    ax.set_xlabel("SoC pair (index, s_start-major)")
    ax.set_ylabel("share of queries")
    ax.set_title("how queries were answered")
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=3)
    fig.suptitle(f"Plot 12 - vertex library, {result.config.name}", fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "12_vertex_library", pdf)


def plot_convexity(result, out_dir: Path, pdf: bool = False) -> Path:
    """Revenue of one day along its three principal price directions (convex by theory)."""
    apply_style()
    c = result.convexity
    k = c.directions.shape[0]
    n_t = c.mean_price.size
    hours = (np.arange(n_t) + 0.5) * 24.0 / n_t
    fig, axes = plt.subplots(2, k, figsize=(4.2 * k, 7), layout="constrained", squeeze=False)
    pair_labels = [f"{a:g} -> {b:g} MWh" for a, b in c.pairs]
    for i in range(k):
        ax = axes[0, i]
        ax.plot(hours, c.mean_price, color=INK_SECONDARY, lw=1.2, label="mean curve")
        ax.plot(hours, c.mean_price + c.sigma[i] * c.directions[i], color=SERIES[0], lw=1.4,
                label="+1 σ along the direction")
        ax.plot(hours, c.mean_price - c.sigma[i] * c.directions[i], color=SERIES[1], lw=1.4,
                label="-1 σ along the direction")
        ax.set_title(f"direction {i + 1}: {100 * c.explained[i]:.0f} % of the price variance")
        hour_axis(ax)
        ax.set_ylabel("price [€/MWh]")
        ax = axes[1, i]
        for j, label in enumerate(pair_labels):
            ax.plot(c.t, c.values[i, :, j], color=SERIES[j], lw=1.6, marker="o", ms=2.5, label=label)
        lo = np.nanmin(c.values[i][np.isfinite(c.values[i])])
        ax.plot(c.scores[i], np.full(c.scores.shape[1], lo), "|", color=INK_SECONDARY, ms=8, alpha=0.5,
                label="out-of-sample curves")
        ok = bool(np.all(c.min_second_difference[i] >= -c.tolerance))
        ax.set_title(f"revenue along direction {i + 1}: {'convex ✓' if ok else 'NOT convex ✗'}")
        ax.set_xlabel("t [multiples of σ]")
        ax.set_ylabel("daily revenue [€]")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    axes[1, 0].legend(loc="best", fontsize=7)
    fig.suptitle(f"Convexity of the daily revenue along the principal price factors, {c.date}, "
                 f"{result.config.name}", fontsize=10, fontweight="bold")
    return save_figure(fig, out_dir, "15_convexity", pdf)


def plot_results(result, out_dir: Path, pdf: bool = False, theme: str = "light") -> list[Path]:
    """Write all valuation figures implemented so far in the given theme; returns their paths."""
    previous = THEME
    apply_style(theme)
    try:
        funcs = [plot_value_fan, plot_value_decomposition, plot_daily_cashflow, plot_soc,
                 plot_dispatch_profile, plot_continuation, plot_regression]
        if result.library is not None and result.library.pairs:
            funcs.append(plot_library)
        if result.convexity is not None:
            funcs.append(plot_convexity)
        return [f(result, out_dir, pdf) for f in funcs]
    finally:
        apply_style(previous)
