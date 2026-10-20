"""Figure style shared by all plots (SPEC section 7).

The valuation figures (plots 1-14) are added from milestone 3 on; this module
currently provides the common style and the helpers the price generator uses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots in fixed order (validated default palette); one per auction stage.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
STAGE_COLORS = {"da": SERIES[0], "ida1": SERIES[1], "ida2": SERIES[2], "ida3": SERIES[3]}
STAGE_LABELS = {"da": "DA", "ida1": "IDA1", "ida2": "IDA2", "ida3": "IDA3"}

# Sequential blue ramp, light -> dark, for quantile fans.
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

DPI = 150


def apply_style() -> None:
    """Set the matplotlib rcParams used by every figure of the package."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "font.size": 9,
    })


def fan(ax: plt.Axes, x: np.ndarray, samples: np.ndarray, color_ramp: tuple[str, ...] = BLUE_RAMP,
        label: str | None = None) -> None:
    """Draw a quantile fan (5/25/50/75/95 %) of ``samples`` along axis 0.

    Args:
        ax: Target axes.
        x: Positions, shape (K,).
        samples: Values, shape (S, K); quantiles are taken over S, NaNs ignored.
        color_ramp: Light-to-dark colours; [1] for the 5-95 band, [2] for 25-75, [-2] median.
        label: Legend label of the median line.
    """
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
