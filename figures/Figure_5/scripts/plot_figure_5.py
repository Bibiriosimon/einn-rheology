from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_STEM = FIGURE_DIR / "Figure_5"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    COLORS,
    FULL_WIDTH_IN,
    apply_npj_style,
    finish_axis,
    panel_label,
    save_figure,
)


def plot_figure_5() -> None:
    apply_npj_style()
    data_path = DATA_DIR / "figure5_extreme_regime_summary.csv"
    frame = pd.read_csv(data_path)

    fraction_order = ["15pct", "25pct", "50pct", "75pct", "100pct"]
    fraction_labels = ["15%", "25%", "50%", "75%", "100%"]
    regime_order = ["low_phi_low_T", "high_phi_highT", "very_low_shear", "low_visc_tail", "high_visc_tail"]
    labels = {
        "low_phi_low_T": "Low filler + low T",
        "high_phi_highT": "High filler + high T",
        "very_low_shear": "Ultra-low shear",
        "low_visc_tail": "Low-viscosity tail",
        "high_visc_tail": "High-viscosity tail",
    }
    fraction_colors = {
        "15pct": "#6BAED6",
        "25pct": "#4292C6",
        "50pct": "#2171B5",
        "75pct": "#08519C",
        "100pct": "#08306B",
    }
    regime_colors = {
        "low_phi_low_T": COLORS["base"],
        "high_phi_highT": COLORS["einn"],
        "very_low_shear": COLORS["xgboost"],
        "low_visc_tail": COLORS["rf"],
        "high_visc_tail": COLORS["svr"],
    }
    regime_markers = dict(zip(regime_order, ["o", "s", "^", "D", "P"]))
    fraction_markers = dict(zip(fraction_order, ["o", "s", "^", "D", "P"]))

    # A compact two-panel layout: each panel has one local legend, while a
    # single note above the axes explains the bar/line encoding shared by both.
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.10))
    fig.subplots_adjust(left=0.075, right=0.955, bottom=0.22, top=0.60, wspace=0.58)

    # Panel (a): Regime-grouped view
    ax = axes[0]
    ax2 = ax.twinx()
    x_regime = np.arange(len(regime_order))
    width = 0.145

    for index, fraction in enumerate(fraction_order):
        sub = frame[frame["TrainFracLabel"] == fraction].set_index("Regime").reindex(regime_order)
        xpos = x_regime + (index - 2) * width
        ax.bar(
            xpos,
            sub["MAE_improvement"].to_numpy(dtype=float),
            width=width,
            color=fraction_colors[fraction],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
            zorder=2,
        )
        ax2.plot(
            xpos,
            sub["Acc10_gain"].to_numpy(dtype=float),
            color=fraction_colors[fraction],
            marker=fraction_markers[fraction],
            markersize=2.8,
            markerfacecolor="white",
            markeredgecolor=fraction_colors[fraction],
            markeredgewidth=0.55,
            linewidth=0.85,
            linestyle="--",
            zorder=4,
        )

    ax.set_xticks(x_regime, [labels[r] for r in regime_order], rotation=22, ha="right")
    ax.set_title("Grouped by extreme regime", pad=6.0)
    ax.set_ylabel(r"MAE reduction, $\Delta$MAE")
    ax2.set_ylabel(r"Acc@0.10 gain")
    ax.set_ylim(-0.03, 0.22)
    ax2.set_ylim(-0.22, 0.42)
    ax.axhline(0, color="#666666", linewidth=0.65, linestyle="--", zorder=1)
    ax2.axhline(0, color="#666666", linewidth=0.55, linestyle=":", zorder=1)
    panel_label(ax, "a")
    finish_axis(ax)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.tick_params(axis="y", width=0.7, labelsize=6.5)
    ax2.grid(False)

    # Panel (b): Fraction-grouped view
    ax = axes[1]
    ax2 = ax.twinx()
    x_fraction = np.arange(len(fraction_order))

    for index, regime in enumerate(regime_order):
        sub = frame[frame["Regime"] == regime].set_index("TrainFracLabel").reindex(fraction_order)
        xpos = x_fraction + (index - 2) * width
        ax.bar(
            xpos,
            sub["MAE_improvement"].to_numpy(dtype=float),
            width=width,
            color=regime_colors[regime],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.90,
            zorder=2,
        )
        ax2.plot(
            xpos,
            sub["Acc10_gain"].to_numpy(dtype=float),
            color=regime_colors[regime],
            marker=regime_markers[regime],
            markersize=2.8,
            markerfacecolor="white",
            markeredgecolor=regime_colors[regime],
            markeredgewidth=0.55,
            linewidth=0.85,
            linestyle="--",
            zorder=4,
        )

    ax.set_xticks(x_fraction, fraction_labels)
    ax.set_title("Grouped by training fraction", pad=6.0)
    ax.set_ylabel(r"MAE reduction, $\Delta$MAE")
    ax2.set_ylabel(r"Acc@0.10 gain")
    ax.set_ylim(-0.03, 0.22)
    ax2.set_ylim(-0.22, 0.42)
    ax.axhline(0, color="#666666", linewidth=0.65, linestyle="--", zorder=1)
    ax2.axhline(0, color="#666666", linewidth=0.55, linestyle=":", zorder=1)
    panel_label(ax, "b")
    finish_axis(ax)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.tick_params(axis="y", width=0.7, labelsize=6.5)
    ax2.grid(False)

    # Legends above
    left_handles = [
        Line2D(
            [],
            [],
            color=fraction_colors[f],
            marker=fraction_markers[f],
            linestyle="--",
            markersize=3.0,
            linewidth=0.85,
            label=l,
        )
        for f, l in zip(fraction_order, fraction_labels)
    ]
    axes[0].legend(
        handles=left_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.52),
        ncol=3,
        title="Training fraction",
        title_fontsize=6.2,
        fontsize=5.7,
        columnspacing=0.95,
        handlelength=1.5,
    )

    right_handles = [
        Line2D(
            [],
            [],
            color=regime_colors[r],
            marker=regime_markers[r],
            linestyle="--",
            markersize=3.0,
            linewidth=0.85,
            label=labels[r],
        )
        for r in regime_order
    ]
    axes[1].legend(
        handles=right_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.52),
        ncol=2,
        title="Extreme regime",
        title_fontsize=6.2,
        fontsize=5.35,
        columnspacing=0.75,
        handlelength=1.45,
    )

    fig.text(
        0.50,
        0.955,
        r"Bars: MAE reduction, $\Delta$MAE   |   Dashed lines: Acc@0.10 gain",
        ha="center",
        va="center",
        fontsize=6.6,
    )

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 5 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_5()
