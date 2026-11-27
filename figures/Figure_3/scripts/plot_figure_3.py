from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_STEM = FIGURE_DIR / "Figure_3"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    COLORS,
    FULL_WIDTH_IN,
    apply_npj_style,
    finish_axis,
    panel_label,
    save_figure,
)


def _plot_curve_family(ax, frame: pd.DataFrame, family: str, panel: str) -> None:
    levels = sorted(frame[frame["series"] == "pred"][family].unique())
    palette = plt.get_cmap("cividis")(np.linspace(0.08, 0.92, len(levels)))
    for index, (level, color) in enumerate(zip(levels, palette)):
        pred = frame[(frame["series"] == "pred") & np.isclose(frame[family], level)].sort_values("logg")
        real = (
            frame[(frame["series"] == "real") & np.isclose(frame[family], level)]
            .drop_duplicates(subset=["logg", "value"])
            .sort_values("logg")
        )
        label = rf"$\phi={level:.3f}$" if family == "phi" else rf"$T^*={level:.1f}$"
        ax.plot(pred["logg"], pred["value"], color=color, linewidth=1.15, label=label, zorder=2)
        ax.scatter(
            real["logg"],
            real["value"],
            s=7,
            color=color,
            edgecolor="#333333",
            linewidth=0.2,
            zorder=3,
        )
    ax.set_xlabel(r"$\log_{10}\dot{\gamma}$")
    ax.set_ylabel(r"$\log_{10}\eta$")
    ax.legend(loc="upper right", ncol=1, fontsize=5.5, handlelength=1.5, labelspacing=0.18)
    panel_label(ax, panel)
    finish_axis(ax)


def plot_figure_3() -> None:
    apply_npj_style()
    convergence = pd.read_csv(DATA_DIR / "figure3a_training_convergence.csv")
    phi_effect = pd.read_csv(DATA_DIR / "figure3b_viscosity_phi_effect.csv")
    temp_effect = pd.read_csv(DATA_DIR / "figure3c_viscosity_temp_effect.csv")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH_IN, 1.88),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.24, top=0.84, wspace=0.52)

    axis = axes[0]
    axis.plot(
        convergence["epoch"],
        convergence["data_fitting_loss"],
        color=COLORS["base"],
        linewidth=1.25,
        label="Data-fitting loss",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Data-fitting loss")

    second = axis.twinx()
    second.plot(
        convergence["epoch"],
        convergence["monotonicity_loss"] * 1.0e4,
        color=COLORS["einn"],
        linewidth=1.2,
        linestyle="--",
        label="Monotonicity loss",
    )
    second.set_ylabel(r"Monotonicity loss ($\times 10^{-4}$)", color=COLORS["einn"], labelpad=1.0)
    second.tick_params(axis="y", colors=COLORS["einn"], labelsize=6.5, width=0.7)
    second.spines["right"].set_visible(True)
    second.spines["right"].set_color(COLORS["einn"])
    second.spines["right"].set_linewidth(0.8)

    handles = axis.get_lines() + second.get_lines()
    axis.legend(handles, [line.get_label() for line in handles], loc="upper right", fontsize=5.6, handlelength=1.6)
    panel_label(axis, "a")
    finish_axis(axis)

    _plot_curve_family(axes[1], phi_effect, "phi", "b")
    _plot_curve_family(axes[2], temp_effect, "T", "c")

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 3 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_3()
