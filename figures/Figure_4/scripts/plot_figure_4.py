from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_STEM = FIGURE_DIR / "Figure_4"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    COLORS,
    FULL_WIDTH_IN,
    apply_npj_style,
    finish_axis,
    panel_label,
    save_figure,
)


def plot_figure_4() -> None:
    apply_npj_style()
    data_path = DATA_DIR / "figure4_final_curves.csv"
    frame = pd.read_csv(data_path)

    panels = ["A", "B", "C"]
    panel_titles = {
        "A": r"$\phi=0.000,\ T^*=0.50$",
        "B": r"$\phi=0.198,\ T^*=0.45$",
        "C": r"$\phi=0.198,\ T^*=0.90$",
    }

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.08))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.21, top=0.72, wspace=0.36)

    for label, axis, p in zip("abc", axes, panels):
        sub = frame[frame["Panel"] == p].sort_values("logg")
        axis.scatter(
            sub["logg"],
            sub["y_true"],
            s=15,
            color=COLORS["nemd"],
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        axis.plot(sub["logg"], sub["base_pred"], color=COLORS["base"], linestyle="--", linewidth=1.35)
        axis.plot(sub["logg"], sub["einn_pred"], color=COLORS["einn"], linewidth=1.5)
        axis.set_title(panel_titles[p], pad=3.0)
        axis.set_xlabel(r"$\log_{10}\dot{\gamma}$")
        axis.set_ylabel(r"$\log_{10}\eta$")

        mae_b = sub["MAE_base"].iloc[0]
        mae_e = sub["MAE_EINN"].iloc[0]
        metric = (
            rf"MAE$_{{\rm base}}$ = {mae_b:.3f}"
            "\n"
            rf"MAE$_{{\rm EINN}}$ = {mae_e:.3f}"
        )
        axis.text(
            0.96,
            0.95,
            metric,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            linespacing=1.0,
            bbox={"facecolor": "white", "edgecolor": "#B8BEC6", "linewidth": 0.4, "pad": 1.4},
        )
        panel_label(axis, label)
        finish_axis(axis)

    handles = [
        Line2D([], [], color=COLORS["nemd"], marker="o", linestyle="none", markersize=3.5, label="NEMD reference"),
        Line2D([], [], color=COLORS["base"], linestyle="--", linewidth=1.35, label="Base"),
        Line2D([], [], color=COLORS["einn"], linewidth=1.5, label="EINN"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985))

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 4 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_4()
