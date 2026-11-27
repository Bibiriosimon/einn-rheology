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
OUTPUT_STEM = FIGURE_DIR / "Figure_2"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    COLORS,
    FULL_WIDTH_IN,
    apply_npj_style,
    finish_axis,
    panel_label,
    save_figure,
)


def _response_panel(
    ax,
    csv_path: Path,
    x_column: str,
    x_label: str,
    title: str,
    panel: str,
) -> None:
    frame = pd.read_csv(csv_path)
    real = frame[frame["series"] == "real"].drop_duplicates(subset=[x_column, "value"]).sort_values(x_column)
    pred = frame[frame["series"] == "pred"].drop_duplicates(subset=[x_column, "value"]).sort_values(x_column)

    if x_column == "logg":
        real_x = np.power(10.0, real[x_column].to_numpy(dtype=float))
        pred_x = np.power(10.0, pred[x_column].to_numpy(dtype=float))
        ax.set_xscale("log")
        ax.set_xlabel(r"$\dot{\gamma}$")
    else:
        real_x = real[x_column].to_numpy(dtype=float)
        pred_x = pred[x_column].to_numpy(dtype=float)
        ax.set_xlabel(x_label)

    real_y = np.power(10.0, real["value"].to_numpy(dtype=float))
    pred_y = np.power(10.0, pred["value"].to_numpy(dtype=float))
    ax.set_yscale("log")

    ax.plot(
        real_x,
        real_y,
        color="#4D4D4D",
        linewidth=1.05,
        marker="s",
        markersize=2.5,
        markerfacecolor="#4D4D4D",
        markeredgecolor="white",
        markeredgewidth=0.25,
        label="NEMD reference",
        zorder=3,
    )
    ax.plot(
        pred_x,
        pred_y,
        color=COLORS["einn"],
        linewidth=1.45,
        marker="o" if len(pred) < 30 else None,
        markersize=2.5,
        markerfacecolor=COLORS["einn"],
        markeredgecolor="white",
        markeredgewidth=0.25,
        label="EINN",
        zorder=2,
    )
    ax.set_ylabel(r"$\eta$")
    ax.set_title(title, pad=3.0)
    panel_label(ax, panel)
    finish_axis(ax)


def plot_figure_2() -> None:
    apply_npj_style()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.05))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.72, wspace=0.34)

    _response_panel(
        axes[0],
        DATA_DIR / "figure2_panel_a_viscosity_vs_shear.csv",
        "logg",
        r"$\log_{10}\dot{\gamma}$",
        r"$\phi=0.198,\ T^*=0.50$",
        "a",
    )
    _response_panel(
        axes[1],
        DATA_DIR / "figure2_panel_b_viscosity_vs_temp.csv",
        "T",
        r"$T^*$",
        r"$\phi=0.198,\ \log_{10}\dot{\gamma}=-3$",
        "b",
    )
    _response_panel(
        axes[2],
        DATA_DIR / "figure2_panel_c_viscosity_vs_phi.csv",
        "phi",
        r"$\phi$",
        r"$T^*=0.50,\ \log_{10}\dot{\gamma}=-5$",
        "c",
    )

    handles = [
        Line2D([], [], color="#4D4D4D", marker="s", linestyle="-", markersize=3.0, linewidth=1.0, label="NEMD reference"),
        Line2D([], [], color=COLORS["einn"], marker="o", linestyle="-", markersize=3.0, linewidth=1.45, label="EINN"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 2 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_2()
