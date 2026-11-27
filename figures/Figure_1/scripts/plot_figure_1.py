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
OUTPUT_STEM = FIGURE_DIR / "Figure_1"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    COLORS,
    FULL_WIDTH_IN,
    apply_npj_style,
    finish_axis,
    panel_label,
    save_figure,
)


def plot_figure_1() -> None:
    apply_npj_style()
    pred_path = DATA_DIR / "figure1_parity_100pct_ood_predictions.csv"
    metric_path = DATA_DIR / "figure1_benchmark_metrics.csv"

    predictions = pd.read_csv(pred_path)
    metrics = pd.read_csv(metric_path).set_index("Model")

    order = ["LLM-TransRheo", "TransRheo", "XGBoost", "SVR", "RF", "LR"]
    display = {
        "LLM-TransRheo": "EINN",
        "TransRheo": "Base",
        "XGBoost": "XGBoost",
        "SVR": "SVR",
        "RF": "Random forest",
        "LR": "Linear regression",
    }
    colors = {
        "LLM-TransRheo": COLORS["einn"],
        "TransRheo": COLORS["base"],
        "XGBoost": COLORS["xgboost"],
        "SVR": COLORS["svr"],
        "RF": COLORS["rf"],
        "LR": COLORS["lr"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH_IN, 4.0))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.105, top=0.95, wspace=0.34, hspace=0.42)
    limits = (-0.12, 4.66)

    for label, ax, model in zip("abcdef", axes.flat, order):
        sub = predictions[predictions["Model"] == model]
        target = sub["target"].to_numpy(dtype=float)
        pred = sub["pred"].to_numpy(dtype=float)

        ax.scatter(
            target,
            pred,
            s=12,
            color=colors[model],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.9,
            zorder=3,
        )
        ax.plot(limits, limits, color="#555555", linewidth=0.85, linestyle="--", zorder=2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(display[model], pad=2.0, fontweight="bold")
        ax.set_xlabel(r"NEMD $\log_{10}\eta$")
        ax.set_ylabel(r"Predicted $\log_{10}\eta$")

        metric = metrics.loc[model]
        annotation = (
            f"$R^2$ = {metric['R2']:.3f}\n"
            f"RMSE = {metric['RMSE']:.3f}\n"
            f"MAE = {metric['MAE']:.3f}\n"
            f"Acc@0.10 = {metric['Acc10']:.3f}"
        )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            linespacing=1.05,
            bbox={"facecolor": "white", "edgecolor": "#B8BEC6", "linewidth": 0.45, "pad": 1.6},
            zorder=5,
        )
        panel_label(ax, label)
        finish_axis(ax)

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 1 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_1()
