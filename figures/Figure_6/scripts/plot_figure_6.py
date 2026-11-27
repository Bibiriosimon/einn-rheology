from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_STEM = FIGURE_DIR / "Figure_6"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    COLORS,
    FULL_WIDTH_IN,
    apply_npj_style,
    finish_axis,
    panel_label,
    save_figure,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wilson_interval(proportion: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def plot_figure_6() -> None:
    apply_npj_style()
    source = read_csv(DATA_DIR / "figure6_accuracy_curves.csv")
    count_rows = read_csv(DATA_DIR / "benchmark_ood_all_models_all_fractions.csv")
    counts = {(row["TrainFracLabel"], row["Model"]): int(row["N"]) for row in count_rows}
    values = {(row["TrainFracLabel"], row["Model"]): row for row in source}

    fraction_order = ["25pct", "50pct", "75pct", "100pct"]
    fraction_labels = ["25%", "50%", "75%", "100%"]
    model_order = ["LLM-TransRheo", "TransRheo", "XGBoost", "SVR", "RF", "LR"]
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
    markers = dict(zip(model_order, ["o", "s", "^", "D", "P", "X"]))
    x = np.arange(len(fraction_order), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.55), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.70, wspace=0.18)

    for axis, metric in zip(axes, ["Acc10", "Acc20"]):
        for model in model_order:
            points = [values[(fraction, model)] for fraction in fraction_order]
            y = np.asarray([float(row[metric]) for row in points])
            if model == "LLM-TransRheo":
                intervals = np.asarray([
                    wilson_interval(p, counts[(fraction, model)])
                    for fraction, p in zip(fraction_order, y)
                ])
                axis.errorbar(
                    x,
                    y,
                    yerr=np.vstack([y - intervals[:, 0], intervals[:, 1] - y]),
                    color=COLORS["einn"],
                    marker="o",
                    markersize=4.6,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    linewidth=2.25,
                    elinewidth=1.15,
                    capsize=3.0,
                    capthick=1.0,
                    label=display[model],
                    zorder=5,
                )
            else:
                axis.plot(
                    x,
                    y,
                    color=colors[model],
                    marker=markers[model],
                    markersize=3.2,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    linewidth=0.95,
                    alpha=0.72,
                    label=display[model],
                    zorder=3,
                )
        axis.set_xticks(x)
        axis.set_xticklabels(fraction_labels)
        axis.set_xlabel("Training-data fraction")
        axis.set_ylim(-0.02, 1.02)
        finish_axis(axis)

    axes[0].set_ylabel(r"Accuracy ($\Delta \leq 0.10$)")
    axes[1].set_ylabel(r"Accuracy ($\Delta \leq 0.20$)")
    axes[0].set_title(r"Accuracy within $\pm 0.10$", pad=4.0)
    axes[1].set_title(r"Accuracy within $\pm 0.20$", pad=4.0)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    handles = [
        Line2D(
            [],
            [],
            color=colors[model],
            marker=markers[model],
            linestyle="-" if model == "LLM-TransRheo" else "-",
            linewidth=2.0 if model == "LLM-TransRheo" else 1.0,
            markersize=4.0,
            label=display[model],
        )
        for model in model_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=6,
        fontsize=6.2,
        columnspacing=0.9,
        handlelength=1.6,
    )

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 6 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_6()
