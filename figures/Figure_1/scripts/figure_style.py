from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


FULL_WIDTH_IN = 6.30
OUTPUT_DPI = 600

COLORS = {
    "einn": "#D55E00",
    "base": "#0072B2",
    "nemd": "#222222",
    "xgboost": "#009E73",
    "svr": "#E69F00",
    "rf": "#CC79A7",
    "lr": "#687386",
    "cyan": "#56B4E9",
    "grid": "#E1E5EA",
}


def apply_npj_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.labelpad": 2.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.0,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.5,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def finish_axis(axis, *, grid: bool = True) -> None:
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(which="both", width=0.7)
    if grid:
        axis.grid(
            True,
            which="major",
            color=COLORS["grid"],
            linewidth=0.45,
            linestyle="-",
            zorder=0,
        )
    axis.set_axisbelow(True)


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.10,
        1.025,
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )


def save_figure(figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=OUTPUT_DPI)
    figure.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=OUTPUT_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(output_stem.with_suffix(".pdf"))
