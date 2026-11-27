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
OUTPUT_STEM = FIGURE_DIR / "Figure_8"

sys.path.insert(0, str(SCRIPT_DIR))
from figure_style import (  # noqa: E402
    apply_npj_style,
    panel_label,
    save_figure,
)


def aggregate(data: np.ndarray, x_idx: int, y_idx: int):
    groups: dict[tuple[float, float], list[float]] = {}
    for row in data:
        key = (float(row[x_idx]), float(row[y_idx]))
        groups.setdefault(key, []).append(float(row[3]))
    points = np.array([(x, y, np.mean(zs)) for (x, y), zs in groups.items()])
    return points[:, 0], points[:, 1], points[:, 2]


def plot_figure_8() -> None:
    apply_npj_style()
    csv_path = DATA_DIR / "new_MDdata_406_records.csv"
    df = pd.read_csv(csv_path)
    data = df[["phi", "logg", "T", "log_eta"]].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.55))
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.20, top=0.83, wspace=0.42)

    specs = [
        (2, 1, r"$T^*$", r"$\log_{10}\dot{\gamma}$", r"$T^*$ vs $\log_{10}\dot{\gamma}$", (0.4, 2.5), (-5, 0)),
        (0, 1, r"$\phi$", r"$\log_{10}\dot{\gamma}$", r"$\phi$ vs $\log_{10}\dot{\gamma}$", (0.0, 0.461), (-5, 0)),
        (0, 2, r"$\phi$", r"$T^*$", r"$\phi$ vs $T^*$", (0.0, 0.461), (0.4, 2.5)),
    ]

    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=4.8)
    contour = None
    for label, ax, (xi, yi, xlabel, ylabel, title, xlim, ylim) in zip("abc", axes, specs):
        x, y, z = aggregate(data, xi, yi)
        contour = ax.tricontourf(x, y, z, levels=np.linspace(0.0, 4.8, 97), cmap="turbo", norm=norm)
        ax.scatter(x, y, s=3.5, c=z, cmap="turbo", norm=norm, edgecolors="none", alpha=0.55, zorder=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4.0)
        panel_label(ax, label)

    cax = fig.add_axes([0.945, 0.20, 0.012, 0.63])
    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label(r"Mean $\log_{10}\eta$")

    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Figure 8 successfully generated at: {OUTPUT_STEM}.[pdf/png/tiff]")


if __name__ == "__main__":
    plot_figure_8()
