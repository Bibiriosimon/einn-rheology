#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


ROOT = Path("benchmark-3.11")
SRC = ROOT / "benchmark_ood_all_models_all_fractions.csv"
OUT_DIR = ROOT / "table1_ood_consistent"
OUT_CSV = OUT_DIR / "table1_ood_consistent.csv"
OUT_TEX = OUT_DIR / "table1_ood_consistent.tex"

FRACTION_ORDER = ["25pct", "50pct", "75pct", "100pct"]
MODEL_ORDER = ["TransRheo", "LLM-TransRheo", "LR", "SVR", "RF", "XGBoost"]


def fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[(df["Split"] == "OOD") & (df["Task"] == "y")].copy()
    df["TrainFracLabel"] = pd.Categorical(df["TrainFracLabel"], categories=FRACTION_ORDER, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["TrainFracLabel", "Model"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    metrics = [
        ("R2", True, "$R^2\\uparrow$"),
        ("RMSE", False, "RMSE$\\downarrow$"),
        ("MAE", False, "MAE$\\downarrow$"),
        ("Acc05", True, "Acc@0.05$\\uparrow$"),
        ("Acc10", True, "Acc@0.10$\\uparrow$"),
        ("Acc20", True, "Acc@0.20$\\uparrow$"),
    ]

    best_map: dict[tuple[str, str], float] = {}
    for frac in FRACTION_ORDER:
        sub = df[df["TrainFracLabel"].astype(str) == frac]
        for metric, higher_is_better, _ in metrics:
            best_val = sub[metric].max() if higher_is_better else sub[metric].min()
            best_map[(frac, metric)] = float(best_val)

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{OOD-only benchmark of $y=\log_{10}\eta$ under different training-data fractions on the latest expanded dataset. "
        r"This table is generated directly from the same fixed outer group-held-out OOD split used in Fig.~2, ensuring strict consistency with the OOD accuracy curves.}"
    )
    lines.append(r"\label{tab:table1_ood_consistent}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{c l ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"TrainFrac & Model & $R^2\uparrow$ & RMSE$\downarrow$ & MAE$\downarrow$ & Acc@0.05$\uparrow$ & Acc@0.10$\uparrow$ & Acc@0.20$\uparrow$ & $N$ \\")
    lines.append(r"\midrule")

    for frac in FRACTION_ORDER:
        sub = df[df["TrainFracLabel"].astype(str) == frac].copy()
        first = True
        for _, row in sub.iterrows():
            frac_cell = frac.replace("pct", r"\%") if first else ""
            first = False
            cells = [frac_cell, str(row["Model"])]
            for metric, _, _ in metrics:
                val = float(row[metric])
                text = fmt(val)
                if abs(val - best_map[(frac, metric)]) < 1e-12:
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
            cells.append(str(int(row["N"])))
            lines.append(" & ".join(cells) + r" \\")
        if frac != FRACTION_ORDER[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {OUT_CSV}")
    print(f"[Saved] {OUT_TEX}")


if __name__ == "__main__":
    main()
