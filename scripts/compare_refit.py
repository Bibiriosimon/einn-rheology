"""Report differences between fresh benchmark fits and the original archived benchmark."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("refit", type=Path)
    args = p.parse_args()
    comparisons = {}
    for label in ("25pct", "50pct", "75pct", "100pct"):
        original = ROOT / "runs" / "benchmark" / label
        refit = args.refit / label
        if not (refit / "benchmark_ood_all_models_y.csv").is_file():
            continue
        rows = []
        for path in (original, refit):
            data = pd.read_csv(path / "all_model_predictions_y.csv")
            rows.append(data[data.Split == "OOD"].set_index(["Model", "row_id"]).sort_index())
        past, new = rows
        if not past.index.equals(new.index):
            raise ValueError(f"Model or row identities differ in {label}")
        if (abs(past.target - new.target) > 1e-10).any():
            raise ValueError(f"Held-out targets differ in {label}")
        results = pd.DataFrame({
            "archived_error": abs(past.target - past.pred),
            "fresh_error": abs(new.target - new.pred),
            "prediction_abs_diff": abs(past.pred - new.pred),
        })
        by_model = results.groupby(level="Model")
        comparisons[label] = {model: {
            "held_out_rows": len(group),
            "archived_mae": float(group.archived_error.mean()),
            "fresh_mae": float(group.fresh_error.mean()),
            "max_abs_prediction_difference": float(group.prediction_abs_diff.max()),
        } for model, group in by_model}
    print(json.dumps({"completed_fractions": comparisons}, indent=2))

if __name__ == "__main__":
    main()
