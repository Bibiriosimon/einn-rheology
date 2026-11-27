"""Recreate the two archived grouped partition sequences and check every record."""
import json

import pandas as pd

from train_march import ROOT, load_archived

def main():
    archived = load_archived()
    base = archived.read_tabular_data(ROOT / "data/new_MDdata.xlsx", None)
    _, _, tasks, groups, _ = archived.load_arrays(archived.build_multitask_records(base))
    outer = archived.split_by_group(groups, tasks, frac=.1, seed=42)
    verified = {}
    sequences = {
        "benchmark": (.25, .50, .75, 1.),
        "benchmark_sparsefractions_run": (.05, .10, .15),
    }
    for directory, fractions in sequences.items():
        for index, fraction in enumerate(fractions):
            label = f"{round(fraction * 100)}pct"
            _, selected = archived.select_fraction_groups(groups, tasks, outer.train_idx, fraction, 1042 + index)
            inner = archived.split_by_group(groups, tasks, frac=.1, seed=59, base_indices=selected)
            known = pd.read_csv(ROOT / "runs" / directory / label / "split_indices.csv")
            record = {"TRAIN": inner.train_idx, "VAL": inner.test_idx, "OOD": outer.test_idx}
            for name, ids in record.items():
                if set(known[known.split == name].record_idx) != set(ids):
                    raise ValueError(f"{directory}/{label} {name} split mismatch")
            verified[f"{directory}/{label}"] = {name: len(ids) for name, ids in record.items()}
    print(json.dumps(verified, indent=2))

if __name__ == "__main__":
    main()
