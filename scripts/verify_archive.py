"""Recompute archived benchmark results using only files in this repository."""
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def sheet_rows(book, name):
    rows = iter(book[name].values)
    header = next(rows)
    return [dict(zip(header, row)) for row in rows]

def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))

def compare_values(left, right):
    if left.keys() != right.keys():
        raise ValueError(f"Mismatched keys: {left.keys() ^ right.keys()}")
    error = max(abs(float(v) - float(right[k])) for k, v in left.items())
    if error > TOL:
        raise ValueError(f"Maximum difference {error} exceeds {TOL}")
    return {"rows": len(left), "max_abs_difference": error}

def main():
    source_manifest = json_file(ROOT / "docs/artifact_provenance.json")
    for name, record in source_manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Archived input changed: {name}")

    bench = ROOT / "runs/benchmark"
    book = load_workbook(ROOT / "data/Supplementary_Data.xlsx", read_only=True, data_only=True)
    data = sheet_rows(book, "Figure8_NEMD_Simulation_406")
    coords = {(r["phi"], r["T"], r["logg"]) for r in data}
    condition_groups = {(phi, temp) for phi, temp, _ in coords}
    if (len(data), len(coords), len(condition_groups)) != (406, 406, 105):
        raise ValueError("Unexpected dataset")
    columns = ("phi", "logg", "T", "log_eta")
    truth = [tuple(float(row[k]) for k in columns) for row in data]
    csv_data = read_csv(ROOT / "data/new_MDdata_406_records.csv")
    numeric_csv = [tuple(float(row[k]) for k in columns) for row in csv_data]
    source_book = load_workbook(ROOT / "data/new_MDdata.xlsx", read_only=True, data_only=True)
    source_rows = iter(source_book["Sheet1"].values)
    next(source_rows)
    numeric_xlsx = [tuple(float(v) for v in row) for row in source_rows]
    source_book.close()
    if truth != numeric_csv or truth != numeric_xlsx:
        raise ValueError("Training workbook, CSV and submission labels disagree in row order or values")
    table1 = {(r["TrainFracLabel"], r["Model"]): r["MAE"]
              for r in sheet_rows(book, "Table1_OOD_Benchmark_Summary")}
    table4 = {(float(r["train_fraction"]), r["variant"]): r["MAE"]
              for r in sheet_rows(book, "Table4_Ablation_Analysis")}
    figure1 = {(r["Model"], int(r["row_id"])): r["pred"]
               for r in sheet_rows(book, "Figure1_Parity_100pct")}
    book.close()

    stored_errors = defaultdict(list)
    for row in read_csv(bench / "all_model_predictions_all_fractions.csv"):
        if row["Split"] == "OOD":
            stored_errors[(row["TrainFracLabel"], row["Model"])].append(
                abs(float(row["target"]) - float(row["pred"])))
    table1_recomputed = {k: mean(e) for k, e in stored_errors.items() if k in table1}
    table4_archived = {(float(r["train_fraction"]), r["variant"]): r["MAE"]
                       for r in read_csv(ROOT / "runs/ablation/ablation_overall_ood.csv")}
    figure1_archived = {(r["Model"], int(r["row_id"])): r["pred"]
                        for r in read_csv(bench / "100pct/all_model_predictions_y.csv") if r["Split"] == "OOD"}

    source = ROOT / "src/llm_transrheo_prior_correction_pipeline.py"
    prior = ast.parse(source.read_text(encoding="utf-8-sig"))
    prior.body = [node for node in prior.body if isinstance(node, ast.FunctionDef)
                  and node.name in {"assign_level", "build_prior_text"}]
    namespace = {}
    exec(compile(prior, "<archived prior text formatter>", "exec"), namespace)
    fractions = {}
    for fraction in (25, 50, 75, 100):
        directory = bench / f"{fraction}pct"
        bank = json_file(directory / "bias_prior_bank.json")
        rows = read_csv(directory / "prior_conditioned_dataset.csv")
        expected_ids = {int(r["row_id"]) for r in rows if r["Split"] == "OOD"}
        formatted = sum(namespace["build_prior_text"](r, bank, bank["bin_spec"]) == r["prior_text"] for r in rows)
        split = read_csv(directory / "split_indices.csv")
        membership = defaultdict(set)
        for row in split:
            membership[row["group_id"]].add(row["split"])
        counts = Counter(s for _, s in {(r["row_id"], r["split"]) for r in split})
        if formatted != 406 or len(expected_ids) != 41 or any(len(s) > 1 for s in membership.values()):
            raise ValueError(f"Unexpected partition or prior text for {fraction}%")
        fractions[fraction] = {"exact_template_matches": formatted, "rows_by_split": dict(counts)}

    print(json.dumps({
        "archival_integrity_files": len(source_manifest),
        "dataset": {"rows": len(data), "unique_coordinates": len(coords), "condition_groups": len(condition_groups),
                    "row_aligned_sources": ["training workbook", "numeric CSV", "supporting workbook"]},
        "table_1_mae_recomputed_from_ood_predictions": compare_values(table1, table1_recomputed),
        "table_4_mae_matched_to_archived_summary": compare_values(table4, table4_archived),
        "figure_1_predicted_points": compare_values(figure1, figure1_archived),
        "fractions": fractions
    }, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
