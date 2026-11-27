"""Rebuild original evidence inputs using local historical licensed resources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_retriever import (
    ChunkKnowledgeBase, build_summary_struct, format_evidence_text, summary_struct_to_text,
)
from llm_transrheo_prior_correction_pipeline import (
    build_summary_cache_key, extract_prior_hint, load_summary_cache,
)

RUNS = {
    "krepc_cached_det_full": None,
    "krepc_cached_ds_flash_full": "deepseek-v4-flash",
    "krepc_cached_ds_full": "deepseek-v4-pro",
}

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--knowledge", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, help="Directory holding original run folders and their local JSONL caches")
    args = p.parse_args()
    kb = ChunkKnowledgeBase(args.knowledge)
    records = []
    for run, backend in RUNS.items():
        if backend and args.cache_root is None:
            continue
        run_dir = ROOT / "runs" / run
        pred = pd.read_csv(run_dir / "krepc_predictions_numeric.csv")
        evidence_index = pd.read_csv(run_dir / "evidence_locators.csv").groupby("row_id")
        bank = json.loads((run_dir / "bias_prior_bank.json").read_text())
        bin_spec = bank["bin_spec"]
        if backend:
            cache_path = args.cache_root / run / "krepc_summary_cache_deepseek.jsonl"
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            cache = load_summary_cache(cache_path)
        else:
            cache = {}
        for row in pred.to_dict("records"):
            id_ = row["row_id"]
            values = {k: row[k] for k in ("phi", "logg", "T", "base_pred")}
            hint = extract_prior_hint(row, bank, bin_spec)
            query, regime, evidence = kb.retrieve(**values, prior_hint=hint, topk=4, per_source_cap=2)
            planned = evidence_index.get_group(id_).sort_values("rank")
            if list(planned.evidence_id) != [entry["id"] for entry in evidence]:
                raise ValueError(f"Retrieval does not match the archived sequence: {run}, row {id_}")
            primary = backend and row["summary_source"] in {"deepseek_cache", "deepseek_live"}
            if primary:
                key = build_summary_cache_key(id_, query, evidence)
                summary = cache[key]["summary_struct"]
            else:
                summary = build_summary_struct(**values, regime=regime, evidence=evidence, prior_hint=hint)
            records.append({
                "run": run, "row_id": id_, "query_sha256": digest(query),
                "evidence_sha256": digest(format_evidence_text(evidence, max_text_chars=600)),
                "summary_sha256": digest(summary_struct_to_text(summary)),
                "archived_provider": backend if primary else None,
            })
    out = pd.DataFrame(records)
    known = ROOT / "registrations/evidence_input_fingerprints.csv"
    if known.is_file():
        ref = pd.read_csv(known).fillna("").set_index(["run", "row_id"]).sort_index()
        current = out.fillna("").set_index(["run", "row_id"]).sort_index()
        pd.testing.assert_frame_equal(ref.loc[current.index], current)
    print(out.to_csv(index=False), end="")

if __name__ == "__main__":
    main()
