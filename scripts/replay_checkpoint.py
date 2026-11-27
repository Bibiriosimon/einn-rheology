"""Recompute OOD predictions from the archived Table 1 template-model checkpoints."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from TransRheo_highfit_multitask import TransRheoHighFit, predict_model
from llm_transrheo_prior_correction_pipeline import PriorCorrectionModel

def compare(directory, weights, tokenizer_path, tokenizer, max_length):
    rows = pd.read_csv(directory / "llm_transrheo_predictions_all_splits.csv")
    rows = rows[rows["Split"] == "OOD"].sort_values("row_id")
    if len(rows) != 41 or not rows.row_id.is_unique:
        raise ValueError("Expected 41 distinct OOD states")

    device = torch.device("cpu")
    source = torch.load(weights / "transrheo_base_ckpt.pt", map_location=device, weights_only=True)
    model = TransRheoHighFit(d_model=256, n_heads=8, n_attn_layers=3, n_mlp_layers=4,
                             mlp_hidden_mult=2.0, dropout=0.05)
    model.load_state_dict(source["model_state"], strict=True)
    x = np.column_stack((rows.phi, rows.logg, 1 / rows["T"]))
    x_base = (x - np.array(source["scaler_mean"])) / np.array(source["scaler_scale"])
    base = predict_model(model, x_base, np.zeros(len(rows), dtype=int), 16, device)
    del model, source

    checkpoint = torch.load(weights / "llm_transrheo_ckpt.pt", map_location=device,
                            weights_only=True, mmap=True)
    config = AutoConfig.from_pretrained(str(tokenizer_path), local_files_only=True)
    with patch.object(AutoModel, "from_pretrained", return_value=AutoModel.from_config(config)):
        corrector = PriorCorrectionModel(str(tokenizer_path)).eval()
    corrector.load_state_dict(checkpoint["model_state"], strict=True)
    with np.load(directory / "llm_correction_scaler.npz", allow_pickle=False) as scaler:
        correction_x = (np.column_stack((x, base)) - scaler["mean"]) / scaler["std"]
    deltas = []
    with torch.inference_mode():
        for i in range(0, len(rows), 8):
            text = rows.iloc[i:i + 8].prior_text.tolist()
            tokens = tokenizer(text, padding="max_length", truncation=True,
                               max_length=max_length, return_tensors="pt")
            delta = corrector(torch.tensor(correction_x[i:i + 8], dtype=torch.float32),
                              tokens["input_ids"], tokens["attention_mask"])
            deltas.extend(delta.numpy().tolist())
    adjusted = base + np.array(deltas)
    report = {
        "fraction": directory.name,
        "count": len(base),
        "base_max_abs_difference": float(np.max(np.abs(base - rows.base_pred))),
        "template_combined_max_abs_difference": float(np.max(np.abs(adjusted - rows.pred))),
        "recomputed_template_mae": float(np.mean(np.abs(adjusted - rows.y_true))),
        "historical_template_mae": float(np.mean(np.abs(rows.pred - rows.y_true))),
        "token_max_length": max_length,
    }
    report["within_tolerance_2e_5"] = report["template_combined_max_abs_difference"] <= 2e-5
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True,
                        help="Directory containing 25pct/50pct/75pct/100pct checkpoint subdirectories")
    parser.add_argument("--scibert", type=Path, required=True,
                        help="Local SciBERT configuration and vocabulary")
    parser.add_argument("--fractions", nargs="+", default=["25pct", "50pct", "75pct", "100pct"],
                        choices=["25pct", "50pct", "75pct", "100pct"])
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(args.scibert.resolve()), local_files_only=True)
    passed = True
    for fraction in args.fractions:
        record = compare(ROOT / "runs/benchmark" / fraction, args.weights_root / fraction,
                         args.scibert.resolve(), tokenizer, args.max_length)
        print(json.dumps(record), flush=True)
        passed &= record["within_tolerance_2e_5"]
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
