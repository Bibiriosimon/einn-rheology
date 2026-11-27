"""Recompute archived KREPC predictions from exact frozen encoder features."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delta_correction_model import KREPCorrectionModel

RUNS = ("krepc_cached_det_full", "krepc_cached_ds_flash_full", "krepc_cached_ds_full")

def load_model(weights, scibert, bins, head_only=False):
    config = AutoConfig.from_pretrained(str(scibert), local_files_only=True)
    with patch.object(AutoModel, "from_pretrained", return_value=AutoModel.from_config(config)):
        model = KREPCorrectionModel(str(scibert), delta_bins=len(bins["centers"])).eval()
    checkpoint = torch.load(weights, weights_only=True, map_location="cpu", mmap=True)
    state = checkpoint["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=not head_only)
    if head_only and (unexpected or set(missing) != {k for k in model.state_dict() if k.startswith("text_encoder.")}):
        raise ValueError("Invalid head-only model state")
    return model

def numeric_inputs(directory, rows):
    with np.load(directory / "llm_correction_scaler.npz", allow_pickle=False) as scaler:
        values = np.column_stack([rows.phi, rows.logg, 1 / rows["T"], rows.base_pred])
        return torch.from_numpy(((values - scaler["mean"]) / scaler["std"]).astype("float32"))

def bin_inputs(directory):
    bins = json.loads((directory / "krepc_delta_bin_spec.json").read_text())
    return bins, torch.tensor(bins["centers"]), torch.tensor(bins["half_widths"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(4)
    for run in RUNS:
        directory = ROOT / "runs" / run
        rows = pd.read_csv(directory / "krepc_predictions_numeric.csv").sort_values("row_id")
        bins, centers, widths = bin_inputs(directory)
        with np.load(directory / "frozen_encoder_features.npz", allow_pickle=False) as f:
            assert np.array_equal(f["row_id"], rows.row_id)
            evidence = torch.from_numpy(f["evidence"])
            summary = torch.from_numpy(f["summary"])
        model = load_model(directory / "head_state.pt", ROOT / "environment/scibert_config", bins,
                           head_only=True)
        with torch.inference_mode(), patch.object(model, "encode_text", side_effect=[evidence, summary]):
            result = model(numeric_inputs(directory, rows), None, None, None, None, centers, widths)
        delta = result["delta_pred"].numpy()
        pred_diff = float(np.max(np.abs(rows.pred - rows.base_pred - delta)))
        gate_diff = float(np.max(np.abs(rows.gate_alpha - result["gate_alpha"].numpy())))
        print(json.dumps({"run": run, "rows": len(rows), "prediction_max_abs_difference": pred_diff,
                          "gate_max_abs_difference": gate_diff}), flush=True)
        if max(pred_diff, gate_diff) > 2e-5:
            raise ValueError("Archived prediction mismatch")
if __name__ == "__main__":
    main()
