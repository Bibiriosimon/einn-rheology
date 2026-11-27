#!/usr/bin/env python3
"""
ablation_epc_evidence_control.py

Ablation experiments for TransRheo-EINN: isolating the contribution of
LLM-derived experiential summaries and the EPC gated correction design.

Variants:
  base              — TransRheo numerical backbone only (no EPC)
  full_einn         — Complete TransRheo-EINN (full EPC + language experience)
  shuffled_summary  — EPC with randomly shuffled summary embeddings (within split)
  random_summary    — EPC with random vectors replacing summary embeddings
  residual_only     — EPC using only residual-pattern prior (no literature evidence)
  no_gate           — EPC with gated fusion replaced by fixed averaging

Outputs:
  results/ablation_overall_ood.csv
  results/ablation_extreme_regimes.csv
  results/ablation_table_main.tex
  figures/ablation_evidence_control.pdf
  figures/ablation_extreme_regime_gain.pdf
  results/ablation_experiment_notes.md
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from TransRheo_highfit_multitask import (
    TransRheoHighFit,
    build_multitask_records,
    evaluate_indices,
    fit_scaler_on_train,
    load_arrays,
    make_lr_lambda,
    read_tabular_data,
    run_training_phase,
    set_seed,
    split_by_group,
    calculate_metrics as base_calc_metrics,
    physics_check_numpy,
)
from llm_transrheo_prior_correction_pipeline import (
    assign_level,
    build_bias_prior_bank,
    build_bin_spec,
    build_delta_bin_spec,
    assign_delta_bins,
    build_group_batches,
    build_summary_cache_key,
    calc_metrics,
    extract_prior_hint,
    format_evidence_text,
    load_summary_cache,
    normalize_summary_struct,
    physics_v_shear,
    summary_struct_to_text,
    tokenize_texts,
    unique_thresholds,
    call_deepseek_summary_with_retry,
    append_summary_cache,
)
from knowledge_retriever import (
    ChunkKnowledgeBase,
    build_summary_struct,
    infer_regime,
)
from delta_correction_model import KREPCorrectionModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASK_NAMES = ["y", "logg", "T", "phi"]
VALID_VARIANTS = [
    "base", "full_einn", "shuffled_summary",
    "random_summary", "residual_only", "no_gate",
]
DEFAULT_FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def get_accuracy(y_true, y_pred, thr):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) <= thr))


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": safe_r2(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Acc05": get_accuracy(y_true, y_pred, 0.05),
        "Acc10": get_accuracy(y_true, y_pred, 0.10),
        "Acc20": get_accuracy(y_true, y_pred, 0.20),
        "N": int(len(y_true)),
    }


def physics_v_shear_from_df(df: pd.DataFrame, tau: float = 1e-6) -> float:
    """Wrapper accepting 'target'/'pred' columns."""
    if len(df) == 0:
        return 0.0
    work = df.copy()
    work["phi_g"] = work["phi_raw"].astype(float).round(4)
    work["T_g"] = work["T_raw"].astype(float).round(4)
    viol, total = 0, 0
    for _, grp in work.groupby(["phi_g", "T_g"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("logg")
        yy = grp["pred"].to_numpy(dtype=float)
        ll = grp["logg"].to_numpy(dtype=float)
        for i in range(len(yy)):
            for j in range(i + 1, len(yy)):
                if ll[j] > ll[i]:
                    total += 1
                    if yy[j] > yy[i] + tau:
                        viol += 1
    return float(viol / max(1, total))


def physics_v_temperature(df: pd.DataFrame, tau: float = 1e-6) -> float:
    """Check viscosity increases as temperature decreases (physical consistency)."""
    if len(df) == 0:
        return 0.0
    work = df.copy()
    work["phi_g"] = work["phi_raw"].astype(float).round(4)
    work["logg_g"] = work["logg"].astype(float).round(4)
    viol, total = 0, 0
    for _, grp in work.groupby(["phi_g", "logg_g"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("T_raw", ascending=False)
        yy = grp["pred"].to_numpy(dtype=float)
        tt = grp["T_raw"].to_numpy(dtype=float)
        for i in range(len(yy)):
            for j in range(i + 1, len(yy)):
                if tt[j] < tt[i]:
                    total += 1
                    if yy[j] < yy[i] - tau:
                        viol += 1
    return float(viol / max(1, total))


def physics_v_phi(df: pd.DataFrame, tau: float = 1e-6) -> float:
    """Check viscosity increases as phi increases (physical consistency)."""
    if len(df) == 0:
        return 0.0
    work = df.copy()
    work["logg_g"] = work["logg"].astype(float).round(4)
    work["T_g"] = work["T_raw"].astype(float).round(4)
    viol, total = 0, 0
    for _, grp in work.groupby(["logg_g", "T_g"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("phi_raw")
        yy = grp["pred"].to_numpy(dtype=float)
        pp = grp["phi_raw"].to_numpy(dtype=float)
        for i in range(len(yy)):
            for j in range(i + 1, len(yy)):
                if pp[j] > pp[i]:
                    total += 1
                    if yy[j] < yy[i] - tau:
                        viol += 1
    return float(viol / max(1, total))


# ---------------------------------------------------------------------------
# Fraction selection (same logic as benchmark_llm_transrheo_datafractions)
# ---------------------------------------------------------------------------
def select_fraction_groups(groups, task_ids, base_indices, frac, seed):
    base_indices = np.asarray(base_indices, dtype=int)
    y_idx = base_indices[task_ids[base_indices] == 0]
    if len(y_idx) == 0:
        raise ValueError("No y-task records in base indices.")
    y_groups = groups[y_idx]
    uniq_groups, counts = np.unique(y_groups, return_counts=True)
    total = int(counts.sum())
    target = total if frac >= 0.999999 else max(1, int(math.ceil(total * frac)))
    rng = np.random.default_rng(seed)
    order = np.arange(len(uniq_groups))
    rng.shuffle(order)
    chosen = []
    running = 0
    for idx in order:
        chosen.append(int(uniq_groups[idx]))
        running += int(counts[idx])
        if running >= target:
            break
    chosen = np.asarray(sorted(set(chosen)), dtype=int)
    sel = base_indices[np.isin(groups[base_indices], chosen)]
    return chosen, sel


# ---------------------------------------------------------------------------
# TransRheo backbone training for a given fraction
# ---------------------------------------------------------------------------
def train_base_transrheo_fraction(args, out_dir, base_df, records, x_feat, y, task_ids, groups, row_ids, outer_ood_idx, selected_id_idx):
    inner = split_by_group(groups, task_ids, frac=args.val_frac, seed=args.seed + 17, base_indices=selected_id_idx)
    train_idx, val_idx, ood_idx = inner.train_idx, inner.test_idx, outer_ood_idx

    scaler = fit_scaler_on_train(x_feat, task_ids, train_idx)
    x_scaled = scaler.transform(x_feat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransRheoHighFit(
        d_model=args.base_d_model,
        n_heads=args.base_n_heads,
        n_attn_layers=args.base_n_attn_layers,
        n_mlp_layers=args.base_n_mlp_layers,
        mlp_hidden_mult=args.base_mlp_hidden_mult,
        dropout=args.base_dropout,
    ).to(device)

    weights = torch.tensor([args.base_weight_y, args.base_weight_logg, args.base_weight_T, args.base_weight_phi], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.base_weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_idx) / max(1, args.base_batch_size)))
    total_steps = max(1, steps_per_epoch * sum(args.base_epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(total_steps=total_steps, warmup_frac=args.base_warmup_frac, min_scale=args.base_min_lr_scale),
    )

    def pin_const(lam):
        return lambda global_step, epoch: lam

    best = {"val_y_mae": float("inf"), "epoch": 0, "stage": "", "state_dict": None}
    log_rows = []
    for phase_name, n_epochs, pin_fn in [
        ("S1", args.base_epochs[0], pin_const(0.0)),
        ("S2", args.base_epochs[1], pin_const(args.base_lambda_pin_s2)),
        ("S3", args.base_epochs[2], pin_const(args.base_lambda_pin_s3)),
    ]:
        best = run_training_phase(
            phase_name=phase_name,
            model=model, optimizer=optimizer, scheduler=scheduler,
            train_idx=train_idx, val_idx=val_idx,
            x_scaled=x_scaled, y=y, task_ids=task_ids, records=records,
            device=device, weights=weights,
            batch_size=args.base_batch_size, infer_batch_size=args.base_eval_batch_size,
            seed=args.seed,
            n_epochs=n_epochs, grad_clip=args.base_grad_clip,
            aug_noise_frac=args.base_aug_noise_frac,
            pin_tau=args.base_pin_tau, pin_lambda_fn=pin_fn,
            best=best, log_rows=log_rows, patience=args.base_patience,
        )
        if best.get("state_dict") is not None:
            model.load_state_dict(best["state_dict"])

    if best.get("state_dict") is not None:
        model.load_state_dict(best["state_dict"])

    if args.base_tail_epochs > 0:
        tail_weights = torch.tensor(
            [args.base_tail_weight_y, args.base_tail_weight_logg, args.base_tail_weight_T, args.base_tail_weight_phi],
            dtype=torch.float32, device=device,
        )
        tail_optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_tail_lr, weight_decay=args.base_weight_decay)
        tail_steps = max(1, steps_per_epoch * args.base_tail_epochs)
        tail_scheduler = torch.optim.lr_scheduler.LambdaLR(
            tail_optimizer,
            lr_lambda=make_lr_lambda(total_steps=tail_steps, warmup_frac=args.base_tail_warmup_frac, min_scale=args.base_tail_min_lr_scale),
        )
        if args.base_tail_ramp_frac <= 0 or args.base_tail_pin_lambda <= 0:
            tail_pin_lambda_fn = lambda step, epoch: 0.0 if args.base_tail_pin_lambda <= 0 else float(args.base_tail_pin_lambda)
        else:
            tail_ramp_steps = max(1, int(args.base_tail_ramp_frac * tail_steps))
            def tail_pin_lambda_fn(step, epoch, _target=args.base_tail_pin_lambda, _ramp_steps=tail_ramp_steps):
                progress = min(step, _ramp_steps)
                return float(_target) * (float(progress) / float(_ramp_steps))

        best = run_training_phase(
            phase_name="TAIL",
            model=model, optimizer=tail_optimizer, scheduler=tail_scheduler,
            train_idx=train_idx, val_idx=val_idx,
            x_scaled=x_scaled, y=y, task_ids=task_ids, records=records,
            device=device, weights=tail_weights,
            batch_size=args.base_tail_batch_size if args.base_tail_batch_size > 0 else args.base_batch_size,
            infer_batch_size=args.base_eval_batch_size,
            seed=args.seed + 9000,
            n_epochs=args.base_tail_epochs, grad_clip=args.base_grad_clip,
            aug_noise_frac=args.base_tail_aug_noise_frac,
            pin_tau=args.base_pin_tau, pin_lambda_fn=tail_pin_lambda_fn,
            best=best, log_rows=log_rows, patience=args.base_tail_patience,
        )
        if best.get("state_dict") is not None:
            model.load_state_dict(best["state_dict"])

    split_df = pd.DataFrame({"record_idx": np.arange(len(records)), "row_id": row_ids, "group_id": groups, "split": "ID_HOLDOUT"})
    split_df.loc[train_idx, "split"] = "TRAIN"
    split_df.loc[val_idx, "split"] = "VAL"
    split_df.loc[ood_idx, "split"] = "OOD"
    split_df.to_csv(out_dir / "split_indices.csv", index=False)

    frames = []
    pred_frames = []
    y_frames = []
    for split_name, idx in [("TRAIN", train_idx), ("VAL", val_idx), ("OOD", ood_idx)]:
        metrics_df, pred_df, _ = evaluate_indices(
            model=model, x_scaled=x_scaled, y=y, task_ids=task_ids,
            records=records, indices=idx,
            batch_size=args.base_eval_batch_size, device=device,
        )
        metrics_df["Split"] = split_name
        pred_df["Split"] = split_name
        frames.append(metrics_df)
        pred_frames.append(pred_df)
        y_frames.append(pred_df[pred_df["target_name"] == "y"].copy())

    summary_df = pd.concat(frames, ignore_index=True)
    pred_all_df = pd.concat(pred_frames, ignore_index=True)
    y_pred_df = pd.concat(y_frames, ignore_index=True)
    pd.DataFrame(log_rows).to_csv(out_dir / "transrheo_training_log.csv", index=False)
    summary_df.to_csv(out_dir / "transrheo_summary_metrics.csv", index=False)
    pred_all_df.to_csv(out_dir / "transrheo_all_predictions_multitask.csv", index=False)
    y_pred_df.to_csv(out_dir / "transrheo_y_predictions_all_splits.csv", index=False)

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "split": {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "ood_idx": ood_idx.tolist()},
            "best": {k: v for k, v in best.items() if k != "state_dict"},
        },
        out_dir / "transrheo_base_ckpt.pt",
    )
    return y_pred_df, split_df


# ---------------------------------------------------------------------------
# EPC correction training (variant-aware)
# ---------------------------------------------------------------------------
def train_llm_correction_variant(
    args, base_df, base_y_df, bank, bin_spec, out_dir, variant="full_einn",
):
    """Train EPC correction head with variant-specific modifications."""
    dataset = base_df.copy().reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=int)
    if "group_id" not in dataset.columns:
        dataset["group_id"] = dataset[["phi", "T"]].astype(str).agg("|".join, axis=1).astype("category").cat.codes
    base_part = base_y_df[["row_id", "Split", "target", "pred"]].rename(columns={"target": "y_true", "pred": "base_pred"})
    dataset = dataset.merge(base_part, on="row_id", how="inner")
    dataset["group_id"] = dataset["group_id"].astype(int)
    dataset["delta_true"] = dataset["y_true"] - dataset["base_pred"]

    # -- residual_only: skip knowledge base retrieval, use only prior_hint --
    use_kb = variant != "residual_only"
    if use_kb:
        kb = ChunkKnowledgeBase(args.knowledge_dir, max_chunks=args.knowledge_max_chunks)
    else:
        kb = None

    summary_cache_path = out_dir / "krepc_summary_cache_deterministic.jsonl"
    summary_cache = load_summary_cache(summary_cache_path)

    evidence_texts = []
    summary_texts = []
    query_texts = []
    summary_structs = []
    summary_sources = []
    prior_directions = []
    prior_risks = []
    evidence_ids = []

    prep_iter = tqdm(dataset.iterrows(), total=len(dataset), desc="[KREPC] knowledge prep", ncols=120, leave=False)
    for _, row in prep_iter:
        prior_hint = extract_prior_hint(row, bank, bin_spec)

        if use_kb:
            query_text, regime, evidence = kb.retrieve(
                phi=float(row["phi"]), logg=float(row["logg"]), T=float(row["T"]),
                base_pred=float(row["base_pred"]), prior_hint=prior_hint,
                topk=args.prior_retrieval_topk, per_source_cap=args.prior_per_source_cap,
            )
        else:
            # residual_only: no knowledge retrieval, empty evidence
            regime = infer_regime(float(row["phi"]), float(row["logg"]), float(row["T"]))
            query_text = f"phi {row['phi']:.4f} logg {row['logg']:.4f} T {row['T']:.4f} base {row['base_pred']:.6f} ; residual prior only"
            evidence = []

        summary_struct = build_summary_struct(
            phi=float(row["phi"]), logg=float(row["logg"]), T=float(row["T"]),
            base_pred=float(row["base_pred"]), regime=regime,
            evidence=evidence, prior_hint=prior_hint,
        )
        summary_text = summary_struct_to_text(summary_struct)
        if use_kb:
            evidence_text = format_evidence_text(evidence, max_text_chars=args.prior_evidence_text_chars)
        else:
            # residual_only: build evidence text from prior hint only
            evidence_text = (
                f"Residual prior only. Direction: {prior_hint.get('direction', 'uncertain')}. "
                f"Risk: {prior_hint.get('risk', 'low')}. "
                f"Regime: phi={row['phi']:.3f} T={row['T']:.2f} logg={row['logg']:.4f}."
            )

        evidence_texts.append(evidence_text)
        summary_texts.append(summary_text)
        query_texts.append(query_text)
        summary_structs.append(summary_struct)
        summary_sources.append("deterministic")
        prior_directions.append(summary_struct.get("likely_bias_direction", "uncertain"))
        prior_risks.append(summary_struct.get("correction_strength", "medium"))
        evidence_ids.append(json.dumps(summary_struct.get("evidence_ids", []), ensure_ascii=False))

    dataset["query_text"] = query_texts
    dataset["evidence_text"] = evidence_texts
    dataset["summary_text"] = summary_texts
    dataset["summary_json"] = [json.dumps(s, ensure_ascii=False) for s in summary_structs]
    dataset["summary_source"] = summary_sources
    dataset["prior_direction"] = prior_directions
    dataset["prior_risk"] = prior_risks
    dataset["evidence_ids"] = evidence_ids
    dataset.to_csv(out_dir / "knowledge_conditioned_dataset.csv", index=False)

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.scibert_dir))
    evidence_ids_tok, evidence_mask = tokenize_texts(
        tokenizer, dataset["evidence_text"].tolist(), args.prior_evidence_max_length,
    )
    summary_ids_tok, summary_mask = tokenize_texts(
        tokenizer, dataset["summary_text"].tolist(), args.prior_summary_max_length,
    )

    x_num = np.stack([
        dataset["phi"].to_numpy(dtype=float),
        dataset["logg"].to_numpy(dtype=float),
        (1.0 / np.clip(dataset["T"].to_numpy(dtype=float), 1e-8, None)),
        dataset["base_pred"].to_numpy(dtype=float),
    ], axis=1)
    train_mask = dataset["Split"].astype(str).eq("TRAIN").to_numpy()
    mean = x_num[train_mask].mean(axis=0)
    std = x_num[train_mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_num = (x_num - mean) / std
    np.savez(out_dir / "llm_correction_scaler.npz", mean=mean, std=std)

    delta_spec = build_delta_bin_spec(
        dataset.loc[train_mask, "delta_true"].to_numpy(dtype=float),
        args.prior_delta_bins,
    )
    dataset["delta_bin_true"] = assign_delta_bins(
        dataset["delta_true"].to_numpy(dtype=float), delta_spec,
    )
    (out_dir / "krepc_delta_bin_spec.json").write_text(json.dumps(delta_spec, indent=2), encoding="utf-8")

    # -- shuffled_summary: permute summary tokens within each split --
    if variant == "shuffled_summary":
        rng = np.random.default_rng(args.seed + 9999)
        for split_name in ["TRAIN", "VAL", "OOD"]:
            mask = (dataset["Split"].astype(str) == split_name).to_numpy()
            idxs = np.where(mask)[0]
            perm = rng.permutation(len(idxs))
            summary_ids_tok[idxs] = summary_ids_tok[idxs[perm]]
            summary_mask[idxs] = summary_mask[idxs[perm]]

    # -- random_summary: compute mean/std of real summary embeddings --
    random_summary_embeds = None
    if variant == "random_summary":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmp_model = KREPCorrectionModel(
            str(args.scibert_dir), hidden_dim=args.prior_hidden_dim,
            dropout=args.prior_dropout, text_pool=args.prior_text_pool,
            delta_bins=args.prior_delta_bins,
        ).to(device)
        tmp_model.eval()
        train_idx = np.where(train_mask)[0]
        real_embeds = []
        with torch.no_grad():
            for s in range(0, len(train_idx), 32):
                sub = train_idx[s:s + 32]
                sb = torch.tensor(summary_ids_tok[sub], dtype=torch.long, device=device)
                sm = torch.tensor(summary_mask[sub], dtype=torch.long, device=device)
                emb = tmp_model.encode_text(sb, sm).detach().cpu().numpy()
                real_embeds.append(emb)
        real_embeds = np.concatenate(real_embeds, axis=0) if real_embeds else np.array([[]])
        emb_mean = real_embeds.mean(axis=0)
        emb_std = real_embeds.std(axis=0)
        n_total = len(dataset)
        random_vecs = np.random.default_rng(args.seed + 8888).normal(
            loc=emb_mean, scale=emb_std, size=(n_total, emb_mean.shape[0]),
        ).astype(np.float32)
        random_summary_embeds = torch.tensor(random_vecs, dtype=torch.float32, device=device)
        del tmp_model

    # -- Model --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KREPCorrectionModel(
        str(args.scibert_dir),
        hidden_dim=args.prior_hidden_dim,
        dropout=args.prior_dropout,
        text_pool=args.prior_text_pool,
        delta_bins=args.prior_delta_bins,
    )
    model.freeze_text(
        freeze_n_layers=args.prior_freeze_n_text_layers,
        freeze_embeddings=args.prior_freeze_text_embeddings,
    )
    model.to(device)

    bin_centers_t = torch.tensor(delta_spec["centers"], dtype=torch.float32, device=device)
    bin_half_widths_t = torch.tensor(delta_spec["half_widths"], dtype=torch.float32, device=device)

    text_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (text_params if name.startswith("text_encoder.") else other_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": text_params, "lr": args.prior_lr_text, "weight_decay": args.prior_weight_decay},
        {"params": other_params, "lr": args.prior_lr_other, "weight_decay": args.prior_weight_decay},
    ])

    train_df = dataset[dataset["Split"] == "TRAIN"].copy()
    val_df = dataset[dataset["Split"] == "VAL"].copy()
    best = {"val_y_mae": float("inf"), "epoch": 0, "state": None}
    patience_left = args.prior_patience
    logs = []
    for epoch in range(1, args.prior_epochs + 1):
        model.train()
        batches = build_group_batches(train_df, args.prior_batch_size, args.seed + epoch * 31)
        pbar = tqdm(batches, desc=f"[KREPC] epoch {epoch}", ncols=120, leave=False)
        loss_sum, seen = 0.0, 0
        for batch in pbar:
            batch = np.asarray(batch, dtype=int)
            xb = torch.tensor(x_num[batch], dtype=torch.float32, device=device)
            eb = torch.tensor(evidence_ids_tok[batch], dtype=torch.long, device=device)
            em = torch.tensor(evidence_mask[batch], dtype=torch.long, device=device)
            sb = torch.tensor(summary_ids_tok[batch], dtype=torch.long, device=device)
            sm = torch.tensor(summary_mask[batch], dtype=torch.long, device=device)
            delta_true = torch.tensor(
                dataset.iloc[batch]["delta_true"].to_numpy(dtype=float),
                dtype=torch.float32, device=device,
            )
            delta_bin_true = torch.tensor(
                dataset.iloc[batch]["delta_bin_true"].to_numpy(dtype=int),
                dtype=torch.long, device=device,
            )
            base_pred = torch.tensor(
                dataset.iloc[batch]["base_pred"].to_numpy(dtype=float),
                dtype=torch.float32, device=device,
            )
            phi_b = torch.tensor(dataset.iloc[batch]["phi"].to_numpy(dtype=float), dtype=torch.float32, device=device)
            logg_b = torch.tensor(dataset.iloc[batch]["logg"].to_numpy(dtype=float), dtype=torch.float32, device=device)
            t_b = torch.tensor(dataset.iloc[batch]["T"].to_numpy(dtype=float), dtype=torch.float32, device=device)

            batch_rse = None
            if random_summary_embeds is not None:
                batch_rse = random_summary_embeds[batch]

            out = model(
                xb, eb, em, sb, sm, bin_centers_t, bin_half_widths_t,
                no_gate=(variant == "no_gate"),
                random_summary_embeds=batch_rse,
            )
            delta_pred = out["delta_pred"]
            y_final = base_pred + delta_pred
            y_true = base_pred + delta_true
            loss_main = F.l1_loss(y_final, y_true)
            loss_bin = F.cross_entropy(out["delta_logits"], delta_bin_true)
            true_bin_centers = bin_centers_t[delta_bin_true]
            refine_target = delta_true - true_bin_centers
            loss_ref = F.smooth_l1_loss(out["delta_refine"], refine_target)
            loss_reg = torch.mean(torch.abs(out["delta_refine"]))
            from TransRheo_highfit_multitask import pin_shear_loss
            pin = pin_shear_loss(y_final, torch.zeros_like(y_final, dtype=torch.long), phi_b, logg_b, t_b, tau=args.prior_pin_tau)
            loss = (
                loss_main
                + args.prior_lambda_bin * loss_bin
                + args.prior_lambda_ref * loss_ref
                + args.prior_lambda_delta * loss_reg
                + args.prior_lambda_pin * pin
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.prior_grad_clip)
            optimizer.step()
            seen += len(batch)
            loss_sum += float(loss.item()) * len(batch)
            pbar.set_postfix(L=f"{loss_sum / max(1, seen):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            idx = val_df.index.to_numpy(dtype=int)
            xb = torch.tensor(x_num[idx], dtype=torch.float32, device=device)
            eb = torch.tensor(evidence_ids_tok[idx], dtype=torch.long, device=device)
            em = torch.tensor(evidence_mask[idx], dtype=torch.long, device=device)
            sb = torch.tensor(summary_ids_tok[idx], dtype=torch.long, device=device)
            sm = torch.tensor(summary_mask[idx], dtype=torch.long, device=device)
            val_rse = random_summary_embeds[idx] if random_summary_embeds is not None else None
            out_val = model(
                xb, eb, em, sb, sm, bin_centers_t, bin_half_widths_t,
                no_gate=(variant == "no_gate"),
                random_summary_embeds=val_rse,
            )
            delta_val = out_val["delta_pred"].detach().cpu().numpy()
            gate_mean = float(out_val["gate_alpha"].mean().item())
            refine_abs = float(out_val["delta_refine"].abs().mean().item())
            top_conf = float(out_val["delta_probs"].max(dim=1).values.mean().item())
            y_val = val_df["base_pred"].to_numpy(dtype=float) + delta_val
            val_mae = float(mean_absolute_error(val_df["y_true"].to_numpy(dtype=float), y_val))
            logs.append({
                "epoch": epoch, "train_loss": loss_sum / max(1, seen),
                "val_y_mae": val_mae, "val_gate_mean": gate_mean,
                "val_refine_abs_mean": refine_abs, "val_top_bin_conf_mean": top_conf,
            })
            if val_mae < best["val_y_mae"] - 1e-6:
                best = {
                    "val_y_mae": val_mae, "epoch": epoch,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                patience_left = args.prior_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[EarlyStop] KREPC stop at epoch {epoch} | best val_y_mae = {best['val_y_mae']:.6f}")
                    break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save({"model_state": model.state_dict(), "best": best}, out_dir / "krepc_ckpt.pt")
    pd.DataFrame(logs).to_csv(out_dir / "krepc_training_log.csv", index=False)

    # Evaluate
    frames = []
    for split_name in ["TRAIN", "VAL", "OOD"]:
        part = dataset[dataset["Split"] == split_name].copy()
        idx = part.index.to_numpy(dtype=int)
        pred_chunks = []
        gate_chunks = []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(idx), args.prior_eval_batch_size):
                sub = idx[s:s + args.prior_eval_batch_size]
                xb = torch.tensor(x_num[sub], dtype=torch.float32, device=device)
                eb = torch.tensor(evidence_ids_tok[sub], dtype=torch.long, device=device)
                em = torch.tensor(evidence_mask[sub], dtype=torch.long, device=device)
                sb = torch.tensor(summary_ids_tok[sub], dtype=torch.long, device=device)
                sm = torch.tensor(summary_mask[sub], dtype=torch.long, device=device)
                sub_rse = random_summary_embeds[sub] if random_summary_embeds is not None else None
                out = model(
                    xb, eb, em, sb, sm, bin_centers_t, bin_half_widths_t,
                    no_gate=(variant == "no_gate"),
                    random_summary_embeds=sub_rse,
                )
                pred_chunks.append(out["delta_pred"].detach().cpu().numpy())
                gate_chunks.append(out["gate_alpha"].detach().cpu().numpy())
        delta = np.concatenate(pred_chunks) if pred_chunks else np.array([], dtype=float)
        gate_alpha = np.concatenate(gate_chunks) if gate_chunks else np.array([], dtype=float)
        part["pred"] = part["base_pred"].to_numpy(dtype=float) + delta
        part["delta_pred"] = delta
        part["gate_alpha"] = gate_alpha
        part["Model"] = "KREPC-TransRheo"
        part["Split"] = split_name
        frames.append(part)
    pred_df = pd.concat(frames, ignore_index=True)
    pred_df.to_csv(out_dir / "krepc_predictions_all_splits.csv", index=False)
    return pred_df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_variant(base_y_df, epc_pred_df, variant_name, train_frac):
    """Compute OOD metrics for a given variant run."""
    # Base model OOD
    base_ood = base_y_df[base_y_df["Split"] == "OOD"].copy()
    base_metrics = calculate_metrics(base_ood["target"], base_ood["pred"])
    base_phys_shear = physics_v_shear_from_df(base_ood[["phi_raw", "T_raw", "logg", "pred"]])
    base_phys_temp = physics_v_temperature(base_ood[["phi_raw", "T_raw", "logg", "pred"]])
    base_phys_phi = physics_v_phi(base_ood[["phi_raw", "T_raw", "logg", "pred"]])

    if epc_pred_df is not None:
        epc_ood = epc_pred_df[epc_pred_df["Split"] == "OOD"].copy()
        # EPC pred_df has 'phi'/'T' (not 'phi_raw'/'T_raw') and 'y_true' (not 'target')
        epc_ood["phi_raw"] = epc_ood["phi"]
        epc_ood["T_raw"] = epc_ood["T"]
        epc_metrics = calculate_metrics(epc_ood["y_true"], epc_ood["pred"])
        epc_phys_shear = physics_v_shear_from_df(epc_ood[["phi_raw", "T_raw", "logg", "pred"]])
        epc_phys_temp = physics_v_temperature(epc_ood[["phi_raw", "T_raw", "logg", "pred"]])
        epc_phys_phi = physics_v_phi(epc_ood[["phi_raw", "T_raw", "logg", "pred"]])
    else:
        epc_metrics = base_metrics
        epc_phys_shear = base_phys_shear
        epc_phys_temp = base_phys_temp
        epc_phys_phi = base_phys_phi

    return {
        "train_fraction": train_frac,
        "variant": variant_name,
        "R2": epc_metrics["R2"],
        "RMSE": epc_metrics["RMSE"],
        "MAE": epc_metrics["MAE"],
        "Acc05": epc_metrics["Acc05"],
        "Acc10": epc_metrics["Acc10"],
        "Acc20": epc_metrics["Acc20"],
        "N": epc_metrics["N"],
        "physical_violation_shear": epc_phys_shear,
        "physical_violation_temperature": epc_phys_temp,
        "physical_violation_phi": epc_phys_phi,
    }


def evaluate_extreme_regimes(base_y_df, epc_pred_df, dataset, variant_name, train_frac):
    """Evaluate extreme-regime performance."""
    if epc_pred_df is not None:
        ood = epc_pred_df[epc_pred_df["Split"] == "OOD"].copy()
        ood["phi_raw"] = ood["phi"]
        ood["T_raw"] = ood["T"]
        y_true_col = "y_true"
        logg_col = "logg"
    else:
        ood = base_y_df[base_y_df["Split"] == "OOD"].copy()
        y_true_col = "target"
        logg_col = "logg"

    if len(ood) == 0:
        return []

    # Build regime bins from dataset
    bin_spec = build_bin_spec(dataset)
    phi_labels = bin_spec["phi"]["labels"]
    T_labels = bin_spec["T"]["labels"]
    logg_labels = bin_spec["logg"]["labels"]

    ood["phi_level"] = ood["phi_raw"].apply(lambda v: assign_level(v, bin_spec["phi"]["thresholds"], phi_labels))
    ood["T_level"] = ood["T_raw"].apply(lambda v: assign_level(v, bin_spec["T"]["thresholds"], T_labels))
    ood["logg_level"] = ood[logg_col].apply(lambda v: assign_level(v, bin_spec["logg"]["thresholds"], logg_labels))

    low_tail = float(ood[y_true_col].quantile(0.10))
    high_tail = float(ood[y_true_col].quantile(0.90))

    regimes = {
        "low_phi_low_T": ((ood["phi_level"] == "low") & (ood["T_level"] == "low")).values,
        "high_phi_highT": ((ood["phi_level"] == "high") & (ood["T_level"].isin(["high", "very_high"]))).values,
        "very_low_shear": (ood["logg_level"] == "very_low").values,
        "low_visc_tail": (ood[y_true_col] <= low_tail).values,
        "high_visc_tail": (ood[y_true_col] >= high_tail).values,
    }

    rows = []
    for regime_name, mask in regimes.items():
        sub = ood[mask].copy()
        if len(sub) == 0:
            continue
        yt = sub[y_true_col].to_numpy(dtype=float)
        yp = sub["pred"].to_numpy(dtype=float)
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        acc10 = float(np.mean(np.abs(yp - yt) <= 0.10))
        acc20 = float(np.mean(np.abs(yp - yt) <= 0.20))
        rows.append({
            "train_fraction": train_frac,
            "regime": regime_name,
            "variant": variant_name,
            "n_samples": len(sub),
            "MAE": mae,
            "RMSE": rmse,
            "Acc10": acc10,
            "Acc20": acc20,
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ablation_evidence_control(all_metrics, save_path):
    """Grouped bar chart: MAE per variant across train fractions."""
    save_path = Path(save_path)
    variants_order = ["base", "full_einn", "shuffled_summary", "random_summary", "residual_only", "no_gate"]
    variant_labels = {
        "base": "TransRheo-base",
        "full_einn": "Full EINN",
        "shuffled_summary": "Shuffled Summary",
        "random_summary": "Random Summary",
        "residual_only": "Residual Only",
        "no_gate": "No Gate Fusion",
    }
    colors = ["#888888", "#2E86AB", "#E5A37D", "#D16B6B", "#7BAE7B", "#B085C4"]

    df = pd.DataFrame(all_metrics)
    fracs = sorted(df["train_fraction"].unique())

    fig, axes = plt.subplots(1, len(fracs), figsize=(5 * len(fracs), 4.5), sharey=True)
    if len(fracs) == 1:
        axes = [axes]

    for ax, frac in zip(axes, fracs):
        sub = df[df["train_fraction"] == frac]
        vals = []
        valid_variants = []
        valid_colors = []
        for v in variants_order:
            row = sub[sub["variant"] == v]
            if len(row) > 0:
                vals.append(row.iloc[0]["MAE"])
                valid_variants.append(variant_labels.get(v, v))
                valid_colors.append(colors[variants_order.index(v)])
        xpos = np.arange(len(valid_variants))
        bars = ax.bar(xpos, vals, color=valid_colors, width=0.6, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_title(f"{int(frac * 100)}% Training Data", fontsize=12)
        ax.set_xticks(xpos)
        ax.set_xticklabels(valid_variants, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved {save_path}")


def plot_extreme_regime_gain(all_extreme, save_path):
    """Delta MAE over TransRheo-base across extreme regimes."""
    save_path = Path(save_path)
    df = pd.DataFrame(all_extreme)
    if len(df) == 0:
        print("[Plot] no extreme regime data, skip.")
        return

    # Compute delta MAE vs base for each (train_fraction, regime)
    base_df = df[df["variant"] == "base"][["train_fraction", "regime", "MAE"]].rename(columns={"MAE": "base_mae"})
    merged = df.merge(base_df, on=["train_fraction", "regime"], how="left")
    merged["delta_MAE"] = merged["MAE"] - merged["base_mae"]

    variants_order = ["full_einn", "shuffled_summary", "random_summary", "residual_only", "no_gate"]
    variant_labels = {
        "full_einn": "Full EINN", "shuffled_summary": "Shuffled",
        "random_summary": "Random", "residual_only": "Residual Only", "no_gate": "No Gate",
    }
    colors = ["#2E86AB", "#E5A37D", "#D16B6B", "#7BAE7B", "#B085C4"]

    regimes = sorted(merged["regime"].unique())
    fracs = sorted(merged["train_fraction"].unique())

    n_cols = len(fracs)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, frac in zip(axes, fracs):
        sub = merged[merged["train_fraction"] == frac]
        xpos = np.arange(len(regimes))
        width = 0.15
        for vi, v in enumerate(variants_order):
            vals = []
            for r in regimes:
                row = sub[(sub["variant"] == v) & (sub["regime"] == r)]
                vals.append(row.iloc[0]["delta_MAE"] if len(row) > 0 else 0.0)
            offset = (vi - len(variants_order) / 2 + 0.5) * width
            bars = ax.bar(xpos + offset, vals, width, label=variant_labels.get(v, v),
                          color=colors[vi], edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{int(frac * 100)}% Training Data", fontsize=12)
        ax.set_xticks(xpos)
        ax.set_xticklabels(regimes, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Delta MAE vs TransRheo-base")
        if frac == fracs[-1]:
            ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved {save_path}")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------
def generate_latex_table(all_metrics, save_path):
    """Generate compact LaTeX table comparing all variants."""
    save_path = Path(save_path)
    df = pd.DataFrame(all_metrics)
    variants_order = ["base", "full_einn", "shuffled_summary", "random_summary", "residual_only", "no_gate"]
    variant_labels = {
        "base": "TransRheo-base",
        "full_einn": "Full TransRheo-EINN",
        "shuffled_summary": "Shuffled-summary EPC",
        "random_summary": "Random-summary EPC",
        "residual_only": "Residual-pattern-only EPC",
        "no_gate": "No-gated-fusion EPC",
    }
    fracs = sorted(df["train_fraction"].unique())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study: overall OOD performance across training fractions.}",
        r"\label{tab:ablation_ood}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Train fraction & Variant & R$^2$ & RMSE & MAE & Acc@0.10 & Acc@0.20 \\",
        r"\midrule",
    ]
    for frac in fracs:
        frac_lines = []
        for v in variants_order:
            row = df[(df["train_fraction"] == frac) & (df["variant"] == v)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            frac_label = f"{int(frac * 100)}{chr(92)}%"
            vname = variant_labels.get(v, v)
            frac_lines.append(
                f"  {frac_label} & {vname} & "
                f"{r['R2']:.4f} & {r['RMSE']:.4f} & {r['MAE']:.4f} & "
                f"{r['Acc10']:.3f} & {r['Acc20']:.3f} \\\\"
            )
            # Only print frac label on first variant per fraction
            frac_label = ""
        lines.extend(frac_lines)
        lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[LaTeX] saved {save_path}")


# ---------------------------------------------------------------------------
# Experiment notes
# ---------------------------------------------------------------------------
def write_experiment_notes(save_path, args, variant, commands):
    save_path = Path(save_path)
    notes = f"""# Ablation Experiment Notes

## Purpose
Ablation experiments isolating the contribution of LLM-derived experiential
summaries and the EPC gated correction design in TransRheo-EINN.

## Data Split Policy
- Group-held-out OOD split: phi+T groups are split at the dataset level.
- OOD fraction: {args.ood_frac} (held-out groups)
- Validation fraction: {args.val_frac} (from training groups)
- Training fractions: {', '.join(str(f) for f in args.fractions)}

## Leakage Prevention
1. Residual patterns (bias prior) computed only from TRAIN fold.
2. Shuffled summaries permuted only within each split (TRAIN/VAL/OOD separately).
3. Random summary mean/std computed only from TRAIN fold embeddings.
4. Regime labels (phi_level, T_level, logg_level) derived from input conditions only.
5. High-viscosity-tail / low-viscosity-tail regimes used only for post-hoc evaluation.
6. Each training fraction uses its own training set and residual statistics.
7. No test labels used in summary construction, evidence retrieval, or EPC inputs.

## Variants Implemented
| Variant | Description |
|---------|-------------|
| base | TransRheo numerical backbone only |
| full_einn | Complete TransRheo-EINN |
| shuffled_summary | EPC with shuffled summary embeddings within split |
| random_summary | EPC with random Gaussian summary embeddings |
| residual_only | EPC using only residual-pattern prior (no literature evidence) |
| no_gate | EPC with fixed averaging instead of gated fusion |

## Generated Output Files
- results/ablation_overall_ood.csv
- results/ablation_extreme_regimes.csv
- results/ablation_table_main.tex
- figures/ablation_evidence_control.pdf
- figures/ablation_extreme_regime_gain.pdf
- results/ablation_experiment_notes.md (this file)

## Exact Commands
"""
    for cmd in commands:
        notes += f"  {cmd}\n"

    notes += f"""
## Seed
Base seed: {args.seed}
Summary shuffle seed: {args.seed + 9999}
Random vector seed: {args.seed + 8888}
Fraction selection seeds: {args.seed + 1000} + fraction index
"""
    save_path.write_text(notes, encoding="utf-8")
    print(f"[Notes] saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_args_proxy(args_ns, variant, seed):
    """Build a Namespace-like object with all needed defaults for sub-functions."""
    class ArgsProxy:
        pass
    proxy = ArgsProxy()
    # Copy all existing attributes
    for k, v in vars(args_ns).items():
        setattr(proxy, k, v)
    # Override seed and variant-specific
    proxy.seed = seed
    proxy.summary_backend = "deterministic"
    proxy.summary_allow_fallback = False
    proxy.deepseek_api_key = ""
    proxy.base_predictions_csv = None  # will use train mode
    proxy.base_mode = "train"
    proxy.prior_lambda_bin = getattr(args_ns, "prior_lambda_bin", 0.20)
    proxy.prior_lambda_ref = getattr(args_ns, "prior_lambda_ref", 0.10)
    proxy.prior_lambda_delta = getattr(args_ns, "prior_lambda_delta", 0.05)
    proxy.prior_lambda_pin = getattr(args_ns, "prior_lambda_pin", 0.02)
    proxy.prior_pin_tau = getattr(args_ns, "prior_pin_tau", 1e-6)
    proxy.prior_delta_bins = getattr(args_ns, "prior_delta_bins", 5)
    proxy.prior_evidence_max_length = getattr(args_ns, "prior_evidence_max_length", 320)
    proxy.prior_summary_max_length = getattr(args_ns, "prior_summary_max_length", 192)
    proxy.prior_retrieval_topk = getattr(args_ns, "prior_retrieval_topk", 5)
    proxy.prior_per_source_cap = getattr(args_ns, "prior_per_source_cap", 2)
    proxy.prior_evidence_text_chars = getattr(args_ns, "prior_evidence_text_chars", 600)
    proxy.prior_hidden_dim = getattr(args_ns, "prior_hidden_dim", 256)
    proxy.prior_dropout = getattr(args_ns, "prior_dropout", 0.10)
    proxy.prior_text_pool = getattr(args_ns, "prior_text_pool", "cls")
    proxy.prior_freeze_text_embeddings = getattr(args_ns, "prior_freeze_text_embeddings", False)
    proxy.prior_freeze_n_text_layers = getattr(args_ns, "prior_freeze_n_text_layers", 10)
    proxy.prior_epochs = getattr(args_ns, "prior_epochs", 40)
    proxy.prior_patience = getattr(args_ns, "prior_patience", 8)
    proxy.prior_batch_size = getattr(args_ns, "prior_batch_size", 16)
    proxy.prior_eval_batch_size = getattr(args_ns, "prior_eval_batch_size", 32)
    proxy.prior_lr_text = getattr(args_ns, "prior_lr_text", 8e-6)
    proxy.prior_lr_other = getattr(args_ns, "prior_lr_other", 2e-4)
    proxy.prior_weight_decay = getattr(args_ns, "prior_weight_decay", 1e-4)
    proxy.prior_grad_clip = getattr(args_ns, "prior_grad_clip", 1.0)
    proxy.prior_min_support_pair = getattr(args_ns, "prior_min_support_pair", 4)
    proxy.prior_min_support_single = getattr(args_ns, "prior_min_support_single", 5)
    proxy.prior_min_mae = getattr(args_ns, "prior_min_mae", 0.04)
    proxy.prior_min_sign_strength = getattr(args_ns, "prior_min_sign_strength", 0.45)
    proxy.knowledge_max_chunks = getattr(args_ns, "knowledge_max_chunks", 0)
    return proxy


def run_variant_for_fraction(args, variant, train_frac, frac_idx, out_dir):
    """Run a single variant at a single training fraction."""
    frac_pct = int(round(train_frac * 100))
    frac_label = f"{frac_pct}pct"
    frac_dir = out_dir / frac_label
    frac_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"[Ablation] variant={variant} fraction={frac_label} seed={args.seed}")
    print(f"{'=' * 70}")

    # Build proxy args
    proxy = build_args_proxy(args, variant, args.seed)

    # Load data & build fixed OOD split
    base_df = read_tabular_data(Path(proxy.data_path), proxy.sheet)
    records = build_multitask_records(base_df)
    x_feat, y, task_ids, groups, row_ids = load_arrays(records)
    outer = split_by_group(groups, task_ids, frac=proxy.ood_frac, seed=proxy.seed)
    id_base_idx = outer.train_idx
    ood_idx = outer.test_idx

    # Select fraction
    chosen_groups, selected_id_idx = select_fraction_groups(
        groups, task_ids, id_base_idx, train_frac, args.seed + 1000 + frac_idx,
    )
    print(f"  Selected {len(chosen_groups)} groups ({len(selected_id_idx)} records) for training")

    # Step 1: Train TransRheo backbone (or skip if cached and --skip-base)
    base_ckpt = frac_dir / "transrheo_base_ckpt.pt"
    base_pred_csv = frac_dir / "transrheo_y_predictions_all_splits.csv"
    split_csv = frac_dir / "split_indices.csv"
    if args.skip_base and base_ckpt.exists() and base_pred_csv.exists() and split_csv.exists():
        print(f"  [Skip-base] loading cached base predictions from {frac_label}")
        base_y_df = pd.read_csv(base_pred_csv)
        split_df = pd.read_csv(split_csv)
    else:
        base_y_df, split_df = train_base_transrheo_fraction(
            proxy, frac_dir, base_df, records, x_feat, y, task_ids, groups, row_ids, ood_idx, selected_id_idx,
        )

    # Step 2: For base variant, done
    if variant == "base":
        epc_pred_df = None
    else:
        # Build bias prior bank
        bin_spec = build_bin_spec(base_df)
        bank = build_bias_prior_bank(base_df, base_y_df, bin_spec, proxy, frac_dir)

        # Train EPC with variant modification
        epc_pred_df = train_llm_correction_variant(
            proxy, base_df, base_y_df, bank, bin_spec, frac_dir, variant=variant,
        )

    # Evaluate
    metrics = evaluate_variant(base_y_df, epc_pred_df, variant, train_frac)
    extreme_rows = evaluate_extreme_regimes(base_y_df, epc_pred_df, base_df, variant, train_frac)

    # Save individual results
    pd.DataFrame([metrics]).to_csv(frac_dir / f"ablation_metrics_{variant}.csv", index=False)
    if extreme_rows:
        pd.DataFrame(extreme_rows).to_csv(frac_dir / f"ablation_extreme_{variant}.csv", index=False)

    return metrics, extreme_rows


def main():
    ap = argparse.ArgumentParser(description="TransRheo-EINN ablation: EPC evidence control experiments.")
    # Ablation-specific args
    ap.add_argument("--variant", type=str, default="full_einn", choices=VALID_VARIANTS,
                    help="Ablation variant to run")
    ap.add_argument("--train-fraction", type=float, default=None,
                    help="Single training fraction (overrides --fractions)")
    ap.add_argument("--fractions", type=str, default="0.25,0.50,0.75,1.00",
                    help="Comma-separated training fractions")
    ap.add_argument("--scibert-dir", type=Path, required=True)
    ap.add_argument("--data-path", type=str, default="new_MDdata.xlsx")
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/ablation"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ood-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--knowledge-dir", type=Path, default=Path("Knowledge"))

    # TransRheo backbone args
    ap.add_argument("--base-epochs", type=str, default="50,50,30")
    ap.add_argument("--base-batch-size", type=int, default=96)
    ap.add_argument("--base-eval-batch-size", type=int, default=1024)
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--base-weight-decay", type=float, default=0.0)
    ap.add_argument("--base-grad-clip", type=float, default=1.0)
    ap.add_argument("--base-aug-noise-frac", type=float, default=0.01)
    ap.add_argument("--base-patience", type=int, default=25)
    ap.add_argument("--base-warmup-frac", type=float, default=0.08)
    ap.add_argument("--base-min-lr-scale", type=float, default=0.15)
    ap.add_argument("--base-pin-tau", type=float, default=1e-6)
    ap.add_argument("--base-lambda-pin-s2", type=float, default=0.15)
    ap.add_argument("--base-lambda-pin-s3", type=float, default=0.25)
    ap.add_argument("--base-weight-y", type=float, default=4.0)
    ap.add_argument("--base-weight-logg", type=float, default=0.5)
    ap.add_argument("--base-weight-T", type=float, default=0.5)
    ap.add_argument("--base-weight-phi", type=float, default=0.5)
    ap.add_argument("--base-d-model", type=int, default=256)
    ap.add_argument("--base-n-heads", type=int, default=8)
    ap.add_argument("--base-n-attn-layers", type=int, default=3)
    ap.add_argument("--base-n-mlp-layers", type=int, default=4)
    ap.add_argument("--base-mlp-hidden-mult", type=float, default=2.0)
    ap.add_argument("--base-dropout", type=float, default=0.05)
    ap.add_argument("--base-tail-epochs", type=int, default=80)
    ap.add_argument("--base-tail-patience", type=int, default=20)
    ap.add_argument("--base-tail-lr", type=float, default=8e-5)
    ap.add_argument("--base-tail-batch-size", type=int, default=0)
    ap.add_argument("--base-tail-warmup-frac", type=float, default=0.05)
    ap.add_argument("--base-tail-min-lr-scale", type=float, default=0.10)
    ap.add_argument("--base-tail-weight-y", type=float, default=10.0)
    ap.add_argument("--base-tail-weight-logg", type=float, default=0.15)
    ap.add_argument("--base-tail-weight-T", type=float, default=0.15)
    ap.add_argument("--base-tail-weight-phi", type=float, default=0.15)
    ap.add_argument("--base-tail-pin-lambda", type=float, default=0.05)
    ap.add_argument("--base-tail-ramp-frac", type=float, default=0.15)
    ap.add_argument("--base-tail-aug-noise-frac", type=float, default=0.0)

    # EPC args
    ap.add_argument("--prior-epochs", type=int, default=40)
    ap.add_argument("--prior-patience", type=int, default=8)
    ap.add_argument("--prior-batch-size", type=int, default=16)
    ap.add_argument("--prior-eval-batch-size", type=int, default=32)
    ap.add_argument("--prior-lr-text", type=float, default=8e-6)
    ap.add_argument("--prior-lr-other", type=float, default=2e-4)
    ap.add_argument("--prior-weight-decay", type=float, default=1e-4)
    ap.add_argument("--prior-grad-clip", type=float, default=1.0)
    ap.add_argument("--prior-hidden-dim", type=int, default=256)
    ap.add_argument("--prior-dropout", type=float, default=0.10)
    ap.add_argument("--prior-text-pool", type=str, default="cls", choices=["cls", "mean"])
    ap.add_argument("--prior-freeze-text-embeddings", action="store_true")
    ap.add_argument("--prior-freeze-n-text-layers", type=int, default=10)
    ap.add_argument("--prior-lambda-delta", type=float, default=0.05)
    ap.add_argument("--prior-lambda-bin", type=float, default=0.20)
    ap.add_argument("--prior-lambda-ref", type=float, default=0.10)
    ap.add_argument("--prior-lambda-pin", type=float, default=0.02)
    ap.add_argument("--prior-pin-tau", type=float, default=1e-6)
    ap.add_argument("--prior-delta-bins", type=int, default=5)
    ap.add_argument("--prior-evidence-max-length", type=int, default=320)
    ap.add_argument("--prior-summary-max-length", type=int, default=192)
    ap.add_argument("--prior-retrieval-topk", type=int, default=5)
    ap.add_argument("--prior-per-source-cap", type=int, default=2)
    ap.add_argument("--prior-evidence-text-chars", type=int, default=600)
    ap.add_argument("--knowledge-max-chunks", type=int, default=0)

    # Flags
    ap.add_argument("--skip-base", action="store_true",
                    help="Skip base TransRheo training (reuse cached)")
    ap.add_argument("--run-all", action="store_true",
                    help="Run all variants sequentially")
    ap.add_argument("--only-plot", action="store_true",
                    help="Only regenerate plots from existing CSV files")

    args = ap.parse_args()
    args.base_epochs = tuple(int(x) for x in args.base_epochs.split(","))
    args.out_dir = args.out_dir.resolve()
    args.fig_dir = args.fig_dir.resolve()
    args.scibert_dir = args.scibert_dir.resolve()
    args.data_path = str(Path(args.data_path).resolve())
    args.knowledge_dir = args.knowledge_dir.resolve()

    if args.train_fraction is not None:
        fractions = [args.train_fraction]
    else:
        fractions = [float(x) for x in args.fractions.split(",") if x.strip()]

    # Ensure output dirs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.only_plot:
        # Re-plot from existing CSV files
        print("[Mode] only-plot: regenerating figures from cached CSVs")
        overall_path = args.out_dir / "ablation_overall_ood.csv"
        extreme_path = args.out_dir / "ablation_extreme_regimes.csv"
        if overall_path.exists():
            all_metrics = pd.read_csv(overall_path).to_dict("records")
            plot_ablation_evidence_control(all_metrics, args.fig_dir / "ablation_evidence_control.pdf")
            generate_latex_table(all_metrics, args.out_dir / "ablation_table_main.tex")
        if extreme_path.exists():
            all_extreme = pd.read_csv(extreme_path).to_dict("records")
            plot_extreme_regime_gain(all_extreme, args.fig_dir / "ablation_extreme_regime_gain.pdf")
        return

    # Determine which variants to run
    if args.run_all:
        variants = VALID_VARIANTS
    else:
        variants = [args.variant]

    all_metrics = []
    all_extreme = []
    all_commands = []

    for variant in variants:
        for fi, frac in enumerate(fractions):
            set_seed(args.seed)
            m, e = run_variant_for_fraction(args, variant, frac, fi, args.out_dir)
            all_metrics.append(m)
            all_extreme.extend(e)

            cmd = (
                f"python ablation_epc_evidence_control.py --variant {variant} "
                f"--train-fraction {frac} --scibert-dir {args.scibert_dir} "
                f"--seed {args.seed} --out-dir {args.out_dir}"
            )
            if cmd not in all_commands:
                all_commands.append(cmd)

    # Save aggregated results (append mode for single-variant calls)
    metrics_df = pd.DataFrame(all_metrics)
    overall_csv = args.out_dir / "ablation_overall_ood.csv"
    if overall_csv.exists():
        existing = pd.read_csv(overall_csv)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        metrics_df = metrics_df.drop_duplicates(subset=["train_fraction", "variant"], keep="last")
    metrics_df.to_csv(overall_csv, index=False)
    print(f"\n[Results] saved {overall_csv}")

    extreme_df = pd.DataFrame(all_extreme)
    if len(extreme_df) > 0:
        extreme_csv = args.out_dir / "ablation_extreme_regimes.csv"
        if extreme_csv.exists():
            existing = pd.read_csv(extreme_csv)
            extreme_df = pd.concat([existing, extreme_df], ignore_index=True)
            extreme_df = extreme_df.drop_duplicates(
                subset=["train_fraction", "regime", "variant"], keep="last"
            )
        extreme_df.to_csv(extreme_csv, index=False)
        print(f"[Results] saved {extreme_csv}")

    # Generate figures and tables
    plot_ablation_evidence_control(all_metrics, args.fig_dir / "ablation_evidence_control.pdf")
    plot_extreme_regime_gain(all_extreme, args.fig_dir / "ablation_extreme_regime_gain.pdf")
    generate_latex_table(all_metrics, args.out_dir / "ablation_table_main.tex")

    # Write experiment notes
    write_experiment_notes(args.out_dir / "ablation_experiment_notes.md", args, variants, all_commands)

    # Print summary
    print(f"\n{'=' * 70}")
    print("Ablation Experiment Complete!")
    print(f"{'=' * 70}")
    print(f"Output directory: {args.out_dir}")
    print(f"Figure directory: {args.fig_dir}")
    print(f"\nVariants run: {', '.join(variants)}")
    print(f"Fractions: {', '.join(str(f) for f in fractions)}")
    print(f"\nGenerated files:")
    print(f"  {args.out_dir / 'ablation_overall_ood.csv'}")
    print(f"  {args.out_dir / 'ablation_extreme_regimes.csv'}")
    print(f"  {args.out_dir / 'ablation_table_main.tex'}")
    print(f"  {args.fig_dir / 'ablation_evidence_control.pdf'}")
    print(f"  {args.fig_dir / 'ablation_extreme_regime_gain.pdf'}")
    print(f"  {args.out_dir / 'ablation_experiment_notes.md'}")


if __name__ == "__main__":
    main()
