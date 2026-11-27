#!/usr/bin/env python3
import argparse
import json
import math
import os
import hashlib
from pathlib import Path
import urllib.request
import urllib.error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from TransRheo_highfit_multitask import (
    TransRheoHighFit,
    build_multitask_records,
    evaluate_indices,
    fit_scaler_on_train,
    load_arrays,
    make_lr_lambda,
    pin_shear_loss,
    plot_parity_y,
    plot_residual_vs_logg,
    read_tabular_data,
    run_training_phase,
    set_seed,
    split_by_group,
)
from knowledge_retriever import (
    ChunkKnowledgeBase,
    build_summary_struct,
    format_evidence_text,
    summary_struct_to_text,
)
from delta_correction_model import KREPCorrectionModel


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mse)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'Acc05': float(np.mean(np.abs(y_pred - y_true) <= 0.05)),
        'Acc10': float(np.mean(np.abs(y_pred - y_true) <= 0.10)),
        'Acc20': float(np.mean(np.abs(y_pred - y_true) <= 0.20)),
        'N': int(len(y_true)),
    }


def load_cached_base_predictions(args):
    base_df = read_tabular_data(Path(args.data_path), args.sheet).reset_index(drop=True)
    base_df['row_id'] = np.arange(len(base_df), dtype=int)
    base_y_df = pd.read_csv(args.base_predictions_csv)
    required = {'Split', 'row_id', 'target', 'pred', 'phi_raw', 'T_raw', 'logg'}
    missing = required - set(base_y_df.columns)
    if missing:
        raise ValueError(f'Cached base predictions missing required columns: {sorted(missing)}')
    base_y_df = base_y_df.copy()
    base_y_df['row_id'] = base_y_df['row_id'].astype(int)
    base_y_df = base_y_df.sort_values(['row_id', 'Split']).reset_index(drop=True)
    if base_y_df['Split'].nunique() < 3:
        raise ValueError('Cached base predictions do not include TRAIN/VAL/OOD splits.')
    check = base_y_df[['row_id', 'target']].drop_duplicates().sort_values('row_id')
    if len(check) != len(base_df):
        raise ValueError(f'Row count mismatch between base data ({len(base_df)}) and cached predictions ({len(check)} unique row_id).')
    y_gap = np.max(np.abs(check['target'].to_numpy(dtype=float) - base_df['y'].to_numpy(dtype=float)))
    if y_gap > 1e-6:
        raise ValueError(f'Cached base predictions appear misaligned with data file; max |target - y| = {y_gap:.6e}')
    split_df = base_y_df[['Split', 'row_id']].drop_duplicates().copy()
    return base_df, base_y_df, split_df


def load_summary_cache(cache_path: Path):
    cache = {}
    if not cache_path.exists():
        return cache
    with cache_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[item['cache_key']] = item
    return cache


def append_summary_cache(cache_path: Path, item: dict):
    with cache_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def normalize_summary_struct(summary: dict, fallback_ids):
    out = dict(summary or {})
    mech = out.get('relevant_mechanism', [])
    if isinstance(mech, str):
        mech = [mech]
    rationale = out.get('evidence_backed_rationale', [])
    if isinstance(rationale, str):
        rationale = [rationale]
    evidence_ids = out.get('evidence_ids', fallback_ids)
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids]
    out['relevant_mechanism'] = [str(x).strip() for x in mech if str(x).strip()]
    out['evidence_backed_rationale'] = [str(x).strip() for x in rationale if str(x).strip()]
    out['evidence_ids'] = [str(x).strip() for x in evidence_ids if str(x).strip()]
    out['regime_assessment'] = str(out.get('regime_assessment', '')).strip()
    out['likely_bias_direction'] = str(out.get('likely_bias_direction', 'uncertain')).strip() or 'uncertain'
    out['correction_strength'] = str(out.get('correction_strength', 'medium')).strip() or 'medium'
    return out


def build_summary_cache_key(row_id, query_text, evidence):
    raw = json.dumps({
        'row_id': int(row_id),
        'query_text': query_text,
        'evidence_ids': [ev.get('id', '') for ev in evidence],
        'evidence_scores': [round(float(ev.get('score', 0.0)), 6) for ev in evidence],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def call_deepseek_summary(sample_payload, evidence, api_key, model_name, timeout_sec=90):
    system = (
        "You are assisting a rheology correction pipeline. Given a sample state, a base prediction, and retrieved "
        "rheology evidence, produce a concise correction summary in strict JSON with keys: regime_assessment, "
        "relevant_mechanism, likely_bias_direction, correction_strength, evidence_backed_rationale, evidence_ids. "
        "Do not invent evidence ids. Keep relevant_mechanism and evidence_backed_rationale as arrays of short strings."
    )
    body = json.dumps({
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps({'sample': sample_payload, 'evidence': evidence}, ensure_ascii=False)},
        ],
        'temperature': 0.2,
        'response_format': {'type': 'json_object'},
    }, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        'https://api.deepseek.com/chat/completions',
        data=body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    content = out['choices'][0]['message']['content']
    parsed = json.loads(content)
    return parsed, out


def call_deepseek_summary_with_retry(sample_payload, evidence, api_key, model_name, timeout_sec=90, max_retries=3, retry_sleep_sec=5):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_deepseek_summary(
                sample_payload=sample_payload,
                evidence=evidence,
                api_key=api_key,
                model_name=model_name,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            import time
            sleep_sec = retry_sleep_sec * attempt
            print(f"[DeepSeek] retry {attempt}/{max_retries - 1} after {type(exc).__name__}: sleep {sleep_sec}s")
            time.sleep(sleep_sec)
    raise last_exc


def physics_v_shear(df_pred_y, tau=1e-6):
    if len(df_pred_y) == 0:
        return 0.0
    work = df_pred_y.copy()
    work['phi_g'] = work['phi_raw'].astype(float).round(4)
    work['T_g'] = work['T_raw'].astype(float).round(4)
    viol, total = 0, 0
    for _, grp in work.groupby(['phi_g', 'T_g']):
        if len(grp) < 2:
            continue
        grp = grp.sort_values('logg')
        yy = grp['pred'].to_numpy(dtype=float)
        ll = grp['logg'].to_numpy(dtype=float)
        for i in range(len(yy)):
            for j in range(i + 1, len(yy)):
                if ll[j] > ll[i]:
                    total += 1
                    if yy[j] > yy[i] + tau:
                        viol += 1
    return float(viol / max(1, total))


def unique_thresholds(values, n_bins):
    vals = sorted(np.unique(np.round(np.asarray(values, dtype=float), 6)).tolist())
    if len(vals) < n_bins:
        return np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1]).tolist()
    block = len(vals) / n_bins
    out = []
    for k in range(1, n_bins):
        i0 = max(0, min(len(vals) - 1, math.ceil(k * block) - 1))
        i1 = max(0, min(len(vals) - 1, math.ceil(k * block)))
        out.append((vals[i0] + vals[i1]) / 2.0)
    return out


def assign_level(value, thresholds, labels):
    for i, thr in enumerate(thresholds):
        if value <= thr:
            return labels[i]
    return labels[-1]


def build_bin_spec(base_df):
    return {
        'phi': {'labels': ['low', 'medium', 'high'], 'thresholds': unique_thresholds(base_df['phi'].values, 3)},
        'T': {'labels': ['low', 'medium', 'high', 'very_high'], 'thresholds': unique_thresholds(base_df['T'].values, 4)},
        'logg': {'labels': ['very_low', 'low', 'medium', 'high'], 'thresholds': unique_thresholds(base_df['logg'].values, 4)},
    }


def build_group_batches(df_rows, batch_size, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, row in df_rows.iterrows():
        groups.setdefault(int(row['group_id']), []).append(int(i))
    batches = []
    for idxs in groups.values():
        rng.shuffle(idxs)
        for s in range(0, len(idxs), batch_size):
            batches.append(idxs[s:s + batch_size])
    rng.shuffle(batches)
    return batches


def make_models():
    models = {
        'LR': LinearRegression(),
        'SVR': SVR(C=20.0, epsilon=0.01, gamma='scale'),
        'RF': RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=900, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    return models


def benchmark_frame(pred_df, model_name):
    rows = []
    for split_name, sub in pred_df.groupby('Split'):
        met = calc_metrics(sub['target'], sub['pred'])
        row = {'Split': split_name, 'Model': model_name, 'Feature': 'Optimized (phi, logg, 1/T)', 'Task': 'y'}
        row.update(met)
        row['V_shear'] = physics_v_shear(sub[['phi_raw', 'T_raw', 'logg', 'pred']].copy())
        rows.append(row)
    return pd.DataFrame(rows)

def train_base_transrheo(args, out_dir):
    base_df = read_tabular_data(Path(args.data_path), args.sheet)
    records = build_multitask_records(base_df)
    x_feat, y, task_ids, groups, row_ids = load_arrays(records)

    outer = split_by_group(groups, task_ids, frac=args.ood_frac, seed=args.seed)
    inner = split_by_group(groups, task_ids, frac=args.val_frac, seed=args.seed + 17, base_indices=outer.train_idx)
    train_idx, val_idx, ood_idx = inner.train_idx, inner.test_idx, outer.test_idx

    scaler = fit_scaler_on_train(x_feat, task_ids, train_idx)
    x_scaled = scaler.transform(x_feat)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    best = {'val_y_mae': float('inf'), 'epoch': 0, 'stage': '', 'state_dict': None}
    log_rows = []
    for phase_name, n_epochs, pin_fn in [
        ('S1', args.base_epochs[0], pin_const(0.0)),
        ('S2', args.base_epochs[1], pin_const(args.base_lambda_pin_s2)),
        ('S3', args.base_epochs[2], pin_const(args.base_lambda_pin_s3)),
    ]:
        best = run_training_phase(
            phase_name=phase_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_idx=train_idx,
            val_idx=val_idx,
            x_scaled=x_scaled,
            y=y,
            task_ids=task_ids,
            records=records,
            device=device,
            weights=weights,
            batch_size=args.base_batch_size,
            infer_batch_size=args.base_eval_batch_size,
            seed=args.seed,
            n_epochs=n_epochs,
            grad_clip=args.base_grad_clip,
            aug_noise_frac=args.base_aug_noise_frac,
            pin_tau=args.base_pin_tau,
            pin_lambda_fn=pin_fn,
            best=best,
            log_rows=log_rows,
            patience=args.base_patience,
        )
        if best.get('state_dict') is not None:
            model.load_state_dict(best['state_dict'])

    if best.get('state_dict') is not None:
        model.load_state_dict(best['state_dict'])

    if args.base_tail_epochs > 0:
        print(
            f"[Tail] start y-priority finetune | epochs={args.base_tail_epochs} | "
            f"lr={args.base_tail_lr:.2e} | y_weight={args.base_tail_weight_y:.2f}"
        )
        tail_weights = torch.tensor(
            [args.base_tail_weight_y, args.base_tail_weight_logg, args.base_tail_weight_T, args.base_tail_weight_phi],
            dtype=torch.float32,
            device=device,
        )
        tail_optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_tail_lr, weight_decay=args.base_weight_decay)
        tail_steps = max(1, steps_per_epoch * args.base_tail_epochs)
        tail_scheduler = torch.optim.lr_scheduler.LambdaLR(
            tail_optimizer,
            lr_lambda=make_lr_lambda(
                total_steps=tail_steps,
                warmup_frac=args.base_tail_warmup_frac,
                min_scale=args.base_tail_min_lr_scale,
            ),
        )

        if args.base_tail_ramp_frac <= 0 or args.base_tail_pin_lambda <= 0:
            tail_pin_lambda_fn = lambda step, epoch: 0.0 if args.base_tail_pin_lambda <= 0 else float(args.base_tail_pin_lambda)
        else:
            tail_ramp_steps = max(1, int(args.base_tail_ramp_frac * tail_steps))

            def tail_pin_lambda_fn(step, epoch, _target=args.base_tail_pin_lambda, _ramp_steps=tail_ramp_steps):
                progress = min(step, _ramp_steps)
                return float(_target) * (float(progress) / float(_ramp_steps))

        best = run_training_phase(
            phase_name='TAIL',
            model=model,
            optimizer=tail_optimizer,
            scheduler=tail_scheduler,
            train_idx=train_idx,
            val_idx=val_idx,
            x_scaled=x_scaled,
            y=y,
            task_ids=task_ids,
            records=records,
            device=device,
            weights=tail_weights,
            batch_size=args.base_tail_batch_size if args.base_tail_batch_size > 0 else args.base_batch_size,
            infer_batch_size=args.base_eval_batch_size,
            seed=args.seed + 9000,
            n_epochs=args.base_tail_epochs,
            grad_clip=args.base_grad_clip,
            aug_noise_frac=args.base_tail_aug_noise_frac,
            pin_tau=args.base_pin_tau,
            pin_lambda_fn=tail_pin_lambda_fn,
            best=best,
            log_rows=log_rows,
            patience=args.base_tail_patience,
        )
        if best.get('state_dict') is not None:
            model.load_state_dict(best['state_dict'])

    split_df = pd.DataFrame({'record_idx': np.arange(len(records)), 'row_id': row_ids, 'group_id': groups, 'split': 'ID_HOLDOUT'})
    split_df.loc[train_idx, 'split'] = 'TRAIN'
    split_df.loc[val_idx, 'split'] = 'VAL'
    split_df.loc[ood_idx, 'split'] = 'OOD'
    split_df.to_csv(out_dir / 'split_indices.csv', index=False)

    frames = []
    pred_frames = []
    y_frames = []
    for split_name, idx in [('TRAIN', train_idx), ('VAL', val_idx), ('OOD', ood_idx)]:
        metrics_df, pred_df, _ = evaluate_indices(
            model=model,
            x_scaled=x_scaled,
            y=y,
            task_ids=task_ids,
            records=records,
            indices=idx,
            batch_size=args.base_eval_batch_size,
            device=device,
        )
        metrics_df['Split'] = split_name
        pred_df['Split'] = split_name
        frames.append(metrics_df)
        pred_frames.append(pred_df)
        y_frames.append(pred_df[pred_df['target_name'] == 'y'].copy())

    summary_df = pd.concat(frames, ignore_index=True)
    pred_all_df = pd.concat(pred_frames, ignore_index=True)
    y_pred_df = pd.concat(y_frames, ignore_index=True)
    pd.DataFrame(log_rows).to_csv(out_dir / 'transrheo_training_log.csv', index=False)
    summary_df.to_csv(out_dir / 'transrheo_summary_metrics.csv', index=False)
    pred_all_df.to_csv(out_dir / 'transrheo_all_predictions_multitask.csv', index=False)
    y_pred_df.to_csv(out_dir / 'transrheo_y_predictions_all_splits.csv', index=False)
    plot_parity_y(y_pred_df[y_pred_df['Split'] == 'OOD'][['target', 'pred']].copy(), out_dir / 'transrheo_parity_ood.png')
    plot_residual_vs_logg(y_pred_df[y_pred_df['Split'] == 'OOD'][['target', 'pred', 'logg']].copy(), out_dir / 'transrheo_residual_ood.png')

    torch.save(
        {
            'model_state': model.state_dict(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'split': {'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist(), 'ood_idx': ood_idx.tolist()},
            'best': {k: v for k, v in best.items() if k != 'state_dict'},
        },
        out_dir / 'transrheo_base_ckpt.pt',
    )
    return base_df, y_pred_df, split_df


def build_bias_prior_bank(base_df, base_y_df, bin_spec, args, out_dir):
    work = base_y_df[base_y_df['Split'].isin(['TRAIN', 'VAL'])].copy()
    work['residual'] = work['target'] - work['pred']
    work['phi_bin'] = work['phi_raw'].map(lambda v: assign_level(float(v), bin_spec['phi']['thresholds'], bin_spec['phi']['labels']))
    work['T_bin'] = work['T_raw'].map(lambda v: assign_level(float(v), bin_spec['T']['thresholds'], bin_spec['T']['labels']))
    work['logg_bin'] = work['logg'].map(lambda v: assign_level(float(v), bin_spec['logg']['thresholds'], bin_spec['logg']['labels']))

    rules = []
    for by_cols, min_n in [(['phi_bin', 'T_bin'], args.prior_min_support_pair), (['logg_bin'], args.prior_min_support_single), (['phi_bin'], args.prior_min_support_single), (['T_bin'], args.prior_min_support_single)]:
        for key, grp in work.groupby(by_cols):
            n = len(grp)
            if n < min_n:
                continue
            bias = float(grp['residual'].mean())
            mae = float(grp['residual'].abs().mean())
            pos_frac = float((grp['residual'] > 0).mean())
            sign_strength = abs(pos_frac - 0.5) * 2.0
            if mae < args.prior_min_mae or sign_strength < args.prior_min_sign_strength:
                continue
            direction = 'positive_correction' if bias > 0 else 'negative_correction'
            risk = 'high' if mae >= 0.12 else ('medium' if mae >= 0.07 else 'low')
            rules.append({
                'scope': '+'.join(by_cols),
                'key': '|'.join([str(x) for x in (key if isinstance(key, tuple) else (key,))]),
                'n': int(n),
                'bias': bias,
                'mae': mae,
                'pos_frac': pos_frac,
                'direction': direction,
                'risk': risk,
            })
    bank = {'bin_spec': bin_spec, 'rules': sorted(rules, key=lambda r: (-r['mae'], -r['n']))}
    (out_dir / 'bias_prior_bank.json').write_text(json.dumps(bank, indent=2), encoding='utf-8')
    pd.DataFrame(bank['rules']).to_csv(out_dir / 'bias_prior_bank.csv', index=False)
    return bank


def extract_prior_hint(row, bank, bin_spec):
    phi_bin = assign_level(float(row['phi']), bin_spec['phi']['thresholds'], bin_spec['phi']['labels'])
    t_bin = assign_level(float(row['T']), bin_spec['T']['thresholds'], bin_spec['T']['labels'])
    logg_bin = assign_level(float(row['logg']), bin_spec['logg']['thresholds'], bin_spec['logg']['labels'])
    matched = []
    for rule in bank['rules']:
        scope = rule['scope']
        parts = rule['key'].split('|')
        ok = False
        if scope == 'phi_bin+T_bin':
            ok = parts[0] == phi_bin and parts[1] == t_bin
        elif scope == 'logg_bin':
            ok = parts[0] == logg_bin
        elif scope == 'phi_bin':
            ok = parts[0] == phi_bin
        elif scope == 'T_bin':
            ok = parts[0] == t_bin
        if ok:
            matched.append(rule)
    order = {'high': 0, 'medium': 1, 'low': 2}
    matched = sorted(matched, key=lambda r: (order.get(r['risk'], 9), -r['mae']))[:3]
    if not matched:
        return {
            'phi_bin': phi_bin,
            'T_bin': t_bin,
            'logg_bin': logg_bin,
            'direction': 'uncertain',
            'risk': 'low',
            'matched_rules': [],
        }
    return {
        'phi_bin': phi_bin,
        'T_bin': t_bin,
        'logg_bin': logg_bin,
        'direction': matched[0]['direction'],
        'risk': matched[0]['risk'],
        'matched_rules': matched,
    }


def build_delta_bin_spec(delta_train, n_bins):
    delta_train = np.asarray(delta_train, dtype=float)
    if len(delta_train) == 0:
        raise ValueError('delta_train is empty.')
    edges = np.quantile(delta_train, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = min(edges[0], delta_train.min() - 1e-6)
    edges[-1] = max(edges[-1], delta_train.max() + 1e-6)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_widths = 0.5 * (edges[1:] - edges[:-1])
    labels = []
    for i in range(n_bins):
        if i == 0:
            labels.append('strong_negative')
        elif i == n_bins - 1:
            labels.append('strong_positive')
        elif i < n_bins // 2:
            labels.append(f'negative_{i}')
        elif i == n_bins // 2 and n_bins % 2 == 1:
            labels.append('neutral')
        else:
            labels.append(f'positive_{i}')
    return {
        'edges': edges.tolist(),
        'centers': centers.tolist(),
        'half_widths': half_widths.tolist(),
        'labels': labels,
    }


def assign_delta_bins(values, spec):
    edges = np.asarray(spec['edges'], dtype=float)
    values = np.asarray(values, dtype=float)
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(np.int64)


def build_prior_text(row, bank, bin_spec):
    phi_bin = assign_level(float(row['phi']), bin_spec['phi']['thresholds'], bin_spec['phi']['labels'])
    t_bin = assign_level(float(row['T']), bin_spec['T']['thresholds'], bin_spec['T']['labels'])
    logg_bin = assign_level(float(row['logg']), bin_spec['logg']['thresholds'], bin_spec['logg']['labels'])
    lines = [
        f'This sample lies in a {phi_bin}-concentration, {t_bin}-temperature, {logg_bin}-shear regime.',
        'The base TransRheo prediction is accurate overall, but some regimes require correction attention.',
    ]
    matched = []
    for rule in bank['rules']:
        scope = rule['scope']
        parts = rule['key'].split('|')
        ok = False
        if scope == 'phi_bin+T_bin':
            ok = parts[0] == phi_bin and parts[1] == t_bin
        elif scope == 'logg_bin':
            ok = parts[0] == logg_bin
        elif scope == 'phi_bin':
            ok = parts[0] == phi_bin
        elif scope == 'T_bin':
            ok = parts[0] == t_bin
        if ok:
            matched.append(rule)
    order = {'high': 0, 'medium': 1, 'low': 2}
    matched = sorted(matched, key=lambda r: (order.get(r['risk'], 9), -r['mae']))[:3]
    for rule in matched:
        direction = 'A positive correction to the base prediction is more likely in this regime.' if rule['direction'] == 'positive_correction' else 'A negative correction to the base prediction is more likely in this regime.'
        lines.append(f"Matched prior: {rule['scope']}={rule['key']} with {rule['risk']} correction risk. {direction}")
    lines.append(f"The current numerical state is phi={float(row['phi']):.3f}, T={float(row['T']):.2f}, and log shear rate={float(row['logg']):.5f}.")
    return ' '.join(lines)

class PriorCorrectionModel(nn.Module):
    def __init__(self, scibert_dir, num_dim=4, hidden_dim=256, dropout=0.1, text_pool='cls'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(scibert_dir)
        self.text_pool = text_pool
        text_dim = int(self.text_encoder.config.hidden_size)
        self.num_net = nn.Sequential(
            nn.Linear(num_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, 1)

    def freeze_text(self, freeze_n_layers=10, freeze_embeddings=True):
        if freeze_embeddings:
            for p in self.text_encoder.embeddings.parameters():
                p.requires_grad = False
        layers = getattr(getattr(self.text_encoder, 'encoder', None), 'layer', None)
        if layers is not None:
            for layer in list(layers)[:max(0, freeze_n_layers)]:
                for p in layer.parameters():
                    p.requires_grad = False

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if self.text_pool == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden[:, 0, :]

    def forward(self, x_num, input_ids, attention_mask):
        h_num = self.num_net(x_num)
        h_text = self.text_proj(self.encode_text(input_ids, attention_mask))
        h = self.fusion(torch.cat([h_num, h_text], dim=1))
        return self.delta_head(h).squeeze(1)


def tokenize_texts(tokenizer, texts, max_length):
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='np')
    return enc['input_ids'].astype(np.int64), enc['attention_mask'].astype(np.int64)


def train_llm_correction(args, base_df, base_y_df, bank, bin_spec, out_dir):
    dataset = base_df.copy().reset_index(drop=True)
    dataset['row_id'] = np.arange(len(dataset), dtype=int)
    if 'group_id' not in dataset.columns:
        dataset['group_id'] = dataset[['phi', 'T']].astype(str).agg('|'.join, axis=1).astype('category').cat.codes
    base_part = base_y_df[['row_id', 'Split', 'target', 'pred']].rename(columns={'target': 'y_true', 'pred': 'base_pred'})
    dataset = dataset.merge(base_part, on='row_id', how='inner')
    dataset['group_id'] = dataset['group_id'].astype(int)
    dataset['delta_true'] = dataset['y_true'] - dataset['base_pred']
    kb = ChunkKnowledgeBase(args.knowledge_dir, max_chunks=args.knowledge_max_chunks)
    summary_cache_path = out_dir / f'krepc_summary_cache_{args.summary_backend}.jsonl'
    summary_cache = load_summary_cache(summary_cache_path)
    deepseek_key = args.deepseek_api_key or os.environ.get('DEEPSEEK_API_KEY', '')
    retrieval_feedback = []
    evidence_texts = []
    summary_texts = []
    query_texts = []
    summary_structs = []
    summary_sources = []
    prior_directions = []
    prior_risks = []
    evidence_ids = []

    prep_iter = tqdm(dataset.iterrows(), total=len(dataset), desc='[KREPC] knowledge prep', ncols=120, leave=False)
    for _, row in prep_iter:
        prior_hint = extract_prior_hint(row, bank, bin_spec)
        query_text, regime, evidence = kb.retrieve(
            phi=float(row['phi']),
            logg=float(row['logg']),
            T=float(row['T']),
            base_pred=float(row['base_pred']),
            prior_hint=prior_hint,
            topk=args.prior_retrieval_topk,
            per_source_cap=args.prior_per_source_cap,
        )
        cache_key = build_summary_cache_key(row['row_id'], query_text, evidence)
        sample_payload = {
            'row_id': int(row['row_id']),
            'phi': float(row['phi']),
            'logg': float(row['logg']),
            'T': float(row['T']),
            'base_prediction_log_viscosity': float(row['base_pred']),
            'query': query_text,
            'regime': regime,
            'prior_hint': prior_hint,
        }
        summary_source = 'deterministic'
        llm_raw = None
        if args.summary_backend == 'deepseek':
            if not deepseek_key:
                raise ValueError('summary-backend=deepseek requires --deepseek-api-key or DEEPSEEK_API_KEY.')
            cached = summary_cache.get(cache_key)
            if cached is not None:
                summary_struct = cached['summary_struct']
                summary_source = 'deepseek_cache'
            else:
                evidence_payload = [
                    {
                        'id': e['id'],
                        'source_hint': e['source_hint'],
                        'law': e['law'],
                        'equation': e['equation'],
                        'text': e['text'][:args.prior_evidence_text_chars],
                        'topic_tags': e['topic_tags'],
                    } for e in evidence
                ]
                try:
                    raw_summary, llm_raw = call_deepseek_summary_with_retry(
                        sample_payload=sample_payload,
                        evidence=evidence_payload,
                        api_key=deepseek_key,
                        model_name=args.deepseek_model,
                        timeout_sec=args.deepseek_timeout_sec,
                        max_retries=args.deepseek_max_retries,
                        retry_sleep_sec=args.deepseek_retry_sleep_sec,
                    )
                    summary_struct = normalize_summary_struct(raw_summary, [e['id'] for e in evidence])
                    summary_source = 'deepseek_live'
                    cache_item = {
                        'cache_key': cache_key,
                        'row_id': int(row['row_id']),
                        'backend': 'deepseek',
                        'summary_struct': summary_struct,
                    }
                    summary_cache[cache_key] = cache_item
                    append_summary_cache(summary_cache_path, cache_item)
                except Exception as exc:
                    if not args.summary_allow_fallback:
                        raise
                    summary_struct = build_summary_struct(
                        phi=float(row['phi']),
                        logg=float(row['logg']),
                        T=float(row['T']),
                        base_pred=float(row['base_pred']),
                        regime=regime,
                        evidence=evidence,
                        prior_hint=prior_hint,
                    )
                    summary_source = f'deterministic_fallback:{type(exc).__name__}'
        else:
            summary_struct = build_summary_struct(
                phi=float(row['phi']),
                logg=float(row['logg']),
                T=float(row['T']),
                base_pred=float(row['base_pred']),
                regime=regime,
                evidence=evidence,
                prior_hint=prior_hint,
            )
        evidence_text = format_evidence_text(evidence, max_text_chars=args.prior_evidence_text_chars)
        summary_text = summary_struct_to_text(summary_struct)
        evidence_texts.append(evidence_text)
        summary_texts.append(summary_text)
        query_texts.append(query_text)
        summary_structs.append(summary_struct)
        summary_sources.append(summary_source)
        prior_directions.append(summary_struct.get('likely_bias_direction', 'uncertain'))
        prior_risks.append(summary_struct.get('correction_strength', 'medium'))
        evidence_ids.append(json.dumps(summary_struct.get('evidence_ids', []), ensure_ascii=False))
        retrieval_feedback.append({
            'row_id': int(row['row_id']),
            'split': str(row['Split']),
            'query_text': query_text,
            'prior_hint': prior_hint,
            'summary_source': summary_source,
            'summary_struct': summary_struct,
            'evidence': evidence,
            'llm_raw': llm_raw,
        })

    dataset['query_text'] = query_texts
    dataset['evidence_text'] = evidence_texts
    dataset['summary_text'] = summary_texts
    dataset['summary_json'] = [json.dumps(s, ensure_ascii=False) for s in summary_structs]
    dataset['summary_source'] = summary_sources
    dataset['prior_direction'] = prior_directions
    dataset['prior_risk'] = prior_risks
    dataset['evidence_ids'] = evidence_ids
    dataset.to_csv(out_dir / 'knowledge_conditioned_dataset.csv', index=False)
    with (out_dir / 'krepc_retrieval_feedback.jsonl').open('w', encoding='utf-8') as f:
        for item in retrieval_feedback:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    tokenizer = AutoTokenizer.from_pretrained(str(args.scibert_dir))
    evidence_ids_tok, evidence_mask = tokenize_texts(tokenizer, dataset['evidence_text'].tolist(), args.prior_evidence_max_length)
    summary_ids_tok, summary_mask = tokenize_texts(tokenizer, dataset['summary_text'].tolist(), args.prior_summary_max_length)
    x_num = np.stack([
        dataset['phi'].to_numpy(dtype=float),
        dataset['logg'].to_numpy(dtype=float),
        (1.0 / np.clip(dataset['T'].to_numpy(dtype=float), 1e-8, None)),
        dataset['base_pred'].to_numpy(dtype=float),
    ], axis=1)
    train_mask = dataset['Split'].astype(str).eq('TRAIN').to_numpy()
    mean = x_num[train_mask].mean(axis=0)
    std = x_num[train_mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_num = (x_num - mean) / std
    np.savez(out_dir / 'llm_correction_scaler.npz', mean=mean, std=std)
    delta_spec = build_delta_bin_spec(dataset.loc[train_mask, 'delta_true'].to_numpy(dtype=float), args.prior_delta_bins)
    dataset['delta_bin_true'] = assign_delta_bins(dataset['delta_true'].to_numpy(dtype=float), delta_spec)
    (out_dir / 'krepc_delta_bin_spec.json').write_text(json.dumps(delta_spec, indent=2), encoding='utf-8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KREPCorrectionModel(
        str(args.scibert_dir),
        hidden_dim=args.prior_hidden_dim,
        dropout=args.prior_dropout,
        text_pool=args.prior_text_pool,
        delta_bins=args.prior_delta_bins,
    )
    model.freeze_text(freeze_n_layers=args.prior_freeze_n_text_layers, freeze_embeddings=args.prior_freeze_text_embeddings)
    model.to(device)
    bin_centers_t = torch.tensor(delta_spec['centers'], dtype=torch.float32, device=device)
    bin_half_widths_t = torch.tensor(delta_spec['half_widths'], dtype=torch.float32, device=device)

    text_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (text_params if name.startswith('text_encoder.') else other_params).append(p)
    optimizer = torch.optim.AdamW([
        {'params': text_params, 'lr': args.prior_lr_text, 'weight_decay': args.prior_weight_decay},
        {'params': other_params, 'lr': args.prior_lr_other, 'weight_decay': args.prior_weight_decay},
    ])

    train_df = dataset[dataset['Split'] == 'TRAIN'].copy()
    val_df = dataset[dataset['Split'] == 'VAL'].copy()
    best = {'val_y_mae': float('inf'), 'epoch': 0, 'state': None}
    patience_left = args.prior_patience
    logs = []
    for epoch in range(1, args.prior_epochs + 1):
        model.train()
        batches = build_group_batches(train_df, args.prior_batch_size, args.seed + epoch * 31)
        pbar = tqdm(batches, desc=f'[KREPC] epoch {epoch}', ncols=120, leave=False)
        loss_sum, seen = 0.0, 0
        for batch in pbar:
            batch = np.asarray(batch, dtype=int)
            xb = torch.tensor(x_num[batch], dtype=torch.float32, device=device)
            eb = torch.tensor(evidence_ids_tok[batch], dtype=torch.long, device=device)
            em = torch.tensor(evidence_mask[batch], dtype=torch.long, device=device)
            sb = torch.tensor(summary_ids_tok[batch], dtype=torch.long, device=device)
            sm = torch.tensor(summary_mask[batch], dtype=torch.long, device=device)
            delta_true = torch.tensor(dataset.iloc[batch]['delta_true'].to_numpy(dtype=float), dtype=torch.float32, device=device)
            delta_bin_true = torch.tensor(dataset.iloc[batch]['delta_bin_true'].to_numpy(dtype=int), dtype=torch.long, device=device)
            base_pred = torch.tensor(dataset.iloc[batch]['base_pred'].to_numpy(dtype=float), dtype=torch.float32, device=device)
            phi_b = torch.tensor(dataset.iloc[batch]['phi'].to_numpy(dtype=float), dtype=torch.float32, device=device)
            logg_b = torch.tensor(dataset.iloc[batch]['logg'].to_numpy(dtype=float), dtype=torch.float32, device=device)
            t_b = torch.tensor(dataset.iloc[batch]['T'].to_numpy(dtype=float), dtype=torch.float32, device=device)
            out = model(xb, eb, em, sb, sm, bin_centers_t, bin_half_widths_t)
            delta_pred = out['delta_pred']
            y_final = base_pred + delta_pred
            y_true = base_pred + delta_true
            loss_main = F.l1_loss(y_final, y_true)
            loss_bin = F.cross_entropy(out['delta_logits'], delta_bin_true)
            true_bin_centers = bin_centers_t[delta_bin_true]
            refine_target = delta_true - true_bin_centers
            loss_ref = F.smooth_l1_loss(out['delta_refine'], refine_target)
            loss_reg = torch.mean(torch.abs(out['delta_refine']))
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
            pbar.set_postfix(L=f'{loss_sum / max(1, seen):.4f}')

        model.eval()
        with torch.no_grad():
            idx = val_df.index.to_numpy(dtype=int)
            xb = torch.tensor(x_num[idx], dtype=torch.float32, device=device)
            eb = torch.tensor(evidence_ids_tok[idx], dtype=torch.long, device=device)
            em = torch.tensor(evidence_mask[idx], dtype=torch.long, device=device)
            sb = torch.tensor(summary_ids_tok[idx], dtype=torch.long, device=device)
            sm = torch.tensor(summary_mask[idx], dtype=torch.long, device=device)
            out_val = model(xb, eb, em, sb, sm, bin_centers_t, bin_half_widths_t)
            delta_val = out_val['delta_pred'].detach().cpu().numpy()
            gate_mean = float(out_val['gate_alpha'].mean().item())
            refine_abs = float(out_val['delta_refine'].abs().mean().item())
            top_conf = float(out_val['delta_probs'].max(dim=1).values.mean().item())
        y_val = val_df['base_pred'].to_numpy(dtype=float) + delta_val
        val_mae = float(mean_absolute_error(val_df['y_true'].to_numpy(dtype=float), y_val))
        logs.append({
            'epoch': epoch,
            'train_loss': loss_sum / max(1, seen),
            'val_y_mae': val_mae,
            'val_gate_mean': gate_mean,
            'val_refine_abs_mean': refine_abs,
            'val_top_bin_conf_mean': top_conf,
        })
        if val_mae < best['val_y_mae'] - 1e-6:
            best = {'val_y_mae': val_mae, 'epoch': epoch, 'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            patience_left = args.prior_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] KREPC stop at epoch {epoch} | best val_y_mae = {best['val_y_mae']:.6f}")
                break

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    torch.save({'model_state': model.state_dict(), 'best': best}, out_dir / 'krepc_ckpt.pt')
    pd.DataFrame(logs).to_csv(out_dir / 'krepc_training_log.csv', index=False)

    frames = []
    node_rows = []
    for split_name in ['TRAIN', 'VAL', 'OOD']:
        part = dataset[dataset['Split'] == split_name].copy()
        idx = part.index.to_numpy(dtype=int)
        pred_chunks = []
        bin_chunks = []
        refine_chunks = []
        gate_chunks = []
        top_bin_chunks = []
        top_prob_chunks = []
        entropy_chunks = []
        evidence_norm_chunks = []
        summary_norm_chunks = []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(idx), args.prior_eval_batch_size):
                sub = idx[s:s + args.prior_eval_batch_size]
                xb = torch.tensor(x_num[sub], dtype=torch.float32, device=device)
                eb = torch.tensor(evidence_ids_tok[sub], dtype=torch.long, device=device)
                em = torch.tensor(evidence_mask[sub], dtype=torch.long, device=device)
                sb = torch.tensor(summary_ids_tok[sub], dtype=torch.long, device=device)
                sm = torch.tensor(summary_mask[sub], dtype=torch.long, device=device)
                out = model(xb, eb, em, sb, sm, bin_centers_t, bin_half_widths_t)
                probs = out['delta_probs']
                pred_chunks.append(out['delta_pred'].detach().cpu().numpy())
                bin_chunks.append(out['delta_bin'].detach().cpu().numpy())
                refine_chunks.append(out['delta_refine'].detach().cpu().numpy())
                gate_chunks.append(out['gate_alpha'].detach().cpu().numpy())
                top_bin_chunks.append(torch.argmax(probs, dim=1).detach().cpu().numpy())
                top_prob_chunks.append(torch.max(probs, dim=1).values.detach().cpu().numpy())
                entropy_chunks.append((-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)).detach().cpu().numpy())
                evidence_norm_chunks.append(out['evidence_norm'].detach().cpu().numpy())
                summary_norm_chunks.append(out['summary_norm'].detach().cpu().numpy())
        delta = np.concatenate(pred_chunks) if pred_chunks else np.array([], dtype=float)
        delta_bin_comp = np.concatenate(bin_chunks) if bin_chunks else np.array([], dtype=float)
        delta_refine_comp = np.concatenate(refine_chunks) if refine_chunks else np.array([], dtype=float)
        gate_alpha = np.concatenate(gate_chunks) if gate_chunks else np.array([], dtype=float)
        top_bin = np.concatenate(top_bin_chunks) if top_bin_chunks else np.array([], dtype=int)
        top_prob = np.concatenate(top_prob_chunks) if top_prob_chunks else np.array([], dtype=float)
        delta_entropy = np.concatenate(entropy_chunks) if entropy_chunks else np.array([], dtype=float)
        evidence_norm = np.concatenate(evidence_norm_chunks) if evidence_norm_chunks else np.array([], dtype=float)
        summary_norm = np.concatenate(summary_norm_chunks) if summary_norm_chunks else np.array([], dtype=float)
        part['pred'] = part['base_pred'].to_numpy(dtype=float) + delta
        part['delta_bin_component'] = delta_bin_comp
        part['delta_refine_component'] = delta_refine_comp
        part['delta_pred'] = delta
        part['gate_alpha'] = gate_alpha
        part['delta_top_bin'] = top_bin
        part['delta_top_bin_label'] = [delta_spec['labels'][int(i)] for i in top_bin.tolist()]
        part['delta_top_bin_prob'] = top_prob
        part['delta_entropy'] = delta_entropy
        part['evidence_norm'] = evidence_norm
        part['summary_norm'] = summary_norm
        part['Model'] = 'KREPC-TransRheo'
        part['Split'] = split_name
        frames.append(part)
        node_rows.append(part[[
            'Split', 'row_id', 'query_text', 'prior_direction', 'prior_risk', 'evidence_ids',
            'delta_bin_component', 'delta_refine_component', 'delta_pred', 'gate_alpha',
            'delta_top_bin', 'delta_top_bin_label', 'delta_top_bin_prob', 'delta_entropy',
            'evidence_norm', 'summary_norm'
        ]].copy())
    pred_df = pd.concat(frames, ignore_index=True)
    pred_df.to_csv(out_dir / 'krepc_predictions_all_splits.csv', index=False)
    pd.concat(node_rows, ignore_index=True).to_csv(out_dir / 'krepc_node_feedback_all_splits.csv', index=False)
    ood = pred_df[pred_df['Split'] == 'OOD'][['y_true', 'pred', 'logg']].rename(columns={'y_true': 'target'}).copy()
    plot_parity_y(ood[['target', 'pred']].copy(), out_dir / 'krepc_parity_ood.png')
    plot_residual_vs_logg(ood[['target', 'pred', 'logg']].copy(), out_dir / 'krepc_residual_ood.png')
    return pred_df


def run_traditional_models(base_df, base_y_df, out_dir):
    split_rows = base_y_df[['row_id', 'Split']].drop_duplicates().copy()
    tr = split_rows[split_rows['Split'] == 'TRAIN']['row_id'].astype(int).to_numpy()
    val = split_rows[split_rows['Split'] == 'VAL']['row_id'].astype(int).to_numpy()
    ood = split_rows[split_rows['Split'] == 'OOD']['row_id'].astype(int).to_numpy()
    trval = np.asarray(sorted(np.concatenate([tr, val]).tolist()), dtype=int)
    X = np.stack([base_df['phi'].to_numpy(dtype=float), base_df['logg'].to_numpy(dtype=float), 1.0 / np.clip(base_df['T'].to_numpy(dtype=float), 1e-12, None)], axis=1)
    y = base_df['y'].to_numpy(dtype=float)
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xs = (X - mean) / std
    rows, pred_rows = [], []
    for name, model in make_models().items():
        model.fit(Xs[trval], y[trval])
        for split_name, idx in [('TRAIN', tr), ('VAL', val), ('OOD', ood)]:
            pred = np.asarray(model.predict(Xs[idx]), dtype=float)
            sub = pd.DataFrame({
                'Split': split_name,
                'Model': name,
                'row_id': idx,
                'phi_raw': base_df.iloc[idx]['phi'].to_numpy(dtype=float),
                'T_raw': base_df.iloc[idx]['T'].to_numpy(dtype=float),
                'gamma_dot_raw': np.power(10.0, base_df.iloc[idx]['logg'].to_numpy(dtype=float)),
                'logg': base_df.iloc[idx]['logg'].to_numpy(dtype=float),
                'target': y[idx],
                'pred': pred,
            })
            pred_rows.append(sub)
            row = {'Split': split_name, 'Model': name, 'Feature': 'Optimized (phi, logg, 1/T)', 'Task': 'y'}
            row.update(calc_metrics(y[idx], pred))
            row['V_shear'] = physics_v_shear(sub[['phi_raw', 'T_raw', 'logg', 'pred']].copy())
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    summary_df.to_csv(out_dir / 'traditional_summary_metrics.csv', index=False)
    pred_df.to_csv(out_dir / 'traditional_predictions_all_splits.csv', index=False)
    return summary_df, pred_df


def save_extreme_regime_outputs(all_pred, bin_spec, out_dir):
    ood = all_pred[all_pred['Split'] == 'OOD'].copy()
    if len(ood) == 0:
        return
    phi_labels = bin_spec['phi']['labels']
    T_labels = bin_spec['T']['labels']
    logg_labels = bin_spec['logg']['labels']
    ood['phi_level'] = ood['phi_raw'].apply(lambda v: assign_level(v, bin_spec['phi']['thresholds'], phi_labels))
    ood['T_level'] = ood['T_raw'].apply(lambda v: assign_level(v, bin_spec['T']['thresholds'], T_labels))
    ood['logg_level'] = ood['logg'].apply(lambda v: assign_level(v, bin_spec['logg']['thresholds'], logg_labels))
    low_tail = float(ood['target'].quantile(0.10))
    high_tail = float(ood['target'].quantile(0.90))
    ood['regime_low_phi_low_T'] = ((ood['phi_level'] == 'low') & (ood['T_level'] == 'low')).astype(int)
    ood['regime_high_phi_highT'] = ((ood['phi_level'] == 'high') & (ood['T_level'].isin(['high', 'very_high']))).astype(int)
    ood['regime_very_low_shear'] = (ood['logg_level'] == 'very_low').astype(int)
    ood['regime_low_visc_tail'] = (ood['target'] <= low_tail).astype(int)
    ood['regime_high_visc_tail'] = (ood['target'] >= high_tail).astype(int)
    ood.to_csv(out_dir / 'ood_predictions_y_with_regimes.csv', index=False)

    regime_defs = [
        ('low_phi_low_T', ood['regime_low_phi_low_T'] == 1),
        ('high_phi_highT', ood['regime_high_phi_highT'] == 1),
        ('very_low_shear', ood['regime_very_low_shear'] == 1),
        ('low_visc_tail', ood['regime_low_visc_tail'] == 1),
        ('high_visc_tail', ood['regime_high_visc_tail'] == 1),
    ]
    rows = []
    for regime_name, mask in regime_defs:
        sub = ood[mask].copy()
        if len(sub) == 0:
            continue
        for model_name, grp in sub.groupby('Model'):
            row = {'Split': 'OOD', 'Regime': regime_name, 'Model': model_name, 'Feature': 'Optimized (phi, logg, 1/T)', 'Task': 'y'}
            row.update(calc_metrics(grp['target'], grp['pred']))
            row['V_shear'] = physics_v_shear(grp[['phi_raw', 'T_raw', 'logg', 'pred']].copy())
            rows.append(row)
    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(out_dir / 'benchmark_ood_extreme_regimes_y.csv', index=False)


def save_final_outputs(base_y_df, llm_pred_df, trad_pred_df, bin_spec, out_dir):
    base_y = base_y_df[['Split','row_id','phi_raw','T_raw','gamma_dot_raw','logg','target','pred']].copy()
    base_y['Model'] = 'TransRheo'
    llm_y = llm_pred_df[['Split','row_id','phi','T','logg','y_true','pred']].copy()
    llm_y = llm_y.rename(columns={'phi':'phi_raw','T':'T_raw','y_true':'target'})
    llm_y['gamma_dot_raw'] = np.power(10.0, llm_y['logg'].to_numpy(dtype=float))
    llm_y['Model'] = 'KREPC-TransRheo'
    llm_y = llm_y[['Split','row_id','phi_raw','T_raw','gamma_dot_raw','logg','target','pred','Model']]
    all_pred = pd.concat([base_y, llm_y, trad_pred_df], ignore_index=True)
    all_pred.to_csv(out_dir / 'all_model_predictions_y.csv', index=False)

    summary = pd.concat([
        benchmark_frame(base_y, 'TransRheo'),
        benchmark_frame(llm_y, 'KREPC-TransRheo'),
        pd.concat([benchmark_frame(trad_pred_df[trad_pred_df['Model'] == m].copy(), m) for m in trad_pred_df['Model'].unique()], ignore_index=True),
    ], ignore_index=True)
    summary.to_csv(out_dir / 'benchmark_all_models_y.csv', index=False)
    ood = summary[(summary['Split'] == 'OOD') & (summary['Task'] == 'y')].sort_values(['MAE','RMSE'])
    ood.to_csv(out_dir / 'benchmark_ood_all_models_y.csv', index=False)
    save_extreme_regime_outputs(all_pred, bin_spec, out_dir)
    print('\n[OOD y comparison]')
    print(ood.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

def parse_args():
    ap = argparse.ArgumentParser(description='Train TransRheo base + LLM prior correction + classical baselines.')
    ap.add_argument('--data-path', type=str, default='new_MDdata.xlsx')
    ap.add_argument('--sheet', type=str, default=None)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--scibert-dir', type=Path, required=True)
    ap.add_argument('--base-mode', type=str, default='train', choices=['train', 'cached'])
    ap.add_argument('--base-predictions-csv', type=Path, default=Path(r'F:\TransRheo\benchmark-3.11\100pct\transrheo_y_predictions_all_splits.csv'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ood-frac', type=float, default=0.10)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--base-epochs', type=str, default='50,50,30')
    ap.add_argument('--base-batch-size', type=int, default=96)
    ap.add_argument('--base-eval-batch-size', type=int, default=1024)
    ap.add_argument('--base-lr', type=float, default=3e-4)
    ap.add_argument('--base-weight-decay', type=float, default=0.0)
    ap.add_argument('--base-grad-clip', type=float, default=1.0)
    ap.add_argument('--base-aug-noise-frac', type=float, default=0.01)
    ap.add_argument('--base-patience', type=int, default=25)
    ap.add_argument('--base-warmup-frac', type=float, default=0.08)
    ap.add_argument('--base-min-lr-scale', type=float, default=0.15)
    ap.add_argument('--base-pin-tau', type=float, default=1e-6)
    ap.add_argument('--base-lambda-pin-s2', type=float, default=0.15)
    ap.add_argument('--base-lambda-pin-s3', type=float, default=0.25)
    ap.add_argument('--base-weight-y', type=float, default=4.0)
    ap.add_argument('--base-weight-logg', type=float, default=0.5)
    ap.add_argument('--base-weight-T', type=float, default=0.5)
    ap.add_argument('--base-weight-phi', type=float, default=0.5)
    ap.add_argument('--base-d-model', type=int, default=256)
    ap.add_argument('--base-n-heads', type=int, default=8)
    ap.add_argument('--base-n-attn-layers', type=int, default=3)
    ap.add_argument('--base-n-mlp-layers', type=int, default=4)
    ap.add_argument('--base-mlp-hidden-mult', type=float, default=2.0)
    ap.add_argument('--base-dropout', type=float, default=0.05)
    ap.add_argument('--base-tail-epochs', type=int, default=80)
    ap.add_argument('--base-tail-patience', type=int, default=20)
    ap.add_argument('--base-tail-lr', type=float, default=8e-5)
    ap.add_argument('--base-tail-batch-size', type=int, default=0)
    ap.add_argument('--base-tail-warmup-frac', type=float, default=0.05)
    ap.add_argument('--base-tail-min-lr-scale', type=float, default=0.10)
    ap.add_argument('--base-tail-weight-y', type=float, default=10.0)
    ap.add_argument('--base-tail-weight-logg', type=float, default=0.15)
    ap.add_argument('--base-tail-weight-T', type=float, default=0.15)
    ap.add_argument('--base-tail-weight-phi', type=float, default=0.15)
    ap.add_argument('--base-tail-pin-lambda', type=float, default=0.05)
    ap.add_argument('--base-tail-ramp-frac', type=float, default=0.15)
    ap.add_argument('--base-tail-aug-noise-frac', type=float, default=0.0)
    ap.add_argument('--prior-min-support-pair', type=int, default=4)
    ap.add_argument('--prior-min-support-single', type=int, default=5)
    ap.add_argument('--prior-min-mae', type=float, default=0.04)
    ap.add_argument('--prior-min-sign-strength', type=float, default=0.45)
    ap.add_argument('--prior-epochs', type=int, default=40)
    ap.add_argument('--prior-patience', type=int, default=8)
    ap.add_argument('--prior-batch-size', type=int, default=16)
    ap.add_argument('--prior-eval-batch-size', type=int, default=32)
    ap.add_argument('--prior-lr-text', type=float, default=8e-6)
    ap.add_argument('--prior-lr-other', type=float, default=2e-4)
    ap.add_argument('--prior-weight-decay', type=float, default=1e-4)
    ap.add_argument('--prior-grad-clip', type=float, default=1.0)
    ap.add_argument('--prior-hidden-dim', type=int, default=256)
    ap.add_argument('--prior-dropout', type=float, default=0.10)
    ap.add_argument('--prior-text-pool', type=str, default='cls', choices=['cls', 'mean'])
    ap.add_argument('--prior-freeze-text-embeddings', action='store_true')
    ap.add_argument('--prior-freeze-n-text-layers', type=int, default=10)
    ap.add_argument('--prior-lambda-delta', type=float, default=0.05)
    ap.add_argument('--prior-lambda-bin', type=float, default=0.20)
    ap.add_argument('--prior-lambda-ref', type=float, default=0.10)
    ap.add_argument('--prior-lambda-pin', type=float, default=0.02)
    ap.add_argument('--prior-pin-tau', type=float, default=1e-6)
    ap.add_argument('--prior-delta-bins', type=int, default=5)
    ap.add_argument('--prior-evidence-max-length', type=int, default=320)
    ap.add_argument('--prior-summary-max-length', type=int, default=192)
    ap.add_argument('--prior-retrieval-topk', type=int, default=5)
    ap.add_argument('--prior-per-source-cap', type=int, default=2)
    ap.add_argument('--prior-evidence-text-chars', type=int, default=600)
    ap.add_argument('--knowledge-dir', type=Path, default=Path(r'F:\TransRheo\Knowledge'))
    ap.add_argument('--knowledge-max-chunks', type=int, default=0)
    ap.add_argument('--summary-backend', type=str, default='deterministic', choices=['deterministic', 'deepseek'])
    ap.add_argument('--summary-allow-fallback', action='store_true')
    ap.add_argument('--deepseek-model', type=str, default='deepseek-v4-flash')
    ap.add_argument('--deepseek-api-key', type=str, default='')
    ap.add_argument('--deepseek-timeout-sec', type=int, default=90)
    ap.add_argument('--deepseek-max-retries', type=int, default=4)
    ap.add_argument('--deepseek-retry-sleep-sec', type=int, default=6)
    args = ap.parse_args()
    args.base_epochs = tuple(int(x) for x in args.base_epochs.split(','))
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.base_mode == 'train':
        print('[Stage] train base TransRheo ...')
        base_df, base_y_df, split_df = train_base_transrheo(args, args.out_dir)
    else:
        print(f'[Stage] load cached base predictions from {args.base_predictions_csv} ...')
        base_df, base_y_df, split_df = load_cached_base_predictions(args)

    print('[Stage] build bias prior bank ...')
    bin_spec = build_bin_spec(base_df)
    bank = build_bias_prior_bank(base_df, base_y_df, bin_spec, args, args.out_dir)

    print(f'[Stage] train KREPC correction head | summary_backend={args.summary_backend} ...')
    llm_pred_df = train_llm_correction(args, base_df, base_y_df, bank, bin_spec, args.out_dir)

    print('[Stage] run traditional models ...')
    _, trad_pred_df = run_traditional_models(base_df, base_y_df, args.out_dir)

    print('[Stage] final comparison ...')
    save_final_outputs(base_y_df, llm_pred_df, trad_pred_df, bin_spec, args.out_dir)
    print(f'[Done] outputs -> {args.out_dir}')


if __name__ == '__main__':
    main()
