#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = [
    'LLM-TransRheo',
    'TransRheo',
    'XGBoost',
    'SVR',
    'RF',
    'LR',
]

DISPLAY_NAME = {
    'LLM-TransRheo': 'LLM-TransRheo',
    'TransRheo': 'TransRheo',
    'XGBoost': 'XGBoost',
    'SVR': 'SVR',
    'RF': 'RF',
    'LR': 'LR',
}

COLORS = {
    'LLM-TransRheo': '#C84C31',
    'TransRheo': '#2B6CB0',
    'XGBoost': '#2F855A',
    'SVR': '#D69E2E',
    'RF': '#805AD5',
    'LR': '#4A5568',
}


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    acc10 = np.mean(np.abs(y_true - y_pred) <= 0.10)
    acc20 = np.mean(np.abs(y_true - y_pred) <= 0.20)
    return r2, rmse, mae, acc10, acc20


def model_limits(y_true, y_pred, pad_ratio=0.08):
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    span = max(1e-6, hi - lo)
    pad = span * pad_ratio
    return lo - pad, hi + pad


def main():
    ap = argparse.ArgumentParser(description='Plot 2x3 parity figure for six models.')
    ap.add_argument('--pred-csv', type=Path, required=True)
    ap.add_argument('--out-png', type=Path, required=True)
    ap.add_argument('--fraction-label', type=str, default='100pct')
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    df = df[(df['Split'] == 'OOD') & (df['TrainFracLabel'] == args.fraction_label)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, model in zip(axes, MODEL_ORDER):
        sub = df[df['Model'] == model].copy()
        if sub.empty:
            ax.axis('off')
            continue
        y_true = sub['target'].to_numpy(dtype=float)
        y_pred = sub['pred'].to_numpy(dtype=float)
        r2, rmse, mae, acc10, acc20 = calc_metrics(y_true, y_pred)
        lo, hi = model_limits(y_true, y_pred)

        ax.scatter(y_true, y_pred, s=24, alpha=0.85, color=COLORS[model], edgecolors='none')
        ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2, color='black', alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(DISPLAY_NAME[model], fontsize=13)
        ax.set_xlabel(r'True $\log_{10}(\eta)$')
        ax.set_ylabel(r'Pred $\log_{10}(\eta)$')
        ax.grid(True, linestyle=':', alpha=0.45)
        txt = (
            f'R2={r2:.3f}\n'
            f'RMSE={rmse:.3f}\n'
            f'MAE={mae:.3f}\n'
            f'Acc10={acc10:.3f}'
        )
        ax.text(
            0.04, 0.96, txt,
            transform=ax.transAxes,
            ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#999999', alpha=0.9)
        )

    fig.suptitle(f'OOD Parity Comparison ({args.fraction_label})', fontsize=16)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220)
    plt.close(fig)
    print(f'[Saved] {args.out_png}')


if __name__ == '__main__':
    main()
