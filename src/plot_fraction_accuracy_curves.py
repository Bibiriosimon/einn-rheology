
#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Plot Acc10/Acc20 vs train fraction for multiple models.')
    ap.add_argument('--benchmark-csv', type=Path, required=True)
    ap.add_argument('--out-png', type=Path, required=True)
    ap.add_argument('--out-csv', type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.benchmark_csv)
    order = ['25pct', '50pct', '75pct', '100pct']
    model_order = ['LLM-TransRheo', 'TransRheo', 'SVR', 'XGBoost', 'RF', 'LR']
    colors = {
        'LLM-TransRheo': '#0B6E4F',
        'TransRheo': '#1D4ED8',
        'SVR': '#D97706',
        'XGBoost': '#B91C1C',
        'RF': '#6D28D9',
        'LR': '#4B5563',
    }
    markers = {
        'LLM-TransRheo': 'o',
        'TransRheo': 's',
        'SVR': '^',
        'XGBoost': 'D',
        'RF': 'P',
        'LR': 'X',
    }

    df = df[df['Split'].eq('OOD') & df['Task'].eq('y')].copy()
    df['TrainFracLabel'] = pd.Categorical(df['TrainFracLabel'], categories=order, ordered=True)
    df = df.sort_values(['TrainFracLabel', 'Model'])

    if args.out_csv is not None:
        export = df[['TrainFrac', 'TrainFracLabel', 'Model', 'Acc10', 'Acc20']].copy()
        export.to_csv(args.out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    metric_specs = [('Acc10', 'Accuracy @ 0.10'), ('Acc20', 'Accuracy @ 0.20')]

    for ax, (metric, title) in zip(axes, metric_specs):
        for model in model_order:
            sub = df[df['Model'].eq(model)].sort_values('TrainFracLabel')
            if sub.empty:
                continue
            ax.plot(
                sub['TrainFracLabel'].astype(str),
                sub[metric],
                label=model,
                color=colors.get(model, None),
                marker=markers.get(model, 'o'),
                linewidth=2.2,
                markersize=6.5,
            )
        ax.set_title(title)
        ax.set_xlabel('Training Data Fraction')
        ax.set_ylabel('OOD Accuracy')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('OOD Accuracy vs Training Data Fraction')
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=320, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {args.out_png}')
    if args.out_csv is not None:
        print(f'[Saved] {args.out_csv}')


if __name__ == '__main__':
    main()
