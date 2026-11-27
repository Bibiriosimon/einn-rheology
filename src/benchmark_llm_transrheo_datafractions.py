
#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

from llm_transrheo_prior_correction_pipeline import (
    benchmark_frame,
    build_bias_prior_bank,
    build_bin_spec,
    run_traditional_models,
    save_extreme_regime_outputs,
    save_final_outputs,
    train_llm_correction,
)
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


def select_fraction_groups(groups, task_ids, base_indices, frac, seed):
    base_indices = np.asarray(base_indices, dtype=int)
    y_idx = base_indices[task_ids[base_indices] == 0]
    if len(y_idx) == 0:
        raise ValueError('No y-task records in base indices.')
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


class NamespaceProxy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_base_transrheo_fraction(args, out_dir, base_df, records, x_feat, y, task_ids, groups, row_ids, outer_ood_idx, selected_id_idx):
    inner = split_by_group(groups, task_ids, frac=args.val_frac, seed=args.seed + 17, base_indices=selected_id_idx)
    train_idx, val_idx, ood_idx = inner.train_idx, inner.test_idx, outer_ood_idx
    selected_groups = np.unique(groups[selected_id_idx])
    id_holdout_idx = np.setdiff1d(
        np.where(np.isin(groups, np.unique(groups[task_ids == 0]))) [0],
        np.concatenate([selected_id_idx, ood_idx]),
        assume_unique=False,
    )

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
    eval_plan = [('TRAIN', train_idx), ('VAL', val_idx), ('ID_HOLDOUT', id_holdout_idx), ('OOD', ood_idx)]
    for split_name, idx in eval_plan:
        if len(idx) == 0:
            continue
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
    ood_plot = y_pred_df[y_pred_df['Split'] == 'OOD'][['target', 'pred']].copy()
    plot_parity_y(ood_plot, out_dir / 'transrheo_parity_ood.png')
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
    subset_meta = {
        'train_record_count': int(len(train_idx)),
        'val_record_count': int(len(val_idx)),
        'ood_record_count': int(len(ood_idx)),
        'id_holdout_record_count': int(len(id_holdout_idx)),
        'train_group_count': int(len(np.unique(groups[train_idx]))),
        'val_group_count': int(len(np.unique(groups[val_idx]))),
        'ood_group_count': int(len(np.unique(groups[ood_idx]))),
        'selected_id_group_count': int(len(selected_groups)),
    }
    (out_dir / 'subset_meta.json').write_text(json.dumps(subset_meta, indent=2), encoding='utf-8')
    return y_pred_df, split_df, subset_meta


def add_fraction_columns(df, frac_label, frac_value):
    out = df.copy()
    out['TrainFrac'] = frac_value
    out['TrainFracLabel'] = frac_label
    return out


def cleanup_nonbenchmark_outputs(frac_dir: Path):
    keep_names = {
        'benchmark_all_models_y.csv',
        'benchmark_ood_all_models_y.csv',
        'benchmark_ood_extreme_regimes_y.csv',
        'subset_meta.json',
        'split_indices.csv',
        'bias_prior_bank.csv',
        'prior_bank_summary.json',
        'transrheo_summary_metrics.csv',
        'llm_transrheo_summary_metrics.csv',
        'transrheo_training_log.csv',
        'llm_transrheo_training_log.csv',
    }
    for path in frac_dir.iterdir():
        if path.name in keep_names:
            continue
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def parse_args():
    ap = argparse.ArgumentParser(description='Benchmark TransRheo / LLM-TransRheo / classical models across train fractions.')
    ap.add_argument('--data-path', type=str, default='new_MDdata.xlsx')
    ap.add_argument('--sheet', type=str, default=None)
    ap.add_argument('--scibert-dir', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ood-frac', type=float, default=0.10)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--fractions', type=str, default='0.25,0.50,0.75,1.00')
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
    ap.add_argument('--prior-max-length', type=int, default=192)
    ap.add_argument('--prior-text-pool', type=str, default='cls', choices=['cls', 'mean'])
    ap.add_argument('--prior-freeze-text-embeddings', action='store_true')
    ap.add_argument('--prior-freeze-n-text-layers', type=int, default=10)
    ap.add_argument('--prior-lambda-delta', type=float, default=0.05)
    ap.add_argument('--prior-lambda-pin', type=float, default=0.02)
    ap.add_argument('--prior-pin-tau', type=float, default=1e-6)
    ap.add_argument('--benchmark-only', action='store_true', help='Only keep benchmark outputs; remove prediction/parity artifacts.')
    args = ap.parse_args()
    args.base_epochs = tuple(int(x) for x in args.base_epochs.split(','))
    args.fractions = [float(x) for x in args.fractions.split(',') if x.strip()]
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out_root = args.out_root.resolve()
    args.scibert_dir = Path(args.scibert_dir).resolve()
    args.data_path = str(Path(args.data_path).resolve())
    args.scibert_dir = args.scibert_dir.as_posix()
    args.out_root.mkdir(parents=True, exist_ok=True)

    print('[Stage] load data and build fixed OOD split ...')
    base_df = read_tabular_data(Path(args.data_path), args.sheet)
    records = build_multitask_records(base_df)
    x_feat, y, task_ids, groups, row_ids = load_arrays(records)
    outer = split_by_group(groups, task_ids, frac=args.ood_frac, seed=args.seed)
    id_base_idx = outer.train_idx
    ood_idx = outer.test_idx
    full_bin_spec = build_bin_spec(base_df)

    all_bench = []
    all_bench_ood = []
    all_extreme = []
    subset_rows = []

    for i, frac in enumerate(args.fractions):
        frac_pct = int(round(frac * 100))
        frac_label = f'{frac_pct}pct'
        frac_dir = args.out_root / frac_label
        frac_dir.mkdir(parents=True, exist_ok=True)
        print(f'[Stage] fraction {frac_label}: select ID groups ...')
        chosen_groups, selected_id_idx = select_fraction_groups(groups, task_ids, id_base_idx, frac, args.seed + 1000 + i)

        base_y_df, split_df, subset_meta = train_base_transrheo_fraction(
            args=args,
            out_dir=frac_dir,
            base_df=base_df,
            records=records,
            x_feat=x_feat,
            y=y,
            task_ids=task_ids,
            groups=groups,
            row_ids=row_ids,
            outer_ood_idx=ood_idx,
            selected_id_idx=selected_id_idx,
        )

        bank = build_bias_prior_bank(base_df, base_y_df, full_bin_spec, args, frac_dir)
        llm_pred_df = train_llm_correction(args, base_df, base_y_df, bank, full_bin_spec, frac_dir)
        trad_summary_df, trad_pred_df = run_traditional_models(base_df, base_y_df, frac_dir)
        save_final_outputs(base_y_df, llm_pred_df, trad_pred_df, full_bin_spec, frac_dir)

        bench_all = pd.read_csv(frac_dir / 'benchmark_all_models_y.csv')
        bench_ood = pd.read_csv(frac_dir / 'benchmark_ood_all_models_y.csv')
        extreme_path = frac_dir / 'benchmark_ood_extreme_regimes_y.csv'
        extreme_df = pd.read_csv(extreme_path) if extreme_path.exists() and extreme_path.stat().st_size > 0 else pd.DataFrame()

        bench_all = add_fraction_columns(bench_all, frac_label, frac)
        bench_ood = add_fraction_columns(bench_ood, frac_label, frac)
        if len(extreme_df) > 0:
            extreme_df = add_fraction_columns(extreme_df, frac_label, frac)
            all_extreme.append(extreme_df)

        all_bench.append(bench_all)
        all_bench_ood.append(bench_ood)
        subset_rows.append({
            'TrainFrac': frac,
            'TrainFracLabel': frac_label,
            'SelectedIDGroupCount': int(len(chosen_groups)),
            **subset_meta,
        })
        if args.benchmark_only:
            cleanup_nonbenchmark_outputs(frac_dir)

    bench_all_df = pd.concat(all_bench, ignore_index=True)
    bench_ood_df = pd.concat(all_bench_ood, ignore_index=True)
    bench_all_df.to_csv(args.out_root / 'benchmark_all_models_all_fractions.csv', index=False)
    bench_ood_df.to_csv(args.out_root / 'benchmark_ood_all_models_all_fractions.csv', index=False)
    if not args.benchmark_only:
        preds_all_df = pd.concat(
            [
                add_fraction_columns(pd.read_csv(args.out_root / f'{int(round(frac * 100))}pct' / 'all_model_predictions_y.csv'),
                                     f'{int(round(frac * 100))}pct', frac)
                for frac in args.fractions
            ],
            ignore_index=True,
        )
        preds_all_df.to_csv(args.out_root / 'all_model_predictions_all_fractions.csv', index=False)
    pd.DataFrame(subset_rows).to_csv(args.out_root / 'subset_fraction_summary.csv', index=False)
    if all_extreme:
        pd.concat(all_extreme, ignore_index=True).to_csv(args.out_root / 'benchmark_ood_extreme_regimes_all_fractions.csv', index=False)

    print('\n[Final OOD benchmark across fractions]')
    print(bench_ood_df.sort_values(['TrainFrac', 'MAE', 'RMSE']).to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'[Done] outputs -> {args.out_root}')


if __name__ == '__main__':
    main()
