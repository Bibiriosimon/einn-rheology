"""Readable reconstruction of the March template correction from its archived bytecode."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from TransRheo_highfit_multitask import pin_shear_loss, plot_parity_y, plot_residual_vs_logg
from llm_transrheo_prior_correction_pipeline import (
    PriorCorrectionModel, benchmark_frame, build_group_batches, build_prior_text,
    save_extreme_regime_outputs, tokenize_texts,
)

def train_llm_correction(args, base_df, base_y_df, bank, bin_spec, out_dir):
    dataset = base_df.copy().reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=int)
    if "group_id" not in dataset.columns:
        dataset["group_id"] = dataset[["phi", "T"]].astype(str).agg("|".join, axis=1).astype("category").cat.codes
    base_part = base_y_df[["row_id", "Split", "target", "pred"]].rename(
        columns={"target": "y_true", "pred": "base_pred"})
    dataset = dataset.merge(base_part, on="row_id", how="inner")
    dataset["group_id"] = dataset["group_id"].astype(int)
    dataset["prior_text"] = dataset.apply(lambda row: build_prior_text(row, bank, bin_spec), axis=1)
    dataset["delta_true"] = dataset["y_true"] - dataset["base_pred"]
    dataset.to_csv(out_dir / "prior_conditioned_dataset.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(str(args.scibert_dir))
    ids, masks = tokenize_texts(tokenizer, dataset.prior_text.tolist(), args.prior_max_length)
    state = np.stack([dataset.phi.to_numpy(dtype=float), dataset.logg.to_numpy(dtype=float),
                      1. / np.clip(dataset["T"].to_numpy(dtype=float), 1e-8, None),
                      dataset.base_pred.to_numpy(dtype=float)], axis=1)
    train_mask = dataset.Split.astype(str).eq("TRAIN").to_numpy()
    mean = state[train_mask].mean(axis=0)
    std = state[train_mask].std(axis=0)
    std = np.where(std < 1e-8, 1., std)
    x = (state - mean) / std
    np.savez(out_dir / "llm_correction_scaler.npz", mean=mean, std=std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PriorCorrectionModel(str(args.scibert_dir), hidden_dim=args.prior_hidden_dim,
                                 dropout=args.prior_dropout, text_pool=args.prior_text_pool)
    model.freeze_text(freeze_n_layers=args.prior_freeze_n_text_layers,
                      freeze_embeddings=args.prior_freeze_text_embeddings)
    model = model.to(device)
    other, text = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (text if name.startswith("text_encoder.") else other).append(parameter)
    opt = torch.optim.AdamW([
        dict(params=text, lr=args.prior_lr_text, weight_decay=args.prior_weight_decay),
        dict(params=other, lr=args.prior_lr_other, weight_decay=args.prior_weight_decay),
    ])
    train = dataset[dataset.Split == "TRAIN"].copy()
    val = dataset[dataset.Split == "VAL"].copy()
    best = dict(val_y_mae=float("inf"), epoch=0, state=None)
    patience = args.prior_patience
    logs = []

    def num(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def forward(index):
        return model(num(x[index]), torch.tensor(ids[index], dtype=torch.long, device=device),
                     torch.tensor(masks[index], dtype=torch.long, device=device))

    for epoch in range(1, args.prior_epochs + 1):
        model.train()
        batches = build_group_batches(train, args.prior_batch_size, args.seed + epoch * 31)
        progress = tqdm(batches, desc=f"[LLM-TransRheo] epoch {epoch}", ncols=120, leave=False)
        loss_sum, seen = 0., 0
        for indices in progress:
            batch = np.asarray(indices, dtype=int)
            values = dataset.iloc[batch]
            residual = num(values.delta_true.to_numpy(dtype=float))
            base_prediction = num(values.base_pred.to_numpy(dtype=float))
            delta = forward(batch)
            prediction = base_prediction + delta
            target = base_prediction + residual
            loss_main = F.l1_loss(prediction, target)
            loss_reg = torch.mean(torch.abs(delta))
            pin = pin_shear_loss(prediction, torch.zeros_like(prediction, dtype=torch.long),
                                 num(values.phi.to_numpy(dtype=float)), num(values.logg.to_numpy(dtype=float)),
                                 num(values["T"].to_numpy(dtype=float)), tau=args.prior_pin_tau)
            loss = loss_main + args.prior_lambda_delta * loss_reg + args.prior_lambda_pin * pin
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.prior_grad_clip)
            opt.step()
            seen += len(batch)
            loss_sum += float(loss.item()) * len(batch)
            progress.set_postfix(L=f"{loss_sum / max(1, seen):.4f}")

        model.eval()
        with torch.no_grad():
            delta_val = forward(val.index.to_numpy(dtype=int)).detach().cpu().numpy()
        validation = val.base_pred.to_numpy(dtype=float) + delta_val
        val_mae = float(mean_absolute_error(val.y_true.to_numpy(dtype=float), validation))
        logs.append(dict(epoch=epoch, train_loss=loss_sum / max(1, seen), val_y_mae=val_mae))
        if val_mae < best["val_y_mae"] - 1e-6:
            # Retain the archived CUDA-to-CPU snapshot semantics for faithful CPU/CUDA reruns.
            best = dict(val_y_mae=val_mae, epoch=epoch,
                        state={key: tensor.detach().cpu() for key, tensor in model.state_dict().items()})
            patience = args.prior_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[EarlyStop] LLM-TransRheo stop at epoch {epoch} | best val_y_mae = {best['val_y_mae']:.6f}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save({"model_state": model.state_dict(), "best": best}, out_dir / "llm_transrheo_ckpt.pt")
    pd.DataFrame(logs).to_csv(out_dir / "llm_transrheo_training_log.csv", index=False)

    frames = []
    for split in ("TRAIN", "VAL", "OOD"):
        part = dataset[dataset.Split == split].copy()
        index = part.index.to_numpy(dtype=int)
        model.eval()
        with torch.no_grad():
            predictions = [
                forward(index[start:start + args.prior_eval_batch_size]).detach().cpu().numpy()
                for start in range(0, len(index), args.prior_eval_batch_size)
            ]
        delta = np.concatenate(predictions) if predictions else np.array([], dtype=float)
        part["pred"] = part.base_pred.to_numpy(dtype=float) + delta
        part["delta_pred"] = delta
        part["Model"] = "LLM-TransRheo"
        part["Split"] = split
        frames.append(part)
    predicted = pd.concat(frames, ignore_index=True)
    predicted.to_csv(out_dir / "llm_transrheo_predictions_all_splits.csv", index=False)
    ood = predicted[predicted.Split == "OOD"][["y_true", "pred", "logg"]].rename(
        columns={"y_true": "target"}).copy()
    plot_parity_y(ood[["target", "pred"]].copy(), out_dir / "llm_transrheo_parity_ood.png")
    plot_residual_vs_logg(ood[["target", "pred", "logg"]].copy(),
                          out_dir / "llm_transrheo_residual_ood.png")
    return predicted

def save_final_outputs(base_y_df, llm_pred_df, trad_pred_df, bin_spec, out_dir):
    columns = ["Split", "row_id", "phi_raw", "T_raw", "gamma_dot_raw", "logg", "target", "pred"]
    base = base_y_df[columns].copy()
    base["Model"] = "TransRheo"
    adjusted = llm_pred_df[["Split", "row_id", "phi", "T", "logg", "y_true", "pred"]].rename(
        columns={"phi": "phi_raw", "T": "T_raw", "y_true": "target"})
    adjusted["gamma_dot_raw"] = np.power(10., adjusted.logg.to_numpy(dtype=float))
    adjusted["Model"] = "LLM-TransRheo"
    all_predictions = pd.concat([base, adjusted[columns + ["Model"]], trad_pred_df], ignore_index=True)
    all_predictions.to_csv(out_dir / "all_model_predictions_y.csv", index=False)
    summaries = [benchmark_frame(table.copy(), name)
                 for table in (base, adjusted, trad_pred_df) for name, table in table.groupby("Model", sort=False)]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "benchmark_all_models_y.csv", index=False)
    ood = summary[(summary.Split == "OOD") & (summary.Task == "y")].sort_values(["MAE", "RMSE"])
    ood.to_csv(out_dir / "benchmark_ood_all_models_y.csv", index=False)
    save_extreme_regime_outputs(all_predictions, bin_spec, out_dir)
    print("\n[OOD y comparison]")
    print(ood.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
