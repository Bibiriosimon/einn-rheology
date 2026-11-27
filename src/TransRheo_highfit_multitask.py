import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


TASK_NAMES = ["y", "logg", "T", "phi"]


@dataclass
class SplitPack:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_col_name(name: str) -> str:
    s = str(name).strip().lower()
    for ch in [" ", "_", "-", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    return s


def infer_standard_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    norm = {c: normalize_col_name(c) for c in cols}

    def pick(keys: List[str]):
        for c in cols:
            name = norm[c]
            if any(k in name for k in keys):
                return c
        return None

    phi_col = pick(["volumefraction", "phi"])
    logg_col = pick(["logshearrate", "logg", "logshearrate", "loggamma"])
    t_col = pick(["temperature", "temp", "t"])
    y_col = pick(["logviscosity", "logeta", "eta", "logvis"])

    if any(v is None for v in [phi_col, logg_col, t_col, y_col]) and len(cols) >= 4:
        phi_col = phi_col or cols[0]
        logg_col = logg_col or cols[1]
        t_col = t_col or cols[2]
        y_col = y_col or cols[3]

    missing = [k for k, v in {"phi": phi_col, "logg": logg_col, "T": t_col, "y": y_col}.items() if v is None]
    if missing:
        raise ValueError(f"Cannot infer required columns {missing}. Found columns: {cols}")

    return {"phi": phi_col, "logg": logg_col, "T": t_col, "y": y_col}


def read_tabular_data(path: Path, sheet: str = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        try:
            if sheet is None:
                xl = pd.ExcelFile(path)
                chosen = None
                for s in xl.sheet_names:
                    if "pred" not in s.lower():
                        chosen = s
                        break
                sheet = chosen or xl.sheet_names[0]
            df = pd.read_excel(path, sheet_name=sheet)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel requires `openpyxl`. Install it or convert the dataset to CSV."
            ) from exc
    else:
        df = pd.read_csv(path)

    col_map = infer_standard_columns(df)
    out = pd.DataFrame(
        {
            "phi": pd.to_numeric(df[col_map["phi"]], errors="coerce"),
            "logg": pd.to_numeric(df[col_map["logg"]], errors="coerce"),
            "T": pd.to_numeric(df[col_map["T"]], errors="coerce"),
            "y": pd.to_numeric(df[col_map["y"]], errors="coerce"),
        }
    )
    out = out.dropna().reset_index(drop=True)
    return out


def build_multitask_records(base_df: pd.DataFrame):
    records = []
    for row_id, row in base_df.reset_index(drop=True).iterrows():
        phi = float(row["phi"])
        logg = float(row["logg"])
        t_raw = float(row["T"])
        y = float(row["y"])
        gamma_dot = 10.0 ** logg
        feature_t = 1.0 / max(t_raw, 1e-8)

        common = {
            "row_id": int(row_id),
            "phi_raw": phi,
            "T_raw": t_raw,
            "gamma_dot_raw": gamma_dot,
            "log10_eta_raw": y,
            "phi": phi,
            "logg": logg,
            "T_feature": feature_t,
        }
        tasks = [
            (0, "y", y),
            (1, "logg", logg),
            (2, "T", t_raw),
            (3, "phi", phi),
        ]
        for task_id, task_name, target in tasks:
            rec = dict(common)
            rec["task_id"] = int(task_id)
            rec["target_name"] = task_name
            rec["target"] = float(target)
            records.append(rec)
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_arrays(records):
    x_feat, y, task_ids, groups, row_ids = [], [], [], [], []
    group_map = {}
    for rec in records:
        key = (round(float(rec["phi_raw"]), 4), round(float(rec["T_raw"]), 4))
        if key not in group_map:
            group_map[key] = len(group_map)
        groups.append(group_map[key])
        x_feat.append([float(rec["phi"]), float(rec["logg"]), float(rec["T_feature"])])
        y.append(float(rec["target"]))
        task_ids.append(int(rec["task_id"]))
        row_ids.append(int(rec["row_id"]))
    return (
        np.asarray(x_feat, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(task_ids, dtype=int),
        np.asarray(groups, dtype=int),
        np.asarray(row_ids, dtype=int),
    )


def split_by_group(groups: np.ndarray, task_ids: np.ndarray, frac: float, seed: int, base_indices: np.ndarray = None):
    if base_indices is None:
        base_indices = np.arange(len(groups), dtype=int)
    base_indices = np.asarray(base_indices, dtype=int)
    y_idx = base_indices[task_ids[base_indices] == 0]
    if len(y_idx) == 0:
        raise ValueError("No y-task samples found for grouped split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    tr_local, te_local = next(splitter.split(y_idx, groups[y_idx], groups[y_idx]))
    tr_groups = np.unique(groups[y_idx[tr_local]])
    te_groups = np.unique(groups[y_idx[te_local]])
    train_idx = base_indices[np.isin(groups[base_indices], tr_groups)]
    test_idx = base_indices[np.isin(groups[base_indices], te_groups)]
    return SplitPack(train_idx=train_idx, test_idx=test_idx, train_groups=tr_groups, test_groups=te_groups)


def get_accuracy(y_true, y_pred, thr):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) <= thr))


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


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


def physics_check_numpy(df: pd.DataFrame, tau: float = 1e-6) -> float:
    if len(df) == 0:
        return 0.0
    work = df.copy()
    work["phi_g"] = work["phi_raw"].astype(float).round(4)
    work["T_g"] = work["T_raw"].astype(float).round(4)
    viol = 0
    total = 0
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


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(self.norm(x))


class TaskHead(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class TransRheoHighFit(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_attn_layers: int = 3,
        n_mlp_layers: int = 4,
        mlp_hidden_mult: float = 2.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.feature_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(3)])
        self.feature_embed = nn.Embedding(3, d_model)
        self.task_embed = nn.Embedding(len(TASK_NAMES), d_model)
        self.attn_blocks = nn.ModuleList(
            [AttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_attn_layers)]
        )
        mlp_hidden = int(round(d_model * mlp_hidden_mult))
        self.mlp_blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model=d_model, d_hidden=mlp_hidden, dropout=dropout) for _ in range(n_mlp_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([TaskHead(d_model=d_model, dropout=dropout) for _ in TASK_NAMES])

    def forward(self, x_num: torch.Tensor, task_id: torch.Tensor):
        batch_size = x_num.size(0)
        tokens = [self.feature_proj[i](x_num[:, i : i + 1]) for i in range(3)]
        tokens = torch.stack(tokens, dim=1)
        feat_ids = torch.arange(3, device=x_num.device).unsqueeze(0).expand(batch_size, -1)
        x = tokens + self.feature_embed(feat_ids)
        for block in self.attn_blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        h = pooled + self.task_embed(task_id.long())
        for block in self.mlp_blocks:
            h = block(h)
        h = self.out_norm(h)
        pred_all = torch.cat([head(h) for head in self.heads], dim=1)
        return torch.gather(pred_all, 1, task_id.long().unsqueeze(1)).squeeze(1)


def bucket_id_from_record(rec: Dict):
    return (round(float(rec["phi_raw"]), 2), round(float(rec["T_raw"]), 2), round(float(rec["logg"]) * 2) / 2.0)


def build_bucket_batches(indices: np.ndarray, records, batch_size: int, seed: int):
    rnd = random.Random(seed)
    buckets = {}
    for idx in np.asarray(indices, dtype=int):
        key = bucket_id_from_record(records[int(idx)])
        buckets.setdefault(key, []).append(int(idx))
    batches = []
    for idxs in buckets.values():
        rnd.shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            batches.append(idxs[start : min(start + batch_size, len(idxs))])
    rnd.shuffle(batches)
    return batches


def pin_shear_loss(pred: torch.Tensor, task_id: torch.Tensor, phi_raw: torch.Tensor, logg: torch.Tensor, t_raw: torch.Tensor, tau: float):
    mask = task_id == 0
    if int(mask.sum().item()) <= 1:
        return pred.new_zeros(())
    rows = []
    for i in torch.where(mask)[0].tolist():
        rows.append(
            (
                i,
                round(float(phi_raw[i].detach().cpu()), 4),
                round(float(t_raw[i].detach().cpu()), 4),
                float(logg[i].detach().cpu()),
            )
        )
    grouped = {}
    for idx, phi_val, t_val, logg_val in rows:
        grouped.setdefault((phi_val, t_val), []).append((idx, logg_val))
    losses = []
    for pairs in grouped.values():
        if len(pairs) < 2:
            continue
        pairs = sorted(pairs, key=lambda x: x[1])
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                left_idx, left_logg = pairs[i]
                right_idx, right_logg = pairs[j]
                if right_logg > left_logg:
                    losses.append(torch.relu(pred[right_idx] - pred[left_idx] - tau))
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def fit_scaler_on_train(x_feat: np.ndarray, task_ids: np.ndarray, train_idx: np.ndarray):
    train_idx = np.asarray(train_idx, dtype=int)
    fit_idx = train_idx[task_ids[train_idx] == 0]
    if len(fit_idx) == 0:
        fit_idx = train_idx
    scaler = StandardScaler()
    scaler.fit(x_feat[fit_idx])
    return scaler


def predict_model(model: nn.Module, x_np: np.ndarray, task_ids_np: np.ndarray, batch_size: int, device: torch.device):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            stop = min(start + batch_size, len(x_np))
            xb = torch.tensor(x_np[start:stop], dtype=torch.float32, device=device)
            tb = torch.tensor(task_ids_np[start:stop], dtype=torch.long, device=device)
            preds.append(model(xb, tb).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=float)


def evaluate_indices(model, x_scaled, y, task_ids, records, indices, batch_size, device):
    idx = np.asarray(indices, dtype=int)
    pred = predict_model(model, x_scaled[idx], task_ids[idx], batch_size=batch_size, device=device)
    rows = []
    pred_rows = []
    for tid, task_name in enumerate(TASK_NAMES):
        mask = task_ids[idx] == tid
        if int(mask.sum()) == 0:
            continue
        part_idx = idx[mask]
        y_true = y[part_idx]
        y_pred = pred[mask]
        met = calculate_metrics(y_true, y_pred)
        row = {
            "Model": "TransRheo",
            "Feature": "Optimized (phi, logg, 1/T)",
            "Task": task_name,
        }
        row.update(met)
        if task_name == "y":
            df_tmp = pd.DataFrame([records[i] for i in part_idx])
            df_tmp["pred"] = y_pred
            row["V_shear"] = physics_check_numpy(df_tmp)
        else:
            row["V_shear"] = np.nan
        rows.append(row)
        for i_local, i_global in enumerate(part_idx.tolist()):
            rec = records[i_global]
            pred_rows.append(
                {
                    "Model": "TransRheo",
                    "Feature": "Optimized (phi, logg, 1/T)",
                    "row_id": int(rec["row_id"]),
                    "split_task": task_name,
                    "task_id": int(rec["task_id"]),
                    "target_name": rec["target_name"],
                    "target": float(rec["target"]),
                    "pred": float(y_pred[i_local]),
                    "phi_raw": float(rec["phi_raw"]),
                    "T_raw": float(rec["T_raw"]),
                    "gamma_dot_raw": float(rec["gamma_dot_raw"]),
                    "logg": float(rec["logg"]),
                    "log10_eta_raw": float(rec["log10_eta_raw"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows), pred


def plot_parity_y(df_y: pd.DataFrame, out_path: Path):
    if len(df_y) == 0:
        return
    y_true = df_y["target"].to_numpy(dtype=float)
    y_pred = df_y["pred"].to_numpy(dtype=float)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.figure(figsize=(6.4, 6.0))
    plt.scatter(y_true, y_pred, s=26, alpha=0.8, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1.5, color="black")
    plt.xlabel("True log10(eta)")
    plt.ylabel("Predicted log10(eta)")
    plt.title("TransRheo OOD Parity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_vs_logg(df_y: pd.DataFrame, out_path: Path):
    if len(df_y) == 0:
        return
    res = df_y["pred"].to_numpy(dtype=float) - df_y["target"].to_numpy(dtype=float)
    plt.figure(figsize=(6.8, 4.8))
    plt.scatter(df_y["logg"].to_numpy(dtype=float), res, s=24, alpha=0.8, edgecolors="none")
    plt.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("log10(gamma_dot)")
    plt.ylabel("Residual")
    plt.title("TransRheo Residual vs log10(gamma_dot)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_lr_lambda(total_steps: int, warmup_frac: float, min_scale: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_scale + (1.0 - min_scale) * cosine)

    return lr_lambda


def run_training_phase(
    *,
    phase_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_scaled: np.ndarray,
    y: np.ndarray,
    task_ids: np.ndarray,
    records,
    device: torch.device,
    weights: torch.Tensor,
    batch_size: int,
    infer_batch_size: int,
    seed: int,
    n_epochs: int,
    grad_clip: float,
    aug_noise_frac: float,
    pin_tau: float,
    pin_lambda_fn,
    best: Dict,
    log_rows: List[Dict],
    patience: int,
):
    patience_left = patience
    global_step = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        batches = build_bucket_batches(
            train_idx,
            records=records,
            batch_size=batch_size,
            seed=seed + epoch * 97,
        )
        pbar = tqdm(batches, desc=f"[TransRheo] {phase_name} epoch {epoch}", ncols=120, leave=False)

        train_loss_sum = 0.0
        train_y_mae_sum = 0.0
        batch_seen = 0
        for batch_idx in pbar:
            batch_idx = np.asarray(batch_idx, dtype=int)
            xb = torch.tensor(x_scaled[batch_idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y[batch_idx], dtype=torch.float32, device=device)
            tb = torch.tensor(task_ids[batch_idx], dtype=torch.long, device=device)
            phi_b = torch.tensor([float(records[i]["phi_raw"]) for i in batch_idx], dtype=torch.float32, device=device)
            logg_b = torch.tensor([float(records[i]["logg"]) for i in batch_idx], dtype=torch.float32, device=device)
            t_b = torch.tensor([float(records[i]["T_raw"]) for i in batch_idx], dtype=torch.float32, device=device)

            if aug_noise_frac > 0:
                xb = xb + torch.randn_like(xb) * aug_noise_frac

            pred = model(xb, tb)
            task_losses = []
            active_weight_sum = pred.new_zeros(())
            for tid in range(len(TASK_NAMES)):
                mask = tb == tid
                if int(mask.sum().item()) == 0:
                    continue
                task_losses.append(weights[tid] * F.l1_loss(pred[mask], yb[mask]))
                active_weight_sum = active_weight_sum + weights[tid]
            loss_main = torch.stack(task_losses).sum() / torch.clamp(active_weight_sum, min=1e-8)
            lam_pin = pin_lambda_fn(global_step, epoch)
            loss_pin = pin_shear_loss(pred, tb, phi_b, logg_b, t_b, tau=pin_tau) if lam_pin > 0 else pred.new_zeros(())
            loss = loss_main + lam_pin * loss_pin

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            batch_seen += 1
            train_loss_sum += float(loss.detach().cpu().item())
            y_mask = tb == 0
            if int(y_mask.sum().item()) > 0:
                train_y_mae_sum += float(F.l1_loss(pred[y_mask].detach(), yb[y_mask]).cpu().item())
            pbar.set_postfix(
                {
                    "L": f"{train_loss_sum / max(1, batch_seen):.4f}",
                    "yMAE": f"{train_y_mae_sum / max(1, batch_seen):.4f}",
                    "pin": f"{lam_pin:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        val_metrics_df, _, _ = evaluate_indices(
            model=model,
            x_scaled=x_scaled,
            y=y,
            task_ids=task_ids,
            records=records,
            indices=val_idx,
            batch_size=infer_batch_size,
            device=device,
        )
        val_y_row = val_metrics_df[val_metrics_df["Task"] == "y"].iloc[0].to_dict()
        val_all_mae = float(val_metrics_df["MAE"].mean())
        log_rows.append(
            {
                "phase": phase_name,
                "stage": phase_name,
                "epoch": epoch,
                "train_loss": train_loss_sum / max(1, batch_seen),
                "train_y_mae": train_y_mae_sum / max(1, batch_seen),
                "val_y_mae": float(val_y_row["MAE"]),
                "val_y_r2": float(val_y_row["R2"]),
                "val_all_mae": val_all_mae,
                "lr": scheduler.get_last_lr()[0],
                "pin_lambda": pin_lambda_fn(global_step, epoch),
            }
        )

        improved = float(val_y_row["MAE"]) + 1e-8 < best["val_y_mae"]
        if improved:
            best = {
                "val_y_mae": float(val_y_row["MAE"]),
                "epoch": epoch,
                "stage": phase_name,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] stop at {phase_name} epoch {epoch} | best val_y_mae = {best['val_y_mae']:.6f}")
                break
    return best


def train(cfg: argparse.Namespace):
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    base_df = read_tabular_data(data_path, cfg.sheet)
    base_df.to_csv(out_dir / "prepared_newdata.csv", index=False)
    records = build_multitask_records(base_df)
    write_jsonl(out_dir / "triples.multitask.transrheo.jsonl", records)
    x_feat, y, task_ids, groups, row_ids = load_arrays(records)

    outer = split_by_group(groups=groups, task_ids=task_ids, frac=cfg.ood_frac, seed=cfg.seed)
    inner = split_by_group(groups=groups, task_ids=task_ids, frac=cfg.val_frac, seed=cfg.seed + 17, base_indices=outer.train_idx)

    scaler = fit_scaler_on_train(x_feat=x_feat, task_ids=task_ids, train_idx=inner.train_idx)
    x_scaled = scaler.transform(x_feat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_epochs = [int(x.strip()) for x in cfg.epochs_stage.split(",") if x.strip()]
    if len(stage_epochs) != 3 or any(v <= 0 for v in stage_epochs):
        raise ValueError("`--epochs-stage` must look like `40,40,40`.")

    model = TransRheoHighFit(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_attn_layers=cfg.n_attn_layers,
        n_mlp_layers=cfg.n_mlp_layers,
        mlp_hidden_mult=cfg.mlp_hidden_mult,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(inner.train_idx) / max(1, cfg.batch_size)))
    total_steps = steps_per_epoch * sum(stage_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(total_steps=total_steps, warmup_frac=cfg.warmup_frac, min_scale=cfg.min_lr_scale),
    )

    weights = torch.tensor([cfg.weight_y, cfg.weight_logg, cfg.weight_T, cfg.weight_phi], dtype=torch.float32, device=device)
    best = {"val_y_mae": float("inf"), "epoch": -1, "stage": "", "state": None}
    log_rows = []

    print(f"[Info] Base rows: {len(base_df)}")
    print(f"[Info] Multitask rows: {len(records)}")
    print(f"[Info] ID rows: {len(outer.train_idx)} | OOD rows: {len(outer.test_idx)}")
    print(f"[Info] Train rows: {len(inner.train_idx)} | Val rows: {len(inner.test_idx)}")
    print(f"[Info] Device: {device}")

    stage_specs = [("S1", stage_epochs[0], 0.0), ("S2", stage_epochs[1], cfg.lambda_pin_s2), ("S3", stage_epochs[2], cfg.lambda_pin_s3)]
    phase_seed = cfg.seed
    for stage_name, n_epochs, stage_target_pin in stage_specs:
        if stage_name == "S1":
            pin_lambda_fn = lambda step, epoch, _target=stage_target_pin: 0.0
        else:
            ramp_steps = max(1, int(cfg.ramp_frac * max(1, steps_per_epoch * n_epochs)))

            def pin_lambda_fn(step, epoch, _target=stage_target_pin, _ramp_steps=ramp_steps):
                progress = min(step, _ramp_steps)
                return float(_target) * (float(progress) / float(_ramp_steps))

        best = run_training_phase(
            phase_name=stage_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_idx=inner.train_idx,
            val_idx=inner.test_idx,
            x_scaled=x_scaled,
            y=y,
            task_ids=task_ids,
            records=records,
            device=device,
            weights=weights,
            batch_size=cfg.batch_size,
            infer_batch_size=cfg.infer_batch_size,
            seed=phase_seed,
            n_epochs=n_epochs,
            grad_clip=cfg.grad_clip,
            aug_noise_frac=cfg.aug_noise_frac,
            pin_tau=cfg.pin_tau,
            pin_lambda_fn=pin_lambda_fn,
            best=best,
            log_rows=log_rows,
            patience=cfg.patience,
        )
        phase_seed += 1000

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    if cfg.tail_epochs > 0:
        print(
            f"[Tail] start y-priority finetune | epochs={cfg.tail_epochs} | "
            f"lr={cfg.tail_lr:.2e} | y_weight={cfg.tail_weight_y:.2f}"
        )
        tail_weights = torch.tensor(
            [cfg.tail_weight_y, cfg.tail_weight_logg, cfg.tail_weight_T, cfg.tail_weight_phi],
            dtype=torch.float32,
            device=device,
        )
        tail_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.tail_lr, weight_decay=cfg.weight_decay)
        tail_steps = max(1, steps_per_epoch * cfg.tail_epochs)
        tail_scheduler = torch.optim.lr_scheduler.LambdaLR(
            tail_optimizer,
            lr_lambda=make_lr_lambda(total_steps=tail_steps, warmup_frac=cfg.tail_warmup_frac, min_scale=cfg.tail_min_lr_scale),
        )
        if best["state"] is not None:
            model.load_state_dict(best["state"])

        if cfg.tail_ramp_frac <= 0 or cfg.tail_pin_lambda <= 0:
            tail_pin_lambda_fn = lambda step, epoch: 0.0 if cfg.tail_pin_lambda <= 0 else float(cfg.tail_pin_lambda)
        else:
            tail_ramp_steps = max(1, int(cfg.tail_ramp_frac * tail_steps))

            def tail_pin_lambda_fn(step, epoch, _target=cfg.tail_pin_lambda, _ramp_steps=tail_ramp_steps):
                progress = min(step, _ramp_steps)
                return float(_target) * (float(progress) / float(_ramp_steps))

        best = run_training_phase(
            phase_name="TAIL",
            model=model,
            optimizer=tail_optimizer,
            scheduler=tail_scheduler,
            train_idx=inner.train_idx,
            val_idx=inner.test_idx,
            x_scaled=x_scaled,
            y=y,
            task_ids=task_ids,
            records=records,
            device=device,
            weights=tail_weights,
            batch_size=cfg.tail_batch_size if cfg.tail_batch_size > 0 else cfg.batch_size,
            infer_batch_size=cfg.infer_batch_size,
            seed=cfg.seed + 9000,
            n_epochs=cfg.tail_epochs,
            grad_clip=cfg.grad_clip,
            aug_noise_frac=cfg.tail_aug_noise_frac,
            pin_tau=cfg.pin_tau,
            pin_lambda_fn=tail_pin_lambda_fn,
            best=best,
            log_rows=log_rows,
            patience=cfg.tail_patience,
        )

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    id_metrics_df, id_pred_df, _ = evaluate_indices(
        model=model,
        x_scaled=x_scaled,
        y=y,
        task_ids=task_ids,
        records=records,
        indices=outer.train_idx,
        batch_size=cfg.infer_batch_size,
        device=device,
    )
    ood_metrics_df, ood_pred_df, _ = evaluate_indices(
        model=model,
        x_scaled=x_scaled,
        y=y,
        task_ids=task_ids,
        records=records,
        indices=outer.test_idx,
        batch_size=cfg.infer_batch_size,
        device=device,
    )

    id_metrics_df.insert(0, "Split", "ID")
    ood_metrics_df.insert(0, "Split", "OOD")
    metrics_df = pd.concat([id_metrics_df, ood_metrics_df], ignore_index=True)

    id_pred_df.insert(0, "Split", "ID")
    ood_pred_df.insert(0, "Split", "OOD")
    pred_df = pd.concat([id_pred_df, ood_pred_df], ignore_index=True)
    pred_y_ood = pred_df[(pred_df["Split"] == "OOD") & (pred_df["task_id"] == 0)].copy()

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "all_predictions.csv", index=False)
    pred_y_ood.to_csv(out_dir / "ood_predictions_y.csv", index=False)

    plot_parity_y(pred_y_ood, out_dir / "parity_transrheo.png")
    plot_residual_vs_logg(pred_y_ood, out_dir / "residual_vs_logg_transrheo.png")

    ckpt_payload = {
        "model_state": model.state_dict(),
        "config": vars(cfg),
        "model_hparams": {
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_attn_layers": cfg.n_attn_layers,
            "n_mlp_layers": cfg.n_mlp_layers,
            "mlp_hidden_mult": cfg.mlp_hidden_mult,
            "dropout": cfg.dropout,
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_val_y_mae": best["val_y_mae"],
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "train_groups": outer.train_groups.tolist(),
        "test_groups": outer.test_groups.tolist(),
    }
    torch.save(ckpt_payload, out_dir / "ckpt_best.pt")
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(cfg), "best": {k: v for k, v in best.items() if k != "state"}}, f, indent=2)

    print("\n" + "=" * 80)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("=" * 80)
    print(f"[Done] Outputs written to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="High-fit multitask TransRheo training on optimized feature space.")
    p.add_argument("--data-path", type=str, default="new_MDdata.xlsx")
    p.add_argument("--sheet", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="transrheo_highfit_outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ood-frac", type=float, default=0.10)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--infer-batch-size", type=int, default=1024)
    p.add_argument("--epochs-stage", type=str, default="50,50,30")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.08)
    p.add_argument("--min-lr-scale", type=float, default=0.15)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-attn-layers", type=int, default=3)
    p.add_argument("--n-mlp-layers", type=int, default=4)
    p.add_argument("--mlp-hidden-mult", type=float, default=2.0)
    p.add_argument("--weight-y", type=float, default=4.0)
    p.add_argument("--weight-logg", type=float, default=0.5)
    p.add_argument("--weight-T", type=float, default=0.5)
    p.add_argument("--weight-phi", type=float, default=0.5)
    p.add_argument("--lambda-pin-s2", type=float, default=0.15)
    p.add_argument("--lambda-pin-s3", type=float, default=0.25)
    p.add_argument("--ramp-frac", type=float, default=0.35)
    p.add_argument("--pin-tau", type=float, default=1e-6)
    p.add_argument("--aug-noise-frac", type=float, default=0.01)
    p.add_argument("--tail-epochs", type=int, default=80)
    p.add_argument("--tail-patience", type=int, default=20)
    p.add_argument("--tail-lr", type=float, default=8e-5)
    p.add_argument("--tail-batch-size", type=int, default=0)
    p.add_argument("--tail-warmup-frac", type=float, default=0.05)
    p.add_argument("--tail-min-lr-scale", type=float, default=0.10)
    p.add_argument("--tail-weight-y", type=float, default=10.0)
    p.add_argument("--tail-weight-logg", type=float, default=0.15)
    p.add_argument("--tail-weight-T", type=float, default=0.15)
    p.add_argument("--tail-weight-phi", type=float, default=0.15)
    p.add_argument("--tail-pin-lambda", type=float, default=0.05)
    p.add_argument("--tail-ramp-frac", type=float, default=0.15)
    p.add_argument("--tail-aug-noise-frac", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
