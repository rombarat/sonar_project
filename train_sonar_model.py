"""
Dual-Input Sonar Classifier — Training & Evaluation Pipeline
============================================================

Goal
----
Binary classification: is there a quiet target masked by the loud one?
    label = 1  ⇔  num_targets == 2   (positive)
    label = 0  ⇔  num_targets == 1   (negative)

Inputs (per sample, from ``synth_dataset_1k/psd/sample_XXXXXX.npz``)
    psd_1d : (F,)        — Welch PSD at the focus steer angle
    psd_2d : (A, F)      — Welch PSD swept across all scan angles

Architecture
    Branch A — 1D-CNN over psd_1d    → 1024-d feature vector
    Branch B — 2D-CNN  +  GAP over psd_2d → 256-d feature vector
    Fusion   — concat → 1280 → 256 → 64 → 1 logit (BCEWithLogitsLoss)

The whole pipeline lives in one file but is split into clearly labelled
classes/functions: TrainConfig · SonarDualDataset · DualBranchNet ·
train_one_epoch / evaluate / fit / final_report.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


# =============================================================================
# CENTRAL CONFIGURATION (edit here — never hard-code below)
# =============================================================================

@dataclass
class TrainConfig:
    # -------- data --------
    dataset_dir: Path = Path("synth_dataset_1k")
    labels_csv:  str  = "labels.csv"

    # -------- split / loading --------
    val_fraction:    float = 0.20
    random_seed:     int   = 42
    batch_size:      int   = 32
    num_workers:     int   = 0          # Windows + spawn: keep 0 for safety
    pin_memory:      bool  = False      # turn on with CUDA

    # -------- model --------
    dropout:         float = 0.30

    # -------- optim --------
    epochs:          int   = 25
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    lr_factor:       float = 0.5
    lr_patience:     int   = 3
    grad_clip:       Optional[float] = 1.0   # set None to disable

    # -------- I/O --------
    out_dir:         Path  = Path("runs/dualnet_v1")
    checkpoint_name: str   = "best_model.pth"

    def __post_init__(self):
        # Be forgiving: callers can pass strings, we'll coerce to Path.
        self.dataset_dir = Path(self.dataset_dir)
        self.out_dir     = Path(self.out_dir)


# =============================================================================
# Device selection (CUDA → MPS → CPU)
# =============================================================================

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Dataset
# =============================================================================
#
# Normalization choice — Global Fixed-Range dB → [0, 1]
# -----------------------------------------------------
# The raw PSDs are linear power (V²/Hz). Per-sample Z-score was tried first
# but it leaked the answer: it stretches every sample to unit variance, so
# a quiet target buried 25 dB below the loud one ends up with the same
# rendered contrast as a 0-dB-SIR target. That destroys the very absolute-
# amplitude difference the classifier must learn from.
#
# The fix: a *global, fixed-range* dB → [0,1] map, identical for every
# sample, so the network sees the true absolute brightness of each ridge.
#
#   1. dB:    y = 10 * log10(max(x, 1e-12))
#   2. Map:   y = (y - GLOBAL_MIN_DB) / (GLOBAL_MAX_DB - GLOBAL_MIN_DB)
#   3. Clip:  y = clamp(y, 0.0, 1.0)
#
# GLOBAL_MIN_DB / GLOBAL_MAX_DB were chosen by surveying the actual dB
# range across 100 random .npz files in synth_dataset_1k and
# synth_dataset_1k_hard:
#       psd_1d / psd_2d min : ≈ −100 dB (deep beam nulls)
#       psd_1d / psd_2d max : ≈ −15  dB (loud target peak)
# A min of −100 dB and a max of −10 dB uses essentially the entire [0,1]
# range without wasting headroom on dB values that never occur in our data.
# If you regenerate the dataset with very different SNR/SIR settings, rerun
# the dB-range survey and retune these two constants.

GLOBAL_MIN_DB: float = -100.0
GLOBAL_MAX_DB: float =  -10.0


def log_minmax_fixed(x: np.ndarray,
                     min_db: float = GLOBAL_MIN_DB,
                     max_db: float = GLOBAL_MAX_DB) -> np.ndarray:
    """
    Global fixed-range dB → [0, 1] normalization (returns float32).
    Identical across every sample, so absolute amplitude differences
    between loud and quiet targets are preserved.
    """
    y = 10.0 * np.log10(np.maximum(x, 1e-12)).astype(np.float32)
    y = (y - np.float32(min_db)) / np.float32(max_db - min_db)
    return np.clip(y, 0.0, 1.0)


class SonarDualDataset(Dataset):
    """
    Reads ``labels.csv``, opens the corresponding ``.npz`` per index, and
    returns ``(psd_1d_tensor, psd_2d_tensor, label_tensor)`` ready for a
    dual-input model.

    Shapes returned by __getitem__:
        psd_1d : torch.float32  (1, F)            — 1 channel
        psd_2d : torch.float32  (1, A, F)         — 1 channel
        label  : torch.float32  ()                — scalar 0./1.

    Notes
    -----
    * Per-sample log10 + Z-score normalization is applied here so the
      DataLoader's worker processes share the CPU work. The simulator's
      raw .npz files are left untouched.
    * The dataset does NOT cache decoded arrays in memory — each __getitem__
      hits disk. With 1k samples this is fast; for 100k+ samples consider
      preloading psd_1d into RAM (cheap) and only streaming psd_2d.
    """

    def __init__(self, dataset_dir: Path | str, labels_csv: str = "labels.csv"):
        self.root = Path(dataset_dir)
        labels_path = self.root / labels_csv
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found at {labels_path}. "
                f"Run sonar_simulator.py first to populate {self.root}/."
            )

        df = pd.read_csv(labels_path)
        # Binary label per the project spec: 2 targets → positive, else negative
        df["label"] = (df["num_targets"].astype(int) == 2).astype(np.float32)
        self.df = df.reset_index(drop=True)

        # Cache simple shapes from sample 0 for sanity logging.
        with np.load(self.root / self.df.iloc[0]["psd_path"]) as z0:
            self.f_bins   = int(z0["psd_1d"].shape[0])
            self.n_angles = int(z0["psd_2d"].shape[0])

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        """Full label array — used for stratified train/val splitting."""
        return self.df["label"].to_numpy().astype(np.int64)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with np.load(self.root / row["psd_path"]) as z:
            psd1 = log_minmax_fixed(z["psd_1d"])   # (F,)   in [0,1]
            psd2 = log_minmax_fixed(z["psd_2d"])   # (A, F) in [0,1]

        x1 = torch.from_numpy(psd1).unsqueeze(0)            # (1, F)
        x2 = torch.from_numpy(psd2).unsqueeze(0)            # (1, A, F)
        # Belt-and-suspenders: ensure the strict [0, 1] guarantee makes
        # it past any future dtype/round-off issues.
        x1 = torch.clamp(x1, 0.0, 1.0)
        x2 = torch.clamp(x2, 0.0, 1.0)
        y  = torch.tensor(row["label"], dtype=torch.float32)
        return x1, x2, y


# =============================================================================
# Model — Dual-Branch CNN with concatenation fusion
# =============================================================================

class Branch1D(nn.Module):
    """
    1D CNN over the focus PSD. Small kernels chase narrow tonal spikes;
    progressive downsampling buys receptive field without blowing up FLOPs.

    Input  : (B, 1, F)
    Output : (B, feat_1d)   where feat_1d = 128 * 8 = 1024 by default
    """

    def __init__(self, out_pool: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32,  64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(out_pool),                 # fixed feature size
        )
        self.out_features = 128 * out_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)                  # (B, 1024)


class Branch2D(nn.Module):
    """
    Small VGG-style 2D CNN with Global Average Pooling.  Treats the
    angle×frequency PSD as a 1-channel image and learns spatial-spectral
    features (sidelobe pattern, second-target ridge, etc.).

    Input  : (B, 1, A, F)
    Output : (B, 256)        after GAP
    """

    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(1,    32),
            block(32,   64),
            block(64,  128),
            block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_features = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)                  # (B, 256)


class DualBranchNet(nn.Module):
    """
    Concatenation-fusion head: [branch1d_feat | branch2d_feat] → MLP → 1 logit.

    Returns a raw logit; pair with ``nn.BCEWithLogitsLoss`` for numerical
    stability. Apply ``torch.sigmoid`` only at inference / metric time.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.branch_1d = Branch1D()
        self.branch_2d = Branch2D()
        fused_dim = self.branch_1d.out_features + self.branch_2d.out_features
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256,        64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64,          1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.branch_1d(x1)
        f2 = self.branch_2d(x2)
        return self.head(torch.cat([f1, f2], dim=1)).squeeze(-1)  # (B,)


# =============================================================================
# Training engine
# =============================================================================

def make_loaders(ds: SonarDualDataset, cfg: TrainConfig
                 ) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Stratified 80/20 train/val split on the binary label.  Returns the two
    DataLoaders plus the train/val index arrays (useful for later analysis).
    """
    idx = np.arange(len(ds))
    train_idx, val_idx = train_test_split(
        idx, test_size=cfg.val_fraction,
        random_state=cfg.random_seed,
        stratify=ds.labels,
    )
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False)
    val_loader = DataLoader(
        Subset(ds, val_idx),   batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False)
    return train_loader, val_loader, train_idx, val_idx


def _run_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module, device: torch.device,
               optimizer: Optional[torch.optim.Optimizer] = None,
               grad_clip: Optional[float] = None,
               ) -> tuple[float, float]:
    """Single pass over `loader`. If optimizer is None → eval mode + no_grad."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n    = 0
    correct    = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x1, x2, y in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = y.to(device,  non_blocking=True)

            logits = model(x1, x2)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs           = y.size(0)
            total_loss  += loss.item() * bs
            total_n     += bs
            preds        = (torch.sigmoid(logits) >= 0.5).float()
            correct     += (preds == y).sum().item()

    return total_loss / total_n, correct / total_n


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        cfg: TrainConfig, device: torch.device) -> dict:
    """
    Full training loop with AdamW + ReduceLROnPlateau on val loss.
    Saves the best checkpoint (lowest val loss) to ``cfg.out_dir/cfg.checkpoint_name``.
    Returns a history dict for plotting / logging.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.out_dir / cfg.checkpoint_name

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor,
        patience=cfg.lr_patience)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device,
                                     optimizer=optimizer, grad_clip=cfg.grad_clip)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        dt = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   va_loss,
                "val_acc":    va_acc,
                "config":     cfg.__dict__,
            }, ckpt_path)

        lr_now = optimizer.param_groups[0]["lr"]
        flag   = "  *best*" if is_best else ""
        print(f"epoch {epoch:3d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
              f"lr {lr_now:.2e} | {dt:5.1f}s{flag}")

    print(f"\nbest val loss = {best_val:.4f}   checkpoint -> {ckpt_path}")
    return history


# =============================================================================
# Evaluation & metrics
# =============================================================================

@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader,
                        device: torch.device, threshold: float = 0.5,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_prob, y_pred) over the full loader."""
    model.eval()
    ys, ps = [], []
    for x1, x2, y in loader:
        x1 = x1.to(device); x2 = x2.to(device)
        logits = model(x1, x2)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return y_true.astype(np.int64), y_prob, y_pred


def final_report(model: nn.Module, val_loader: DataLoader,
                 device: torch.device, out_dir: Path,
                 threshold: float = 0.5) -> dict:
    """
    Compute Accuracy / Precision / Recall / F1 on the validation set and
    save a confusion-matrix PNG. Recall is highlighted because a missed
    quiet vessel is the costly failure mode for this task.
    """
    y_true, y_prob, y_pred = collect_predictions(model, val_loader, device, threshold)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n========== Validation report ==========")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}   <-- critical: detection of quiet vessel")
    print(f"  F1        : {f1:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"              pred-Neg  pred-Pos")
    print(f"  true-Neg :   {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  true-Pos :   {cm[1,0]:5d}    {cm[1,1]:5d}")

    # Save confusion matrix figure
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative (1-tgt)", "Positive (2-tgt)"],
                yticklabels=["Negative (1-tgt)", "Positive (2-tgt)"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  —  Acc={acc:.3f}  Prec={p:.3f}  Rec={r:.3f}")
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"\n  saved -> {cm_path}")

    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "confusion_matrix": cm, "y_true": y_true, "y_prob": y_prob,
            "y_pred": y_pred}


def plot_history(history: dict, out_dir: Path) -> None:
    """Loss & accuracy curves over epochs → PNG."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("Loss")
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Accuracy")
    ax2.grid(alpha=0.3); ax2.legend()
    fig.savefig(out_dir / "history.png", dpi=120)
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    # Windows console (cp1252) chokes on Unicode in print() — force UTF-8.
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # =========================================================================
    # >>>   USER-EDITABLE PARAMETERS   <<<
    # =========================================================================
    CFG = TrainConfig(
        dataset_dir = Path(__file__).resolve().parent / "synth_dataset_1k",
        epochs      = 25,
        batch_size  = 32,
        lr          = 1e-3,
        dropout     = 0.30,
        random_seed = 42,
        out_dir     = Path(__file__).resolve().parent / "runs" / "dualnet_v1",
    )
    # =========================================================================

    device = pick_device()
    print(f"device         : {device}")
    print(f"dataset dir    : {CFG.dataset_dir}")

    ds = SonarDualDataset(CFG.dataset_dir, CFG.labels_csv)
    pos = int((ds.labels == 1).sum()); neg = int((ds.labels == 0).sum())
    print(f"samples        : {len(ds)}  (pos={pos}, neg={neg})")
    print(f"psd_1d bins    : {ds.f_bins}")
    print(f"psd_2d shape   : ({ds.n_angles}, {ds.f_bins})")

    train_loader, val_loader, train_idx, val_idx = make_loaders(ds, CFG)
    print(f"train / val    : {len(train_idx)} / {len(val_idx)}")

    model = DualBranchNet(dropout=CFG.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params   : {n_params:,}")

    history = fit(model, train_loader, val_loader, CFG, device)
    plot_history(history, CFG.out_dir)

    # Reload best weights for final reporting
    best = torch.load(CFG.out_dir / CFG.checkpoint_name,
                      map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    print(f"\nloaded best checkpoint from epoch {best['epoch']} "
          f"(val_loss={best['val_loss']:.4f})")
    metrics = final_report(model, val_loader, device, CFG.out_dir)
