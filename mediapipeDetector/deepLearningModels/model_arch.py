"""
model_arch.py
=============
Shared model architectures, data loaders, and training utilities for deepLearningModels/.

Models:
  - SequenceLSTM      : Flexible LSTM for (N, T, F) binary classification
  - TouchCNN1D        : 1D CNN over feature channels
  - TouchResNet1D     : 1D ResNet with residual skip connections
  - TouchAttentionNet : Multi-Head Self-Attention Transformer encoder
  - TouchTCN          : Temporal Convolutional Network (TCN)
"""

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    import asciichartpy as ac
    HAS_ASCIICHART = True
except ImportError:
    HAS_ASCIICHART = False


def print_terminal_curves(history: dict, title: str = ""):
    """Prints terminal-rendered ASCII charts for Loss and Accuracy curves."""
    if not HAS_ASCIICHART:
        return

    tr_loss = history.get("train_loss", [])
    te_loss = history.get("test_loss", [])
    tr_acc  = history.get("train_acc", [])
    te_acc  = history.get("test_acc", [])

    if not tr_loss or not te_loss or not tr_acc or not te_acc:
        return

    print(f"\n{'='*70}", flush=True)
    print(f"  {title} — TRAINING & EVALUATION CURVES", flush=True)
    print(f"{'='*70}", flush=True)

    # Loss Curve
    print("\n  [LOSS CURVE]  🔴 Red: Train Loss | 🔵 Cyan: Test Loss", flush=True)
    loss_chart = ac.plot([tr_loss, te_loss], {'height': 8, 'colors': [ac.red, ac.cyan]})
    for line in loss_chart.split("\n"):
        print("   " + line, flush=True)

    # Accuracy Curve
    print("\n  [ACCURACY CURVE %]  🟢 Green: Train Acc | 🟡 Yellow: Test Acc", flush=True)
    acc_chart = ac.plot([tr_acc, te_acc], {'height': 8, 'colors': [ac.green, ac.yellow]})
    for line in acc_chart.split("\n"):
        print("   " + line, flush=True)

    print(f"{'='*70}\n", flush=True)


def save_history(history: dict, json_path: Path):
    """Saves training history dictionary to JSON."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)


def plot_matplotlib_curves(history: dict, title: str = "", save_path: Path = None, show: bool = False):
    """Generates publication-quality Jupyter notebook style Matplotlib plots for Loss and Accuracy curves."""
    import matplotlib.pyplot as plt

    tr_loss = history.get("train_loss", [])
    te_loss = history.get("test_loss", [])
    tr_acc  = history.get("train_acc", [])
    te_acc  = history.get("test_acc", [])

    if not tr_loss or not te_loss or not tr_acc or not te_acc:
        return

    epochs = list(range(1, len(tr_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

    # --- Loss Plot ---
    axes[0].plot(epochs, tr_loss, label="Train Loss", color="#e74c3c", linewidth=2.0, marker="o", markersize=3)
    axes[0].plot(epochs, te_loss, label="Test Loss", color="#3498db", linewidth=2.0, linestyle="--", marker="s", markersize=3)
    axes[0].set_title(f"{title} — Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("BCE Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    # --- Accuracy Plot ---
    axes[1].plot(epochs, tr_acc, label="Train Acc", color="#2ecc71", linewidth=2.0, marker="o", markersize=3)
    axes[1].plot(epochs, te_acc, label="Test Acc", color="#f39c12", linewidth=2.0, linestyle="--", marker="s", markersize=3)
    axes[1].set_title(f"{title} — Accuracy Curve (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Matplotlib curve plot saved → results/plots/{save_path.name}", flush=True)

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)


def analyze_fit_quality(history: dict) -> dict:
    """Analyzes training history to detect Overfitting, Underfitting, or Good Fit,
    determining the onset epoch and overfitting gap percentage scale.
    """
    tr_loss = history.get("train_loss", [])
    te_loss = history.get("test_loss", [])
    tr_acc  = history.get("train_acc", [])
    te_acc  = history.get("test_acc", [])

    if not tr_loss or not te_loss or not tr_acc or not te_acc:
        return {
            "fit_status": "UNKNOWN",
            "fit_scale": "N/A",
            "onset_epoch": 0,
            "max_gap_pct": 0.0,
            "final_gap_pct": 0.0,
            "min_loss_epoch": 0,
            "recommendation": "N/A",
        }

    n_epochs = len(tr_loss)
    min_te_loss_idx = int(np.argmin(te_loss))
    min_te_loss_epoch = min_te_loss_idx + 1

    gaps = [tr - te for tr, te in zip(tr_acc, te_acc)]
    max_gap_pct   = max(gaps)
    final_gap_pct = gaps[-1]
    best_te_acc   = max(te_acc)

    loss_increase_after_min = te_loss[-1] - te_loss[min_te_loss_idx] if min_te_loss_idx < n_epochs - 1 else 0.0

    if best_te_acc < 70.0 or (tr_acc[-1] < 72.0 and te_acc[-1] < 72.0):
        status = "UNDERFITTING"
        scale = "High"
        onset_epoch = 1
        recommendation = "Increase model capacity (units/layers) or reduce regularization."

    elif max_gap_pct > 5.0 or loss_increase_after_min > 0.03 or (min_te_loss_epoch < n_epochs - 3 and gaps[-1] > 4.0):
        status = "OVERFITTING"
        onset_epoch = min_te_loss_epoch
        for ep_idx in range(min_te_loss_idx, n_epochs):
            if gaps[ep_idx] > 4.0 or te_loss[ep_idx] > te_loss[min_te_loss_idx] + 0.015:
                onset_epoch = ep_idx + 1
                break

        if max_gap_pct >= 12.0:
            scale = "Severe"
        elif max_gap_pct >= 7.0:
            scale = "Moderate"
        else:
            scale = "Mild"

        recommendation = f"Increase dropout (gap: +{max_gap_pct:.1f}%), stop early at epoch {min_te_loss_epoch}, or add weight decay."

    else:
        status = "GOOD FIT (OPTIMAL)"
        scale = "None"
        onset_epoch = min_te_loss_epoch
        recommendation = "Model generalization is well-balanced."

    return {
        "fit_status": status,
        "fit_scale": scale,
        "onset_epoch": onset_epoch,
        "max_gap_pct": round(max_gap_pct, 2),
        "final_gap_pct": round(final_gap_pct, 2),
        "min_loss_epoch": min_te_loss_epoch,
        "recommendation": recommendation,
    }


def print_overfit_analytics(analytics: dict, title: str = ""):
    """Prints overfitting/underfitting diagnostic report to standard output."""
    status   = analytics["fit_status"]
    scale    = analytics["fit_scale"]
    onset    = analytics["onset_epoch"]
    max_gap  = analytics["max_gap_pct"]
    final_gap= analytics["final_gap_pct"]
    min_ep   = analytics["min_loss_epoch"]
    rec      = analytics["recommendation"]

    print(f"\n{'='*70}", flush=True)
    print(f"  {title} — OVERFITTING & DIAGNOSTIC ANALYTICS", flush=True)
    print(f"{'='*70}", flush=True)

    if status == "OVERFITTING":
        tag = f"⚠️  OVERFITTING ({scale})"
    elif status == "UNDERFITTING":
        tag = f"⚠️  UNDERFITTING ({scale})"
    else:
        tag = f"✅  GOOD FIT (OPTIMAL)"

    print(f"  Diagnosis Status:  {tag}", flush=True)
    print(f"  Overfit Onset:     Epoch {onset} (Test loss minimum at epoch {min_ep})", flush=True)
    print(f"  Max Acc Gap:       +{max_gap:.2f}% (Train Acc - Test Acc)", flush=True)
    print(f"  Final Acc Gap:     +{final_gap:.2f}% (Train Acc - Test Acc)", flush=True)
    print(f"  Recommendation:    {rec}", flush=True)
    print(f"{'='*70}\n", flush=True)


def print_terminal_confusion_matrix(cm: np.ndarray, title: str = ""):
    """Prints formatted 2x2 confusion matrix with detailed classification metrics in standard output."""
    if cm.shape != (2, 2):
        return

    tn, fp, fn, tp = cm.ravel()
    total          = tn + fp + fn + tp
    actual_untouch = tn + fp
    actual_touch   = fn + tp

    acc  = (tp + tn) / total * 100.0 if total > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    f1   = 2 * (prec * rec) / (prec + rec) / 100.0 if (prec + rec) > 0 else 0.0

    print(f"\n{'='*70}", flush=True)
    print(f"  {title} — CONFUSION MATRIX & METRICS REPORT", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"                 Predicted UNTOUCH    Predicted TOUCH     Total Actual", flush=True)
    print(f"  Actual UNTOUCH     TN = {tn:<8d}     FP = {fp:<8d}     {actual_untouch:<8d}", flush=True)
    print(f"  Actual TOUCH       FN = {fn:<8d}     TP = {tp:<8d}     {actual_touch:<8d}", flush=True)
    print(f"  ------------------------------------------------------------------", flush=True)
    print(f"  Overall Accuracy : {acc:.2f}% ({tp + tn}/{total})", flush=True)
    print(f"  Precision (Touch): {prec:.2f}% ({tp}/{tp + fp if (tp + fp) > 0 else 1})", flush=True)
    print(f"  Recall (Touch)   : {rec:.2f}% ({tp}/{actual_touch if actual_touch > 0 else 1})", flush=True)
    print(f"  Specificity      : {spec:.2f}% ({tn}/{actual_untouch if actual_untouch > 0 else 1})", flush=True)
    print(f"  F1-Score (Touch) : {f1:.4f}", flush=True)
    print(f"{'='*70}\n", flush=True)


def plot_matplotlib_confusion_matrix(cm: np.ndarray, title: str = "", save_path: Path = None, show: bool = False):
    """Generates Jupyter notebook style Matplotlib confusion matrix heatmap visualization."""
    import matplotlib.pyplot as plt

    if cm.shape != (2, 2):
        return

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Untouch (0)", "Touch (1)"],
        yticklabels=["Untouch (0)", "Touch (1)"],
        title=f"{title} — Confusion Matrix",
        ylabel="Actual Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    labels_matrix = [["TN", "FP"], ["FN", "TP"]]

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = (count / total) * 100.0 if total > 0 else 0.0
            text  = f"{labels_matrix[i][j]}\n{count}\n({pct:.1f}%)"
            color = "white" if count > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  📊 Matplotlib Confusion Matrix saved → results/plots/{save_path.name}", flush=True)

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset & Loaders
# ─────────────────────────────────────────────────────────────────────────────

class TouchDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, y_train, X_test, y_test, batch_size: int):
    train_loader = DataLoader(TouchDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(TouchDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def normalize(X_train_np: np.ndarray, X_test_np: np.ndarray):
    """Fit StandardScaler on train, transform both train and test."""
    scaler = StandardScaler()
    N_tr, T, C = X_train_np.shape
    N_te       = X_test_np.shape[0]
    X_tr = scaler.fit_transform(X_train_np.reshape(N_tr, -1)).reshape(N_tr, T, C)
    X_te = scaler.transform(X_test_np.reshape(N_te, -1)).reshape(N_te, T, C)
    return torch.from_numpy(X_tr).float(), torch.from_numpy(X_te).float()


def parse_labels(df):
    """Parse binary touch label from DataFrame."""
    tc = "touch_finger" if "touch_finger" in df.columns else "touch"
    y  = df[tc].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"]).values.astype(np.float32)
    return y.reshape(-1, 1)


def get_data_paths(base_dir: Path):
    """Resolve train and test CSV file paths with robust fallback search."""
    candidates = [
        (base_dir / "training_testing_data" / "train_dataset.csv", base_dir / "training_testing_data" / "test_dataset.csv"),
        (base_dir / "dataprocessing" / "11_train_test_split" / "training_dataset.csv", base_dir / "dataprocessing" / "11_train_test_split" / "testing_dataset.csv"),
        (base_dir / "dataprocessing" / "12_train_test_split" / "training_dataset.csv", base_dir / "dataprocessing" / "12_train_test_split" / "testing_dataset.csv"),
        (base_dir / "data" / "training_data.csv", base_dir / "data" / "test_data.csv"),
    ]
    for tr, te in candidates:
        if tr.exists() and te.exists():
            return tr, te
    return candidates[0][0], candidates[0][1]


def load_variant_data(variant_name: str, base_dir: Path):
    """Parses and returns normalized X_train, y_train, X_test, y_test tensors for a specified feature representation."""
    train_csv, test_csv = get_data_paths(base_dir)

    def parse_file(csv_path):
        df = pd.read_csv(csv_path)
        n  = len(df)
        y  = parse_labels(df)

        if variant_name == "coords_2d":
            seq_len, feature_dim = 5, 8
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for k in range(1, 6):
                cols = [f"wrist{k}_x", f"wrist{k}_y", f"pip{k}_x", f"pip{k}_y", f"dip{k}_x", f"dip{k}_y", f"tip{k}_x", f"tip{k}_y"]
                X[:, k - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
            return X, y, seq_len, feature_dim

        elif variant_name == "coords_3d":
            seq_len, feature_dim = 5, 12
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for k in range(1, 6):
                cols = [f"wrist{k}_x", f"wrist{k}_y", f"wrist{k}_z", f"pip{k}_x", f"pip{k}_y", f"pip{k}_z", f"dip{k}_x", f"dip{k}_y", f"dip{k}_z", f"tip{k}_x", f"tip{k}_y", f"tip{k}_z"]
                X[:, k - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
            return X, y, seq_len, feature_dim

        elif variant_name in ("vel_2d", "vel_velocities"):
            seq_len, feature_dim = 4, 8
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"pip{v}_vx", f"pip{v}_vy", f"dip{v}_vx", f"dip{v}_vy", f"tip{v}_vx", f"tip{v}_vy"]
                X[:, v - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
            return X, y, seq_len, feature_dim

        elif variant_name == "vel_3d":
            seq_len, feature_dim = 4, 12
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"wrist{v}_vz", f"pip{v}_vx", f"pip{v}_vy", f"pip{v}_vz", f"dip{v}_vx", f"dip{v}_vy", f"dip{v}_vz", f"tip{v}_vx", f"tip{v}_vy", f"tip{v}_vz"]
                X[:, v - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
            return X, y, seq_len, feature_dim

        elif variant_name in ("vel_speed_2d", "vel_speed"):
            seq_len, feature_dim = 4, 12
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"pip{v}_vx", f"pip{v}_vy", f"dip{v}_vx", f"dip{v}_vy", f"tip{v}_vx", f"tip{v}_vy"]
                speed_cols = [f"wrist{v}_speed_2d", f"pip{v}_speed_2d", f"dip{v}_speed_2d", f"tip{v}_speed_2d"]
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                s_vals = df[speed_cols].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([v_vals, s_vals])
            return X, y, seq_len, feature_dim

        elif variant_name == "vel_speed_3d":
            seq_len, feature_dim = 4, 16
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"wrist{v}_vz", f"pip{v}_vx", f"pip{v}_vy", f"pip{v}_vz", f"dip{v}_vx", f"dip{v}_vy", f"dip{v}_vz", f"tip{v}_vx", f"tip{v}_vy", f"tip{v}_vz"]
                speed_cols = [f"wrist{v}_speed_3d", f"pip{v}_speed_3d", f"dip{v}_speed_3d", f"tip{v}_speed_3d"]
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                s_vals = df[speed_cols].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([v_vals, s_vals])
            return X, y, seq_len, feature_dim

        elif variant_name in ("combined_2d", "combined"):
            seq_len, feature_dim = 4, 16
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                pos_cols = [f"wrist{v}_x", f"wrist{v}_y", f"pip{v}_x", f"pip{v}_y", f"dip{v}_x", f"dip{v}_y", f"tip{v}_x", f"tip{v}_y"]
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"pip{v}_vx", f"pip{v}_vy", f"dip{v}_vx", f"dip{v}_vy", f"tip{v}_vx", f"tip{v}_vy"]
                p_vals = df[pos_cols].fillna(0.0).values.astype(np.float32)
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([p_vals, v_vals])
            return X, y, seq_len, feature_dim

        elif variant_name == "combined_3d":
            seq_len, feature_dim = 4, 24
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                pos_cols = [f"wrist{v}_x", f"wrist{v}_y", f"wrist{v}_z", f"pip{v}_x", f"pip{v}_y", f"pip{v}_z", f"dip{v}_x", f"dip{v}_y", f"dip{v}_z", f"tip{v}_x", f"tip{v}_y", f"tip{v}_z"]
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"wrist{v}_vz", f"pip{v}_vx", f"pip{v}_vy", f"pip{v}_vz", f"dip{v}_vx", f"dip{v}_vy", f"dip{v}_vz", f"tip{v}_vx", f"tip{v}_vy", f"tip{v}_vz"]
                p_vals = df[pos_cols].fillna(0.0).values.astype(np.float32)
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([p_vals, v_vals])
            return X, y, seq_len, feature_dim

        elif variant_name == "all_joints_vel":
            seq_len, feature_dim = 4, 18
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"thumb_cmc{v}_vx", f"thumb_cmc{v}_vy", f"index_mcp{v}_vx", f"index_mcp{v}_vy", f"middle_mcp{v}_vx", f"middle_mcp{v}_vy", f"ring_mcp{v}_vx", f"ring_mcp{v}_vy", f"pinky_mcp{v}_vx", f"pinky_mcp{v}_vy", f"pip{v}_vx", f"pip{v}_vy", f"dip{v}_vx", f"dip{v}_vy", f"tip{v}_vx", f"tip{v}_vy"]
                X[:, v - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
            return X, y, seq_len, feature_dim

        elif variant_name in ("all_joints_coords_vel", "all_combined"):
            seq_len, feature_dim = 4, 36
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                pos_cols = [f"wrist{v}_x", f"wrist{v}_y", f"thumb_cmc{v}_x", f"thumb_cmc{v}_y", f"index_mcp{v}_x", f"index_mcp{v}_y", f"middle_mcp{v}_x", f"middle_mcp{v}_y", f"ring_mcp{v}_x", f"ring_mcp{v}_y", f"pinky_mcp{v}_x", f"pinky_mcp{v}_y", f"pip{v}_x", f"pip{v}_y", f"dip{v}_x", f"dip{v}_y", f"tip{v}_x", f"tip{v}_y"]
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"thumb_cmc{v}_vx", f"thumb_cmc{v}_vy", f"index_mcp{v}_vx", f"index_mcp{v}_vy", f"middle_mcp{v}_vx", f"middle_mcp{v}_vy", f"ring_mcp{v}_vx", f"ring_mcp{v}_vy", f"pinky_mcp{v}_vx", f"pinky_mcp{v}_vy", f"pip{v}_vx", f"pip{v}_vy", f"dip{v}_vx", f"dip{v}_vy", f"tip{v}_vx", f"tip{v}_vy"]
                p_vals = df[pos_cols].fillna(0.0).values.astype(np.float32)
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([p_vals, v_vals])
            return X, y, seq_len, feature_dim

        elif variant_name == "z_kinematics":
            seq_len, feature_dim = 4, 8
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                z_pos = [f"wrist{v}_z", f"pip{v}_z", f"dip{v}_z", f"tip{v}_z"]
                z_vel = [f"wrist{v}_vz", f"pip{v}_vz", f"dip{v}_vz", f"tip{v}_vz"]
                zp = df[z_pos].fillna(0.0).values.astype(np.float32)
                zv = df[z_vel].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([zp, zv])
            return X, y, seq_len, feature_dim

        elif variant_name in ("super_combined", "super"):
            seq_len, feature_dim = 4, 28
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                pos_cols = [f"wrist{v}_x", f"wrist{v}_y", f"wrist{v}_z", f"pip{v}_x", f"pip{v}_y", f"pip{v}_z", f"dip{v}_x", f"dip{v}_y", f"dip{v}_z", f"tip{v}_x", f"tip{v}_y", f"tip{v}_z"]
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"wrist{v}_vz", f"pip{v}_vx", f"pip{v}_vy", f"pip{v}_vz", f"dip{v}_vx", f"dip{v}_vy", f"dip{v}_vz", f"tip{v}_vx", f"tip{v}_vy", f"tip{v}_vz"]
                speed_cols = [f"wrist{v}_speed_3d", f"pip{v}_speed_3d", f"dip{v}_speed_3d", f"tip{v}_speed_3d"]
                p_vals = df[pos_cols].fillna(0.0).values.astype(np.float32)
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                s_vals = df[speed_cols].fillna(0.0).values.astype(np.float32)
                X[:, v - 1, :] = np.hstack([p_vals, v_vals, s_vals])
            return X, y, seq_len, feature_dim

        elif variant_name in ("wrist_relative_3d", "wrist_rel_3d"):
            seq_len, feature_dim = 4, 21
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                wx, wy, wz = df[f"wrist{v}_x"].fillna(0.0).values, df[f"wrist{v}_y"].fillna(0.0).values, df[f"wrist{v}_z"].fillna(0.0).values
                px, py, pz = df[f"pip{v}_x"].fillna(0.0).values - wx, df[f"pip{v}_y"].fillna(0.0).values - wy, df[f"pip{v}_z"].fillna(0.0).values - wz
                dx, dy, dz = df[f"dip{v}_x"].fillna(0.0).values - wx, df[f"dip{v}_y"].fillna(0.0).values - wy, df[f"dip{v}_z"].fillna(0.0).values - wz
                tx, ty, tz = df[f"tip{v}_x"].fillna(0.0).values - wx, df[f"tip{v}_y"].fillna(0.0).values - wy, df[f"tip{v}_z"].fillna(0.0).values - wz
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"wrist{v}_vz", f"pip{v}_vx", f"pip{v}_vy", f"pip{v}_vz", f"dip{v}_vx", f"dip{v}_vy", f"dip{v}_vz", f"tip{v}_vx", f"tip{v}_vy", f"tip{v}_vz"]
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                rel_coords = np.column_stack([px, py, pz, dx, dy, dz, tx, ty, tz])
                X[:, v - 1, :] = np.hstack([rel_coords, v_vals])
            return X, y, seq_len, feature_dim

        elif variant_name in ("fingertip_velocity_ratios", "tip_vel_ratios"):
            seq_len, feature_dim = 4, 16
            X = np.zeros((n, seq_len, feature_dim), dtype=np.float32)
            for v in range(1, 5):
                w_vx, w_vy, w_vz = df[f"wrist{v}_vx"].fillna(0.0).values, df[f"wrist{v}_vy"].fillna(0.0).values, df[f"wrist{v}_vz"].fillna(0.0).values
                t_vx, t_vy, t_vz = df[f"tip{v}_vx"].fillna(0.0).values, df[f"tip{v}_vy"].fillna(0.0).values, df[f"tip{v}_vz"].fillna(0.0).values
                rel_vx, rel_vy, rel_vz = t_vx - w_vx, t_vy - w_vy, t_vz - w_vz
                tip_speed = df[f"tip{v}_speed_3d"].fillna(0.0).values
                wrist_speed = df[f"wrist{v}_speed_3d"].fillna(0.0).values
                speed_ratio = (tip_speed + 1e-5) / (wrist_speed + 1e-5)
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy", f"wrist{v}_vz", f"pip{v}_vx", f"pip{v}_vy", f"pip{v}_vz", f"dip{v}_vx", f"dip{v}_vy", f"dip{v}_vz", f"tip{v}_vx", f"tip{v}_vy", f"tip{v}_vz"]
                v_vals = df[vel_cols].fillna(0.0).values.astype(np.float32)
                rel_kinematics = np.column_stack([rel_vx, rel_vy, rel_vz, speed_ratio])
                X[:, v - 1, :] = np.hstack([v_vals, rel_kinematics])
            return X, y, seq_len, feature_dim

        else:
            raise ValueError(f"Unknown data variant: '{variant_name}'")

    X_tr_raw, y_tr, seq_len, feature_dim = parse_file(train_csv)
    X_te_raw, y_te, _, _                 = parse_file(test_csv)

    X_tr, X_te = normalize(X_tr_raw, X_te_raw)
    return X_tr, torch.from_numpy(y_tr).float(), X_te, torch.from_numpy(y_te).float(), seq_len, feature_dim


# ─────────────────────────────────────────────────────────────────────────────
#  Model Classes
# ─────────────────────────────────────────────────────────────────────────────

class SequenceLSTM(nn.Module):
    """Flexible LSTM binary classifier for sequences (N, T, F)."""
    def __init__(self, input_features: int, hidden_units: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class BiLSTM(nn.Module):
    """Bidirectional LSTM binary classifier for sequences (N, T, F)."""
    def __init__(self, input_features: int, hidden_units: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TouchCNN1D(nn.Module):
    """1D CNN classifier operating over temporal sequence channels."""
    def __init__(self, input_features: int, conv_channels: int = 32, fc_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_features, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * 2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        return self.head(self.net(x.permute(0, 2, 1)))


class _ResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class TouchResNet1D(nn.Module):
    """Residual 1D CNN with skip connections for temporal sequences."""
    def __init__(self, input_features: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.b1   = _ResBlock1D(hidden_dim, dropout)
        self.b2   = _ResBlock1D(hidden_dim, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        z = self.in_conv(x.permute(0, 2, 1))
        z = self.b1(z)
        z = self.b2(z)
        return self.head(self.pool(z))


class TouchAttentionNet(nn.Module):
    """Transformer Encoder classifier with Multi-Head Self-Attention."""
    def __init__(self, input_features: int, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Linear(input_features, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
        h   = self.in_proj(x)
        out = self.transformer(h)
        return self.head(out[:, -1, :])


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
        super().__init__()
        padding    = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act1  = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act2  = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res  = self.down(x)
        out1 = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out2 = self.drop2(self.act2(self.bn2(self.conv2(out1))))
        L    = res.shape[-1]
        return out2[:, :, :L] + res


class TouchTCN(nn.Module):
    """Temporal Convolutional Network with dilated causal 1D convolutions."""
    def __init__(self, input_features: int, tcn_channels: int = 32, num_levels: int = 2, dropout: float = 0.2):
        super().__init__()
        blocks = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch    = input_features if i == 0 else tcn_channels
            blocks.append(_TCNBlock(in_ch, tcn_channels, dilation, dropout))
        self.network = nn.Sequential(*blocks)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        fc_mid       = max(tcn_channels // 2, 8)
        self.head    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tcn_channels, fc_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_mid, 1),
        )

    def forward(self, x):
        z = self.network(x.permute(0, 2, 1))
        return self.head(self.pool(z))


# ─────────────────────────────────────────────────────────────────────────────
#  Training / Evaluation Utilities & Universal Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    """Single train or eval epoch. Returns (avg_loss, accuracy_%)."""
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    ctx = torch.enable_grad if train else torch.inference_mode
    with ctx():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits    = model(X_b)
            loss      = loss_fn(logits, y_b)
            preds     = torch.round(torch.sigmoid(logits))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss    += loss.item() * len(X_b)
            total_correct += torch.eq(y_b, preds).sum().item()
            total_n       += len(X_b)

    return total_loss / total_n, (total_correct / total_n) * 100.0


def train_config(model, train_loader, test_loader, epochs: int, lr: float, device,
                 patience: int = 8, verbose: bool = True):
    """Full training loop with early stopping and best-epoch weights preservation."""
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, min_lr=1e-6)

    best_acc        = 0.0
    best_loss       = float("inf")
    best_state_dict = None
    no_improve      = 0
    history         = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        te_loss, te_acc = _run_epoch(model, test_loader,  loss_fn, None,      device, train=False)
        scheduler.step(te_loss)

        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)

        if te_acc > best_acc or (math.isclose(te_acc, best_acc, abs_tol=1e-4) and te_loss < best_loss):
            best_acc        = te_acc
            best_loss       = te_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve      = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"      [Early stop @ epoch {epoch}]", flush=True)
                break

        if verbose:
            print(f"Epoch: {epoch:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%", flush=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_acc, history["test_acc"][-1], history


def evaluate_model(model, test_loader, device):
    """Returns (confusion_matrix, classification_report_dict)."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for X_b, y_b in test_loader:
            preds = torch.round(torch.sigmoid(model(X_b.to(device)))).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())

    p   = np.array(all_preds).squeeze()
    t   = np.array(all_targets).squeeze()
    cm  = confusion_matrix(t, p)
    rpt = classification_report(t, p, target_names=["Untouch", "Touch"], output_dict=True)
    return cm, rpt


def save_result(row: dict, csv_path: str):
    """Append one result row to csv_path (creates file/header on first write)."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_model_benchmark(
    arch_name: str,
    variant_name: str,
    create_model_fn,
    configs: list[dict],
    default_epochs: int = 70,
    seed: int = 42,
):
    """Universal benchmark execution runner for any deep learning model variant script."""
    parser = argparse.ArgumentParser(description=f"Train {arch_name}")
    parser.add_argument("--epochs", "-e", type=int, default=default_epochs, help="Number of training epochs")
    parser.add_argument("--dropout", "-d", type=float, default=None, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", "-bs", type=int, default=None, help="Batch size")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--plot", "-p", "--plot-curves", dest="plot", action="store_true", help="Generate Matplotlib curve plots and Confusion Matrix heatmaps")
    args = parser.parse_args()

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE        = Path(__file__).resolve().parent.parent
    RESULTS_CSV = Path(__file__).resolve().parent / "results" / f"{arch_name.lower()}.csv"
    WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

    X_tr, y_tr, X_te, y_te, seq_len, feature_dim = load_variant_data(variant_name, BASE)

    print(f"\n{'='*65}")
    print(f"  {arch_name}  |  Input: (N, {seq_len}, {feature_dim})  |  Device: {device}")
    print(f"{'='*65}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        if args.dropout is not None:
            cfg["dropout"] = args.dropout
        if args.lr is not None:
            cfg["lr"] = args.lr
        if args.batch_size is not None:
            cfg["bs"] = args.batch_size
        if args.hidden is not None:
            cfg["hidden"] = args.hidden

        hid_dim = cfg.get("hidden", cfg.get("conv_ch", cfg.get("embed_dim", cfg.get("tcn_channels", 32))))

        print(f"\n  [Config {cfg['id']:02d}]  hidden/channels={hid_dim}  "
              f"dropout={cfg['dropout']}  lr={cfg['lr']}  bs={cfg['bs']}")

        train_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te, cfg["bs"])
        model = create_model_fn(feature_dim, cfg).to(device)

        t0                        = time.time()
        best_acc, final_acc, hist = train_config(model, train_loader, test_loader, args.epochs, cfg["lr"], device, verbose=True)
        elapsed                   = time.time() - t0

        print_terminal_curves(hist, title=arch_name)
        analytics = analyze_fit_quality(hist)
        print_overfit_analytics(analytics, title=arch_name)

        history_path = Path(__file__).resolve().parent / "results" / f"{arch_name.lower()}_history.json"
        save_history(hist, history_path)

        cm, rpt = evaluate_model(model, test_loader, device)
        print_terminal_confusion_matrix(cm, title=arch_name)

        if args.plot:
            plot_path = Path(__file__).resolve().parent / "results" / "plots" / f"{arch_name.lower()}.png"
            plot_matplotlib_curves(hist, title=arch_name, save_path=plot_path, show=False)

            cm_plot_path = Path(__file__).resolve().parent / "results" / "plots" / f"{arch_name.lower()}_confusion_matrix.png"
            plot_matplotlib_confusion_matrix(cm, title=arch_name, save_path=cm_plot_path, show=False)

        wf = WEIGHTS_DIR / f"{arch_name}_cfg{cfg['id']:02d}.pth"
        torch.save(model.state_dict(), wf)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        row = {
            "arch":             arch_name,
            "config_id":        cfg["id"],
            "hidden":           hid_dim,
            "layers":           cfg.get("layers", cfg.get("num_levels", 2)),
            "dropout":          cfg["dropout"],
            "lr":               cfg["lr"],
            "batch_size":       cfg["bs"],
            "best_test_acc":    round(best_acc, 4),
            "final_test_acc":   round(final_acc, 4),
            "precision_touch":  round(rpt["Touch"]["precision"], 4),
            "recall_touch":     round(rpt["Touch"]["recall"], 4),
            "f1_touch":         round(rpt["Touch"]["f1-score"], 4),
            "tn":               tn,
            "fp":               fp,
            "fn":               fn,
            "tp":               tp,
            "fit_status":       f"{analytics['fit_status']} ({analytics['fit_scale']})" if analytics['fit_scale'] != "None" else analytics['fit_status'],
            "onset_epoch":      analytics["onset_epoch"],
            "max_gap_pct":      analytics["max_gap_pct"],
            "train_time_s":     round(elapsed, 1),
            "weight_file":      wf.name,
        }
        save_result(row, str(RESULTS_CSV))
        print(f"  → Best: {best_acc:.2f}%  |  F1(Touch): {rpt['Touch']['f1-score']:.4f}  |  {elapsed:.0f}s")

    print(f"\n  ✓ {arch_name} complete. Results → {RESULTS_CSV.name}\n")
