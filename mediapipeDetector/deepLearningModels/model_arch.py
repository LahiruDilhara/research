"""
model_arch.py
=============
Shared model architectures and training utilities for deepLearningModels/.

Models:
  - SequenceLSTM      : Flexible LSTM for (N, T, F) binary classification
  - TouchCNN1D        : 1D CNN over feature channels
  - TouchResNet1D     : 1D ResNet with residual skip connections
  - TouchAttentionNet : Multi-Head Self-Attention Transformer encoder
"""

import csv
import time
from pathlib import Path

import numpy as np
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
    import json
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


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
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
    p1_tr = base_dir / "training_testing_data" / "train_dataset.csv"
    p1_te = base_dir / "training_testing_data" / "test_dataset.csv"
    if p1_tr.exists() and p1_te.exists():
        return p1_tr, p1_te

    p2_tr = base_dir / "dataprocessing" / "11_train_test_split" / "training_dataset.csv"
    p2_te = base_dir / "dataprocessing" / "11_train_test_split" / "testing_dataset.csv"
    if p2_tr.exists() and p2_te.exists():
        return p2_tr, p2_te

    p3_tr = base_dir / "data" / "training_data.csv"
    p3_te = base_dir / "data" / "test_data.csv"
    return p3_tr, p3_te


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
        fc_mid = max(hidden_units // 2, 8)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_units),
            nn.Linear(hidden_units, fc_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_mid, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # Last timestep


class TouchCNN1D(nn.Module):
    """Two-layer 1D CNN with global average pooling."""
    def __init__(self, in_channels: int, conv_channels: int, fc_hidden: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_channels * 2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        # x: (N, T, F) → permute → (N, F, T) for Conv1d
        return self.net(x.permute(0, 2, 1))


class _ResBlock(nn.Module):
    """Residual block: F(x) + x."""
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TouchResNet1D(nn.Module):
    """1D ResNet with two residual blocks and global average pooling."""
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.res1 = _ResBlock(hidden_dim, dropout)
        self.res2 = _ResBlock(hidden_dim, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        fc_mid    = max(hidden_dim // 2, 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, fc_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_mid, 1),
        )

    def forward(self, x):
        z = self.stem(x.permute(0, 2, 1))   # (N, T, F) → (N, F, T)
        z = self.res1(z)
        z = self.res2(z)
        return self.head(self.pool(z))


class TouchAttentionNet(nn.Module):
    """Single-layer Transformer encoder with mean pooling."""
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.embed  = nn.Linear(input_dim, embed_dim)
        self.attn   = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1  = nn.LayerNorm(embed_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2  = nn.LayerNorm(embed_dim)
        self.head   = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        e         = self.embed(x)
        a, _      = self.attn(e, e, e)
        e         = self.norm1(e + a)
        e         = self.norm2(e + self.ffn(e))
        return self.head(e.mean(dim=1))    # Global mean pooling


class TouchBiLSTM(nn.Module):
    """
    Bidirectional LSTM — processes sequence in both directions.
    Last timestep from forward + backward passes are concatenated (2×hidden).
    """
    def __init__(self, input_features: int, hidden_units: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        out_dim = hidden_units * 2   # bidirectional doubles the output dim
        fc_mid  = max(out_dim // 2, 8)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, fc_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_mid, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # Last timestep (fwd + bwd concatenated)


class _TCNBlock(nn.Module):
    """Single dilated TCN block with residual skip connection."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = dilation   # kernel_size=3, so padding=dilation keeps length same
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class TouchTCN(nn.Module):
    """
    Temporal Convolutional Network with exponentially growing dilations.
    Each level doubles the dilation: [1, 2, 4, ...].
    Input: (N, T, F) → permuted to (N, F, T) for Conv1d.
    """
    def __init__(self, in_channels: int, tcn_channels: int, num_levels: int, dropout: float = 0.2):
        super().__init__()
        blocks = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch    = in_channels if i == 0 else tcn_channels
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
        z = self.network(x.permute(0, 2, 1))   # (N, T, F) → (N, F, T)
        return self.head(self.pool(z))


# ─────────────────────────────────────────────────────────────────────────────
#  Training / Evaluation Utilities
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
    """
    Full training loop with early stopping.
    Set verbose=False when running multiple configs in parallel threads.
    Returns: (best_test_acc, final_test_acc, history_dict)
    """
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, min_lr=1e-6)

    best_acc   = 0.0
    no_improve = 0
    history    = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        te_loss, te_acc = _run_epoch(model, test_loader,  loss_fn, None,      device, train=False)
        scheduler.step(te_loss)

        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)

        if te_acc > best_acc:
            best_acc   = te_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"      [Early stop @ epoch {epoch}]", flush=True)
                break

        if verbose:
            print(f"Epoch: {epoch:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%", flush=True)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Result Persistence
# ─────────────────────────────────────────────────────────────────────────────

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
