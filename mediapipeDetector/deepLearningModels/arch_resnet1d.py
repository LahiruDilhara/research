"""
arch_resnet1d.py
================
Architecture 5: 1D ResNet with residual skip connections on combined input.
Input shape: (N, 5, 16)  — permuted to (N, 16, 5) for Conv1d.

10 hyperparameter configurations vary hidden_dim, dropout, lr, batch_size.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import TouchResNet1D, normalize, make_loaders, train_config, evaluate_model, save_result

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
TRAIN_CSV   = BASE / "data" / "training_data.csv"
TEST_CSV    = BASE / "data" / "test_data.csv"
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "resnet1d.csv"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCH_NAME   = "ResNet1D"
SEQ_LEN     = 5
FEATURE_DIM = 16
EPOCHS      = 60
SEED        = 42

# ── 10 Hyperparameter Configurations ──────────────────────────────────────────
# hidden_dim: channels after stem conv (residual blocks keep this same width)
CONFIGS = [
    {"id": 1, "hidden": 32,  "dropout": 0.20, "lr": 1e-3,  "bs": 32},  # small
    {"id": 2, "hidden": 64,  "dropout": 0.20, "lr": 1e-3,  "bs": 32},  # medium
    {"id": 3, "hidden": 128, "dropout": 0.20, "lr": 1e-3,  "bs": 32},  # large
    {"id": 4, "hidden": 128, "dropout": 0.30, "lr": 5e-4,  "bs": 64},  # large, low lr
]

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    def parse(path):
        df = pd.read_csv(path)
        n  = len(df)
        X  = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for k in range(1, 6):
            coord_cols = [f"wrist{k}_x", f"wrist{k}_y",
                          f"mcp{k}_x",   f"mcp{k}_y",
                          f"pip{k}_x",   f"pip{k}_y",
                          f"dip{k}_x",   f"dip{k}_y"]
            coords = df[coord_cols].fillna(0.0).values.astype(np.float32)
            if k == 1:
                vels = np.zeros((n, 8), dtype=np.float32)
            else:
                v = k - 1
                vel_cols = [f"wrist{v}_vx", f"wrist{v}_vy",
                            f"mcp{v}_vx",   f"mcp{v}_vy",
                            f"pip{v}_vx",   f"pip{v}_vy",
                            f"dip{v}_vx",   f"dip{v}_vy"]
                vels = df[vel_cols].fillna(0.0).values.astype(np.float32)
            X[:, k - 1, :] = np.hstack([coords, vels])
        tc = "touch_finger" if "touch_finger" in df.columns else "touch"
        y  = df[tc].astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"]).values.astype(np.float32)
        return X, y.reshape(-1, 1)

    X_tr, y_tr = parse(TRAIN_CSV)
    X_te, y_te = parse(TEST_CSV)
    X_tr_t, X_te_t = normalize(X_tr, X_te)
    return X_tr_t, torch.from_numpy(y_tr).float(), X_te_t, torch.from_numpy(y_te).float()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  {ARCH_NAME}  |  Input: (N, {SEQ_LEN}, {FEATURE_DIM})  |  Device: {device}")
    print(f"{'='*65}")

    X_tr, y_tr, X_te, y_te = load_data()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        print(f"\n  [Config {cfg['id']:02d}/10]  hidden={cfg['hidden']}  "
              f"dropout={cfg['dropout']}  lr={cfg['lr']}  bs={cfg['bs']}")

        train_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te, cfg["bs"])

        model = TouchResNet1D(
            in_channels=FEATURE_DIM,
            hidden_dim=cfg["hidden"],
            dropout=cfg["dropout"],
        ).to(device)

        t0                        = time.time()
        best_acc, final_acc, hist = train_config(model, train_loader, test_loader, EPOCHS, cfg["lr"], device)
        elapsed                   = time.time() - t0

        _, rpt = evaluate_model(model, test_loader, device)

        wf = WEIGHTS_DIR / f"{ARCH_NAME}_cfg{cfg['id']:02d}.pth"
        torch.save(model.state_dict(), wf)

        row = {
            "arch":             ARCH_NAME,
            "config_id":        cfg["id"],
            "hidden_dim":       cfg["hidden"],
            "dropout":          cfg["dropout"],
            "lr":               cfg["lr"],
            "batch_size":       cfg["bs"],
            "best_test_acc":    round(best_acc, 4),
            "final_test_acc":   round(final_acc, 4),
            "precision_touch":  round(rpt["Touch"]["precision"], 4),
            "recall_touch":     round(rpt["Touch"]["recall"], 4),
            "f1_touch":         round(rpt["Touch"]["f1-score"], 4),
            "train_time_s":     round(elapsed, 1),
            "weight_file":      wf.name,
        }
        save_result(row, str(RESULTS_CSV))
        print(f"  → Best: {best_acc:.2f}%  |  F1(Touch): {rpt['Touch']['f1-score']:.4f}  |  {elapsed:.0f}s")

    print(f"\n  ✓ {ARCH_NAME} complete. Results → {RESULTS_CSV.name}\n")


if __name__ == "__main__":
    main()
