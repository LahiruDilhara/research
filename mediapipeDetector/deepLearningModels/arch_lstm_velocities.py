"""
arch_lstm_velocities.py
=======================
Architecture 1: LSTM trained on VELOCITIES ONLY.
Input shape: (N, 4, 8)  — 4 transition steps × 8 velocity features.

Features per step:
  wrist_vx, wrist_vy, mcp_vx, mcp_vy, pip_vx, pip_vy, dip_vx, dip_vy

10 hyperparameter configurations are trained sequentially.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import SequenceLSTM, normalize, make_loaders, train_config, evaluate_model, save_result

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
TRAIN_CSV   = BASE / "data" / "training_data.csv"
TEST_CSV    = BASE / "data" / "test_data.csv"
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "lstm_velocities.csv"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCH_NAME   = "LSTM_Velocities"
SEQ_LEN     = 4      # 4 transition velocity steps
FEATURE_DIM = 8      # 8 velocity features per step
EPOCHS      = 60
SEED        = 42

# ── 10 Hyperparameter Configurations ──────────────────────────────────────────
CONFIGS = [
    # id   hidden  layers  dropout    lr      bs
    {"id": 1,  "hidden": 16,  "layers": 1, "dropout": 0.10, "lr": 1e-3,  "bs": 32},
    {"id": 2,  "hidden": 32,  "layers": 1, "dropout": 0.20, "lr": 1e-3,  "bs": 32},
    {"id": 3,  "hidden": 64,  "layers": 1, "dropout": 0.20, "lr": 1e-3,  "bs": 16},
    {"id": 4,  "hidden": 128, "layers": 1, "dropout": 0.30, "lr": 5e-4,  "bs": 32},
    {"id": 5,  "hidden": 32,  "layers": 2, "dropout": 0.20, "lr": 1e-3,  "bs": 32},
    {"id": 6,  "hidden": 64,  "layers": 2, "dropout": 0.20, "lr": 1e-3,  "bs": 32},
    {"id": 7,  "hidden": 64,  "layers": 2, "dropout": 0.30, "lr": 5e-4,  "bs": 64},
    {"id": 8,  "hidden": 128, "layers": 2, "dropout": 0.20, "lr": 1e-3,  "bs": 64},
    {"id": 9,  "hidden": 64,  "layers": 3, "dropout": 0.30, "lr": 5e-4,  "bs": 32},
    {"id": 10, "hidden": 128, "layers": 3, "dropout": 0.40, "lr": 1e-4,  "bs": 16},
]

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    def parse(path):
        df = pd.read_csv(path)
        n  = len(df)
        X  = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for v in range(1, 5):   # velocity indices 1..4
            cols = [f"wrist{v}_vx", f"wrist{v}_vy",
                    f"mcp{v}_vx",   f"mcp{v}_vy",
                    f"pip{v}_vx",   f"pip{v}_vy",
                    f"dip{v}_vx",   f"dip{v}_vy"]
            X[:, v - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
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
        print(f"\n  [Config {cfg['id']:02d}/10]  hidden={cfg['hidden']}  layers={cfg['layers']}  "
              f"dropout={cfg['dropout']}  lr={cfg['lr']}  bs={cfg['bs']}")

        train_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te, cfg["bs"])

        model = SequenceLSTM(
            input_features=FEATURE_DIM,
            hidden_units=cfg["hidden"],
            num_layers=cfg["layers"],
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
            "hidden":           cfg["hidden"],
            "layers":           cfg["layers"],
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
