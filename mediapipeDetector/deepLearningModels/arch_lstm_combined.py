"""
arch_lstm_combined.py
=====================
Architecture 3: LSTM trained on BOTH COORDINATES + VELOCITIES.
Input shape: (N, 5, 16)  — 5 timesteps × 16 features.

Features per step:
  [wrist_x/y, mcp_x/y, pip_x/y, dip_x/y]  (8 coords)
  [wrist_vx/vy, mcp_vx/vy, pip_vx/vy, dip_vx/vy]  (8 vels, step-1 padded with zeros)

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
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "lstm_combined.csv"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCH_NAME   = "LSTM_Combined"
SEQ_LEN     = 5      # 5 frame timesteps
FEATURE_DIM = 16     # 8 coords + 8 velocities per step
EPOCHS      = 60
SEED        = 42

# ── 10 Hyperparameter Configurations ──────────────────────────────────────────
CONFIGS = [
    {"id": 1, "hidden": 64, "layers": 2, "dropout": 0.20, "lr": 1e-3, "bs": 32},
]

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    def parse(path):
        df = pd.read_csv(path)
        n  = len(df)
        X  = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)

        for k in range(1, 6):
            # 8 coordinates
            coord_cols = [f"wrist{k}_x", f"wrist{k}_y",
                          f"mcp{k}_x",   f"mcp{k}_y",
                          f"pip{k}_x",   f"pip{k}_y",
                          f"dip{k}_x",   f"dip{k}_y"]
            coords = df[coord_cols].fillna(0.0).values.astype(np.float32)

            # 8 velocities (step k=1 → zero-padded)
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
