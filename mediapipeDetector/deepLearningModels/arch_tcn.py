"""
arch_tcn.py
===========
Architecture 8: Temporal Convolutional Network (TCN) on combined input.
Input shape: (N, 5, 16).

TCN uses stacked dilated convolutions with exponentially growing receptive fields:
  Level 1: dilation=1  (receptive field = 3)
  Level 2: dilation=2  (receptive field = 5)
  Level 3: dilation=4  (receptive field = 9)

Each block has a residual skip connection. Global average pooling at the end.
All 4 hyperparameter configs are trained IN PARALLEL using GPU threads.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchTCN, normalize, make_loaders,
    train_config, evaluate_model, save_result
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
TRAIN_CSV   = BASE / "data" / "training_data.csv"
TEST_CSV    = BASE / "data" / "test_data.csv"
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "tcn.csv"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCH_NAME   = "TCN"
SEQ_LEN     = 5
FEATURE_DIM = 16     # 8 coords + 8 velocities
EPOCHS      = 60
SEED        = 42

_print_lock = threading.Lock()

# ── 4 Hyperparameter Configurations ───────────────────────────────────────────
# tcn_ch : number of channels in each dilated conv layer
# levels  : number of dilation levels (dilations = [1, 2, 4, ...])
CONFIGS = [
    {"id": 1, "tcn_ch": 64, "levels": 3, "dropout": 0.20, "lr": 1e-3, "bs": 32},
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


# ── Per-config training (runs in a thread) ─────────────────────────────────────
def train_one_config(cfg, X_tr, y_tr, X_te, y_te, device):
    train_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te, cfg["bs"])
    model = TouchTCN(
        in_channels=FEATURE_DIM,
        tcn_channels=cfg["tcn_ch"],
        num_levels=cfg["levels"],
        dropout=cfg["dropout"],
    ).to(device)

    t0 = time.time()
    best_acc, final_acc, _ = train_config(
        model, train_loader, test_loader, EPOCHS, cfg["lr"], device, verbose=False
    )
    elapsed = time.time() - t0

    _, rpt = evaluate_model(model, test_loader, device)
    wf = WEIGHTS_DIR / f"{ARCH_NAME}_cfg{cfg['id']:02d}.pth"
    torch.save(model.state_dict(), wf)

    return {"cfg": cfg, "best_acc": best_acc, "final_acc": final_acc,
            "rpt": rpt, "elapsed": elapsed, "wf": wf}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  {ARCH_NAME}  |  Input: (N, {SEQ_LEN}, {FEATURE_DIM})  |  Device: {device}")
    print(f"  Launching all {len(CONFIGS)} configs in parallel on {device} ...")
    print(f"{'='*65}\n")

    X_tr, y_tr, X_te, y_te = load_data()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    done    = 0
    t_wall  = time.time()

    with ThreadPoolExecutor(max_workers=len(CONFIGS)) as executor:
        futures = {
            executor.submit(train_one_config, cfg, X_tr, y_tr, X_te, y_te, device): cfg
            for cfg in CONFIGS
        }
        for future in as_completed(futures):
            r     = future.result()
            done += 1
            cfg   = r["cfg"]
            with _print_lock:
                print(
                    f"  ✓ cfg{cfg['id']:02d}  tcn_ch={cfg['tcn_ch']:3d}  levels={cfg['levels']}  "
                    f"lr={cfg['lr']}  →  Best: {r['best_acc']:.2f}%  "
                    f"F1: {r['rpt']['Touch']['f1-score']:.4f}  "
                    f"({r['elapsed']:.0f}s)  [{done}/{len(CONFIGS)} done]"
                )
            results.append(r)

    wall_elapsed = time.time() - t_wall
    results.sort(key=lambda r: r["cfg"]["id"])

    for r in results:
        cfg = r["cfg"]
        save_result({
            "arch":             ARCH_NAME,
            "config_id":        cfg["id"],
            "tcn_channels":     cfg["tcn_ch"],
            "num_levels":       cfg["levels"],
            "dropout":          cfg["dropout"],
            "lr":               cfg["lr"],
            "batch_size":       cfg["bs"],
            "best_test_acc":    round(r["best_acc"],  4),
            "final_test_acc":   round(r["final_acc"], 4),
            "precision_touch":  round(r["rpt"]["Touch"]["precision"], 4),
            "recall_touch":     round(r["rpt"]["Touch"]["recall"],    4),
            "f1_touch":         round(r["rpt"]["Touch"]["f1-score"],  4),
            "train_time_s":     round(r["elapsed"], 1),
            "weight_file":      r["wf"].name,
        }, str(RESULTS_CSV))

    print(f"\n  {'─'*60}")
    print(f"  {ARCH_NAME}  —  All {len(CONFIGS)} Results (wall time: {wall_elapsed:.0f}s)")
    print(f"  {'─'*60}")
    for r in results:
        cfg = r["cfg"]
        print(f"  cfg{cfg['id']:02d}  Best: {r['best_acc']:6.2f}%  "
              f"F1: {r['rpt']['Touch']['f1-score']:.4f}  "
              f"tcn_ch={cfg['tcn_ch']:3d}  levels={cfg['levels']}  lr={cfg['lr']}")
    print(f"  {'─'*60}")
    print(f"\n  ✓ {ARCH_NAME} complete. Results → {RESULTS_CSV.name}\n")


if __name__ == "__main__":
    main()
