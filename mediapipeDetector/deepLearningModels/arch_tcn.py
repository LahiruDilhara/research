"""
arch_tcn.py
===========
Temporal Convolutional Network (TCN) on combined landmark coordinates + velocities.
Input shape: (N, 4, 16).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchTCN, normalize, make_loaders, train_config,
    evaluate_model, save_result, get_data_paths, parse_labels,
    print_terminal_curves, save_history, plot_matplotlib_curves,
    analyze_fit_quality, print_overfit_analytics,
    print_terminal_confusion_matrix, plot_matplotlib_confusion_matrix
)

BASE        = Path(__file__).resolve().parent.parent
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "tcn.csv"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCH_NAME   = "TCN"
SEQ_LEN     = 4
FEATURE_DIM = 16
EPOCHS      = 70
SEED        = 42

CONFIGS = [
    {"id": 1, "tcn_channels": 32, "num_levels": 2, "dropout": 0.20, "lr": 1e-3, "bs": 32},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Train {ARCH_NAME}")
    parser.add_argument("--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--dropout", "-d", type=float, default=None, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", "-bs", type=int, default=None, help="Batch size")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden dimension (tcn channels)")
    parser.add_argument("--plot", "-p", "--plot-curves", dest="plot", action="store_true", help="Generate Matplotlib Loss & Acc curve plots and Confusion Matrix heatmaps")
    return parser.parse_args()


def load_data():
    train_csv, test_csv = get_data_paths(BASE)
    def parse(path):
        df = pd.read_csv(path)
        n  = len(df)
        X  = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for v in range(1, 5):
            coord_cols = [
                f"wrist{v}_x", f"wrist{v}_y",
                f"pip{v}_x",   f"pip{v}_y",
                f"dip{v}_x",   f"dip{v}_y",
                f"tip{v}_x",   f"tip{v}_y"
            ]
            vel_cols = [
                f"wrist{v}_vx", f"wrist{v}_vy",
                f"pip{v}_vx",   f"pip{v}_vy",
                f"dip{v}_vx",   f"dip{v}_vy",
                f"tip{v}_vx",   f"tip{v}_vy"
            ]
            coords = df[coord_cols].fillna(0.0).values.astype(np.float32)
            vels   = df[vel_cols].fillna(0.0).values.astype(np.float32)
            X[:, v - 1, :] = np.hstack([coords, vels])
        y = parse_labels(df)
        return X, y

    X_tr, y_tr = parse(train_csv)
    X_te, y_te = parse(test_csv)
    X_tr_t, X_te_t = normalize(X_tr, X_te)
    return X_tr_t, torch.from_numpy(y_tr).float(), X_te_t, torch.from_numpy(y_te).float()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  {ARCH_NAME}  |  Input: (N, {SEQ_LEN}, {FEATURE_DIM})  |  Device: {device}")
    print(f"{'='*65}")

    X_tr, y_tr, X_te, y_te = load_data()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        if args.dropout is not None:
            cfg["dropout"] = args.dropout
        if args.lr is not None:
            cfg["lr"] = args.lr
        if args.batch_size is not None:
            cfg["bs"] = args.batch_size
        if args.hidden is not None:
            cfg["tcn_channels"] = args.hidden

        print(f"\n  [Config {cfg['id']:02d}]  hidden={cfg['tcn_channels']}  "
              f"dropout={cfg['dropout']}  lr={cfg['lr']}  bs={cfg['bs']}")

        train_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te, cfg["bs"])

        model = TouchTCN(
            in_channels=FEATURE_DIM,
            tcn_channels=cfg["tcn_channels"],
            num_levels=cfg["num_levels"],
            dropout=cfg["dropout"],
        ).to(device)

        t0                        = time.time()
        best_acc, final_acc, hist = train_config(model, train_loader, test_loader, args.epochs, cfg["lr"], device, verbose=True)
        elapsed                   = time.time() - t0

        print_terminal_curves(hist, title=ARCH_NAME)
        analytics = analyze_fit_quality(hist)
        print_overfit_analytics(analytics, title=ARCH_NAME)

        history_path = Path(__file__).resolve().parent / "results" / f"{ARCH_NAME.lower()}_history.json"
        save_history(hist, history_path)

        cm, rpt = evaluate_model(model, test_loader, device)
        print_terminal_confusion_matrix(cm, title=ARCH_NAME)

        if args.plot:
            plot_path = Path(__file__).resolve().parent / "results" / "plots" / f"{ARCH_NAME.lower()}.png"
            plot_matplotlib_curves(hist, title=ARCH_NAME, save_path=plot_path, show=False)

            cm_plot_path = Path(__file__).resolve().parent / "results" / "plots" / f"{ARCH_NAME.lower()}_confusion_matrix.png"
            plot_matplotlib_confusion_matrix(cm, title=ARCH_NAME, save_path=cm_plot_path, show=False)

        wf = WEIGHTS_DIR / f"{ARCH_NAME}_cfg{cfg['id']:02d}.pth"
        torch.save(model.state_dict(), wf)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        row = {
            "arch":             ARCH_NAME,
            "config_id":        cfg["id"],
            "tcn_channels":     cfg["tcn_channels"],
            "num_levels":       cfg["num_levels"],
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

    print(f"\n  ✓ {ARCH_NAME} complete. Results → {RESULTS_CSV.name}\n")


if __name__ == "__main__":
    main()
