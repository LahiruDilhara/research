"""
arch_lstm_coords.py
===================
LSTM trained on COORDINATES ONLY.
Input shape: (N, 5, 8) — 5 frame timesteps × 8 landmark coordinate features.
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
    SequenceLSTM, normalize, make_loaders, train_config,
    evaluate_model, save_result, get_data_paths, parse_labels,
    print_terminal_curves, save_history, plot_matplotlib_curves
)

BASE        = Path(__file__).resolve().parent.parent
RESULTS_CSV = Path(__file__).resolve().parent / "results" / "lstm_coords.csv"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCH_NAME   = "LSTM_Coords"
SEQ_LEN     = 5
FEATURE_DIM = 8
EPOCHS      = 70
SEED        = 42

CONFIGS = [
    {"id": 1, "hidden": 32, "layers": 2, "dropout": 0.20, "lr": 1e-3, "bs": 32},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Train {ARCH_NAME}")
    parser.add_argument("--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--dropout", "-d", type=float, default=None, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", "-bs", type=int, default=None, help="Batch size")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--plot", "-p", "--plot-curves", dest="plot", action="store_true", help="Generate Matplotlib Loss & Acc curve plots")
    return parser.parse_args()


def load_data():
    train_csv, test_csv = get_data_paths(BASE)
    def parse(path):
        df = pd.read_csv(path)
        n  = len(df)
        X  = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for k in range(1, 6):
            cols = [
                f"wrist{k}_x", f"wrist{k}_y",
                f"pip{k}_x",   f"pip{k}_y",
                f"dip{k}_x",   f"dip{k}_y",
                f"tip{k}_x",   f"tip{k}_y"
            ]
            X[:, k - 1, :] = df[cols].fillna(0.0).values.astype(np.float32)
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
            cfg["hidden"] = args.hidden

        print(f"\n  [Config {cfg['id']:02d}]  hidden={cfg['hidden']}  layers={cfg['layers']}  "
              f"dropout={cfg['dropout']}  lr={cfg['lr']}  bs={cfg['bs']}")

        train_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te, cfg["bs"])

        model = SequenceLSTM(
            input_features=FEATURE_DIM,
            hidden_units=cfg["hidden"],
            num_layers=cfg["layers"],
            dropout=cfg["dropout"],
        ).to(device)

        t0                        = time.time()
        best_acc, final_acc, hist = train_config(model, train_loader, test_loader, args.epochs, cfg["lr"], device, verbose=True)
        elapsed                   = time.time() - t0

        print_terminal_curves(hist, title=ARCH_NAME)

        history_path = Path(__file__).resolve().parent / "results" / f"{ARCH_NAME.lower()}_history.json"
        save_history(hist, history_path)

        if args.plot:
            plot_path = Path(__file__).resolve().parent / "results" / "plots" / f"{ARCH_NAME.lower()}.png"
            plot_matplotlib_curves(hist, title=ARCH_NAME, save_path=plot_path, show=False)

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
