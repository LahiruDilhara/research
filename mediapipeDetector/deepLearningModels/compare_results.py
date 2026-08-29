"""
compare_results.py
==================
Reads all result CSVs from deepLearningModels/results/ and prints a
ranked summary table across all training runs.

Appends a single-line master experiment run entry to results/experiment_history.log
containing model metrics, dataset lengths, hyperparameters, and process.sh pipeline lines.

Optionally generates a master grid plot of Loss & Accuracy curve pairs for
all models when --plot flag is supplied.

Usage:
  python deepLearningModels/compare_results.py
  python deepLearningModels/compare_results.py --plot
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_CSV = RESULTS_DIR / "summary_all.csv"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_history.log"

# Columns to display
DISPLAY_COLS = [
    "rank", "arch", "config_id",
    "best_test_acc", "f1_touch", "precision_touch", "recall_touch",
    "fit_status", "onset_epoch", "max_gap_pct",
    "train_time_s",
]

PARAM_COLS_PER_ARCH = {
    "LSTM_Velocities":      ["hidden", "layers", "dropout", "lr", "batch_size"],
    "LSTM_Coords":          ["hidden", "layers", "dropout", "lr", "batch_size"],
    "LSTM_Combined":        ["hidden", "layers", "dropout", "lr", "batch_size"],
    "LSTM_Vel_Speed":       ["hidden", "layers", "dropout", "lr", "batch_size"],
    "LSTM_All_Joints_Vel":  ["hidden", "layers", "dropout", "lr", "batch_size"],
    "CNN1D":                ["conv_ch", "fc_hid", "dropout", "lr", "batch_size"],
    "ResNet1D":             ["hidden_dim", "dropout", "lr", "batch_size"],
    "Attention":            ["embed_dim", "num_heads", "dropout", "lr", "batch_size"],
    "BiLSTM":               ["hidden", "layers", "dropout", "lr", "batch_size"],
    "TCN":                  ["tcn_channels", "num_levels", "dropout", "lr", "batch_size"],
}

BAR  = "=" * 90
DASH = "─" * 90


def parse_args():
    parser = argparse.ArgumentParser(description="Compare deep learning benchmark results.")
    parser.add_argument("--plot", "-p", "--plot-curves", dest="plot", action="store_true", help="Generate master Matplotlib curves comparison grid")
    return parser.parse_args()


def load_all_results() -> pd.DataFrame:
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != SUMMARY_CSV.name]

    if not csv_files:
        print("\n  No result CSVs found in results/. Run run_all.py first.\n")
        sys.exit(0)

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [Warning] Could not read {f.name}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _get_acc_pct(val) -> float:
    f = float(val)
    return f if f > 1.0 else f * 100.0


def fmt_row(df: pd.DataFrame) -> pd.DataFrame:
    if "best_test_acc" in df.columns:
        df["best_test_acc"] = df["best_test_acc"].apply(lambda v: f"{_get_acc_pct(v):.2f}%")
    if "final_test_acc" in df.columns:
        df["final_test_acc"] = df["final_test_acc"].apply(lambda v: f"{_get_acc_pct(v):.2f}%")
    for col in ["f1_touch", "precision_touch", "recall_touch"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{float(v):.4f}")
    if "train_time_s" in df.columns:
        df["train_time_s"] = df["train_time_s"].apply(lambda v: f"{float(v):.1f}s")
    return df


def print_full_table(df: pd.DataFrame):
    df_sorted = df.sort_values("best_test_acc_raw", ascending=False).reset_index(drop=True)
    df_sorted.insert(0, "rank", df_sorted.index + 1)

    print(f"\n{BAR}")
    print("  FULL BENCHMARK RESULTS  (sorted by Test Accuracy)")
    print(f"{BAR}")

    display = df_sorted.copy()
    display = fmt_row(display)
    cols    = [c for c in DISPLAY_COLS if c in display.columns]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(display[cols].to_string(index=False))
    print(f"{BAR}")


def print_top_n(df: pd.DataFrame, n: int = 5):
    top = df.sort_values("best_test_acc_raw", ascending=False).head(n)
    print(f"\n{BAR}")
    print(f"  TOP {n} CONFIGURATIONS OVERALL")
    print(f"{BAR}")
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        arch = row.get("arch", "?")
        cid  = int(row.get("config_id", 0))
        acc  = _get_acc_pct(row["best_test_acc_raw"])
        f1   = float(row.get("f1_touch", 0))
        prec = float(row.get("precision_touch", 0))
        rec  = float(row.get("recall_touch", 0))
        t    = row.get("train_time_s", "?")
        print(f"  #{rank}  {arch:22s}  cfg{cid:02d}  →  "
              f"Acc={acc:6.2f}%  |  F1={f1:.4f}  |  "
              f"Prec={prec:.4f}  Rec={rec:.4f}  |  {t}s")
    print(f"{BAR}")


def print_per_arch_best(df: pd.DataFrame):
    print(f"\n{DASH}")
    print("  BEST CONFIG PER ARCHITECTURE")
    print(f"{DASH}")
    for arch, grp in df.groupby("arch"):
        best = grp.loc[grp["best_test_acc_raw"].idxmax()]
        cid  = int(best.get("config_id", 0))
        acc  = _get_acc_pct(best["best_test_acc_raw"])
        f1   = float(best.get("f1_touch", 0))
        fit  = best.get("fit_status", "N/A")
        ep   = best.get("onset_epoch", "?")

        param_keys = PARAM_COLS_PER_ARCH.get(arch, [])
        params = "  ".join(f"{k}={best[k]}" for k in param_keys if k in best)
        print(f"  {arch:22s}  cfg{cid:02d}  Acc={acc:6.2f}%  F1={f1:.4f}  |  Fit={fit} (Onset: Ep {ep})  |  {params}")
    print(f"{DASH}")


def print_arch_ranking(df: pd.DataFrame):
    print(f"\n{DASH}")
    print("  ARCHITECTURE RANKING  (by best single config accuracy)")
    print(f"{DASH}")
    arch_best = (
        df.groupby("arch")["best_test_acc_raw"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
    )
    arch_best.columns = ["arch", "best_acc"]
    for i, row in arch_best.iterrows():
        acc_val = float(row["best_acc"])
        acc_pct = acc_val * 100.0 if acc_val <= 1.0 else acc_val
        bar_len = int((acc_pct / 100.0) * 15)
        bar_vis = "█" * bar_len
        print(f"  #{i+1}  {row['arch']:22s}  {acc_pct:6.2f}%  {bar_vis}")
    print(f"{DASH}")


def plot_all_models_grid():
    """Generates a master grid image of Loss & Accuracy curves for all models."""
    import matplotlib.pyplot as plt

    history_files = sorted(RESULTS_DIR.glob("*_history.json"))
    if not history_files:
        return

    n_models = len(history_files)
    fig, axes = plt.subplots(n_models, 2, figsize=(13, 3.2 * n_models), dpi=150)
    if n_models == 1:
        axes = [axes]

    for idx, hfile in enumerate(history_files):
        arch_name = hfile.name.replace("_history.json", "").upper()
        try:
            with open(hfile, "r") as f:
                hist = json.load(f)
        except Exception:
            continue

        tr_loss = hist.get("train_loss", [])
        te_loss = hist.get("test_loss", [])
        tr_acc  = hist.get("train_acc", [])
        te_acc  = hist.get("test_acc", [])

        if not tr_loss or not te_loss:
            continue

        epochs  = list(range(1, len(tr_loss) + 1))
        ax_loss = axes[idx][0]
        ax_acc  = axes[idx][1]

        # Loss Plot
        ax_loss.plot(epochs, tr_loss, label="Train Loss", color="#e74c3c", linewidth=1.8, marker="o", markersize=2)
        ax_loss.plot(epochs, te_loss, label="Test Loss", color="#3498db", linewidth=1.8, linestyle="--", marker="s", markersize=2)
        ax_loss.set_title(f"{arch_name} — Loss Curve", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("BCE Loss", fontsize=9)
        ax_loss.grid(True, linestyle=":", alpha=0.6)
        ax_loss.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

        # Accuracy Plot
        ax_acc.plot(epochs, tr_acc, label="Train Acc", color="#2ecc71", linewidth=1.8, marker="o", markersize=2)
        ax_acc.plot(epochs, te_acc, label="Test Acc", color="#f39c12", linewidth=1.8, linestyle="--", marker="s", markersize=2)
        ax_acc.set_title(f"{arch_name} — Accuracy Curve (%)", fontsize=11, fontweight="bold")
        ax_acc.set_xlabel("Epoch", fontsize=9)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
        ax_acc.grid(True, linestyle=":", alpha=0.6)
        ax_acc.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    grid_path = plot_dir / "all_models_curves_grid.png"
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")
    print(f"\n  📊 Master Matplotlib comparison grid saved → results/plots/{grid_path.name}\n", flush=True)
    plt.close(fig)


def get_process_sh_commands(base_dir: Path) -> str:
    process_sh = base_dir / "process.sh"
    if not process_sh.exists():
        return "process.sh=N/A"
    cmds = []
    with open(process_sh, "r") as f:
        for line in f:
            l = line.strip()
            if l.startswith("python3 "):
                cmds.append(l)
    return " ; ".join(cmds) if cmds else "process.sh=empty"


def get_dataset_info(base_dir: Path) -> str:
    p1_tr = base_dir / "training_testing_data" / "train_dataset.csv"
    p1_te = base_dir / "training_testing_data" / "test_dataset.csv"
    if not p1_tr.exists() or not p1_te.exists():
        p1_tr = base_dir / "dataprocessing" / "11_train_test_split" / "training_dataset.csv"
        p1_te = base_dir / "dataprocessing" / "11_train_test_split" / "testing_dataset.csv"

    if not p1_tr.exists() or not p1_te.exists():
        return "train_len=N/A, test_len=N/A"

    try:
        df_tr = pd.read_csv(p1_tr)
        df_te = pd.read_csv(p1_te)
        tr_len = len(df_tr)
        te_len = len(df_te)

        tc_tr = [c for c in df_tr.columns if "touch" in c.lower() or "label" in c.lower() or "target" in c.lower()]
        tc_te = [c for c in df_te.columns if "touch" in c.lower() or "label" in c.lower() or "target" in c.lower()]

        tr_touch = int(df_tr[tc_tr[0]].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y", "1.0"]).sum()) if tc_tr else "?"
        te_touch = int(df_te[tc_te[0]].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y", "1.0"]).sum()) if tc_te else "?"

        return f"train_len={tr_len} (touch={tr_touch}), test_len={te_len} (touch={te_touch})"
    except Exception as e:
        return f"train_len=err, test_len=err ({e})"


def append_master_log_entry(df: pd.DataFrame):
    base_dir = RESULTS_DIR.parent.parent
    master_log_path = RESULTS_DIR / "experiment_history.log"
    master_log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_entries = []
    hyper_entries = []

    for arch, grp in df.groupby("arch"):
        best = grp.loc[grp["best_test_acc_raw"].idxmax()]
        acc  = _get_acc_pct(best["best_test_acc_raw"])
        f1   = float(best.get("f1_touch", 0))
        prec = float(best.get("precision_touch", 0))
        rec  = float(best.get("recall_touch", 0))
        t    = float(best.get("train_time_s", 0))

        model_entries.append(f"{arch}: Acc={acc:.2f}%, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Time={t:.1f}s")

        param_keys = PARAM_COLS_PER_ARCH.get(arch, [])
        p_str = ", ".join(f"{k}={best[k]}" for k in param_keys if k in best)
        hyper_entries.append(f"{arch}({p_str})")

    models_summary = " | ".join(model_entries)
    hypers_summary = " ; ".join(hyper_entries)
    dataset_summary = get_dataset_info(base_dir)
    process_sh_summary = get_process_sh_commands(base_dir)

    line = (
        f"[{timestamp}] MODELS_PERFORMANCE: [{models_summary}] | "
        f"HYPERPARAMS: [{hypers_summary}] | "
        f"DATASET: [{dataset_summary}] | "
        f"PIPELINE (process.sh): [{process_sh_summary}]\n"
    )

    with open(master_log_path, "a") as f:
        f.write(line)

    print(f"  📝 Master run entry logged to → results/experiment_history.log", flush=True)


def main():
    args = parse_args()
    df   = load_all_results()

    if df.empty:
        print("  No data to compare.\n")
        return

    df["best_test_acc_raw"] = pd.to_numeric(df["best_test_acc"], errors="coerce")
    df["f1_touch"]          = pd.to_numeric(df.get("f1_touch", pd.Series()), errors="coerce").fillna(0)
    df["precision_touch"]   = pd.to_numeric(df.get("precision_touch", pd.Series()), errors="coerce").fillna(0)
    df["recall_touch"]      = pd.to_numeric(df.get("recall_touch", pd.Series()), errors="coerce").fillna(0)

    print_arch_ranking(df)
    print_per_arch_best(df)
    print_top_n(df, n=5)
    print_full_table(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["best_test_acc_raw"], errors="ignore").sort_values(
        "best_test_acc", ascending=False
    ).to_csv(SUMMARY_CSV, index=False)
    print(f"\n  Full ranked summary saved → {SUMMARY_CSV}\n")

    append_master_log_entry(df)

    if args.plot:
        plot_all_models_grid()


if __name__ == "__main__":
    main()
