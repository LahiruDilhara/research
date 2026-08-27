"""
compare_results.py
==================
Reads all result CSVs from deepLearningModels/results/ and prints a
ranked summary table across all 60 training runs (6 archs × 10 configs).

Usage:
  python deepLearningModels/compare_results.py
  (also called automatically by run_all.py after training)
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_CSV = RESULTS_DIR / "summary_all.csv"

# Columns to display (gracefully skip any that are missing for certain arches)
DISPLAY_COLS = [
    "rank", "arch", "config_id",
    "best_test_acc", "f1_touch", "precision_touch", "recall_touch",
    "train_time_s",
]

PARAM_COLS_PER_ARCH = {
    "LSTM_Velocities": ["hidden", "layers", "dropout", "lr", "batch_size"],
    "LSTM_Coords":     ["hidden", "layers", "dropout", "lr", "batch_size"],
    "LSTM_Combined":   ["hidden", "layers", "dropout", "lr", "batch_size"],
    "CNN1D":           ["conv_ch", "fc_hid", "dropout", "lr", "batch_size"],
    "ResNet1D":        ["hidden_dim", "dropout", "lr", "batch_size"],
    "Attention":       ["embed_dim", "num_heads", "dropout", "lr", "batch_size"],
}

BAR  = "=" * 90
DASH = "─" * 90


def load_all_results() -> pd.DataFrame:
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    # Exclude any previously generated summary
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


def fmt_row(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for display."""
    for col in ["best_test_acc", "final_test_acc", "f1_touch", "precision_touch", "recall_touch"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2).astype(str) + "%"
    if "train_time_s" in df.columns:
        df["train_time_s"] = df["train_time_s"].round(1).astype(str) + "s"
    return df


def print_full_table(df: pd.DataFrame):
    """Print all 60 rows sorted by test accuracy."""
    df_sorted = df.sort_values("best_test_acc_raw", ascending=False).reset_index(drop=True)
    df_sorted.insert(0, "rank", df_sorted.index + 1)

    print(f"\n{BAR}")
    print("  FULL BENCHMARK RESULTS  ─  All 60 Runs  (sorted by Test Accuracy)")
    print(f"{BAR}")

    display = df_sorted.copy()
    display = fmt_row(display)
    cols    = [c for c in DISPLAY_COLS if c in display.columns]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(display[cols].to_string(index=False))
    print(f"{BAR}")


def print_top_n(df: pd.DataFrame, n: int = 5):
    """Print Top-N configurations overall."""
    top = df.sort_values("best_test_acc_raw", ascending=False).head(n)
    print(f"\n{BAR}")
    print(f"  TOP {n} CONFIGURATIONS OVERALL")
    print(f"{BAR}")
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        arch = row.get("arch", "?")
        cid  = int(row.get("config_id", 0))
        acc  = float(row["best_test_acc_raw"]) * 100
        f1   = float(row.get("f1_touch", 0))
        prec = float(row.get("precision_touch", 0))
        rec  = float(row.get("recall_touch", 0))
        t    = row.get("train_time_s", "?")
        print(f"  #{rank}  {arch:22s}  cfg{cid:02d}  →  "
              f"Acc={acc:6.2f}%  |  F1={f1:.4f}  |  "
              f"Prec={prec:.4f}  Rec={rec:.4f}  |  {t}s")
    print(f"{BAR}")


def print_per_arch_best(df: pd.DataFrame):
    """Print best config per architecture."""
    print(f"\n{DASH}")
    print("  BEST CONFIG PER ARCHITECTURE")
    print(f"{DASH}")
    for arch, grp in df.groupby("arch"):
        best = grp.loc[grp["best_test_acc_raw"].idxmax()]
        cid  = int(best.get("config_id", 0))
        acc  = float(best["best_test_acc_raw"]) * 100
        f1   = float(best.get("f1_touch", 0))
        t    = best.get("train_time_s", "?")

        # Collect config params for this arch
        param_keys = PARAM_COLS_PER_ARCH.get(arch, [])
        params = "  ".join(f"{k}={best[k]}" for k in param_keys if k in best)
        print(f"  {arch:22s}  cfg{cid:02d}  Acc={acc:6.2f}%  F1={f1:.4f}  |  {params}")
    print(f"{DASH}")


def print_arch_ranking(df: pd.DataFrame):
    """Print architectures ranked by their best accuracy."""
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
        bar_len = int(row["best_acc"] * 60)
        bar_vis = "█" * bar_len
        print(f"  #{i+1}  {row['arch']:22s}  {row['best_acc']*100:6.2f}%  {bar_vis}")
    print(f"{DASH}")


def main():
    df = load_all_results()

    if df.empty:
        print("  No data to compare.\n")
        return

    # Keep a raw numeric copy of best_test_acc for sorting
    df["best_test_acc_raw"] = pd.to_numeric(df["best_test_acc"], errors="coerce")
    df["f1_touch"]          = pd.to_numeric(df.get("f1_touch", pd.Series()), errors="coerce").fillna(0)
    df["precision_touch"]   = pd.to_numeric(df.get("precision_touch", pd.Series()), errors="coerce").fillna(0)
    df["recall_touch"]      = pd.to_numeric(df.get("recall_touch", pd.Series()), errors="coerce").fillna(0)

    print_arch_ranking(df)
    print_per_arch_best(df)
    print_top_n(df, n=5)
    print_full_table(df)

    # Save summary CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["best_test_acc_raw"], errors="ignore").sort_values(
        "best_test_acc", ascending=False
    ).to_csv(SUMMARY_CSV, index=False)
    print(f"\n  Full ranked summary saved → {SUMMARY_CSV}\n")


if __name__ == "__main__":
    main()
