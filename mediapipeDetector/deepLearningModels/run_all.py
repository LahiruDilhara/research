"""
run_all.py
==========
Master runner: runs all architecture training scripts ONE AT A TIME (sequential).

CLI Flags:
  --epochs, -e      Override training epochs for all models (default: use script default)
  --dropout, -d     Override dropout for all models (default: use script default)
  --lr              Override learning rate for all models (default: use script default)
  --batch-size, -bs Override batch size for all models (default: use script default)
  --hidden          Override hidden dimension for all models (default: use script default)

Usage:
  python3 deepLearningModels/run_all.py
  python3 deepLearningModels/run_all.py --epochs 50 --dropout 0.3 --lr 0.001 -bs 64
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Architecture scripts ──────────────────────────────────────────────────────
SCRIPTS = [
    "arch_lstm_velocities.py",      # LSTM (vel only      4×8 )
    "arch_lstm_coords.py",          # LSTM (coords        5×8 )
    "arch_lstm_combined.py",        # LSTM (combined      4×16)
    "arch_lstm_vel_speed.py",       # LSTM (vel + speed   4×12)
    "arch_lstm_all_joints_vel.py",  # LSTM (all 9 joints  4×18)
    "arch_bilstm.py",               # BiLSTM (combined     4×16)
    "arch_cnn1d.py",                # 1D CNN (combined      4×16)
    "arch_resnet1d.py",             # 1D ResNet (combined   4×16)
    "arch_attention.py",            # Transformer (combined  4×16)
    "arch_tcn.py",                  # TCN (combined        4×16)
]

ARCH_DESCRIPTIONS = {
    "arch_lstm_velocities.py":     "LSTM  (vel only      4×8 )",
    "arch_lstm_coords.py":         "LSTM  (coords        5×8 )",
    "arch_lstm_combined.py":       "LSTM  (combined      4×16)",
    "arch_lstm_vel_speed.py":      "LSTM  (vel + speed   4×12)",
    "arch_lstm_all_joints_vel.py": "LSTM  (all 9 joints  4×18)",
    "arch_bilstm.py":              "BiLSTM (combined     4×16)",
    "arch_cnn1d.py":               "CNN1D (combined      4×16)",
    "arch_resnet1d.py":            "ResNet1D (combined   4×16)",
    "arch_attention.py":           "Attention (combined  4×16)",
    "arch_tcn.py":                 "TCN   (combined      4×16)",
}


def _find_python() -> str:
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description="Run all deep learning model benchmarks sequentially.")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override default training epochs for all models")
    parser.add_argument("--dropout", "-d", type=float, default=None, help="Override default dropout probability for all models")
    parser.add_argument("--lr", type=float, default=None, help="Override default learning rate for all models")
    parser.add_argument("--batch-size", "-bs", type=int, default=None, help="Override default batch size for all models")
    parser.add_argument("--hidden", type=int, default=None, help="Override default hidden dimension for all models")
    return parser.parse_args()


def main():
    cli_args = parse_args()

    base    = Path(__file__).resolve().parent
    python  = _find_python()
    log_dir = base / "results"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous CSV result files before running new benchmarks
    for old_csv in log_dir.glob("*.csv"):
        try:
            old_csv.unlink()
        except Exception:
            pass

    # Build forward CLI flags to pass to internal scripts
    extra_flags = []
    if cli_args.epochs is not None:
        extra_flags.extend(["--epochs", str(cli_args.epochs)])
    if cli_args.dropout is not None:
        extra_flags.extend(["--dropout", str(cli_args.dropout)])
    if cli_args.lr is not None:
        extra_flags.extend(["--lr", str(cli_args.lr)])
    if cli_args.batch_size is not None:
        extra_flags.extend(["--batch-size", str(cli_args.batch_size)])
    if cli_args.hidden is not None:
        extra_flags.extend(["--hidden", str(cli_args.hidden)])

    bar = "=" * 70
    print(f"\n{bar}")
    print("  DEEP LEARNING BENCHMARK RUNNER")
    print(f"  {len(SCRIPTS)} Architectures (Sequential Training)")
    if extra_flags:
        print(f"  Active Overrides: {' '.join(extra_flags)}")
    else:
        print(f"  Running with default script parameters (no overrides passed)")
    print(f"{bar}\n")

    completed = []
    t_total_start = time.time()

    for idx, script in enumerate(SCRIPTS, 1):
        arch_desc = ARCH_DESCRIPTIONS.get(script, script)
        log_path  = log_dir / f"{script.replace('.py', '')}.log"

        print(f"\n{'─'*70}")
        print(f"  [{idx}/{len(SCRIPTS)}]  {arch_desc}")
        print(f"  Output also saved → results/{log_path.name}")
        print(f"{'─'*70}\n")

        t_start = time.time()

        with open(log_path, "w") as log_f:
            cmd = [python, "-u", str(base / script)] + extra_flags
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(base.parent),
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
                log_f.flush()

            proc.wait()

        elapsed = time.time() - t_start
        minutes, secs = divmod(int(elapsed), 60)
        rc     = proc.returncode
        status = "✓ DONE" if rc == 0 else f"✗ FAILED (exit {rc})"
        completed.append((arch_desc, status, minutes, secs))
        print(f"\n  {status}  —  {arch_desc}  ({minutes}m {secs}s)\n")

    total_elapsed = time.time() - t_total_start
    total_min, total_sec = divmod(int(total_elapsed), 60)

    print(f"\n{bar}")
    print("  ALL ARCHITECTURES COMPLETE")
    print(f"{bar}")
    for arch_desc, status, m, s in completed:
        print(f"  {status:10s}  {arch_desc}  ({m}m {s}s)")
    print(f"\n  Total time: {total_min}m {total_sec}s")
    print(f"{bar}\n")

    print("  Generating results comparison report...\n")
    result = subprocess.run(
        [python, str(base / "compare_results.py")],
        cwd=str(base.parent),
    )

    if result.returncode != 0:
        print("\n  [Warning] compare_results.py exited with an error.")


if __name__ == "__main__":
    main()
