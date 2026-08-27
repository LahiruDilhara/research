"""
run_all.py
==========
Master runner: runs all 6 architecture training scripts ONE AT A TIME (sequential).

Each script trains 10 hyperparameter configs sequentially.
Output is streamed live to the terminal and also saved to results/<arch>.log

Usage:
  cd /path/to/mediapipeDetector
  .venv/bin/python3 deepLearningModels/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Architecture scripts (ordered for readability) ────────────────────────────
SCRIPTS = [
    "arch_lstm_velocities.py",   # LSTM on velocities only      (N, 4, 8)
    "arch_lstm_coords.py",       # LSTM on coordinates only     (N, 5, 8)
    "arch_lstm_combined.py",     # LSTM on coords + velocities  (N, 5, 16)
    "arch_cnn1d.py",             # 1D CNN on combined input     (N, 5, 16)
    "arch_resnet1d.py",          # 1D ResNet on combined input  (N, 5, 16)
    "arch_attention.py",         # Transformer on combined      (N, 5, 16)
    "arch_bilstm.py",            # Bidirectional LSTM           (N, 5, 16)
    "arch_tcn.py",               # Temporal Conv Network (TCN)  (N, 5, 16)
]

ARCH_DESCRIPTIONS = {
    "arch_lstm_velocities.py": "LSTM  (vel only  4×8 )",
    "arch_lstm_coords.py":     "LSTM  (coords    5×8 )",
    "arch_lstm_combined.py":   "LSTM  (combined  5×16)",
    "arch_cnn1d.py":           "CNN1D (combined  5×16)",
    "arch_resnet1d.py":        "ResNet1D (combined 5×16)",
    "arch_attention.py":       "Attention (combined 5×16)",
    "arch_bilstm.py":          "BiLSTM (combined 5×16)",
    "arch_tcn.py":             "TCN   (combined  5×16)",
}


def _find_python() -> str:
    """
    Prefer .venv/bin/python3 in the project root (one level up from this file).
    Falls back to the interpreter that launched this script.
    """
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main():
    base    = Path(__file__).resolve().parent
    python  = _find_python()
    log_dir = base / "results"
    log_dir.mkdir(parents=True, exist_ok=True)

    bar = "=" * 70
    print(f"\n{bar}")
    print("  DEEP LEARNING BENCHMARK RUNNER")
    print(f"  {len(SCRIPTS)} Architectures × 4 Configs = {len(SCRIPTS)*4} total training runs")
    print(f"  Running ONE architecture at a time  (sequential mode)")
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

        # Stream output live to terminal AND save to log file simultaneously
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                [python, str(base / script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(base.parent),   # project root so ./data/ paths resolve
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)

            proc.wait()

        elapsed = time.time() - t_start
        minutes, secs = divmod(int(elapsed), 60)
        rc     = proc.returncode
        status = "✓ DONE" if rc == 0 else f"✗ FAILED (exit {rc})"
        completed.append((arch_desc, status, minutes, secs))
        print(f"\n  {status}  —  {arch_desc}  ({minutes}m {secs}s)\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total_start
    total_min, total_sec = divmod(int(total_elapsed), 60)

    print(f"\n{bar}")
    print("  ALL ARCHITECTURES COMPLETE")
    print(f"{bar}")
    for arch_desc, status, m, s in completed:
        print(f"  {status:10s}  {arch_desc}  ({m}m {s}s)")
    print(f"\n  Total time: {total_min}m {total_sec}s")
    print(f"{bar}\n")

    # ── Auto-run the comparison report ────────────────────────────────────────
    print("  Generating results comparison report...\n")
    result = subprocess.run(
        [python, str(base / "compare_results.py")],
        cwd=str(base.parent),
    )

    if result.returncode != 0:
        print("\n  [Warning] compare_results.py exited with an error.")
        print("  Run it manually: .venv/bin/python3 deepLearningModels/compare_results.py")


if __name__ == "__main__":
    main()
