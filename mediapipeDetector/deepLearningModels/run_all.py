"""
run_all.py
==========
Master runner for deep learning model benchmarks.

Supports architecture pool filtering (-m), input representation filtering (-i),
randomized sampling (-r), and model count control (-n).

CLI Flags:
  -n, --num-models INT   Number of models to run from pool (e.g. -n 5, -n 20, -n 1000). Runs all if > pool size.
  -r, --random           Randomize model selection from pool.
  -m, --models STR       Comma-separated architecture name filter (e.g. -m resnet,lstm,bilstm,cnn,tcn,attention).
  -i, --inputs STR       Comma-separated input representation filter (e.g. -i coords,coords_3d,vel_3d,super_combined,wrist_rel_3d,tip_vel_ratios).
  --epochs, -e INT       Override training epochs for all selected models.
  --dropout, -d FLOAT    Override dropout probability.
  --lr FLOAT             Override learning rate.
  --batch-size, -bs INT  Override batch size.
  --hidden INT           Override hidden dimension.
  --plot, -p             Generate Jupyter notebook style Matplotlib Loss/Acc curves and Confusion Matrix plots.
  --seed INT             Random seed for model shuffling (default: 42).

Usage Examples:
  python3 deepLearningModels/run_all.py --epochs 3
  python3 deepLearningModels/run_all.py -n 5 -r --epochs 10
  python3 deepLearningModels/run_all.py -m resnet -i super_combined,wrist_rel_3d -n 2
  python3 deepLearningModels/run_all.py -n 1000 --epochs 50 --plot
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

# Complete Expanded Model Pool (44 distinct architecture & feature variants)
ALL_SCRIPTS = [
    # LSTM variants (15)
    ("arch_lstm_velocities.py",        "LSTM_Velocities (2D Vels 4×8)",               "vel_2d"),
    ("arch_lstm_coords.py",            "LSTM_Coords (2D Coords 5×8)",                 "coords_2d"),
    ("arch_lstm_combined.py",          "LSTM_Combined (2D Coords+Vels 4×16)",         "combined_2d"),
    ("arch_lstm_vel_speed.py",         "LSTM_Vel_Speed (2D Vels+Speeds 4×12)",        "vel_speed_2d"),
    ("arch_lstm_all_joints_vel.py",    "LSTM_All_Joints_Vel (All 9 Joints Vels 4×18)","all_joints_vel"),
    ("arch_lstm_all_combined.py",      "LSTM_All_Combined (All Joints Coords+Vels 4×36)", "all_joints_coords_vel"),
    ("arch_lstm_coords_3d.py",         "LSTM_Coords_3D (3D Coords 5×12)",             "coords_3d"),
    ("arch_lstm_vel_3d.py",            "LSTM_Vel_3D (3D Vels 4×12)",                  "vel_3d"),
    ("arch_lstm_vel_speed_3d.py",      "LSTM_Vel_Speed_3D (3D Vels+Speeds 4×16)",    "vel_speed_3d"),
    ("arch_lstm_combined_3d.py",       "LSTM_Combined_3D (3D Coords+Vels 4×24)",     "combined_3d"),
    ("arch_lstm_z_kinematics.py",      "LSTM_Z_Kinematics (Z-Depth+Vels 4×8)",        "z_kinematics"),
    ("arch_lstm_super_combined.py",    "LSTM_Super_Combined (3D Coords+Vels+Speeds 4×28)", "super_combined"),
    ("arch_lstm_wrist_rel_3d.py",      "LSTM_Wrist_Rel_3D (Wrist-Rel 3D Coords+Vels 4×21)", "wrist_relative_3d"),
    ("arch_lstm_tip_vel_ratios.py",    "LSTM_Tip_Vel_Ratios (Tip Vel Ratios 4×16)",    "fingertip_velocity_ratios"),

    # BiLSTM variants (6)
    ("arch_bilstm.py",                 "BiLSTM (2D Coords+Vels 4×16)",                "combined_2d"),
    ("arch_bilstm_all_combined.py",     "BiLSTM_All_Combined (All Joints Coords+Vels 4×36)", "all_joints_coords_vel"),
    ("arch_bilstm_vel_3d.py",          "BiLSTM_Vel_3D (3D Vels+Speeds 4×16)",         "vel_speed_3d"),
    ("arch_bilstm_super_combined.py",  "BiLSTM_Super_Combined (Super Combined 4×28)", "super_combined"),
    ("arch_bilstm_wrist_rel_3d.py",    "BiLSTM_Wrist_Rel_3D (Wrist-Rel 3D Coords+Vels 4×21)", "wrist_relative_3d"),
    ("arch_bilstm_tip_vel_ratios.py",  "BiLSTM_Tip_Vel_Ratios (Tip Vel Ratios 4×16)", "fingertip_velocity_ratios"),

    # 1D CNN variants (6)
    ("arch_cnn1d.py",                  "CNN1D (2D Coords+Vels 4×16)",                 "combined_2d"),
    ("arch_cnn1d_all_combined.py",      "CNN1D_All_Combined (All Joints Coords+Vels 4×36)", "all_joints_coords_vel"),
    ("arch_cnn1d_vel_3d.py",           "CNN1D_Vel_3D (3D Vels+Speeds 4×16)",          "vel_speed_3d"),
    ("arch_cnn1d_super_combined.py",   "CNN1D_Super_Combined (Super Combined 4×28)",  "super_combined"),
    ("arch_cnn1d_wrist_rel_3d.py",     "CNN1D_Wrist_Rel_3D (Wrist-Rel 3D Coords+Vels 4×21)", "wrist_relative_3d"),
    ("arch_cnn1d_tip_vel_ratios.py",   "CNN1D_Tip_Vel_Ratios (Tip Vel Ratios 4×16)", "fingertip_velocity_ratios"),

    # 1D ResNet variants (6)
    ("arch_resnet1d.py",               "ResNet1D (2D Coords+Vels 4×16)",              "combined_2d"),
    ("arch_resnet1d_all_combined.py",   "ResNet1D_All_Combined (All Joints Coords+Vels 4×36)", "all_joints_coords_vel"),
    ("arch_resnet1d_vel_3d.py",        "ResNet1D_Vel_3D (3D Vels+Speeds 4×16)",       "vel_speed_3d"),
    ("arch_resnet1d_super_combined.py","ResNet1D_Super_Combined (Super Combined 4×28)", "super_combined"),
    ("arch_resnet1d_wrist_rel_3d.py",  "ResNet1D_Wrist_Rel_3D (Wrist-Rel 3D Coords+Vels 4×21)", "wrist_relative_3d"),
    ("arch_resnet1d_tip_vel_ratios.py","ResNet1D_Tip_Vel_Ratios (Tip Vel Ratios 4×16)", "fingertip_velocity_ratios"),

    # Transformer Attention variants (6)
    ("arch_attention.py",              "Attention (2D Coords+Vels 4×16)",             "combined_2d"),
    ("arch_attention_all_combined.py",  "Attention_All_Combined (All Joints Coords+Vels 4×36)", "all_joints_coords_vel"),
    ("arch_attention_vel_3d.py",       "Attention_Vel_3D (3D Vels+Speeds 4×16)",      "vel_speed_3d"),
    ("arch_attention_super_combined.py","Attention_Super_Combined (Super Combined 4×28)", "super_combined"),
    ("arch_attention_wrist_rel_3d.py", "Attention_Wrist_Rel_3D (Wrist-Rel 3D Coords+Vels 4×21)", "wrist_relative_3d"),
    ("arch_attention_tip_vel_ratios.py","Attention_Tip_Vel_Ratios (Tip Vel Ratios 4×16)", "fingertip_velocity_ratios"),

    # TCN variants (6)
    ("arch_tcn.py",                    "TCN (2D Coords+Vels 4×16)",                   "combined_2d"),
    ("arch_tcn_all_combined.py",        "TCN_All_Combined (All Joints Coords+Vels 4×36)", "all_joints_coords_vel"),
    ("arch_tcn_vel_3d.py",             "TCN_Vel_3D (3D Vels+Speeds 4×16)",            "vel_speed_3d"),
    ("arch_tcn_super_combined.py",     "TCN_Super_Combined (Super Combined 4×28)",    "super_combined"),
    ("arch_tcn_wrist_rel_3d.py",       "TCN_Wrist_Rel_3D (Wrist-Rel 3D Coords+Vels 4×21)", "wrist_relative_3d"),
    ("arch_tcn_tip_vel_ratios.py",     "TCN_Tip_Vel_Ratios (Tip Vel Ratios 4×16)",   "fingertip_velocity_ratios"),
]


def _find_python() -> str:
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description="Run deep learning model benchmarks with filtering and sampling.")
    parser.add_argument("-n", "--num-models", type=int, default=None, help="Number of models to run from pool (e.g. 5, 20, 1000)")
    parser.add_argument("-r", "--random", action="store_true", help="Randomize model selection from the pool")
    parser.add_argument("-m", "--models", type=str, default=None, help="Comma-separated architecture name filter (e.g. -m resnet,lstm,cnn)")
    parser.add_argument("-i", "--inputs", type=str, default=None, help="Comma-separated input representation filter (e.g. -i coords,vel_3d,super_combined,wrist_rel_3d)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model sampling (default: 42)")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override default training epochs for all models")
    parser.add_argument("--dropout", "-d", type=float, default=None, help="Override default dropout probability for all models")
    parser.add_argument("--lr", type=float, default=None, help="Override default learning rate for all models")
    parser.add_argument("--batch-size", "-bs", type=int, default=None, help="Override default batch size for all models")
    parser.add_argument("--hidden", type=int, default=None, help="Override default hidden dimension for all models")
    parser.add_argument("--plot", "-p", "--plot-curves", dest="plot", action="store_true", help="Generate Jupyter notebook style Matplotlib Loss/Acc curves and Confusion Matrix plots")
    return parser.parse_args()


def filter_and_sample_models(cli_args) -> list[tuple[str, str, str]]:
    pool = list(ALL_SCRIPTS)

    # 1. Apply architecture filter -m / --models
    if cli_args.models:
        terms = [t.strip().lower() for t in cli_args.models.split(",") if t.strip()]
        matched = []
        for item in pool:
            script, desc, input_var = item
            script_stem = script.replace(".py", "").lower()
            desc_lower = desc.lower()
            if any(term in script_stem or term in desc_lower for term in terms):
                matched.append(item)
        if not matched:
            print(f"[Error] No models matched architecture filter: '{cli_args.models}'")
            sys.exit(1)
        pool = matched

    # 2. Apply input representation filter -i / --inputs
    if cli_args.inputs:
        terms = [t.strip().lower() for t in cli_args.inputs.split(",") if t.strip()]
        matched = []
        for item in pool:
            script, desc, input_var = item
            script_stem = script.replace(".py", "").lower()
            var_lower = input_var.lower()
            desc_lower = desc.lower()
            if any(term in var_lower or term in script_stem or term in desc_lower for term in terms):
                matched.append(item)
        if not matched:
            print(f"[Error] No models matched input representation filter: '{cli_args.inputs}'")
            sys.exit(1)
        pool = matched

    # 3. Randomize pool if -r / --random is passed
    if cli_args.random:
        rnd = random.Random(cli_args.seed)
        rnd.shuffle(pool)

    # 4. Limit count if -n / --num-models is passed
    if cli_args.num_models is not None and cli_args.num_models > 0:
        pool = pool[:cli_args.num_models]

    return pool


def main():
    cli_args = parse_args()

    base    = Path(__file__).resolve().parent
    python  = _find_python()
    log_dir = base / "results"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter and sample models
    selected_scripts = filter_and_sample_models(cli_args)

    # Clean previous CSV result files before running new benchmark suite
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
    if cli_args.plot:
        extra_flags.append("--plot")

    bar = "=" * 70
    print(f"\n{bar}")
    print("  DEEP LEARNING BENCHMARK RUNNER")
    print(f"  Selected Models to Run : {len(selected_scripts)} / {len(ALL_SCRIPTS)} Total in Pool")
    if cli_args.models:
        print(f"  Architecture Filter (-m): {cli_args.models}")
    if cli_args.inputs:
        print(f"  Input Representation (-i): {cli_args.inputs}")
    if cli_args.random:
        print(f"  Random Sampling (-r)   : Enabled (Seed: {cli_args.seed})")
    if extra_flags:
        print(f"  Active Overrides       : {' '.join(extra_flags)}")
    print(f"{bar}\n")

    completed = []
    t_total_start = time.time()

    for idx, (script, arch_desc, input_var) in enumerate(selected_scripts, 1):
        log_path  = log_dir / f"{script.replace('.py', '')}.log"

        print(f"\n{'─'*70}")
        print(f"  [{idx}/{len(selected_scripts)}]  {arch_desc}")
        print(f"  Output saved → results/{log_path.name}")
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
    print("  ALL SELECTED MODELS COMPLETE")
    print(f"{bar}")
    for arch_desc, status, m, s in completed:
        print(f"  {status:10s}  {arch_desc}  ({m}m {s}s)")
    print(f"\n  Total execution time: {total_min}m {total_sec}s")
    print(f"{bar}\n")

    print("  Generating results comparison report...\n")
    compare_cmd = [python, str(base / "compare_results.py")]
    if cli_args.plot:
        compare_cmd.append("--plot")

    result = subprocess.run(
        compare_cmd,
        cwd=str(base.parent),
    )

    if result.returncode != 0:
        print("\n  [Warning] compare_results.py exited with an error.")


if __name__ == "__main__":
    main()
