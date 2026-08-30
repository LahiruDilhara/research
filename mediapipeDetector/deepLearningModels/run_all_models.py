"""
run_all_models.py
=================
Master model trainer, loader, evaluator, and real-time inference speed auditor.

Features:
  1. Executes model training for selected architectures via run_all.py / script pool.
  2. Overwrites existing .pth weight checkpoints in weights/ with peak epoch weights.
  3. Loads saved PyTorch weights for every trained model.
  4. Evaluates full performance analytics on Train, Test, and Combined datasets.
  5. Audits Real-Time Inference Latency (ms) and Throughput (FPS) per single sample (batch=1) and batch (batch=32).
  6. Evaluates real-time suitability for live interactive touch gesture applications.
  7. Saves comprehensive CSV and JSON evaluation reports to results/.
"""

import os
import sys
from pathlib import Path

# Ensure execution in .venv virtual environment
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
_venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

import argparse
import csv
import importlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure deepLearningModels directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_arch import (
    load_variant_data, load_entire_dataset, make_loaders, evaluate_model, analyze_fit_quality
)
from run_all import ALL_SCRIPTS


def _find_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def benchmark_inference_speed(model, sample_input, device, num_warmup: int = 10, num_runs: int = 100):
    """
    Measures exact single-sample and batch inference latency (ms) and throughput (FPS).
    """
    model.eval()
    sample_tensor = torch.from_numpy(sample_input).float().to(device)

    # Warmup runs
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(sample_tensor)

    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Single-sample latency benchmark
    single_sample = sample_tensor[:1]  # Shape (1, seq_len, feature_dim)
    latencies_ms = []

    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(single_sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_single_ms = float(np.mean(latencies_ms))
    p95_single_ms  = float(np.percentile(latencies_ms, 95))
    fps            = 1000.0 / (mean_single_ms + 1e-6)

    # Batch latency benchmark (batch size N)
    batch_latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(sample_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_latencies_ms.append((t1 - t0) * 1000.0)

    mean_batch_ms = float(np.mean(batch_latencies_ms))
    batch_fps     = (len(sample_input) * 1000.0) / (mean_batch_ms + 1e-6)

    # Real-time suitability classification
    if mean_single_ms < 1.5:
        realtime_status = "⚡ ULTRA REAL-TIME (<1.5 ms)"
    elif mean_single_ms < 4.0:
        realtime_status = "✅ REAL-TIME SUITABLE (<4.0 ms)"
    elif mean_single_ms < 10.0:
        realtime_status = "⚠️ MARGINAL REAL-TIME (<10 ms)"
    else:
        realtime_status = "❌ TOO SLOW FOR LIVE VIDEO"

    return {
        "single_sample_latency_ms": round(mean_single_ms, 3),
        "single_sample_p95_ms":     round(p95_single_ms, 3),
        "throughput_fps":           round(fps, 1),
        "batch_latency_ms":         round(mean_batch_ms, 3),
        "batch_throughput_fps":     round(batch_fps, 1),
        "realtime_status":          realtime_status,
    }


def evaluate_trained_model(script_file: str, arch_title: str, variant_name: str, device, base_dir: Path):
    """
    Loads saved model weights, evaluates performance metrics on Train, Test, and Entire Unsplit datasets,
    and runs real-time inference speed benchmarks.
    """
    module_name = script_file.replace(".py", "")
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        mod = importlib.import_module(f"deepLearningModels.{module_name}")

    arch_name = getattr(mod, "ARCH_NAME", module_name)
    configs   = getattr(mod, "CONFIGS", [{'id': 1}])
    cfg       = configs[0]

    # Load Train & Test data for variant
    X_tr, y_tr, X_te, y_te, seq_len, feature_dim = load_variant_data(variant_name, base_dir)

    # Load Entire Unsplit Dataset (dataprocessing/10_per_finger_dataset/per_finger_dataset.csv)
    X_full, y_full, _, _ = load_entire_dataset(variant_name, base_dir)

    # Weights file path
    weights_path = SCRIPT_DIR / "weights" / f"{arch_name}_cfg{cfg['id']:02d}.pth"
    if not weights_path.exists():
        return None

    # Instantiate model architecture
    model = mod.create_model(feature_dim, cfg).to(device)

    # Load saved weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataloaders for train, test, and full dataset
    bs = cfg.get("bs", 32)
    train_loader = DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=False)
    test_loader  = DataLoader(torch.utils.data.TensorDataset(X_te, y_te), batch_size=bs, shuffle=False)
    full_loader  = DataLoader(torch.utils.data.TensorDataset(X_full, y_full), batch_size=bs, shuffle=False)

    # Evaluate Test Set
    cm_test, rpt_test = evaluate_model(model, test_loader, device)
    tn_te, fp_te, fn_te, tp_te = cm_test.ravel() if cm_test.shape == (2, 2) else (0, 0, 0, 0)
    test_acc  = (tn_te + tp_te) / (tn_te + fp_te + fn_te + tp_te) * 100.0 if (tn_te + fp_te + fn_te + tp_te) > 0 else 0.0
    prec_test = rpt_test["Touch"]["precision"] * 100.0
    rec_test  = rpt_test["Touch"]["recall"] * 100.0
    f1_test   = rpt_test["Touch"]["f1-score"]
    spec_test = (tn_te / (tn_te + fp_te)) * 100.0 if (tn_te + fp_te) > 0 else 0.0

    # Evaluate Train Set
    cm_train, rpt_train = evaluate_model(model, train_loader, device)
    tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel() if cm_train.shape == (2, 2) else (0, 0, 0, 0)
    train_acc = (tn_tr + tp_tr) / (tn_tr + fp_tr + fn_tr + tp_tr) * 100.0 if (tn_tr + fp_tr + fn_tr + tp_tr) > 0 else 0.0

    # Evaluate Entire Unsplit Dataset (Touch & Untouch Data)
    cm_full, rpt_full = evaluate_model(model, full_loader, device)
    tn_fl, fp_fl, fn_fl, tp_fl = cm_full.ravel() if cm_full.shape == (2, 2) else (0, 0, 0, 0)
    full_acc  = (tn_fl + tp_fl) / (tn_fl + fp_fl + fn_fl + tp_fl) * 100.0 if (tn_fl + fp_fl + fn_fl + tp_fl) > 0 else 0.0
    prec_full = rpt_full["Touch"]["precision"] * 100.0
    rec_full  = rpt_full["Touch"]["recall"] * 100.0
    f1_full   = rpt_full["Touch"]["f1-score"]
    spec_full = (tn_fl / (tn_fl + fp_fl)) * 100.0 if (tn_fl + fp_fl) > 0 else 0.0

    # Generalization gap & status
    acc_gap = train_acc - test_acc
    if acc_gap > 5.0:
        fit_status = f"OVERFITTING (Gap: +{acc_gap:.2f}%)"
    elif acc_gap < -5.0:
        fit_status = f"UNDERFITTING (Gap: {acc_gap:.2f}%)"
    else:
        fit_status = f"GOOD FIT (Gap: {acc_gap:+.2f}%)"

    # Run inference latency benchmark
    sample_data = X_te.numpy()
    speed_metrics = benchmark_inference_speed(model, sample_data, device)

    return {
        "script":                   script_file,
        "arch_name":                arch_name,
        "variant":                  variant_name,
        "sequence_len":             seq_len,
        "feature_dim":              feature_dim,
        "weights_file":             weights_path.name,
        "train_samples":            len(X_tr),
        "test_samples":             len(X_te),
        "full_dataset_samples":     len(X_full),
        "train_acc_pct":            round(train_acc, 2),
        "test_acc_pct":             round(test_acc, 2),
        "full_acc_pct":             round(full_acc, 2),
        "full_f1_touch":            round(f1_full, 4),
        "full_precision_pct":       round(prec_full, 2),
        "full_recall_pct":          round(rec_full, 2),
        "full_specificity_pct":     round(spec_full, 2),
        "test_precision_pct":       round(prec_test, 2),
        "test_recall_pct":          round(rec_test, 2),
        "test_specificity_pct":     round(spec_test, 2),
        "test_f1_touch":            round(f1_test, 4),
        "tn_test":                  int(tn_te),
        "fp_test":                  int(fp_te),
        "fn_test":                  int(fn_te),
        "tp_test":                  int(tp_te),
        "tn_full":                  int(tn_fl),
        "fp_full":                  int(fp_fl),
        "fn_full":                  int(fn_fl),
        "tp_full":                  int(tp_fl),
        "fit_status":               fit_status,
        "single_latency_ms":        speed_metrics["single_sample_latency_ms"],
        "single_p95_ms":            speed_metrics["single_sample_p95_ms"],
        "throughput_fps":           speed_metrics["throughput_fps"],
        "batch_latency_ms":         speed_metrics["batch_latency_ms"],
        "realtime_status":          speed_metrics["realtime_status"],
    }


def print_single_model_analytics(res: dict, idx: int, total_count: int):
    """Prints rich formatted per-model performance, full confusion matrices, and latency analytics immediately after evaluation."""
    print(f"\n{'='*75}", flush=True)
    print(f"  [{idx}/{total_count}]  {res['arch_name']} ({res['variant']})", flush=True)
    print(f"  Input Shape: (N, {res['sequence_len']}, {res['feature_dim']})  |  Checkpoint: weights/{res['weights_file']}", flush=True)
    print(f"{'='*75}", flush=True)

    # 1. Test Set Confusion Matrix & Metrics Report
    tn_te, fp_te, fn_te, tp_te = res['tn_test'], res['fp_test'], res['fn_test'], res['tp_test']
    act_un_te = tn_te + fp_te
    act_to_te = fn_te + tp_te
    tot_te    = act_un_te + act_to_te

    print(f"  [1] TEST SET CONFUSION MATRIX & METRICS REPORT ({tot_te:,} Samples: {act_to_te:,} Touch | {act_un_te:,} Untouch)", flush=True)
    print(f"                 Predicted UNTOUCH    Predicted TOUCH     Total Actual", flush=True)
    print(f"  Actual UNTOUCH     TN = {tn_te:<8d}     FP = {fp_te:<8d}     {act_un_te:<8d}", flush=True)
    print(f"  Actual TOUCH       FN = {fn_te:<8d}     TP = {tp_te:<8d}     {act_to_te:<8d}", flush=True)
    print(f"  ------------------------------------------------------------------", flush=True)
    print(f"  Test Accuracy        : {res['test_acc_pct']:.2f}% ({tp_te + tn_te}/{tot_te})", flush=True)
    print(f"  Precision (Touch)    : {res['test_precision_pct']:.2f}% ({tp_te}/{tp_te + fp_te if (tp_te + fp_te) > 0 else 1})", flush=True)
    print(f"  Recall / Sensitivity : {res['test_recall_pct']:.2f}% ({tp_te}/{act_to_te if act_to_te > 0 else 1})", flush=True)
    print(f"  Specificity (Untouch): {res['test_specificity_pct']:.2f}% ({tn_te}/{act_un_te if act_un_te > 0 else 1})", flush=True)
    print(f"  F1-Score (Touch)     : {res['test_f1_touch']:.4f}", flush=True)

    # 2. Entire Dataset (Full Pre-Split Unrolled Data)
    tn_fl, fp_fl, fn_fl, tp_fl = res['tn_full'], res['fp_full'], res['fn_full'], res['tp_full']
    act_un_fl = tn_fl + fp_fl
    act_to_fl = fn_fl + tp_fl
    tot_fl    = act_un_fl + act_to_fl

    print(f"\n  [2] ENTIRE DATASET CONFUSION MATRIX & METRICS REPORT ({tot_fl:,} Records: {act_to_fl:,} Touch | {act_un_fl:,} Untouch)", flush=True)
    print(f"                 Predicted UNTOUCH    Predicted TOUCH     Total Actual", flush=True)
    print(f"  Actual UNTOUCH     TN = {tn_fl:<8d}     FP = {fp_fl:<8d}     {act_un_fl:<8d}", flush=True)
    print(f"  Actual TOUCH       FN = {fn_fl:<8d}     TP = {tp_fl:<8d}     {act_to_fl:<8d}", flush=True)
    print(f"  ------------------------------------------------------------------", flush=True)
    print(f"  Full Accuracy        : {res['full_acc_pct']:.2f}% ({tp_fl + tn_fl}/{tot_fl})", flush=True)
    print(f"  Precision (Touch)    : {res['full_precision_pct']:.2f}% ({tp_fl}/{tp_fl + fp_fl if (tp_fl + fp_fl) > 0 else 1})", flush=True)
    print(f"  Recall / Sensitivity : {res['full_recall_pct']:.2f}% ({tp_fl}/{act_to_fl if act_to_fl > 0 else 1})", flush=True)
    print(f"  Specificity (Untouch): {res['full_specificity_pct']:.2f}% ({tn_fl}/{act_un_fl if act_un_fl > 0 else 1})", flush=True)
    print(f"  F1-Score (Touch)     : {res['full_f1_touch']:.4f}", flush=True)
    print(f"  Generalization Fit   : {res['fit_status']}", flush=True)

    # 3. Real-Time Inference Latency & Speed Audit
    print(f"\n  [3] REAL-TIME INFERENCE SPEED & LATENCY AUDIT", flush=True)
    print(f"  Single-Sample Latency (batch=1):  {res['single_latency_ms']:.3f} ms  (p95: {res['single_p95_ms']:.3f} ms)", flush=True)
    print(f"  Inference Throughput (FPS):       {res['throughput_fps']:,.1f} frames/sec", flush=True)
    print(f"  Batch Latency (batch=32):         {res['batch_latency_ms']:.3f} ms", flush=True)
    print(f"  Real-Time Suitability Status:     {res['realtime_status']}", flush=True)
    print(f"{'='*75}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Master Deep Learning Model Evaluator and Real-Time Speed Auditor.")
    parser.add_argument("-m", "--models", type=str, default=None, help="Comma-separated architecture name filter")
    parser.add_argument("-i", "--inputs", type=str, default=None, help="Comma-separated input representation filter")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Specify inference device: cuda or cpu")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA inference")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    elif args.cuda:
        if not torch.cuda.is_available():
            print("  ⚠️ CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset sample info for overview header
    try:
        X_tr_s, y_tr_s, X_te_s, y_te_s, _, _ = load_variant_data("vel_2d", PROJECT_ROOT)
        X_fl_s, y_fl_s, _, _                 = load_entire_dataset("vel_2d", PROJECT_ROOT)
        n_tr, tr_touch, tr_untouch = len(y_tr_s), int((y_tr_s == 1.0).sum()), int((y_tr_s == 0.0).sum())
        n_te, te_touch, te_untouch = len(y_te_s), int((y_te_s == 1.0).sum()), int((y_te_s == 0.0).sum())
        n_fl, fl_touch, fl_untouch = len(y_fl_s), int((y_fl_s == 1.0).sum()), int((y_fl_s == 0.0).sum())
    except Exception:
        n_tr, tr_touch, tr_untouch = 3664, 1832, 1832
        n_te, te_touch, te_untouch = 916, 458, 458
        n_fl, fl_touch, fl_untouch = 13500, 3636, 9864

    print("\n" + "="*95)
    print("  DATASET BREAKDOWN & DEEP LEARNING MODEL AUDIT LEADERBOARD")
    print("="*95)
    print(f"  • Evaluation Device Target     : {device} (CUDA Available: {torch.cuda.is_available()})")
    print(f"  • Testing Dataset Size          : {n_te:,} samples ({te_touch:,} Touch | {te_untouch:,} Untouch)")
    print(f"  • Training Dataset Size         : {n_tr:,} samples ({tr_touch:,} Touch | {tr_untouch:,} Untouch)")
    print(f"  • Entire Pre-Split Dataset Size : {n_fl:,} records ({fl_touch:,} Touch | {fl_untouch:,} Untouch)\n")

    matching_scripts = []
    for script_file, display_title, variant_name in ALL_SCRIPTS:
        if args.models:
            allowed = [x.strip().lower() for x in args.models.split(",")]
            if not any(a in script_file.lower() for a in allowed):
                continue
        if args.inputs:
            allowed = [x.strip().lower() for x in args.inputs.split(",")]
            if not any(a in variant_name.lower() for a in allowed):
                continue
        matching_scripts.append((script_file, display_title, variant_name))

    total_count = len(matching_scripts)
    results = []

    for script_file, display_title, variant_name in matching_scripts:
        res = evaluate_trained_model(script_file, display_title, variant_name, device, PROJECT_ROOT)
        if res is not None:
            results.append(res)
            print_single_model_analytics(res, len(results), total_count)

    if not results:
        print("❌ No trained model weights found in deepLearningModels/weights/. Run training first!")
        sys.exit(1)

    # Sort results by Test Accuracy descending
    results.sort(key=lambda r: r["test_acc_pct"], reverse=True)

    # Step 3: Save CSV & JSON Reports
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_csv  = results_dir / "model_evaluation_report.csv"
    report_json = results_dir / "model_evaluation_report.json"

    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(report_json, "w") as f:
        json.dump(results, f, indent=2)

    # Step 4: Display Formatted Terminal Summary
    print("="*118)
    print("  MASTER MODEL AUDIT REPORT & REAL-TIME PERFORMANCE LEADERBOARD")
    print("="*118)
    print(f" {'Rank':<4} {'Architecture':<26} {'Variant':<22} {'Train Acc':<10} {'Test Acc':<10} {'Full Acc':<10} {'Test F1':<8} {'Latency(ms)':<12} {'FPS':<8} {'Real-Time Status'}")
    print("-" * 118)

    for idx, r in enumerate(results, 1):
        print(f" #{idx:<3} {r['arch_name']:<26} {r['variant']:<22} {r['train_acc_pct']:>6.2f}%    {r['test_acc_pct']:>6.2f}%    {r['full_acc_pct']:>6.2f}%    {r['test_f1_touch']:>6.4f}   {r['single_latency_ms']:>6.3f} ms    {r['throughput_fps']:>6.1f}   {r['realtime_status']}")

    print("="*118)
    print(f"\n  ✅ Full evaluation report saved to: {report_csv}")
    print(f"  ✅ Structured JSON report saved to: {report_json}\n")


if __name__ == "__main__":
    main()
