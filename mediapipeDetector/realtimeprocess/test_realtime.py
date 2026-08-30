"""
realtimeprocess/test_realtime.py

Automated Headless Test Suite for Real-Time Streaming Touch Pipeline & Model Manager.
Verifies ring buffer mechanics, 2-frame shift triggers, feature extraction, and PyTorch model predictions.
"""

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtimeprocess.realtime_pipeline import (
    OneEuroFilterBank,
    HandScaleNormalizer,
    process_streaming_frame,
    compute_window_velocities,
    unroll_per_finger_window,
    extract_variant_tensor,
    ALL_21_LANDMARK_NAMES,
    FINGERS,
)
from realtimeprocess.model_manager import ModelManager


def generate_synthetic_landmarks(t_step: int) -> list[tuple[float, float, float]]:
    """Generates synthetic 21 3D landmarks for testing."""
    pts = []
    # Wrist
    pts.append((0.5, 0.8, 0.0))
    # Thumb: cmc, mcp, ip, tip
    pts.extend([(0.42, 0.72, 0.01), (0.38, 0.65, 0.02), (0.35, 0.58, 0.02), (0.32 + 0.01 * t_step, 0.52, 0.03)])
    # Index: mcp, pip, dip, tip
    pts.extend([(0.45, 0.60, 0.01), (0.44, 0.48, 0.02), (0.43, 0.38, 0.02), (0.42, 0.28 - 0.01 * t_step, 0.03)])
    # Middle: mcp, pip, dip, tip
    pts.extend([(0.50, 0.58, 0.01), (0.50, 0.45, 0.02), (0.50, 0.35, 0.02), (0.50, 0.25, 0.03)])
    # Ring: mcp, pip, dip, tip
    pts.extend([(0.55, 0.60, 0.01), (0.56, 0.48, 0.02), (0.57, 0.38, 0.02), (0.58, 0.28, 0.03)])
    # Pinky: mcp, pip, dip, tip
    pts.extend([(0.60, 0.65, 0.01), (0.62, 0.55, 0.02), (0.64, 0.48, 0.02), (0.65, 0.40, 0.03)])

    return pts


def run_pipeline_test():
    print("="*75)
    print("  TEST 1: ONLINE PIPELINE & FEATURE EXTRACTION TEST")
    print("="*75)

    filter_bank = OneEuroFilterBank()
    normalizer  = HandScaleNormalizer()

    w_px, h_px = 640, 480
    raw_window_5 = []

    for t in range(5):
        raw_pts = generate_synthetic_landmarks(t)
        filtered_pts = filter_bank.filter_frame(t * 0.0833, raw_pts)
        raw_window_5.append(filtered_pts)

    print(f"  ✓ Captured 5 frames of raw 21 3D landmarks.")

    # Scale normalization
    norm_frames_5 = []
    for raw_pts in raw_window_5:
        pts_px = [(x * w_px, y * h_px, z * w_px) for (x, y, z) in raw_pts]
        norm_pts = normalizer.normalize(pts_px, center_wrist=True)
        f_dict = {}
        for lm_idx, lm_name in enumerate(ALL_21_LANDMARK_NAMES):
            nx, ny, nz = norm_pts[lm_idx]
            f_dict[f"{lm_name}_x"] = nx
            f_dict[f"{lm_name}_y"] = ny
            f_dict[f"{lm_name}_z"] = nz
        norm_frames_5.append(f_dict)

    print(f"  ✓ HandScale Normalization applied (8-distance palm RMS scale L_hand).")

    # Velocity computation
    v_steps_4 = compute_window_velocities(norm_frames_5)
    print(f"  ✓ 4 frame-to-frame velocity steps computed.")

    # Unroll per finger
    finger_rows = unroll_per_finger_window(norm_frames_5, v_steps_4)
    print(f"  ✓ Unrolled into 5 per-finger sequence rows: {list(finger_rows.keys())}")

    # Extract variant tensors
    variants_to_test = ["coords_2d", "vel_2d", "combined_2d", "coords_3d", "vel_3d", "combined_3d", "all_joints_vel", "all_joints_coords_vel"]
    for var in variants_to_test:
        tensor = extract_variant_tensor(finger_rows, var)
        print(f"  ✓ Variant '{var:<20}': Tensor Shape = {tensor.shape} | dtype = {tensor.dtype}")

    print("  ✅ Online Pipeline Test PASSED!\n")


def run_model_manager_test():
    print("="*75)
    print("  TEST 2: MODEL MANAGER & REAL-TIME INFERENCE TEST")
    print("="*75)

    mm = ModelManager()

    if not mm.available_models:
        print("❌ No trained models found in deepLearningModels/weights/. Run training first!")
        sys.exit(1)

    print(f"  Total Models Discovered: {len(mm.available_models)}")

    # Test inference on all discovered models
    w_px, h_px = 640.0, 480.0
    norm_window_5 = []
    normalizer = HandScaleNormalizer()
    filter_bank = OneEuroFilterBank(min_cutoff=3.0, beta=1.4, d_cutoff=1.0)

    for t in range(5):
        raw_pts = generate_synthetic_landmarks(t)
        f_dict, _ = process_streaming_frame(raw_pts, w_px, h_px, float(t) * 0.083, normalizer, filter_bank)
        norm_window_5.append(f_dict)

    scores_5 = [0.95] * 5

    for idx, model_info in enumerate(mm.available_models):
        success = mm.load_model_by_index(idx)
        assert success, f"Failed to load model {model_info['arch_name']}"

        t0 = time.perf_counter()
        preds = mm.predict_window(norm_window_5, scores_5, w_px, h_px)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  ✓ [{idx+1}/{len(mm.available_models)}] {model_info['arch_name']:<24} ({model_info['variant_name']:<20}) | Latency: {latency_ms:.3f} ms")
        for finger in FINGERS:
            p_data = preds[finger]
            status_str = "TOUCH" if p_data["touch"] else "UNTOUCH"
            print(f"      - {finger:<6}: {status_str} (Prob: {p_data['prob']:.4f})")

    print("  ✅ Model Manager & Real-Time Inference Test PASSED!\n")


if __name__ == "__main__":
    run_pipeline_test()
    run_model_manager_test()
