"""
analyze_pipeline.py
===================
Comprehensive Pipeline Audit Breakdown Aggregator & Reporting Tool.

Pure Python standard library implementation (zero external dependencies)
that aggregates pre-computed JSON summaries from ./dataprocessing/summaries/
and outputs a rich, highly detailed step-by-step audit report.

Usage:
  python3 datacreator/analyze_pipeline.py
  python3 datacreator/analyze_pipeline.py -d ./dataprocessing
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

BAR = "=" * 80
DASH = "─" * 80


def parse_args():
    parser = argparse.ArgumentParser(description="Read step summary JSONs and output pipeline audit report.")
    parser.add_argument("-d", "--data-dir", type=str, default="./dataprocessing", help="Root data processing directory (default: ./dataprocessing)")
    return parser.parse_args()


def load_summary_json(summaries_dir: Path, filename: str) -> dict | None:
    p = summaries_dir / filename
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_summary_json(summaries_dir: Path, filename: str, data: dict):
    try:
        summaries_dir.mkdir(parents=True, exist_ok=True)
        with open(summaries_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_step1_data(base_dir: Path, summaries_dir: Path) -> dict | None:
    cached = load_summary_json(summaries_dir, "step_1_raw_files.json")
    if cached:
        return cached

    raw_dir = base_dir / "1_rawCSVFiles"
    if not raw_dir.exists():
        return None

    lmk_files = list(raw_dir.glob("*.raw_landmarks.*"))
    ann_files = list(raw_dir.glob("*.window_annotations.*"))

    if not lmk_files and not ann_files:
        return None

    total_frames = 0
    x_vals, y_vals, z_vals = [], [], []

    for f in lmk_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                rows = list(reader)
                total_frames += len(rows)
                for r in rows:
                    for k, v in r.items():
                        if v:
                            try:
                                val = float(v)
                                if k.endswith("_x"): x_vals.append(val)
                                elif k.endswith("_y"): y_vals.append(val)
                                elif k.endswith("_z"): z_vals.append(val)
                            except ValueError:
                                pass
        except Exception:
            pass

    tot_windows = 0
    tot_touch = 0
    for f in ann_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                rows = list(reader)
                tot_windows += len(rows)
                for r in rows:
                    if r.get("any_touch", "0") in ("1", "true", "True"):
                        tot_touch += 1
        except Exception:
            pass

    avg_frames = round(total_frames / len(lmk_files), 1) if lmk_files else 0.0

    def calc_stats(vals):
        if not vals: return {}
        return {
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
        }

    data = {
        "step": 1,
        "name": "raw_files",
        "landmark_files": len(lmk_files),
        "annotation_files": len(ann_files),
        "total_frames": total_frames,
        "avg_frames_per_video": avg_frames,
        "total_windows": tot_windows,
        "touch_windows": tot_touch,
        "touch_pct": round(tot_touch / tot_windows * 100.0, 2) if tot_windows > 0 else 0.0,
        "x_bounds": calc_stats(x_vals),
        "y_bounds": calc_stats(y_vals),
        "z_bounds": calc_stats(z_vals),
    }

    save_summary_json(summaries_dir, "step_1_raw_files.json", data)
    return data


def main():
    args = parse_args()
    base_dir = Path(args.data_dir).resolve()
    summaries_dir = base_dir / "summaries"

    print(f"\n{BAR}")
    print("      COMPREHENSIVE PIPELINE AUDIT & BROAD DATA ANALYTICS REPORT")
    print(f"      Root Directory: {base_dir}")
    print(f"      Summary Dir   : {summaries_dir}")
    print(f"{BAR}")

    # Stage 1
    stg1 = get_step1_data(base_dir, summaries_dir)
    print(f"\n{BAR}")
    print("  STAGE 1: RAW DATASET INGESTION & BOUNDS (dataprocessing/1_rawCSVFiles)")
    print(f"{BAR}")
    if stg1:
        print(f"  Landmark Video CSV Files : {stg1.get('landmark_files', 0)}")
        print(f"  Annotation CSV Files     : {stg1.get('annotation_files', 0)}")
        print(f"  Total Raw Frames Ingested: {stg1.get('total_frames', 0):,} (Avg {stg1.get('avg_frames_per_video', 0)} frames/video)")
        if "x_bounds" in stg1 and stg1["x_bounds"]:
            xb, yb, zb = stg1["x_bounds"], stg1["y_bounds"], stg1.get("z_bounds", {})
            print(f"  Landmark X Bounds        : Min = {xb.get('min', 'N/A'):<8} | Max = {xb.get('max', 'N/A'):<8} | Mean = {xb.get('mean', 'N/A')}")
            print(f"  Landmark Y Bounds        : Min = {yb.get('min', 'N/A'):<8} | Max = {yb.get('max', 'N/A'):<8} | Mean = {yb.get('mean', 'N/A')}")
            if zb:
                print(f"  Landmark Z Bounds        : Min = {zb.get('min', 'N/A'):<8} | Max = {zb.get('max', 'N/A'):<8} | Mean = {zb.get('mean', 'N/A')}")
        print(f"  Raw Window Sequence Count: {stg1.get('total_windows', 0):,}")
        print(f"  Raw Touch Annotations    : {stg1.get('touch_windows', 0):,} ({stg1.get('touch_pct', 0)}% of raw windows)")
    else:
        print("  [Notice] Step 1 raw data summary not available.")

    # Stage 2
    stg2 = load_summary_json(summaries_dir, "step_2_normalize_landmarks.json")
    print(f"\n{DASH}")
    print("  STAGE 2: SCALE NORMALIZATION & WRIST CENTERING (dataprocessing/2_normalized_coordinates)")
    print(f"{DASH}")
    if stg2:
        print(f"  Files Processed          : {stg2.get('success_count', 0)} / {stg2.get('total_files', 0)} success")
        print(f"  Normalization Algorithm  : {stg2.get('normalization_method', '8-distance palm RMS scale L_hand')}")
        print(f"  Wrist Origin Centering   : {stg2.get('wrist_origin', 'Centered at (0,0,0)')}")
        print("  Spatial Transformation   : Translation invariant (centered at wrist) & Scale invariant (divided by L_hand)")
    else:
        print("  [Notice] Step 2 JSON summary not available.")

    # Stage 3
    stg3 = load_summary_json(summaries_dir, "step_3_filter_landmarks.json")
    print(f"\n{DASH}")
    print("  STAGE 3: 1EURO ADAPTIVE SMOOTHING FILTER (dataprocessing/3_euroFilter_coordinates)")
    print(f"{DASH}")
    if stg3:
        print(f"  Files Filtered           : {stg3.get('success_count', 0)} / {stg3.get('total_files', 0)} success")
        print(f"  Filter Parameters        : min_cutoff = {stg3.get('min_cutoff')} Hz | beta = {stg3.get('beta')} | d_cutoff = {stg3.get('d_cutoff')} Hz")
        print(f"  Filtered Feature Fields  : {', '.join(stg3.get('filtered_coordinates', ['x', 'y', 'z']))}")
        print(f"  Preserved Raw Metadata   : {', '.join(stg3.get('unfiltered_metadata', ['hand_score', 'visibility', 'presence']))}")
    else:
        print("  [Notice] Step 3 JSON summary not available.")

    # Stage 5 & 6
    stg6 = load_summary_json(summaries_dir, "step_6_merge_windows.json")
    print(f"\n{DASH}")
    print("  STAGE 5 & 6: WINDOWED SEQUENCE MERGE & CLASS BALANCE (dataprocessing/6_merged_windowed_dataset)")
    print(f"{DASH}")
    if stg6:
        print(f"  Files Merged Into Dataset: {stg6.get('files_merged', 0)}")
        print(f"  Total Sequence Windows   : {stg6.get('total_windows', 0):,}")
        print(f"  Class Distribution       : TOUCH = {stg6.get('touch_windows', 0):,} ({stg6.get('touch_pct', 0)}%) | UNTOUCH = {stg6.get('untouch_windows', 0):,}")
        print(f"  Multi-Touch Windows      : {stg6.get('multi_touch_windows', 0):,} windows (simultaneous finger touch)")
        print(f"  Handedness Breakdown     : Right Hand = {stg6.get('right_hand_count', 0):,} | Left Hand = {stg6.get('left_hand_count', 0):,}")
        if "per_finger_touches" in stg6:
            print("  Per-Finger Touch Breakdown:")
            for fg, cnt in stg6["per_finger_touches"].items():
                print(f"    - {fg.capitalize():8s} Touch Count: {cnt:,}")
    else:
        print("  [Notice] Step 6 JSON summary not available.")

    # Stage 7
    stg7 = load_summary_json(summaries_dir, "step_7_calculate_velocities.json")
    print(f"\n{DASH}")
    print("  STAGE 7: 4-STEP TRANSITION VELOCITIES & KINEMATICS (dataprocessing/7_dataset_with_velocities)")
    print(f"{DASH}")
    if stg7:
        print(f"  Window Sequences Processed: {stg7.get('total_windows', 0):,}")
        if "speed_2d_stats" in stg7 and "speed_3d_stats" in stg7:
            s2d, s3d = stg7["speed_2d_stats"], stg7["speed_3d_stats"]
            print(f"  2D Speed Statistics (px/frame): Min = {s2d.get('min'):<6} | Max = {s2d.get('max'):<6} | Mean = {s2d.get('mean')}")
            print(f"  3D Speed Statistics (px/frame): Min = {s3d.get('min'):<6} | Max = {s3d.get('max'):<6} | Mean = {s3d.get('mean')}")
    else:
        print("  [Notice] Step 7 JSON summary not available.")

    # Stage 8
    stg8 = load_summary_json(summaries_dir, "step_8_filter_dataset.json")
    print(f"\n{DASH}")
    print("  STAGE 8: FLAG-BASED DATASET CLEANING (dataprocessing/8_cleaned_dataset)")
    print(f"{DASH}")
    if stg8:
        print(f"  Input Window Sequences   : {stg8.get('total_input_windows', 0):,}")
        print(f"  Retained Sequences       : {stg8.get('retained_windows', 0):,} ({stg8.get('retention_pct', 0)}% retained)")
        print(f"  Dropped Sequences        : {stg8.get('dropped_windows', 0):,}")
        print(f"  Filter Removal Breakdown :")
        print(f"    - Zero-Velocity Touch  : {stg8.get('removed_zero_vel_touch_cnt', 0)} sequences")
        print(f"    - Out-of-Sync          : {stg8.get('removed_out_of_sync_cnt', 0)} sequences")
        print(f"    - Hand Invisible       : {stg8.get('removed_hand_invisible_cnt', 0)} sequences")
        print(f"  Cleaned Class Balance    : TOUCH = {stg8.get('touch_windows', 0):,} ({stg8.get('touch_pct', 0)}%) | UNTOUCH = {stg8.get('untouch_windows', 0):,}")
    else:
        print("  [Notice] Step 8 JSON summary not available.")

    # Stage 9
    stg9 = load_summary_json(summaries_dir, "step_9_filter_window_quality.json")
    print(f"\n{DASH}")
    print("  STAGE 9: COMPREHENSIVE WINDOW QUALITY & CONFIDENCE FILTER (dataprocessing/9_quality_filtered_dataset)")
    print(f"{DASH}")
    if stg9:
        print(f"  Input Window Sequences   : {stg9.get('total_input_windows', 0):,}")
        print(f"  Retained Sequences       : {stg9.get('retained_windows', 0):,} ({stg9.get('retained_pct', 0)}% retained)")
        print(f"  Dropped Sequences        : {stg9.get('dropped_windows', 0):,}")
        print(f"  Quality Cutoff Removal Breakdown:")
        print(f"    - Low Avg Hand Score   : {stg9.get('drop_min_avg_score_cnt', 0)} sequences")
        print(f"    - Low Frame Hand Score : {stg9.get('drop_min_frame_score_cnt', 0)} sequences")
        print(f"    - High Score Drop      : {stg9.get('drop_max_score_drop_cnt', 0)} sequences")
        print(f"    - 2D Speed Anomaly     : {stg9.get('drop_max_speed_2d_cnt', 0)} sequences")
        print(f"    - 3D Speed Anomaly     : {stg9.get('drop_max_speed_3d_cnt', 0)} sequences")
        print(f"  Class Balance Transition : Touch: {stg9.get('touch_before', 0)} -> {stg9.get('touch_after', 0)} | Untouch: {stg9.get('untouch_before', 0)} -> {stg9.get('untouch_after', 0)}")
    else:
        print("  [Notice] Step 9 JSON summary not available.")

    # Stage 10
    stg10 = load_summary_json(summaries_dir, "step_10_split_fingers.json")
    print(f"\n{DASH}")
    print("  STAGE 10: PER-FINGER SEQUENCE UNROLLING (dataprocessing/10_per_finger_dataset)")
    print(f"{DASH}")
    if stg10:
        print(f"  Input Window Sequences   : {stg10.get('input_window_rows', 0):,}")
        print(f"  Unrolled Per-Finger Rows : {stg10.get('unrolled_rows', 0):,} (5× multiplier)")
        print(f"  Touch Finger Records     : {stg10.get('touch_records', 0):,} ({stg10.get('touch_records', 0)/stg10.get('unrolled_rows', 1)*100.0:.2f}%)")
        print(f"  Untouch Finger Records   : {stg10.get('untouch_records', 0):,}")
        if "per_finger_touches" in stg10:
            print("  Per-Finger Touch Breakdown:")
            for fg, cnt in stg10["per_finger_touches"].items():
                print(f"    - {fg.capitalize():8s} Touch Records: {cnt:,}")
    else:
        print("  [Notice] Step 10 JSON summary not available.")

    # Stage 11 & 12
    stg11 = load_summary_json(summaries_dir, "step_11_split_touch.json")
    stg12 = load_summary_json(summaries_dir, "step_12_train_test_split.json")
    print(f"\n{DASH}")
    print("  STAGE 11 & 12: TOUCH SPLIT & BALANCED TRAIN / TEST PARTITIONING (dataprocessing/12_train_test_split)")
    print(f"{DASH}")
    if stg12:
        tr_cnt = stg12.get("train_records", 0)
        te_cnt = stg12.get("test_records", 0)
        tot = tr_cnt + te_cnt
        tr_pct = round(tr_cnt / tot * 100.0, 2) if tot > 0 else 0
        te_pct = round(te_cnt / tot * 100.0, 2) if tot > 0 else 0
        print(f"  Training Set Records     : {tr_cnt:,} ({tr_pct}% of total dataset)")
        print(f"  Testing Set Records      : {te_cnt:,} ({te_pct}% of total dataset)")
        print(f"  Training Set Touch Ratio : TOUCH = {stg12.get('train_touch', 0):,} | UNTOUCH = {stg12.get('train_untouch', 0):,}")
        print(f"  Testing Set Touch Ratio  : TOUCH = {stg12.get('test_touch', 0):,} | UNTOUCH = {stg12.get('test_untouch', 0):,}")
        print(f"  Video Partition Audit    : {'PASSED (0 Video Leakage between Train and Test Sets)' if stg12.get('no_video_leak') else 'Disabled'}")
    else:
        print("  [Notice] Step 12 JSON summary not available.")

    print(f"\n{BAR}")
    print("      PIPELINE AUDIT COMPLETE")
    print(f"{BAR}\n")


if __name__ == "__main__":
    main()
