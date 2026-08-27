#!/usr/bin/env python3
"""
analyze_dataset.py — Analyzes dataset CSV files (per-finger schema) and prints a comprehensive statistical summary.

Usage:
    python3 analysis/analyze_dataset.py -i ./data/finger_split_all.csv
    python3 analysis/analyze_dataset.py -i ./data/cleaned_data.csv ./data/training_data.csv
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def parse_bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def analyze_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    total_rows = len(rows)
    print("=" * 80)
    print(f" DATASET ANALYSIS REPORT: {csv_path.name}")
    print(f" Path: {csv_path.resolve()}")
    print("=" * 80)
    print(f"Total Rows (Records): {total_rows}")

    if total_rows == 0:
        print("Dataset is empty.\n")
        return

    # Touch vs Untouch breakdown
    touch_col = "touch_finger" if "touch_finger" in fieldnames else ("touch" if "touch" in fieldnames else None)
    
    if touch_col:
        touch_cnt = sum(1 for r in rows if parse_bool(r.get(touch_col, "0")))
        untouch_cnt = total_rows - touch_cnt
        touch_pct = (touch_cnt / total_rows) * 100 if total_rows > 0 else 0
        untouch_pct = (untouch_cnt / total_rows) * 100 if total_rows > 0 else 0
        ratio_str = f"1 : {untouch_cnt / touch_cnt:.2f}" if touch_cnt > 0 else "N/A"

        print("\n--- [1] TARGET LABEL SUMMARY (Touch vs Untouch) ---")
        print(f"  Touch Rows (Positive):   {touch_cnt:6d} ({touch_pct:6.2f}%)")
        print(f"  Untouch Rows (Negative): {untouch_cnt:6d} ({untouch_pct:6.2f}%)")
        print(f"  Touch to Untouch Ratio:  {ratio_str}")
    else:
        print("\n--- [1] TARGET LABEL SUMMARY ---")
        print("  Warning: No touch column found in CSV.")

    # Breakdown by Finger Name
    if "finger_name" in fieldnames:
        print("\n--- [2] PER-FINGER BREAKDOWN ---")
        finger_counts = Counter(r.get("finger_name", "unknown") for r in rows)
        print(f"  {'Finger':<12} | {'Total Rows':<12} | {'Touch Rows':<12} | {'Untouch Rows':<12} | {'Touch %':<8}")
        print("  " + "-" * 66)
        for finger, count in sorted(finger_counts.items()):
            if touch_col:
                f_touch = sum(1 for r in rows if r.get("finger_name") == finger and parse_bool(r.get(touch_col, "0")))
                f_untouch = count - f_touch
                f_touch_pct = (f_touch / count) * 100 if count > 0 else 0
                print(f"  {finger:<12} | {count:<12d} | {f_touch:<12d} | {f_untouch:<12d} | {f_touch_pct:<7.2f}%")
            else:
                print(f"  {finger:<12} | {count:<12d} | N/A          | N/A            | N/A")

    # Handedness (Right vs Left Hand)
    if "rightHand" in fieldnames:
        print("\n--- [3] HANDEDNESS DISTRIBUTION ---")
        right_cnt = sum(1 for r in rows if parse_bool(r.get("rightHand", "0")))
        left_cnt = total_rows - right_cnt
        print(f"  Right Hand (rightHand=1): {right_cnt:6d} ({(right_cnt/total_rows)*100:6.2f}%)")
        print(f"  Left Hand  (rightHand=0): {left_cnt:6d} ({(left_cnt/total_rows)*100:6.2f}%)")

    # Context Flags
    context_flags = ["hand_move", "hand_point_of_view", "hand_closer", "hovering", "daylight", "hand_visible", "out_of_sync", "any_difference"]
    available_flags = [col for col in context_flags if col in fieldnames]

    if available_flags:
        print("\n--- [4] CONTEXT FLAGS SUMMARY ---")
        print(f"  {'Flag Name':<20} | {'True Count':<12} | {'Percentage':<10}")
        print("  " + "-" * 48)
        for flag in available_flags:
            cnt = sum(1 for r in rows if parse_bool(r.get(flag, "0")))
            pct = (cnt / total_rows) * 100
            print(f"  {flag:<20} | {cnt:<12d} | {pct:<9.2f}%")

    # Video Breakdown
    if "video_file" in fieldnames:
        video_counts = Counter(r.get("video_file", "unknown") for r in rows)
        print("\n--- [5] VIDEO FILE SOURCE BREAKDOWN ---")
        print(f"  Total Unique Videos: {len(video_counts)}")
        print(f"  {'Video File':<30} | {'Total Rows':<12} | {'Touch Rows':<12} | {'Untouch Rows':<12}")
        print("  " + "-" * 72)
        for vfile, count in sorted(video_counts.items()):
            if touch_col:
                v_touch = sum(1 for r in rows if r.get("video_file") == vfile and parse_bool(r.get(touch_col, "0")))
                v_untouch = count - v_touch
                print(f"  {vfile:<30} | {count:<12d} | {v_touch:<12d} | {v_untouch:<12d}")
            else:
                print(f"  {vfile:<30} | {count:<12d} | N/A          | N/A")

    # Zero Fingertip Velocity Analysis
    if touch_col and "dip1_vx" in fieldnames:
        zero_vel_touch_cnt = 0
        for r in rows:
            if parse_bool(r.get(touch_col, "0")):
                all_zero = True
                for v in range(1, 5):
                    vx = float(r.get(f"dip{v}_vx", "0") or 0.0)
                    vy = float(r.get(f"dip{v}_vy", "0") or 0.0)
                    if vx != 0.0 or vy != 0.0:
                        all_zero = False
                        break
                if all_zero:
                    zero_vel_touch_cnt += 1

        print("\n--- [6] ZERO FINGERTIP VELOCITY TOUCH ANOMALIES ---")
        print(f"  Touch Rows with 0 Fingertip Velocity (dip1-4): {zero_vel_touch_cnt} rows")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset CSV file and print summary statistics."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input dataset CSV file path"
    )

    args = parser.parse_args()
    analyze_csv(Path(args.input))


if __name__ == "__main__":
    main()
