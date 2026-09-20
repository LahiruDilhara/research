# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/filter_hand_movement.py

Filters whole-hand transit movement windows from merged window dataset CSV.
Evaluates the net coordinate displacement of stationary hand landmarks (Wrist and MCP knuckles)
between Frame 1 and Frame 5 relative to the rigid palm scale factor L_hand.

If max stationary displacement across the 5-frame window exceeds --threshold (default: 0.2),
the window is classified as HAND MOVING and dropped from the dataset.

Stationary reference landmarks:
  - Wrist (Joint 0)
  - Index MCP (Joint 5)
  - Middle MCP (Joint 9)
  - Ring MCP (Joint 13)
  - Pinky MCP (Joint 17)
"""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacreator.summary_utils import save_step_summary

STATIONARY_NAMES = ["wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]


def calculate_palm_scale(pts_px: dict[str, tuple[float, float]]) -> float:
    """Calculates rigid palm scale L_hand using Root-Mean-Square (RMS) of 8 symmetric palm segments."""
    w_x, w_y = pts_px["wrist"]
    i_x, i_y = pts_px["index_mcp"]
    m_x, m_y = pts_px["middle_mcp"]
    r_x, r_y = pts_px["ring_mcp"]
    p_x, p_y = pts_px["pinky_mcp"]

    d_sq = [
        (i_x - w_x) ** 2 + (i_y - w_y) ** 2,
        (m_x - w_x) ** 2 + (m_y - w_y) ** 2,
        (r_x - w_x) ** 2 + (r_y - w_y) ** 2,
        (p_x - w_x) ** 2 + (p_y - w_y) ** 2,
        (m_x - i_x) ** 2 + (m_y - i_y) ** 2,
        (r_x - m_x) ** 2 + (r_y - m_y) ** 2,
        (p_x - r_x) ** 2 + (p_y - r_y) ** 2,
        (p_x - i_x) ** 2 + (p_y - i_y) ** 2,
    ]
    l_hand = math.sqrt(sum(d_sq) / 8.0)
    return max(1.0, l_hand)


def load_raw_landmarks_cache(raw_dir: str) -> dict:
    """
    Preloads stationary landmark coordinates (in pixels) and palm scales from raw CSV files.
    Returns: cache[(video_file, video_hash)][frame_idx] = (pts_px, l_hand)
    """
    cache = {}
    csv_files = glob.glob(os.path.join(raw_dir, "*.raw_landmarks.*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(raw_dir, "*.raw_landmarks.csv"))

    print(f"  [Cache Loader] Reading {len(csv_files)} raw landmarks CSVs from: {raw_dir}")

    for f_path in csv_files:
        with open(f_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v_file = row.get("video_file", "")
                v_hash = row.get("video_hash", "")
                try:
                    f_idx = int(row.get("frame_idx", -1))
                    w = float(row.get("video_width", 640))
                    h = float(row.get("video_height", 480))
                except (ValueError, TypeError):
                    continue

                if f_idx < 0:
                    continue

                key = (v_file, v_hash)
                if key not in cache:
                    cache[key] = {}

                pts_px = {}
                for name in STATIONARY_NAMES:
                    try:
                        rx = float(row.get(f"{name}_x", 0.0))
                        ry = float(row.get(f"{name}_y", 0.0))
                        pts_px[name] = (rx * w, ry * h)
                    except (ValueError, TypeError):
                        pts_px[name] = (0.0, 0.0)

                l_hand = calculate_palm_scale(pts_px)
                cache[key][f_idx] = (pts_px, l_hand)

    return cache


def filter_hand_movement_csv(
    input_csv: str,
    output_csv: str,
    raw_dir: str,
    threshold: float = 0.20
) -> str:
    """
    Reads combined window dataset CSV, checks stationary landmark displacement across each
    5-frame window using preloaded raw landmarks, and filters out moving hand windows.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input window dataset not found: {input_csv}")

    print(f"\n{'='*75}")
    print(f"  HAND MOVEMENT DISPLACEMENT FILTER (THRESHOLD: {threshold:.3f} L_hand)")
    print(f"{'='*75}")
    print(f"  Input Window CSV  : {input_csv}")
    print(f"  Output Window CSV : {output_csv}")
    print(f"  Raw Landmarks Dir : {raw_dir}")
    print(f"  Displacement Cap  : {threshold:.3f} L_hand")

    raw_cache = load_raw_landmarks_cache(raw_dir)

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    total_windows = len(rows)
    kept_rows = []
    dropped_count = 0
    missing_cache_count = 0

    max_displacements = []

    for row in rows:
        v_file = row.get("video_file", "")
        v_hash = row.get("video_hash", "")
        try:
            start_f = int(row.get("start_frame", -1))
            end_f = int(row.get("end_frame", -1))
        except (ValueError, TypeError):
            start_f, end_f = -1, -1

        key = (v_file, v_hash)
        video_frames = raw_cache.get(key)

        # Fallback search by video_file only if hash doesn't match
        if not video_frames:
            for (kf, kh), f_dict in raw_cache.items():
                if kf == v_file:
                    video_frames = f_dict
                    break

        if not video_frames or start_f not in video_frames or end_f not in video_frames:
            # If raw landmarks are missing for this window, keep row safely
            missing_cache_count += 1
            kept_rows.append(row)
            continue

        pts_f1, l_hand_f1 = video_frames[start_f]
        pts_f5, _ = video_frames[end_f]

        # Calculate stationary displacement across Wrist and 4 MCPs
        max_stationary_disp = 0.0
        for name in STATIONARY_NAMES:
            p1 = pts_f1[name]
            p5 = pts_f5[name]
            if p1 != (0.0, 0.0) and p5 != (0.0, 0.0):
                dist = math.sqrt((p5[0] - p1[0]) ** 2 + (p5[1] - p1[1]) ** 2) / l_hand_f1
                if dist > max_stationary_disp:
                    max_stationary_disp = dist

        max_displacements.append(max_stationary_disp)

        # Apply threshold filter on stationary joints
        if max_stationary_disp > threshold:
            dropped_count += 1
        else:
            kept_rows.append(row)

    kept_count = len(kept_rows)
    kept_pct = (kept_count / total_windows * 100.0) if total_windows > 0 else 0.0
    dropped_pct = (dropped_count / total_windows * 100.0) if total_windows > 0 else 0.0

    print(f"\n  [Audit Summary]")
    print(f"  Total Windows Evaluated : {total_windows:,}")
    print(f"  Stationary Windows Kept : {kept_count:,} ({kept_pct:.2f}%)")
    print(f"  Hand Moving Dropped     : {dropped_count:,} ({dropped_pct:.2f}%)")
    if missing_cache_count > 0:
        print(f"  Missing Raw Landmarks   : {missing_cache_count:,} (Kept safely)")

    if max_displacements:
        avg_disp = sum(max_displacements) / len(max_displacements)
        peak_disp = max(max_displacements)
        print(f"  Mean Stationary Disp    : {avg_disp:.4f} L_hand")
        print(f"  Peak Stationary Disp    : {peak_disp:.4f} L_hand")

    # Write output CSV (using safe temporary file in case input == output)
    tmp_output = f"{output_csv}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(tmp_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(kept_rows)

    os.replace(tmp_output, output_csv)
    print(f"  Filtered dataset saved  → {output_csv}")
    print(f"{'='*75}\n")

    # Record stage summary JSON
    summary_data = {
        "step": 7,
        "step_name": "step_7_filter_hand_movement",
        "threshold": threshold,
        "total_windows": total_windows,
        "kept_windows": kept_count,
        "dropped_windows": dropped_count,
        "kept_percentage": round(kept_pct, 2),
        "dropped_percentage": round(dropped_pct, 2),
    }
    save_step_summary("step_7_filter_hand_movement.json", summary_data)

    return output_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter hand moving windows based on stationary landmark displacement threshold."
    )
    parser.add_argument(
        "-i", "--input-csv",
        default="./dataprocessing/6_merged_windowed_dataset/all_windowed_dataset.csv",
        help="Path to combined window dataset CSV"
    )
    parser.add_argument(
        "-o", "--output-csv",
        default="./dataprocessing/7_hand_movement_filtered/hand_movement_filtered_dataset.csv",
        help="Path to output filtered window dataset CSV"
    )
    parser.add_argument(
        "-r", "--raw-dir",
        default="./dataprocessing/1_rawCSVFiles/",
        help="Path to directory containing raw landmark CSV files"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.20,
        help="Displacement threshold in hand-lengths (default: 0.20)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filter_hand_movement_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        raw_dir=args.raw_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
