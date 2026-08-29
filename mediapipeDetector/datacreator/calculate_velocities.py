# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/calculate_velocities.py

Calculates 4 frame-to-frame velocity components (vx, vy, vz) and speeds (2D & 3D) across 5-frame sequence windows
for all 21 MediaPipe hand landmarks.

Input: Window dataset CSV file(s) containing 5 frame landmark coordinates.
Output: Enhanced CSV dataset file(s) with velocity & speed columns added before annotation labels.

Supports batch execution over single CSVs, glob patterns, or directories. Overwrites target file if it already exists.
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

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

REQUIRED_METADATA_KEYS = [
    "video_file", "video_hash", "duration_ms", "start_ms", "end_ms",
    "start_frame", "end_frame", "window_idx", "window_size", "window_overlap"
]

REQUIRED_LABEL_KEYS = [
    "right_hand", "hand_move", "hand_closer", "hovering", "daylight",
    "hand_visible", "out_of_sync", "thumb_touch", "index_touch",
    "middle_touch", "ring_touch", "pinky_touch", "any_touch"
]


def build_velocity_csv_headers(input_headers: list[str]) -> list[str]:
    """
    Constructs output CSV headers by inserting velocity & speed columns
    after frame coordinate columns and before annotation labels.
    """
    meta_and_coords = []
    labels = []

    for h in input_headers:
        if h in REQUIRED_LABEL_KEYS:
            labels.append(h)
        else:
            meta_and_coords.append(h)

    for l_key in REQUIRED_LABEL_KEYS:
        if l_key not in labels:
            labels.append(l_key)

    velocity_headers = []
    # 4 Velocity steps (v = 1..4) between 5 frame steps
    for v in range(1, 5):
        for lm_name in ALL_21_LANDMARK_NAMES:
            velocity_headers.append(f"{lm_name}{v}_vx")
            velocity_headers.append(f"{lm_name}{v}_vy")
            velocity_headers.append(f"{lm_name}{v}_vz")
            velocity_headers.append(f"{lm_name}{v}_speed_2d")
            velocity_headers.append(f"{lm_name}{v}_speed_3d")

    return meta_and_coords + velocity_headers + labels


def validate_window_columns(headers: list[str], csv_path: str):
    """Validates that input CSV contains required metadata, 5-frame coordinates, and label columns."""
    missing = []
    for col in REQUIRED_METADATA_KEYS:
        if col not in headers:
            missing.append(col)

    for k in [1, 5]:
        if f"wrist{k}_x" not in headers or f"wrist{k}_y" not in headers:
            missing.append(f"wrist{k}_x/y")

    for col in REQUIRED_LABEL_KEYS:
        if col not in headers:
            missing.append(col)

    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required window dataset column(s): {', '.join(missing)}"
        )


def calculate_window_velocities_csv(input_csv: str, output_csv: str = None) -> str:
    """
    Reads window dataset CSV, calculates 4 frame-to-frame velocities (vx, vy, vz) and speeds (2D & 3D)
    for all 21 joints, and writes output CSV with velocity columns inserted.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    dir_name = os.path.dirname(os.path.abspath(input_csv))
    base_name = os.path.basename(input_csv)

    if not output_csv:
        if ".csv" in base_name:
            out_name = base_name.replace(".csv", ".velocities.csv")
        else:
            out_name = f"{base_name}.velocities.csv"
        output_csv = os.path.join(dir_name, out_name)

    print(f"[1/3] Reading window dataset from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    validate_window_columns(headers, input_csv)
    output_headers = build_velocity_csv_headers(headers)

    print(f"[2/3] Calculating 4-step 2D/3D velocities & speeds across {len(rows)} window sequence records...")

    output_rows = []
    for row in rows:
        out_row = dict(row)

        for v in range(1, 5):
            curr_frame = v
            next_frame = v + 1

            for lm_name in ALL_21_LANDMARK_NAMES:
                x1_str = row.get(f"{lm_name}{curr_frame}_x", "0.0")
                y1_str = row.get(f"{lm_name}{curr_frame}_y", "0.0")
                z1_str = row.get(f"{lm_name}{curr_frame}_z", "0.0")
                x2_str = row.get(f"{lm_name}{next_frame}_x", "0.0")
                y2_str = row.get(f"{lm_name}{next_frame}_y", "0.0")
                z2_str = row.get(f"{lm_name}{next_frame}_z", "0.0")

                try:
                    x1, y1, z1 = float(x1_str), float(y1_str), float(z1_str)
                    x2, y2, z2 = float(x2_str), float(y2_str), float(z2_str)

                    vx = x2 - x1
                    vy = y2 - y1
                    vz = z2 - z1
                    speed_2d = math.sqrt(vx * vx + vy * vy)
                    speed_3d = math.sqrt(vx * vx + vy * vy + vz * vz)
                except ValueError:
                    vx, vy, vz, speed_2d, speed_3d = 0.0, 0.0, 0.0, 0.0, 0.0

                out_row[f"{lm_name}{v}_vx"] = f"{vx:.6f}"
                out_row[f"{lm_name}{v}_vy"] = f"{vy:.6f}"
                out_row[f"{lm_name}{v}_vz"] = f"{vz:.6f}"
                out_row[f"{lm_name}{v}_speed_2d"] = f"{speed_2d:.6f}"
                out_row[f"{lm_name}{v}_speed_3d"] = f"{speed_3d:.6f}"

        output_rows.append(out_row)

    print(f"[3/3] Saving dataset with velocities to: {output_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv}': {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_headers)
        writer.writeheader()
        writer.writerows(output_rows)

    # Collect speed statistics across output rows
    speeds_2d = []
    speeds_3d = []
    for r in output_rows:
        for v in range(1, 5):
            for lm_name in ALL_21_LANDMARK_NAMES:
                try:
                    speeds_2d.append(float(r.get(f"{lm_name}{v}_speed_2d", "0.0")))
                    speeds_3d.append(float(r.get(f"{lm_name}{v}_speed_3d", "0.0")))
                except ValueError:
                    pass

    s2d_min = round(min(speeds_2d), 4) if speeds_2d else 0.0
    s2d_max = round(max(speeds_2d), 4) if speeds_2d else 0.0
    s2d_mean = round(sum(speeds_2d) / len(speeds_2d), 4) if speeds_2d else 0.0

    s3d_min = round(min(speeds_3d), 4) if speeds_3d else 0.0
    s3d_max = round(max(speeds_3d), 4) if speeds_3d else 0.0
    s3d_mean = round(sum(speeds_3d) / len(speeds_3d), 4) if speeds_3d else 0.0

    print(f"[Success] Saved dataset with velocities to: {output_csv}")

    # Save summary JSON for pipeline audit
    try:
        from summary_utils import save_step_summary
        save_step_summary("step_7_calculate_velocities.json", {
            "step": 7,
            "name": "calculate_velocities",
            "total_windows": len(output_rows),
            "speed_2d_stats": {"min": s2d_min, "max": s2d_max, "mean": s2d_mean},
            "speed_3d_stats": {"min": s3d_min, "max": s3d_max, "mean": s3d_mean}
        })
    except Exception as e:
        pass

    return output_csv


def collect_input_files(input_patterns: list[str]) -> list[str]:
    """Expands glob patterns, directory paths, or file lists into a list of CSV file paths."""
    matched_files = []
    for pattern in input_patterns:
        search_pattern = os.path.join(pattern, "*.csv") if os.path.isdir(pattern) else pattern
        glob_matches = glob.glob(search_pattern, recursive=True)
        if glob_matches:
            for filepath in sorted(glob_matches):
                if os.path.isfile(filepath) and filepath.endswith(".csv"):
                    if filepath not in matched_files:
                        matched_files.append(filepath)
        elif os.path.isfile(pattern) and pattern not in matched_files:
            matched_files.append(pattern)

    return matched_files


def main():
    parser = argparse.ArgumentParser(
        description="Calculates 4-step frame-to-frame velocities (vx, vy, vz) and speeds for all 21 hand joints"
    )
    parser.add_argument(
        "pos_args",
        nargs="*",
        help="Input CSV file(s), glob pattern(s), or output directory"
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        default=None,
        help="Input CSV file path(s) or glob pattern(s)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory path"
    )

    args = parser.parse_args()

    input_patterns = []
    output_target = None

    if args.output:
        output_target = args.output
        if args.input:
            input_patterns = args.input
        elif args.pos_args:
            input_patterns = args.pos_args
    else:
        combined = []
        if args.input:
            combined.extend(args.input)
        if args.pos_args:
            combined.extend(args.pos_args)

        if len(combined) >= 2:
            output_target = combined[-1]
            input_patterns = combined[:-1]
        elif len(combined) == 1:
            input_patterns = combined
            output_target = None
        else:
            parser.print_help()
            sys.exit(1)

    input_files = collect_input_files(input_patterns)
    if not input_files:
        print("[Error] No valid input CSV files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(input_files)} CSV file(s) to process:")
    for f in input_files:
        print(f"  - {f}")

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Calculating velocities: {input_file}")
        try:
            out_file_path = None
            if output_target:
                if os.path.isdir(output_target) or output_target.endswith(os.sep) or output_target.endswith("/"):
                    os.makedirs(output_target, exist_ok=True)
                    base_name = os.path.basename(input_file)
                    if ".csv" in base_name:
                        out_name = base_name.replace(".csv", ".velocities.csv")
                    else:
                        out_name = f"{base_name}.velocities.csv"
                    out_file_path = os.path.join(output_target, out_name)
                elif len(input_files) == 1:
                    out_file_path = output_target
                else:
                    os.makedirs(output_target, exist_ok=True)
                    out_file_path = os.path.join(output_target, os.path.basename(input_file))
            else:
                out_file_path = None

            calculate_window_velocities_csv(input_csv=input_file, output_csv=out_file_path)
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Velocity Calculation Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    # Save summary JSON for pipeline audit
    try:
        from summary_utils import save_step_summary
        save_step_summary("step_7_calculate_velocities.json", {
            "step": 7,
            "name": "calculate_velocities",
            "total_files": len(input_files),
            "success_count": success_count,
            "fail_count": fail_count
        })
    except Exception as e:
        pass

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
