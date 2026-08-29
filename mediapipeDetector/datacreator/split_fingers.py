# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/split_fingers.py

Unrolls 5-finger sequence window dataset CSV rows into 5 individual per-finger dataset rows (thumb, index, middle, ring, pinky).

Each unrolled per-finger row preserves:
1. Common palm joints (wrist, thumb_cmc, index_mcp, middle_mcp, ring_mcp, pinky_mcp) across 5 frames & 4 velocity steps.
2. Finger-specific joints (pip, dip, tip) mapped consistently:
   - For Thumb: thumb_mcp -> pip, thumb_ip -> dip, thumb_tip -> tip
   - For Index: index_pip -> pip, index_dip -> dip, index_tip -> tip
   - For Middle: middle_pip -> pip, middle_dip -> dip, middle_tip -> tip
   - For Ring: ring_pip -> pip, ring_dip -> dip, ring_tip -> tip
   - For Pinky: pinky_pip -> pip, pinky_dip -> dip, pinky_tip -> tip
3. 5-frame hand confidence scores (hand_score1..5), 3D depth (z), visibility, presence, and 3D velocities/speeds.
4. Per-finger touch label (touch_finger) and context flags.

Supports batch execution over single CSVs, glob patterns, or directories. Overwrites target output file if it already exists.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

COMMON_PALM_JOINTS = [
    "wrist", "thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
]

# Joint mapping for finger-specific joints (pip, dip, tip)
FINGER_JOINT_MAP = {
    "thumb": {"pip": "thumb_mcp", "dip": "thumb_ip", "tip": "thumb_tip"},
    "index": {"pip": "index_pip", "dip": "index_dip", "tip": "index_tip"},
    "middle": {"pip": "middle_pip", "dip": "middle_dip", "tip": "middle_tip"},
    "ring": {"pip": "ring_pip", "dip": "ring_dip", "tip": "ring_tip"},
    "pinky": {"pip": "pinky_pip", "dip": "pinky_dip", "tip": "pinky_tip"},
}

METADATA_KEYS = [
    "video_file", "video_hash", "duration_ms", "start_ms", "end_ms",
    "start_frame", "end_frame", "window_idx", "window_size", "window_overlap"
]

LABEL_FLAGS = [
    "right_hand", "hand_move", "hand_closer", "hovering", "daylight",
    "hand_visible", "out_of_sync", "any_touch"
]


def build_split_finger_headers() -> list[str]:
    """Constructs CSV fieldnames for per-finger dataset rows."""
    headers = list(METADATA_KEYS)
    headers.append("finger_name")

    # 1. 5-frame coordinates & confidence scores (k = 1..5)
    for k in range(1, 6):
        headers.append(f"hand_score{k}")
        # Common palm joints
        for j in COMMON_PALM_JOINTS:
            headers.append(f"{j}{k}_x")
            headers.append(f"{j}{k}_y")
            headers.append(f"{j}{k}_z")
            headers.append(f"{j}{k}_visibility")
            headers.append(f"{j}{k}_presence")
        # Finger-specific joints
        for j in ["pip", "dip", "tip"]:
            headers.append(f"{j}{k}_x")
            headers.append(f"{j}{k}_y")
            headers.append(f"{j}{k}_z")
            headers.append(f"{j}{k}_visibility")
            headers.append(f"{j}{k}_presence")

    # 2. 4-step velocities & speeds (v = 1..4)
    for v in range(1, 5):
        # Common palm joints
        for j in COMMON_PALM_JOINTS:
            headers.append(f"{j}{v}_vx")
            headers.append(f"{j}{v}_vy")
            headers.append(f"{j}{v}_vz")
            headers.append(f"{j}{v}_speed_2d")
            headers.append(f"{j}{v}_speed_3d")
        # Finger-specific joints
        for j in ["pip", "dip", "tip"]:
            headers.append(f"{j}{v}_vx")
            headers.append(f"{j}{v}_vy")
            headers.append(f"{j}{v}_vz")
            headers.append(f"{j}{v}_speed_2d")
            headers.append(f"{j}{v}_speed_3d")

    # 3. Target per-finger touch label & context flags
    headers.append("touch_finger")
    for flag in LABEL_FLAGS:
        headers.append(flag)

    return headers


def parse_bool_flag(val: str) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def split_fingers_csv(input_csv: str, output_csv: str = None) -> str:
    """
    Unrolls 5-finger window dataset CSV rows into per-finger dataset CSV rows.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    dir_name = os.path.dirname(os.path.abspath(input_csv))
    base_name = os.path.basename(input_csv)

    if not output_csv:
        if ".csv" in base_name:
            out_name = base_name.replace(".csv", ".per_finger.csv")
        else:
            out_name = f"{base_name}.per_finger.csv"
        output_csv = os.path.join(dir_name, out_name)

    print(f"[1/3] Reading combined window dataset from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    out_headers = build_split_finger_headers()
    unrolled_rows = []

    print(f"[2/3] Unrolling {len(rows)} window rows into {len(rows) * 5} per-finger dataset records...")

    touch_finger_counts = {fg: 0 for fg in FINGERS}
    total_touch_records = 0

    for row in rows:
        meta_dict = {k: row.get(k, "") for k in METADATA_KEYS}
        label_dict = {}
        for flag in LABEL_FLAGS:
            if flag == "right_hand" and "right_hand" not in row and "rightHand" in row:
                label_dict["right_hand"] = row.get("rightHand", "0")
            else:
                label_dict[flag] = row.get(flag, "0")

        # Unroll into 5 per-finger rows
        for finger in FINGERS:
            f_row = dict(meta_dict)
            f_row["finger_name"] = finger

            # 1. Populate 5-frame coordinates & confidence (k = 1..5)
            for k in range(1, 6):
                f_row[f"hand_score{k}"] = row.get(f"hand_score{k}", "0.0")

                # Common palm joints
                for j in COMMON_PALM_JOINTS:
                    f_row[f"{j}{k}_x"] = row.get(f"{j}{k}_x", "0.0")
                    f_row[f"{j}{k}_y"] = row.get(f"{j}{k}_y", "0.0")
                    f_row[f"{j}{k}_z"] = row.get(f"{j}{k}_z", "0.0")
                    f_row[f"{j}{k}_visibility"] = row.get(f"{j}{k}_visibility", "0.0")
                    f_row[f"{j}{k}_presence"] = row.get(f"{j}{k}_presence", "0.0")

                # Finger-specific joints (pip, dip, tip)
                jmap = FINGER_JOINT_MAP[finger]
                for target_j, src_j in jmap.items():
                    f_row[f"{target_j}{k}_x"] = row.get(f"{src_j}{k}_x", "0.0")
                    f_row[f"{target_j}{k}_y"] = row.get(f"{src_j}{k}_y", "0.0")
                    f_row[f"{target_j}{k}_z"] = row.get(f"{src_j}{k}_z", "0.0")
                    f_row[f"{target_j}{k}_visibility"] = row.get(f"{src_j}{k}_visibility", "0.0")
                    f_row[f"{target_j}{k}_presence"] = row.get(f"{src_j}{k}_presence", "0.0")

            # 2. Populate 4-step velocities & speeds (v = 1..4)
            for v in range(1, 5):
                # Common palm joints
                for j in COMMON_PALM_JOINTS:
                    f_row[f"{j}{v}_vx"] = row.get(f"{j}{v}_vx", "0.000000")
                    f_row[f"{j}{v}_vy"] = row.get(f"{j}{v}_vy", "0.000000")
                    f_row[f"{j}{v}_vz"] = row.get(f"{j}{v}_vz", "0.000000")
                    f_row[f"{j}{v}_speed_2d"] = row.get(f"{j}{v}_speed_2d", "0.000000")
                    f_row[f"{j}{v}_speed_3d"] = row.get(f"{j}{v}_speed_3d", "0.000000")

                # Finger-specific joints (pip, dip, tip)
                jmap = FINGER_JOINT_MAP[finger]
                for target_j, src_j in jmap.items():
                    f_row[f"{target_j}{v}_vx"] = row.get(f"{src_j}{v}_vx", "0.000000")
                    f_row[f"{target_j}{v}_vy"] = row.get(f"{src_j}{v}_vy", "0.000000")
                    f_row[f"{target_j}{v}_vz"] = row.get(f"{src_j}{v}_vz", "0.000000")
                    f_row[f"{target_j}{v}_speed_2d"] = row.get(f"{src_j}{v}_speed_2d", "0.000000")
                    f_row[f"{target_j}{v}_speed_3d"] = row.get(f"{src_j}{v}_speed_3d", "0.000000")

            # 3. Finger touch label
            finger_touch_val = row.get(f"{finger}_touch", "0")
            is_touch = parse_bool_flag(finger_touch_val)
            f_row["touch_finger"] = "1" if is_touch else "0"

            if is_touch:
                touch_finger_counts[finger] += 1
                total_touch_records += 1

            # 4. Context & label flags
            for flag in LABEL_FLAGS:
                f_row[flag] = label_dict[flag]

            unrolled_rows.append(f_row)

    total_unrolled = len(unrolled_rows)
    non_touch_records = total_unrolled - total_touch_records

    print(f"\n==========================================")
    print(f"   PER-FINGER DATASET ANALYTICS REPORT")
    print(f"==========================================")
    print(f"  Input Window Rows       : {len(rows)}")
    print(f"  Unrolled Per-Finger Rows: {total_unrolled}")
    print(f"  Touch Rows (touch_finger): {total_touch_records} ({total_touch_records / total_unrolled * 100.0:.2f}%)")
    print(f"  Non-Touch Rows           : {non_touch_records} ({non_touch_records / total_unrolled * 100.0:.2f}%)")
    print(f"  Per-Finger Touch Breakdown:")
    for fg in FINGERS:
        cnt = touch_finger_counts[fg]
        pct = (cnt / total_unrolled * 100.0) if total_unrolled > 0 else 0.0
        print(f"    - {fg.capitalize():<6} finger touch : {cnt} ({pct:.2f}%)")
    print(f"==========================================\n")

    print(f"[3/3] Saving per-finger dataset to: {output_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv}': {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()
        writer.writerows(unrolled_rows)

    print(f"[Success] Saved per-finger dataset to: {output_csv}")
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
        description="Unrolls 5-finger sequence window dataset CSV rows into 5 individual per-finger dataset rows"
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
        print(f"[{idx}/{len(input_files)}] Unrolling per-finger dataset: {input_file}")
        try:
            out_file_path = None
            if output_target:
                if os.path.isdir(output_target) or output_target.endswith(os.sep) or output_target.endswith("/"):
                    os.makedirs(output_target, exist_ok=True)
                    base_name = os.path.basename(input_file)
                    if ".csv" in base_name:
                        out_name = base_name.replace(".csv", ".per_finger.csv")
                    else:
                        out_name = f"{base_name}.per_finger.csv"
                    out_file_path = os.path.join(output_target, out_name)
                elif len(input_files) == 1:
                    out_file_path = output_target
                else:
                    os.makedirs(output_target, exist_ok=True)
                    out_file_path = os.path.join(output_target, os.path.basename(input_file))
            else:
                out_file_path = None

            split_fingers_csv(input_csv=input_file, output_csv=out_file_path)
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Per-Finger Dataset Unrolling Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
