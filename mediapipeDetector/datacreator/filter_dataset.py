# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/filter_dataset.py

Filters sequence window dataset CSV rows based on configurable cleaning flags:
- --remove-zero-vel-touch: Removes rows where touch occurs (any_touch=1) but all velocity components across 4 transition points are zero.
- --remove-right-hand: Removes right-hand rows (right_hand=1).
- --remove-left-hand: Removes left-hand rows (right_hand=0).
- --remove-out-of-sync: Removes out-of-sync rows (out_of_sync=1).
- --remove-hand-invisible: Removes rows where hand is not visible (hand_visible=0).

Accepts single CSV files, glob patterns, or directory inputs. Overwrites output file if it already exists.
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

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

REQUIRED_FILTER_KEYS = [
    "right_hand", "hand_visible", "out_of_sync", "any_touch"
]


def parse_bool_flag(val: str) -> bool:
    """Parses boolean value from CSV cell string or int."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def validate_filter_headers(headers: list[str], csv_path: str):
    """Ensures input CSV contains required filtering flag columns."""
    missing = []
    for col in REQUIRED_FILTER_KEYS:
        if col not in headers and col != "right_hand":
            missing.append(col)

    # Accept either right_hand or rightHand
    if "right_hand" not in headers and "rightHand" not in headers:
        missing.append("right_hand")

    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required filtering flag column(s): {', '.join(missing)}"
        )


def filter_dataset_csv(
    input_csv: str,
    output_csv: str = None,
    remove_zero_vel_touch: bool = False,
    remove_right_hand: bool = False,
    remove_left_hand: bool = False,
    remove_out_of_sync: bool = False,
    remove_hand_invisible: bool = False,
) -> str:
    """
    Reads dataset CSV, applies configurable filtering rules, and writes cleaned CSV to output_csv.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    dir_name = os.path.dirname(os.path.abspath(input_csv))
    base_name = os.path.basename(input_csv)

    if not output_csv:
        if ".csv" in base_name:
            out_name = base_name.replace(".csv", ".cleaned.csv")
        else:
            out_name = f"{base_name}.cleaned.csv"
        output_csv = os.path.join(dir_name, out_name)

    print(f"[1/3] Reading dataset for filtering from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    validate_filter_headers(headers, input_csv)

    total_rows = len(rows)
    retained_rows = 0
    removed_zero_vel_touch_cnt = 0
    removed_right_hand_cnt = 0
    removed_left_hand_cnt = 0
    removed_out_of_sync_cnt = 0
    removed_hand_invisible_cnt = 0

    print(f"[2/3] Filtering {total_rows} sequence window dataset rows...")

    filtered_rows = []
    for row in rows:
        keep = True

        # 1. Filter zero velocity touch
        if keep and remove_zero_vel_touch:
            touch = parse_bool_flag(row.get("any_touch", "0"))
            if not touch:
                for t_key in ["thumb_touch", "index_touch", "middle_touch", "ring_touch", "pinky_touch"]:
                    if parse_bool_flag(row.get(t_key, "0")):
                        touch = True
                        break

            if touch:
                all_zero = True
                # Check 4 velocity steps for all fingertip/landmark velocity components
                for v in range(1, 5):
                    for lm_name in ALL_21_LANDMARK_NAMES:
                        vx_str = row.get(f"{lm_name}{v}_vx", "0")
                        vy_str = row.get(f"{lm_name}{v}_vy", "0")
                        try:
                            vx = float(vx_str) if vx_str != "" else 0.0
                            vy = float(vy_str) if vy_str != "" else 0.0
                        except ValueError:
                            vx, vy = 0.0, 0.0

                        if vx != 0.0 or vy != 0.0:
                            all_zero = False
                            break
                    if not all_zero:
                        break

                if all_zero:
                    keep = False
                    removed_zero_vel_touch_cnt += 1

        # 2. Filter right hand rows
        if keep and remove_right_hand:
            is_right = parse_bool_flag(row.get("right_hand", row.get("rightHand", "0")))
            if is_right:
                keep = False
                removed_right_hand_cnt += 1

        # 3. Filter left hand rows
        if keep and remove_left_hand:
            is_right = parse_bool_flag(row.get("right_hand", row.get("rightHand", "0")))
            if not is_right:
                keep = False
                removed_left_hand_cnt += 1

        # 4. Filter out of sync rows
        if keep and remove_out_of_sync:
            is_out_of_sync = parse_bool_flag(row.get("out_of_sync", "0"))
            if is_out_of_sync:
                keep = False
                removed_out_of_sync_cnt += 1

        # 5. Filter hand invisible rows
        if keep and remove_hand_invisible:
            is_hand_visible = parse_bool_flag(row.get("hand_visible", "1"))
            if not is_hand_visible:
                keep = False
                removed_hand_invisible_cnt += 1

        if keep:
            filtered_rows.append(row)
            retained_rows += 1

    # ── Compute Analytics Report ──────────────────────────────────────────────
    retained_cnt = len(filtered_rows)
    dropped_cnt = total_rows - retained_cnt
    retention_pct = (retained_cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    touch_cnt = 0
    non_touch_cnt = 0
    right_cnt = 0
    left_cnt = 0
    finger_touch_cnts = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}

    for r in filtered_rows:
        is_touch = parse_bool_flag(r.get("any_touch", "0"))
        if not is_touch:
            for fg in ["thumb", "index", "middle", "ring", "pinky"]:
                if parse_bool_flag(r.get(f"{fg}_touch", "0")):
                    is_touch = True
                    break

        if is_touch:
            touch_cnt += 1
        else:
            non_touch_cnt += 1

        is_right = parse_bool_flag(r.get("right_hand", r.get("rightHand", "0")))
        if is_right:
            right_cnt += 1
        else:
            left_cnt += 1

        for fg in ["thumb", "index", "middle", "ring", "pinky"]:
            if parse_bool_flag(r.get(f"{fg}_touch", "0")):
                finger_touch_cnts[fg] += 1

    touch_pct = (touch_cnt / retained_cnt * 100.0) if retained_cnt > 0 else 0.0
    non_touch_pct = (non_touch_cnt / retained_cnt * 100.0) if retained_cnt > 0 else 0.0

    print(f"\n==========================================")
    print(f"   CLEANED DATASET ANALYTICS REPORT")
    print(f"==========================================")
    print(f"  Input Dataset Rows     : {total_rows}")
    print(f"  Retained Rows          : {retained_cnt} ({retention_pct:.2f}%)")
    print(f"  Dropped Rows           : {dropped_cnt} ({100.0 - retention_pct:.2f}%)")
    print(f"  Filtering Breakdown    :")
    if remove_zero_vel_touch:
        print(f"    - Removed Zero-Velocity Touch: {removed_zero_vel_touch_cnt}")
    if remove_right_hand:
        print(f"    - Removed Right Hand Rows    : {removed_right_hand_cnt}")
    if remove_left_hand:
        print(f"    - Removed Left Hand Rows     : {removed_left_hand_cnt}")
    if remove_out_of_sync:
        print(f"    - Removed Out-of-Sync Rows   : {removed_out_of_sync_cnt}")
    if remove_hand_invisible:
        print(f"    - Removed Hand Invisible Rows: {removed_hand_invisible_cnt}")
    print(f"  Retained Class Distribution:")
    print(f"    - Touch Windows (any_touch)  : {touch_cnt} ({touch_pct:.2f}%)")
    print(f"    - Non-Touch Windows        : {non_touch_cnt} ({non_touch_pct:.2f}%)")
    print(f"  Retained Handedness Breakdown  : Right: {right_cnt} | Left: {left_cnt}")
    print(f"  Retained Per-Finger Touches    :")
    for fg, cnt in finger_touch_cnts.items():
        fg_pct = (cnt / retained_cnt * 100.0) if retained_cnt > 0 else 0.0
        print(f"    - {fg.capitalize():<6} touch         : {cnt} ({fg_pct:.2f}%)")
    print(f"==========================================\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if os.path.exists(output_csv):
        print(f"[Info] Overwriting existing file: {output_csv}")
        try:
            os.remove(output_csv)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv}': {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[Success] Cleaned dataset saved to: {output_csv}")
    return output_csv


def collect_csv_files(input_patterns: list[str]) -> list[str]:
    """Expands glob patterns, directory paths, or file lists into a sorted list of CSV file paths."""
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
        description="Filters sequence window dataset CSV rows based on configurable cleaning flags"
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
        help="Input window dataset CSV file path(s), glob pattern(s), or input directory"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory path (or output CSV file path for single input)"
    )
    parser.add_argument(
        "--remove-zero-vel-touch",
        action="store_true",
        help="Remove rows where touch occurs but all velocity components are 0.0"
    )
    parser.add_argument(
        "--remove-right-hand",
        action="store_true",
        help="Remove right-hand rows (right_hand=1)"
    )
    parser.add_argument(
        "--remove-left-hand",
        action="store_true",
        help="Remove left-hand rows (right_hand=0)"
    )
    parser.add_argument(
        "--remove-out-of-sync",
        action="store_true",
        help="Remove out-of-sync rows (out_of_sync=1)"
    )
    parser.add_argument(
        "--remove-hand-invisible",
        action="store_true",
        help="Remove rows where hand is not visible (hand_visible=0)"
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

    input_files = collect_csv_files(input_patterns)
    if not input_files:
        print("[Error] No valid CSV files found for dataset filtering. Exiting.")
        sys.exit(1)

    print(f"Found {len(input_files)} CSV file(s) to filter:")
    for f in input_files:
        print(f"  - {f}")
    print()

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Filtering dataset: {input_file}")
        try:
            out_file_path = None
            if output_target:
                if os.path.isdir(output_target) or output_target.endswith(os.sep) or output_target.endswith("/"):
                    os.makedirs(output_target, exist_ok=True)
                    base_name = os.path.basename(input_file)
                    out_file_path = os.path.join(output_target, base_name)
                elif len(input_files) == 1:
                    out_file_path = output_target
                else:
                    os.makedirs(output_target, exist_ok=True)
                    out_file_path = os.path.join(output_target, os.path.basename(input_file))
            else:
                out_file_path = None

            filter_dataset_csv(
                input_csv=input_file,
                output_csv=out_file_path,
                remove_zero_vel_touch=args.remove_zero_vel_touch,
                remove_right_hand=args.remove_right_hand,
                remove_left_hand=args.remove_left_hand,
                remove_out_of_sync=args.remove_out_of_sync,
                remove_hand_invisible=args.remove_hand_invisible,
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not filter dataset '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Dataset Filtering Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
