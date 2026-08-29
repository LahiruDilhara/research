# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/split_touch.py

Splits per-finger sequence window dataset CSV rows into two separate CSV files:
1. touch_dataset.csv: Contains all records where touch_finger == 1
2. untouch_dataset.csv: Contains all records where touch_finger == 0

Accepts single CSV files, glob patterns, or directory inputs. Overwrites output files if they already exist.
Includes detailed per-finger analytics for touch vs untouch splits.
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


def parse_bool_flag(val: str) -> bool:
    """Parses boolean value from CSV cell string or int."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def split_touch_csv(
    input_csv: str,
    output_dir: str = None,
    touch_out_path: str = None,
    untouch_out_path: str = None,
) -> tuple[str, str]:
    """
    Reads input per-finger CSV, splits records into touch and untouch CSV datasets.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    if not touch_out_path or not untouch_out_path:
        if not output_dir:
            output_dir = os.path.dirname(os.path.abspath(input_csv))
        os.makedirs(output_dir, exist_ok=True)
        touch_out_path = touch_out_path or os.path.join(output_dir, "touch_dataset.csv")
        untouch_out_path = untouch_out_path or os.path.join(output_dir, "untouch_dataset.csv")

    print(f"[1/3] Reading per-finger dataset for touch splitting from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    # Check for touch label column
    touch_col = None
    if "touch_finger" in headers:
        touch_col = "touch_finger"
    elif "any_touch" in headers:
        touch_col = "any_touch"
    elif "touch" in headers:
        touch_col = "touch"
    else:
        raise ValueError(
            f"CSV '{input_csv}' does not contain 'touch_finger', 'any_touch', or 'touch' column!"
        )

    touch_rows = []
    untouch_rows = []

    touch_finger_counts = {fg: 0 for fg in FINGERS}
    untouch_finger_counts = {fg: 0 for fg in FINGERS}

    print(f"[2/3] Splitting {len(rows)} records using column '{touch_col}'...")

    for r in rows:
        is_touch = parse_bool_flag(r.get(touch_col, "0"))
        fg_name = r.get("finger_name", "").lower()

        if is_touch:
            touch_rows.append(r)
            if fg_name in touch_finger_counts:
                touch_finger_counts[fg_name] += 1
        else:
            untouch_rows.append(r)
            if fg_name in untouch_finger_counts:
                untouch_finger_counts[fg_name] += 1

    total_cnt = len(rows)
    touch_cnt = len(touch_rows)
    untouch_cnt = len(untouch_rows)

    touch_pct = (touch_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0
    untouch_pct = (untouch_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0

    print(f"\n==========================================")
    print(f"    TOUCH SPLIT DATASET ANALYTICS REPORT")
    print(f"==========================================")
    print(f"  Total Input Rows       : {total_cnt}")
    print(f"  Touch Rows (touch=1)   : {touch_cnt} ({touch_pct:.2f}%)")
    print(f"  Untouch Rows (touch=0) : {untouch_cnt} ({untouch_pct:.2f}%)")
    print(f"  Touch Finger Breakdown :")
    for fg in FINGERS:
        cnt = touch_finger_counts[fg]
        pct = (cnt / touch_cnt * 100.0) if touch_cnt > 0 else 0.0
        print(f"    - {fg.capitalize():<6} finger touch : {cnt} ({pct:.2f}%)")
    print(f"  Untouch Finger Breakdown:")
    for fg in FINGERS:
        cnt = untouch_finger_counts[fg]
        pct = (cnt / untouch_cnt * 100.0) if untouch_cnt > 0 else 0.0
        print(f"    - {fg.capitalize():<6} finger untouch: {cnt} ({pct:.2f}%)")
    print(f"==========================================\n")

    print(f"[3/3] Saving split datasets:")
    print(f"      - Touch dataset  : {touch_out_path}")
    print(f"      - Untouch dataset: {untouch_out_path}")

    # Write Touch Dataset
    os.makedirs(os.path.dirname(os.path.abspath(touch_out_path)), exist_ok=True)
    if os.path.exists(touch_out_path):
        print(f"[Info] Overwriting existing file: {touch_out_path}")
        try:
            os.remove(touch_out_path)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{touch_out_path}': {e}")

    with open(touch_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(touch_rows)

    # Write Untouch Dataset
    os.makedirs(os.path.dirname(os.path.abspath(untouch_out_path)), exist_ok=True)
    if os.path.exists(untouch_out_path):
        print(f"[Info] Overwriting existing file: {untouch_out_path}")
        try:
            os.remove(untouch_out_path)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{untouch_out_path}': {e}")

    with open(untouch_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(untouch_rows)

    print(f"[Success] Touch split datasets saved to:\n  - {touch_out_path}\n  - {untouch_out_path}")

    # Save summary JSON for pipeline audit
    try:
        from summary_utils import save_step_summary
        save_step_summary("step_11_split_touch.json", {
            "step": 11,
            "name": "split_touch",
            "total_input_rows": total_cnt,
            "touch_records": len(touch_rows),
            "untouch_records": len(untouch_rows),
            "touch_pct": round(touch_pct, 2)
        })
    except Exception as e:
        pass

    return touch_out_path, untouch_out_path


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
        description="Splits per-finger sequence window dataset CSV rows into touch and untouch CSV files"
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
        help="Input per-finger dataset CSV file path(s), glob pattern(s), or input directory"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory path"
    )
    parser.add_argument(
        "--touch-out",
        default=None,
        help="Explicit output CSV file path for touch dataset"
    )
    parser.add_argument(
        "--untouch-out",
        default=None,
        help="Explicit output CSV file path for untouch dataset"
    )

    args = parser.parse_args()

    input_patterns = []
    output_dir = args.output

    if args.input:
        input_patterns = args.input
    elif args.pos_args:
        if len(args.pos_args) >= 2 and not output_dir:
            output_dir = args.pos_args[-1]
            input_patterns = args.pos_args[:-1]
        else:
            input_patterns = args.pos_args

    if not input_patterns:
        parser.print_help()
        sys.exit(1)

    input_files = collect_csv_files(input_patterns)
    if not input_files:
        print("[Error] No valid CSV files found for touch splitting. Exiting.")
        sys.exit(1)

    print(f"Found {len(input_files)} CSV file(s) to process:")
    for f in input_files:
        print(f"  - {f}")
    print()

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Splitting touch dataset for: {input_file}")
        try:
            split_touch_csv(
                input_csv=input_file,
                output_dir=output_dir,
                touch_out_path=args.touch_out,
                untouch_out_path=args.untouch_out,
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not split touch dataset for '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Touch Splitting Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
