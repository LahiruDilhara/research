# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/merge_windows.py

Scans an input directory or list of sequence window dataset CSV files (*.window_dataset.*.csv),
validates column headers, merges all rows into a single unified CSV file, and writes it to the output location.
Overwrites destination file if it already exists.
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

EXPECTED_METADATA_KEYS = [
    "video_file", "video_hash", "duration_ms", "start_ms", "end_ms",
    "start_frame", "end_frame", "window_idx", "window_size", "window_overlap"
]

EXPECTED_LABEL_KEYS = [
    "right_hand", "hand_move", "hand_closer", "hovering", "daylight",
    "hand_visible", "out_of_sync", "thumb_touch", "index_touch",
    "middle_touch", "ring_touch", "pinky_touch", "any_touch"
]


def validate_window_headers(headers: list[str], csv_path: str):
    """Ensures input window CSV contains essential metadata and label columns."""
    missing = []
    for col in EXPECTED_METADATA_KEYS:
        if col not in headers:
            missing.append(col)
    for col in EXPECTED_LABEL_KEYS:
        if col not in headers:
            missing.append(col)

    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required window dataset column(s): {', '.join(missing)}"
        )


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


def merge_window_csv_files(input_files: list[str], output_csv_path: str) -> str:
    """
    Reads all input window CSV files, validates columns, merges rows into a single CSV,
    and writes to output_csv_path.
    """
    if not input_files:
        raise ValueError("No input CSV files provided for merging!")

    if os.path.isdir(output_csv_path) or output_csv_path.endswith(os.sep) or output_csv_path.endswith("/"):
        os.makedirs(output_csv_path, exist_ok=True)
        output_csv_path = os.path.join(output_csv_path, "all_windowed_dataset.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    print(f"[1/3] Reading and validating {len(input_files)} window dataset CSV file(s)...")

    combined_rows = []
    canonical_headers = None
    total_files_merged = 0

    for idx, csv_path in enumerate(input_files, start=1):
        print(f"  [{idx}/{len(input_files)}] Reading: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        if not rows:
            print(f"  [Warning] CSV '{csv_path}' is empty, skipping.")
            continue

        validate_window_headers(headers, csv_path)

        if canonical_headers is None:
            canonical_headers = headers
        elif headers != canonical_headers:
            print(f"  [Info] Header mismatch in '{csv_path}', aligning fields with canonical schema.")

        combined_rows.extend(rows)
        total_files_merged += 1

    if not combined_rows or canonical_headers is None:
        raise ValueError("No valid rows found to merge across input CSV files!")

    print(f"[2/3] Total sequence window records collected: {len(combined_rows)} from {total_files_merged} file(s).")
    print(f"[3/3] Saving merged dataset to: {output_csv_path}")

    if os.path.exists(output_csv_path):
        print(f"[Info] Overwriting existing file: {output_csv_path}")
        try:
            os.remove(output_csv_path)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv_path}': {e}")

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=canonical_headers)
        writer.writeheader()
        writer.writerows(combined_rows)

    # ── Compute Analytics Report ──────────────────────────────────────────────
    total_cnt = len(combined_rows)
    touch_cnt = 0
    non_touch_cnt = 0
    right_cnt = 0
    left_cnt = 0
    finger_touch_cnts = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}

    def _parse_b(val):
        return str(val).strip().lower() in ("1", "true", "t", "yes", "y")

    for r in combined_rows:
        is_touch = _parse_b(r.get("any_touch", "0"))
        if not is_touch:
            for fg in ["thumb", "index", "middle", "ring", "pinky"]:
                if _parse_b(r.get(f"{fg}_touch", "0")):
                    is_touch = True
                    break

        if is_touch:
            touch_cnt += 1
        else:
            non_touch_cnt += 1

        is_right = _parse_b(r.get("right_hand", r.get("rightHand", "0")))
        if is_right:
            right_cnt += 1
        else:
            left_cnt += 1

        for fg in ["thumb", "index", "middle", "ring", "pinky"]:
            if _parse_b(r.get(f"{fg}_touch", "0")):
                finger_touch_cnts[fg] += 1

    touch_pct = (touch_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0
    non_touch_pct = (non_touch_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0

    print(f"\n==========================================")
    print(f"   MERGED WINDOW DATASET ANALYTICS REPORT")
    print(f"==========================================")
    print(f"  Files Merged           : {total_files_merged}")
    print(f"  Total Window Sequence  : {total_cnt} windows")
    print(f"  Touch Windows (any_touch): {touch_cnt} ({touch_pct:.2f}%)")
    print(f"  Non-Touch Windows      : {non_touch_cnt} ({non_touch_pct:.2f}%)")
    print(f"  Handedness Breakdown   : Right: {right_cnt} | Left: {left_cnt}")
    print(f"  Per-Finger Touches     :")
    for fg, cnt in finger_touch_cnts.items():
        fg_pct = (cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0
        print(f"    - {fg.capitalize():<6} touch       : {cnt} ({fg_pct:.2f}%)")
    print(f"==========================================\n")

    print(f"[Success] Merged {len(combined_rows)} window records into: {output_csv_path}")
    return output_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Merges multiple window dataset CSV files into a single unified dataset CSV"
    )
    parser.add_argument(
        "pos_args",
        nargs="*",
        help="Input CSV file(s), glob pattern(s), or output directory/file"
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        default=None,
        help="Input window CSV file path(s), glob pattern(s), or input directory (e.g. 'dataprocessing/5_windowed_dataset/')"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output merged CSV file path or directory (default: 'dataprocessing/6_merged_windowed_dataset/all_windowed_dataset.csv')"
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
            output_target = "./dataprocessing/6_merged_windowed_dataset/all_windowed_dataset.csv"
        else:
            parser.print_help()
            sys.exit(1)

    input_files = collect_csv_files(input_patterns)
    if not input_files:
        print("[Error] No valid CSV files found for merging. Exiting.")
        sys.exit(1)

    output_target = output_target or "./dataprocessing/6_merged_windowed_dataset/all_windowed_dataset.csv"

    try:
        merge_window_csv_files(input_files, output_target)
    except Exception as e:
        print(f"[Error] Window dataset merging failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
