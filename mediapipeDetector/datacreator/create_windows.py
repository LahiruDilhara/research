# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/create_windows.py

Combines matching pairs of landmark CSVs (*.filtered_landmarks.*.csv, *.normalize_landmarks.*.csv,
or *.raw_landmarks.*.csv) and window annotation CSVs (*.window_annotations.*.csv) into sequence window CSV datasets.

For each 5-frame window in the annotation CSV, extracts all 21 landmark (x, y, z, visibility, presence) coordinates
and hand confidence score for all 5 frames (k = 1..5) along with window metadata and touch/environment labels.

Outputs window dataset CSV files to a specified directory (or beside source files), renaming '.window_annotations.'
(or landmark identifier) to '.window_dataset.'. Overwrites target file if it already exists.
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

ANNOTATION_METADATA_KEYS = [
    "video_file", "video_hash", "duration_ms", "start_ms", "end_ms",
    "start_frame", "end_frame", "window_idx", "window_size", "window_overlap"
]

ANNOTATION_LABEL_KEYS = [
    "right_hand", "hand_move", "hand_closer", "hovering", "daylight",
    "hand_visible", "out_of_sync", "thumb_touch", "index_touch",
    "middle_touch", "ring_touch", "pinky_touch", "any_touch"
]


def build_window_csv_headers() -> list[str]:
    """Builds CSV headers for sequence window dataset (metadata + 5-frame hand_score & 21 landmark x,y,z,vis,pres + labels)."""
    headers = list(ANNOTATION_METADATA_KEYS)

    # 5 Frame steps (k = 1..5)
    for k in range(1, 6):
        headers.append(f"hand_score{k}")
        for lm_name in ALL_21_LANDMARK_NAMES:
            headers.append(f"{lm_name}{k}_x")
            headers.append(f"{lm_name}{k}_y")
            headers.append(f"{lm_name}{k}_z")
            headers.append(f"{lm_name}{k}_visibility")
            headers.append(f"{lm_name}{k}_presence")

    # Touch and environment annotation flags
    headers.extend(ANNOTATION_LABEL_KEYS)
    return headers


def parse_filename_key(filename: str) -> tuple[str, str]:
    """
    Extracts (prefix, hash) from file name.
    Example: 'record1_2.filtered_landmarks.1a5008d3e2e358a6.csv' -> ('record1_2', '1a5008d3e2e358a6')
    """
    base = os.path.basename(filename)
    if base.endswith(".csv"):
        base = base[:-4]

    parts = base.split(".")
    if len(parts) >= 3:
        prefix = parts[0]
        file_hash = parts[-1]
        return prefix, file_hash
    elif len(parts) == 2:
        return parts[0], parts[1]
    return base, ""


def find_csv_pairs(input_patterns: list[str]) -> list[tuple[str, str]]:
    """
    Finds pairs of (landmarks_csv, annotations_csv) from given input patterns or directories.
    Prioritizes filtered_landmarks > normalize_landmarks > raw_landmarks.
    """
    all_files = []
    for pattern in input_patterns:
        search_pattern = os.path.join(pattern, "*.csv") if os.path.isdir(pattern) else pattern
        matches = glob.glob(search_pattern, recursive=True)
        for m in matches:
            if os.path.isfile(m) and m not in all_files:
                all_files.append(m)

    ann_files = [f for f in all_files if ".window_annotations." in os.path.basename(f)]
    lm_files = [f for f in all_files if any(k in os.path.basename(f) for k in [".filtered_landmarks.", ".normalize_landmarks.", ".raw_landmarks."])]

    lm_dict: dict[tuple[str, str], list[str]] = {}
    for lmf in lm_files:
        key = parse_filename_key(lmf)
        lm_dict.setdefault(key, []).append(lmf)

    pairs = []
    for af in sorted(ann_files):
        key = parse_filename_key(af)
        matched_lms = lm_dict.get(key, [])

        if not matched_lms:
            prefix, hash_val = key
            for (p, h), flist in lm_dict.items():
                if p == prefix:
                    matched_lms = flist
                    break

        if matched_lms:
            best_lm = matched_lms[0]
            for candidate in matched_lms:
                if ".filtered_landmarks." in candidate:
                    best_lm = candidate
                    break
                elif ".normalize_landmarks." in candidate and ".filtered_landmarks." not in best_lm:
                    best_lm = candidate
            pairs.append((best_lm, af))
        else:
            print(f"[Warning] Could not find matching landmark CSV for annotation file: '{af}'")

    return pairs


def get_default_output_path(ann_csv_path: str) -> str:
    """Derives default output path <video_name>.window_dataset.<hash>.csv in the same location."""
    dir_name = os.path.dirname(os.path.abspath(ann_csv_path))
    base_name = os.path.basename(ann_csv_path)

    if ".window_annotations." in base_name:
        out_name = base_name.replace(".window_annotations.", ".window_dataset.")
    else:
        name_no_ext = os.path.splitext(base_name)[0]
        out_name = f"{name_no_ext}.window_dataset.csv"

    return os.path.join(dir_name, out_name)


def create_windowed_dataset_csv(
    lm_csv_path: str,
    ann_csv_path: str,
    output_csv_path: str = None
) -> str:
    """
    Combines landmark frame CSV and window annotations CSV into a window dataset CSV.
    """
    output_csv_path = output_csv_path or get_default_output_path(ann_csv_path)

    print(f"[1/3] Reading landmarks from: {lm_csv_path}")
    print(f"      Reading annotations from: {ann_csv_path}")

    with open(lm_csv_path, "r", encoding="utf-8") as f:
        lm_reader = csv.DictReader(f)
        lm_rows = list(lm_reader)

    with open(ann_csv_path, "r", encoding="utf-8") as f:
        ann_reader = csv.DictReader(f)
        ann_rows = list(ann_reader)

    if not ann_rows:
        raise ValueError(f"Window annotations CSV '{ann_csv_path}' is empty!")

    lm_by_frame: dict[int, dict] = {}
    for r in lm_rows:
        try:
            f_idx = int(r.get("frame_idx", -1))
            if f_idx >= 0:
                lm_by_frame[f_idx] = r
        except ValueError:
            continue

    headers = build_window_csv_headers()
    output_rows = []

    print(f"[2/3] Processing {len(ann_rows)} window annotation sequence records...")

    for ann_row in ann_rows:
        try:
            start_frame = int(ann_row.get("start_frame", 0))
            end_frame = int(ann_row.get("end_frame", start_frame + 4))
        except ValueError:
            start_frame = 0
            end_frame = 4

        start_ms = ann_row.get("start_ms", "0")
        end_ms = ann_row.get("end_ms", "0")

        try:
            duration_ms = int(end_ms) - int(start_ms)
        except ValueError:
            duration_ms = 0

        out_row = {
            "video_file": ann_row.get("video_file", ""),
            "video_hash": ann_row.get("video_hash", ""),
            "duration_ms": str(duration_ms),
            "start_ms": str(start_ms),
            "end_ms": str(end_ms),
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
            "window_idx": ann_row.get("window_idx", "0"),
            "window_size": ann_row.get("window_size", "5"),
            "window_overlap": ann_row.get("window_overlap", "2"),
        }

        # Populate landmark features & hand_score for all 5 frames in the window (k = 1..5)
        expected_window_size = int(ann_row.get("window_size", 5))
        for k in range(1, expected_window_size + 1):
            target_frame_idx = start_frame + (k - 1)
            frame_lm = lm_by_frame.get(target_frame_idx, {})

            out_row[f"hand_score{k}"] = frame_lm.get("hand_score", "0.0")

            for lm_name in ALL_21_LANDMARK_NAMES:
                x_val = frame_lm.get(f"{lm_name}_x", "0.0")
                y_val = frame_lm.get(f"{lm_name}_y", "0.0")
                z_val = frame_lm.get(f"{lm_name}_z", "0.0")
                vis_val = frame_lm.get(f"{lm_name}_visibility", "0.0")
                pres_val = frame_lm.get(f"{lm_name}_presence", "0.0")

                out_row[f"{lm_name}{k}_x"] = x_val
                out_row[f"{lm_name}{k}_y"] = y_val
                out_row[f"{lm_name}{k}_z"] = z_val
                out_row[f"{lm_name}{k}_visibility"] = vis_val
                out_row[f"{lm_name}{k}_presence"] = pres_val

        # Populate touch & environment annotation flags
        for key in ANNOTATION_LABEL_KEYS:
            out_row[key] = ann_row.get(key, "0")

        output_rows.append(out_row)

    print(f"[3/3] Saving window dataset CSV to: {output_csv_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    if os.path.exists(output_csv_path):
        try:
            os.remove(output_csv_path)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv_path}': {e}")

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[Success] Saved window dataset CSV to: {output_csv_path}")
    return output_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Combines landmark CSVs and window annotation CSVs into sequence window datasets"
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
    output_dir = None

    if args.output:
        output_dir = args.output
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
            output_dir = combined[-1]
            input_patterns = combined[:-1]
        elif len(combined) == 1:
            input_patterns = combined
            output_dir = None
        else:
            parser.print_help()
            sys.exit(1)

    pairs = find_csv_pairs(input_patterns)
    if not pairs:
        print("[Error] No matching landmark & annotation CSV pairs found. Exiting.")
        sys.exit(1)

    print(f"Found {len(pairs)} landmark & annotation CSV pair(s) to process:")
    for lm_f, ann_f in pairs:
        print(f"  - Landmark:   {lm_f}")
        print(f"    Annotation: {ann_f}")
    if output_dir:
        print(f"Output directory override: {output_dir}")
    print()

    success_count = 0
    fail_count = 0

    for idx, (lm_file, ann_file) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] Creating sequence window dataset...")
        try:
            out_file_path = None
            if output_dir:
                if os.path.isdir(output_dir) or output_dir.endswith(os.sep) or output_dir.endswith("/"):
                    os.makedirs(output_dir, exist_ok=True)
                    out_name = os.path.basename(get_default_output_path(ann_file))
                    out_file_path = os.path.join(output_dir, out_name)
                elif len(pairs) == 1:
                    out_file_path = output_dir
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    out_name = os.path.basename(get_default_output_path(ann_file))
                    out_file_path = os.path.join(output_dir, out_name)
            else:
                out_file_path = get_default_output_path(ann_file)

            create_windowed_dataset_csv(
                lm_csv_path=lm_file,
                ann_csv_path=ann_file,
                output_csv_path=out_file_path
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process pair ('{lm_file}', '{ann_file}'): {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Window Dataset Creation Finished:")
    print(f"  Success: {success_count}/{len(pairs)}")
    print(f"  Failed : {fail_count}/{len(pairs)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
