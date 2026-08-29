# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/filter_window_quality.py

Comprehensive sequence window quality filtering script.
Filters 5-frame sequence window dataset CSV rows based on multiple configurable quality flags:

1. Hand Confidence & Tracking Stability:
   - --min-avg-score FLOAT             : Removes window if average hand confidence score across 5 frames < FLOAT.
   - --min-frame-score FLOAT           : Removes window if ANY frame in 5-frame window has hand confidence score < FLOAT.
   - --max-score-drop FLOAT            : Removes window if confidence score fluctuation max(score) - min(score) > FLOAT.

2. Landmark Visibility & Presence Quality:
   - --min-avg-visibility FLOAT        : Removes window if average landmark visibility across 5 frames < FLOAT.
   - --min-avg-presence FLOAT          : Removes window if average landmark presence across 5 frames < FLOAT.
   - --min-fingertip-visibility FLOAT  : Removes window if ANY fingertip landmark (thumb/index/middle/ring/pinky tip) visibility < FLOAT.
   - --min-wrist-visibility FLOAT      : Removes window if wrist landmark visibility < FLOAT.

3. Kinematic Speed & Acceleration Anomalies:
   - --max-speed-2d FLOAT              : Removes window if any joint 2D speed exceeds FLOAT (detects 2D glitch jumps).
   - --max-speed-3d FLOAT              : Removes window if any joint 3D speed exceeds FLOAT (detects 3D glitch jumps).
   - --max-acceleration FLOAT          : Removes window if joint acceleration step-to-step speed jump > FLOAT.

4. 3D Depth Kinematics:
   - --max-z-range FLOAT               : Removes window if z-depth range max(z) - min(z) > FLOAT across 5 frames.
   - --max-z-variance FLOAT            : Removes window if z-depth variance across 5 frames > FLOAT.

5. Environment & Context Flags:
   - --remove-daylight                 : Removes daylight annotated windows (daylight=1).
   - --remove-hovering                 : Removes hovering annotated windows (hovering=1).
   - --remove-hand-move                : Removes hand moving annotated windows (hand_move=1).
   - --remove-hand-closer              : Removes hand closer annotated windows (hand_closer=1).

Accepts single CSV files, glob patterns, or directory inputs. Overwrites target output file if it already exists.
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

FINGERTIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


def parse_bool_flag(val: str) -> bool:
    """Parses boolean value from CSV cell string or int."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def extract_frame_hand_scores(row: dict) -> list[float]:
    """Extracts hand confidence scores across 5 frames (hand_score1..5) from row."""
    scores = []
    for k in range(1, 6):
        col = f"hand_score{k}"
        if col in row:
            try:
                scores.append(float(row[col]))
            except ValueError:
                scores.append(0.0)
    if not scores and "hand_score" in row:
        try:
            scores.append(float(row["hand_score"]))
        except ValueError:
            scores.append(0.0)
    # If hand_score is unpopulated in legacy datasets (all 0.0s), treat as missing to prevent dropping all rows
    if scores and max(scores) == 0.0:
        return []
    return scores


def extract_visibility_presence_values(row: dict) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Extracts landmark visibility and presence values across 5 frames.
    Returns (all_vis, all_pres, fingertip_vis, wrist_vis).
    """
    all_vis = []
    all_pres = []
    fingertip_vis = []
    wrist_vis = []

    for k in range(1, 6):
        for lm_name in ALL_21_LANDMARK_NAMES:
            v_col = f"{lm_name}{k}_visibility"
            p_col = f"{lm_name}{k}_presence"

            if v_col in row:
                try:
                    v = float(row[v_col])
                    all_vis.append(v)
                    if lm_name in FINGERTIP_NAMES:
                        fingertip_vis.append(v)
                    elif lm_name == "wrist":
                        wrist_vis.append(v)
                except ValueError:
                    pass

            if p_col in row:
                try:
                    all_pres.append(float(row[p_col]))
                except ValueError:
                    pass

    return all_vis, all_pres, fingertip_vis, wrist_vis


def extract_joint_speeds(row: dict) -> tuple[list[float], list[float]]:
    """
    Extracts 2D and 3D joint speeds across 4 velocity steps.
    Returns (speeds_2d, speeds_3d).
    """
    speeds_2d = []
    speeds_3d = []
    for v in range(1, 5):
        for lm_name in ALL_21_LANDMARK_NAMES:
            col2d = f"{lm_name}{v}_speed_2d"
            col3d = f"{lm_name}{v}_speed_3d"
            if col2d in row:
                try:
                    speeds_2d.append(float(row[col2d]))
                except ValueError:
                    pass
            if col3d in row:
                try:
                    speeds_3d.append(float(row[col3d]))
                except ValueError:
                    pass
    return speeds_2d, speeds_3d


def extract_z_depths(row: dict) -> list[float]:
    """Extracts z-depth values across all 5 frames for all 21 joints."""
    z_vals = []
    for k in range(1, 6):
        for lm_name in ALL_21_LANDMARK_NAMES:
            col = f"{lm_name}{k}_z"
            if col in row:
                try:
                    z_vals.append(float(row[col]))
                except ValueError:
                    pass
    return z_vals


def filter_window_quality_csv(
    input_csv: str,
    output_csv: str = None,
    min_avg_score: float | None = None,
    min_frame_score: float | None = None,
    max_score_drop: float | None = None,
    min_avg_visibility: float | None = None,
    min_avg_presence: float | None = None,
    min_fingertip_visibility: float | None = None,
    min_wrist_visibility: float | None = None,
    max_speed_2d: float | None = None,
    max_speed_3d: float | None = None,
    max_acceleration: float | None = None,
    max_z_range: float | None = None,
    max_z_variance: float | None = None,
    remove_daylight: bool = False,
    remove_hovering: bool = False,
    remove_hand_move: bool = False,
    remove_hand_closer: bool = False,
) -> str:
    """
    Reads dataset CSV, applies all enabled window quality filters, and writes cleaned CSV.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    dir_name = os.path.dirname(os.path.abspath(input_csv))
    base_name = os.path.basename(input_csv)

    if not output_csv:
        if ".csv" in base_name:
            out_name = base_name.replace(".csv", ".quality_filtered.csv")
        else:
            out_name = f"{base_name}.quality_filtered.csv"
        output_csv = os.path.join(dir_name, out_name)

    print(f"[1/3] Reading dataset for window quality filtering from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    total_rows = len(rows)
    retained_rows = 0

    drop_min_avg_score_cnt = 0
    drop_min_frame_score_cnt = 0
    drop_max_score_drop_cnt = 0
    drop_min_avg_vis_cnt = 0
    drop_min_avg_pres_cnt = 0
    drop_min_fingertip_vis_cnt = 0
    drop_min_wrist_vis_cnt = 0
    drop_max_speed_2d_cnt = 0
    drop_max_speed_3d_cnt = 0
    drop_max_accel_cnt = 0
    drop_max_z_range_cnt = 0
    drop_max_z_var_cnt = 0
    drop_daylight_cnt = 0
    drop_hovering_cnt = 0
    drop_hand_move_cnt = 0
    drop_hand_closer_cnt = 0

    init_touch_cnt = 0
    init_untouch_cnt = 0
    final_touch_cnt = 0
    final_untouch_cnt = 0

    print(f"[2/3] Applying Window Quality Filter Flags:")
    print(f"      - Min Average Hand Score Cutoff    : {min_avg_score if min_avg_score is not None else 'Disabled'}")
    print(f"      - Min Per-Frame Hand Score Cutoff  : {min_frame_score if min_frame_score is not None else 'Disabled'}")
    print(f"      - Max Hand Score Fluctuation/Drop  : {max_score_drop if max_score_drop is not None else 'Disabled'}")
    print(f"      - Min Average Landmark Visibility  : {min_avg_visibility if min_avg_visibility is not None else 'Disabled'}")
    print(f"      - Min Average Landmark Presence    : {min_avg_presence if min_avg_presence is not None else 'Disabled'}")
    print(f"      - Min Fingertip Visibility Cutoff  : {min_fingertip_visibility if min_fingertip_visibility is not None else 'Disabled'}")
    print(f"      - Min Wrist Visibility Cutoff      : {min_wrist_visibility if min_wrist_visibility is not None else 'Disabled'}")
    print(f"      - Max 2D Speed Cutoff              : {max_speed_2d if max_speed_2d is not None else 'Disabled'}")
    print(f"      - Max 3D Speed Cutoff              : {max_speed_3d if max_speed_3d is not None else 'Disabled'}")
    print(f"      - Max Acceleration Jump Cutoff     : {max_acceleration if max_acceleration is not None else 'Disabled'}")
    print(f"      - Max Z-Depth Range Cutoff         : {max_z_range if max_z_range is not None else 'Disabled'}")
    print(f"      - Max Z-Depth Variance Cutoff      : {max_z_variance if max_z_variance is not None else 'Disabled'}")
    print(f"      - Remove Daylight Windows          : {remove_daylight}")
    print(f"      - Remove Hovering Windows          : {remove_hovering}")
    print(f"      - Remove Hand Moving Windows       : {remove_hand_move}")
    print(f"      - Remove Hand Approaching Windows  : {remove_hand_closer}")
    print(f"      Initial input window records       : {total_rows}")

    filtered_rows = []

    for row in rows:
        touch_val = row.get("any_touch", row.get("touch_finger", "0"))
        is_touch = parse_bool_flag(touch_val)
        if is_touch:
            init_touch_cnt += 1
        else:
            init_untouch_cnt += 1

        keep = True
        scores = extract_frame_hand_scores(row)
        all_vis, all_pres, fingertip_vis, wrist_vis = extract_visibility_presence_values(row)
        speeds_2d, speeds_3d = extract_joint_speeds(row)
        z_vals = extract_z_depths(row)

        # 1. Hand Confidence Filters
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            score_diff = max_score - min_score

            if min_avg_score is not None and avg_score < min_avg_score:
                keep = False
                drop_min_avg_score_cnt += 1

            if keep and min_frame_score is not None and min_score < min_frame_score:
                keep = False
                drop_min_frame_score_cnt += 1

            if keep and max_score_drop is not None and score_diff > max_score_drop:
                keep = False
                drop_max_score_drop_cnt += 1

        # 2. Visibility & Presence Filters
        if keep and min_avg_visibility is not None and all_vis:
            avg_vis = sum(all_vis) / len(all_vis)
            if avg_vis < min_avg_visibility:
                keep = False
                drop_min_avg_vis_cnt += 1

        if keep and min_avg_presence is not None and all_pres:
            avg_pres = sum(all_pres) / len(all_pres)
            if avg_pres < min_avg_presence:
                keep = False
                drop_min_avg_pres_cnt += 1

        if keep and min_fingertip_visibility is not None and fingertip_vis:
            if any(v < min_fingertip_visibility for v in fingertip_vis):
                keep = False
                drop_min_fingertip_vis_cnt += 1

        if keep and min_wrist_visibility is not None and wrist_vis:
            if any(v < min_wrist_visibility for v in wrist_vis):
                keep = False
                drop_min_wrist_vis_cnt += 1

        # 3. Kinematic Speed & Acceleration Anomaly Filters
        if keep and max_speed_2d is not None and speeds_2d:
            if any(s > max_speed_2d for s in speeds_2d):
                keep = False
                drop_max_speed_2d_cnt += 1

        if keep and max_speed_3d is not None and speeds_3d:
            if any(s > max_speed_3d for s in speeds_3d):
                keep = False
                drop_max_speed_3d_cnt += 1

        if keep and max_acceleration is not None:
            target_speeds = speeds_3d if speeds_3d else speeds_2d
            if len(target_speeds) > 21:
                accel_exceeded = False
                for step in range(len(target_speeds) - 21):
                    s_diff = abs(target_speeds[step + 21] - target_speeds[step])
                    if s_diff > max_acceleration:
                        accel_exceeded = True
                        break
                if accel_exceeded:
                    keep = False
                    drop_max_accel_cnt += 1

        # 4. Z-Depth Variance & Range Filters
        if keep and z_vals:
            if max_z_range is not None:
                z_range = max(z_vals) - min(z_vals)
                if z_range > max_z_range:
                    keep = False
                    drop_max_z_range_cnt += 1

            if keep and max_z_variance is not None:
                mean_z = sum(z_vals) / len(z_vals)
                var_z = sum((zv - mean_z) ** 2 for zv in z_vals) / len(z_vals)
                if var_z > max_z_variance:
                    keep = False
                    drop_max_z_var_cnt += 1

        # 5. Environment & Context Flags
        if keep and remove_daylight and parse_bool_flag(row.get("daylight", "0")):
            keep = False
            drop_daylight_cnt += 1

        if keep and remove_hovering and parse_bool_flag(row.get("hovering", "0")):
            keep = False
            drop_hovering_cnt += 1

        if keep and remove_hand_move and parse_bool_flag(row.get("hand_move", "0")):
            keep = False
            drop_hand_move_cnt += 1

        if keep and remove_hand_closer and parse_bool_flag(row.get("hand_closer", "0")):
            keep = False
            drop_hand_closer_cnt += 1

        if keep:
            filtered_rows.append(row)
            retained_rows += 1
            if is_touch:
                final_touch_cnt += 1
            else:
                final_untouch_cnt += 1

    dropped_rows = total_rows - retained_rows
    retained_pct = (retained_rows / total_rows * 100.0) if total_rows > 0 else 0.0
    dropped_pct = (dropped_rows / total_rows * 100.0) if total_rows > 0 else 0.0

    print(f"\n==========================================")
    print(f"   WINDOW QUALITY FILTERING ANALYTICS REPORT")
    print(f"==========================================")
    print(f"  Initial Window Records             : {total_rows}")
    print(f"  Retained Records                   : {retained_rows} ({retained_pct:.2f}%)")
    print(f"  Dropped Records                    : {dropped_rows} ({dropped_pct:.2f}%)")
    print(f"  Filter Removal Breakdown:")
    if min_avg_score is not None:
        print(f"    - Low Avg Hand Score (<{min_avg_score})  : {drop_min_avg_score_cnt} records")
    if min_frame_score is not None:
        print(f"    - Low Frame Hand Score (<{min_frame_score}): {drop_min_frame_score_cnt} records")
    if max_score_drop is not None:
        print(f"    - High Score Drop (>{max_score_drop})   : {drop_max_score_drop_cnt} records")
    if min_avg_visibility is not None:
        print(f"    - Low Avg Visibility (<{min_avg_visibility}): {drop_min_avg_vis_cnt} records")
    if min_avg_presence is not None:
        print(f"    - Low Avg Presence (<{min_avg_presence})  : {drop_min_avg_pres_cnt} records")
    if min_fingertip_visibility is not None:
        print(f"    - Low Fingertip Vis (<{min_fingertip_visibility}): {drop_min_fingertip_vis_cnt} records")
    if min_wrist_visibility is not None:
        print(f"    - Low Wrist Vis (<{min_wrist_visibility}): {drop_min_wrist_vis_cnt} records")
    if max_speed_2d is not None:
        print(f"    - Max 2D Speed Exceeded (>{max_speed_2d}): {drop_max_speed_2d_cnt} records")
    if max_speed_3d is not None:
        print(f"    - Max 3D Speed Exceeded (>{max_speed_3d}): {drop_max_speed_3d_cnt} records")
    if max_acceleration is not None:
        print(f"    - Max Accel Jump Exceeded (>{max_acceleration}): {drop_max_accel_cnt} records")
    if max_z_range is not None:
        print(f"    - Max Z Range Exceeded (>{max_z_range}): {drop_max_z_range_cnt} records")
    if max_z_variance is not None:
        print(f"    - Max Z Variance Exceeded (>{max_z_variance}): {drop_max_z_var_cnt} records")
    if remove_daylight:
        print(f"    - Daylight Annotated Windows     : {drop_daylight_cnt} records")
    if remove_hovering:
        print(f"    - Hovering Annotated Windows     : {drop_hovering_cnt} records")
    if remove_hand_move:
        print(f"    - Hand Move Annotated Windows    : {drop_hand_move_cnt} records")
    if remove_hand_closer:
        print(f"    - Hand Closer Annotated Windows  : {drop_hand_closer_cnt} records")
    print(f"  Class Distribution (Before -> After):")
    print(f"    - Touch Records   : {init_touch_cnt} -> {final_touch_cnt}")
    print(f"    - Untouch Records : {init_untouch_cnt} -> {final_untouch_cnt}")
    print(f"==========================================\n")

    print(f"[3/3] Saving quality filtered dataset to: {output_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv}': {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[Success] Saved quality filtered dataset to: {output_csv}")

    # Save summary JSON for pipeline audit
    try:
        from summary_utils import save_step_summary
        save_step_summary("step_9_filter_window_quality.json", {
            "step": 9,
            "name": "filter_window_quality",
            "total_input_windows": total_rows,
            "retained_windows": retained_rows,
            "dropped_windows": dropped_rows,
            "retained_pct": round(retained_pct, 2),
            "drop_min_avg_score_cnt": drop_min_avg_score_cnt,
            "drop_min_frame_score_cnt": drop_min_frame_score_cnt,
            "drop_max_score_drop_cnt": drop_max_score_drop_cnt,
            "drop_max_speed_2d_cnt": drop_max_speed_2d_cnt,
            "drop_max_speed_3d_cnt": drop_max_speed_3d_cnt,
            "touch_before": init_touch_cnt,
            "touch_after": final_touch_cnt,
            "untouch_before": init_untouch_cnt,
            "untouch_after": final_untouch_cnt
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
        description="Filters sequence window dataset CSV rows based on multi-parameter window quality flags"
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
        help="Output directory or file path"
    )
    parser.add_argument(
        "--min-avg-score",
        type=float,
        default=None,
        help="Removes window if average hand confidence score across 5 frames < FLOAT (e.g. 0.60)"
    )
    parser.add_argument(
        "--min-frame-score",
        type=float,
        default=None,
        help="Removes window if ANY frame in 5-frame window has hand confidence score < FLOAT (e.g. 0.40)"
    )
    parser.add_argument(
        "--max-score-drop",
        type=float,
        default=None,
        help="Removes window if score fluctuation max(hand_score) - min(hand_score) > FLOAT (e.g. 0.30)"
    )
    parser.add_argument(
        "--min-avg-visibility",
        type=float,
        default=None,
        help="Removes window if average landmark visibility across 5 frames < FLOAT"
    )
    parser.add_argument(
        "--min-avg-presence",
        type=float,
        default=None,
        help="Removes window if average landmark presence across 5 frames < FLOAT"
    )
    parser.add_argument(
        "--min-fingertip-visibility",
        type=float,
        default=None,
        help="Removes window if any fingertip landmark visibility < FLOAT"
    )
    parser.add_argument(
        "--min-wrist-visibility",
        type=float,
        default=None,
        help="Removes window if wrist landmark visibility < FLOAT"
    )
    parser.add_argument(
        "--max-speed-2d",
        type=float,
        default=None,
        help="Removes window if any joint 2D speed exceeds FLOAT (detects tracking jumps)"
    )
    parser.add_argument(
        "--max-speed-3d",
        type=float,
        default=None,
        help="Removes window if any joint 3D speed exceeds FLOAT"
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=None,
        help="Removes window if any joint speed jump exceeds FLOAT"
    )
    parser.add_argument(
        "--max-z-range",
        type=float,
        default=None,
        help="Removes window if z-depth range max(z) - min(z) exceeds FLOAT"
    )
    parser.add_argument(
        "--max-z-variance",
        type=float,
        default=None,
        help="Removes window if z-depth variance exceeds FLOAT"
    )
    parser.add_argument(
        "--remove-daylight",
        action="store_true",
        help="Removes daylight annotated windows (daylight=1)"
    )
    parser.add_argument(
        "--remove-hovering",
        action="store_true",
        help="Removes hovering annotated windows (hovering=1)"
    )
    parser.add_argument(
        "--remove-hand-move",
        action="store_true",
        help="Removes hand moving annotated windows (hand_move=1)"
    )
    parser.add_argument(
        "--remove-hand-closer",
        action="store_true",
        help="Removes hand closer annotated windows (hand_closer=1)"
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

    print(f"Found {len(input_files)} CSV file(s) to filter:")
    for f in input_files:
        print(f"  - {f}")

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Quality filtering: {input_file}")
        try:
            out_file_path = None
            if output_target:
                if os.path.isdir(output_target) or output_target.endswith(os.sep) or output_target.endswith("/"):
                    os.makedirs(output_target, exist_ok=True)
                    base_name = os.path.basename(input_file)
                    if ".csv" in base_name:
                        out_name = base_name.replace(".csv", ".quality_filtered.csv")
                    else:
                        out_name = f"{base_name}.quality_filtered.csv"
                    out_file_path = os.path.join(output_target, out_name)
                elif len(input_files) == 1:
                    out_file_path = output_target
                else:
                    os.makedirs(output_target, exist_ok=True)
                    out_file_path = os.path.join(output_target, os.path.basename(input_file))
            else:
                out_file_path = None

            filter_window_quality_csv(
                input_csv=input_file,
                output_csv=out_file_path,
                min_avg_score=args.min_avg_score,
                min_frame_score=args.min_frame_score,
                max_score_drop=args.max_score_drop,
                min_avg_visibility=args.min_avg_visibility,
                min_avg_presence=args.min_avg_presence,
                min_fingertip_visibility=args.min_fingertip_visibility,
                min_wrist_visibility=args.min_wrist_visibility,
                max_speed_2d=args.max_speed_2d,
                max_speed_3d=args.max_speed_3d,
                max_acceleration=args.max_acceleration,
                max_z_range=args.max_z_range,
                max_z_variance=args.max_z_variance,
                remove_daylight=args.remove_daylight,
                remove_hovering=args.remove_hovering,
                remove_hand_move=args.remove_hand_move,
                remove_hand_closer=args.remove_hand_closer,
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Window Quality Filtering Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
