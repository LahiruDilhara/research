# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/normalize_landmarks.py

Landmark scale normalization script for MediaPipe landmark CSVs.
Converts normalized [0, 1] 2D/3D coordinates to true pixel coordinates using video width W and height H,
calculates rigid palm scale L_hand using the Root-Mean-Square (RMS) of 8 symmetric rigid palm skeleton segments
(Wrist to Index/Middle/Ring/Pinky MCPs, contiguous MCP knuckle joints, and outer MCP width) for perfectly
balanced noise resistance without finger-side bias.

Strictly deterministic per-frame normalization (no EMA history across frames), ensuring that identical
hand poses produce identical normalized coordinates regardless of frame order or preceding video history.
Subtracts wrist position to make coordinates zero-centered at the wrist (translation-invariant).

Throws an explicit error and halts if resolution metadata (video_width / video_height) is missing or invalid.
Preserves all extra landmark columns (hand_score, z, visibility, presence) dynamically.
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

# ── Landmark Indices for Symmetric Palm Skeleton Box ─────────────────────────
WRIST_INDEX = 0
INDEX_MCP_INDEX = 5
MIDDLE_MCP_INDEX = 9
RING_MCP_INDEX = 13
PINKY_MCP_INDEX = 17

REQUIRED_METADATA_COLUMNS = [
    "video_file", "video_hash", "video_width", "video_height",
    "video_fps", "total_video_frames", "video_duration_sec",
    "frame_idx", "timestamp_ms", "hand"
]

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class HandScaleNormalizer:
    """
    Calculates rigid palm scale L_hand from 8 symmetric palm segments and normalizes 2D/3D landmarks.
    Pure per-frame function (no EMA history across frames).
    """

    def __init__(self, center_wrist: bool = True):
        self.center_wrist = center_wrist

    def normalize(
        self,
        pts_px: list[tuple[float, float, float]]
    ) -> tuple[list[tuple[float, float, float]], float]:
        """
        Given 21 coordinates [(x_px, y_px, z_px), ...]:
        1. Computes 8 symmetric rigid palm skeleton distances in 2D pixel space.
        2. Computes robust palm scale L_hand = RMS(d1..d8) = sqrt(mean(d_i^2)).
        3. Subtracts wrist position (if self.center_wrist is True) and divides x, y, z by L_hand.
        Returns (normalized_points, L_hand).
        """
        w_x, w_y, w_z = pts_px[WRIST_INDEX]
        i_x, i_y, _   = pts_px[INDEX_MCP_INDEX]
        m_x, m_y, _   = pts_px[MIDDLE_MCP_INDEX]
        r_x, r_y, _   = pts_px[RING_MCP_INDEX]
        p_x, p_y, _   = pts_px[PINKY_MCP_INDEX]

        # 8 Symmetric Palm Skeleton Segments
        d1_sq = (i_x - w_x) ** 2 + (i_y - w_y) ** 2  # Wrist -> Index
        d2_sq = (m_x - w_x) ** 2 + (m_y - w_y) ** 2  # Wrist -> Middle
        d3_sq = (r_x - w_x) ** 2 + (r_y - w_y) ** 2  # Wrist -> Ring
        d4_sq = (p_x - w_x) ** 2 + (p_y - w_y) ** 2  # Wrist -> Pinky
        d5_sq = (m_x - i_x) ** 2 + (m_y - i_y) ** 2  # Index -> Middle
        d6_sq = (r_x - m_x) ** 2 + (r_y - m_y) ** 2  # Middle -> Ring
        d7_sq = (p_x - r_x) ** 2 + (p_y - r_y) ** 2  # Ring -> Pinky
        d8_sq = (p_x - i_x) ** 2 + (p_y - i_y) ** 2  # Index -> Pinky

        # Root-Mean-Square (RMS) of 8 symmetric palm distances for maximum noise resistance
        l_hand = math.sqrt((d1_sq + d2_sq + d3_sq + d4_sq + d5_sq + d6_sq + d7_sq + d8_sq) / 8.0)

        if l_hand <= 0:
            l_hand = 1.0

        offset_x = w_x if self.center_wrist else 0.0
        offset_y = w_y if self.center_wrist else 0.0
        offset_z = w_z if self.center_wrist else 0.0

        normalized_pts = [
            ((px - offset_x) / l_hand, (py - offset_y) / l_hand, (pz - offset_z) / l_hand)
            for px, py, pz in pts_px
        ]
        return normalized_pts, l_hand


def get_default_output_path(input_csv_path: str) -> str:
    """Derives default output path <video_name>.normalize_landmarks.<hash>.csv in the same location."""
    dir_name = os.path.dirname(os.path.abspath(input_csv_path))
    base_name = os.path.basename(input_csv_path)

    if ".raw_landmarks." in base_name:
        out_name = base_name.replace(".raw_landmarks.", ".normalize_landmarks.")
    elif ".filtered_landmarks." in base_name:
        out_name = base_name.replace(".filtered_landmarks.", ".normalize_landmarks.")
    else:
        name_no_ext = os.path.splitext(base_name)[0]
        out_name = f"{name_no_ext}.normalize_landmarks.csv"

    return os.path.join(dir_name, out_name)


def validate_columns(headers: list[str], csv_path: str):
    """Ensures input CSV contains required metadata and landmark coordinate columns."""
    missing = []
    for col in REQUIRED_METADATA_COLUMNS:
        if col not in headers:
            missing.append(col)

    for lm_name in ALL_21_LANDMARK_NAMES:
        x_col = f"{lm_name}_x"
        y_col = f"{lm_name}_y"
        if x_col not in headers:
            missing.append(x_col)
        if y_col not in headers:
            missing.append(y_col)

    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required column(s): {', '.join(missing)}"
        )


def normalize_landmarks_csv(
    input_csv: str,
    output_csv: str = None,
    keep_raw: bool = False,
    center_wrist: bool = True
) -> str:
    """
    Reads landmark CSV file, validates columns, converts MediaPipe coords to pixels using video width/height,
    calculates rigid 8-distance symmetric palm scale L_hand, applies scale & translation normalization, and writes output CSV.
    Preserves all extra columns (hand_score, z, visibility, presence).
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    output_csv = output_csv or get_default_output_path(input_csv)

    print(f"[1/3] Reading landmarks from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    validate_columns(headers, input_csv)

    if keep_raw:
        mode_str = "KEEP RAW (no normalization)"
    else:
        centering_str = "centered at wrist (0,0)" if center_wrist else "uncentered"
        mode_str = f"Pure Per-Frame Scale Normalization (8-distance symmetric palm RMS, {centering_str})"

    print(f"[2/3] Processing {len(rows)} frames with {mode_str}...")

    normalizer = HandScaleNormalizer(center_wrist=center_wrist)
    output_rows = []

    for row_idx, row in enumerate(rows):
        out_row = dict(row)
        hand_type = row.get("hand", "None")

        if hand_type == "None" or keep_raw:
            output_rows.append(out_row)
            continue

        raw_w = row.get("video_width")
        raw_h = row.get("video_height")
        if raw_w is None or raw_w == "" or raw_h is None or raw_h == "":
            raise ValueError(
                f"Missing video resolution metadata ('video_width' or 'video_height') in row {row_idx + 1} of '{input_csv}'"
            )

        try:
            w = float(raw_w)
            h = float(raw_h)
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Invalid non-positive video resolution ({w}x{h}) in row {row_idx + 1} of '{input_csv}'"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid video resolution values ('video_width': '{raw_w}', 'video_height': '{raw_h}') in row {row_idx + 1} of '{input_csv}': {e}"
            )

        pts_px = []
        is_all_zero = True

        for lm_name in ALL_21_LANDMARK_NAMES:
            raw_x = float(row.get(f"{lm_name}_x", 0.0))
            raw_y = float(row.get(f"{lm_name}_y", 0.0))
            raw_z = float(row.get(f"{lm_name}_z", 0.0))

            if raw_x != 0.0 or raw_y != 0.0:
                is_all_zero = False

            x_px = raw_x * w
            y_px = raw_y * h
            z_px = raw_z * w  # Depth scaled proportionally to width
            pts_px.append((x_px, y_px, z_px))

        if is_all_zero:
            output_rows.append(out_row)
            continue

        norm_pts, l_hand = normalizer.normalize(pts_px)

        for lm_idx, lm_name in enumerate(ALL_21_LANDMARK_NAMES):
            nx, ny, nz = norm_pts[lm_idx]
            out_row[f"{lm_name}_x"] = f"{nx:.6f}"
            out_row[f"{lm_name}_y"] = f"{ny:.6f}"
            if f"{lm_name}_z" in row:
                out_row[f"{lm_name}_z"] = f"{nz:.6f}"

        output_rows.append(out_row)

    print(f"[3/3] Saving normalized landmarks to: {output_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{output_csv}': {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[Success] Saved landmark data to: {output_csv}")
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
                    if ".normalize_landmarks." in filepath and ".normalize_landmarks." not in pattern:
                        continue
                    if filepath not in matched_files:
                        matched_files.append(filepath)
        elif os.path.isfile(pattern) and pattern not in matched_files:
            matched_files.append(pattern)
        else:
            print(f"[Warning] No files found matching input pattern/path: '{pattern}'")

    return matched_files


def main():
    parser = argparse.ArgumentParser(
        description="Applies pure per-frame hand scale normalization (L_hand) to MediaPipe landmark CSV(s)"
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
        help="Input CSV file path(s) or glob pattern(s) (e.g. '*.raw_landmarks.*.csv', 'videos/')"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory path (or output file path for single input)"
    )
    parser.add_argument(
        "-k", "--keep-raw", "--no-normalize",
        action="store_true",
        help="Bypass scale normalization and keep original coordinates"
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Bypass wrist translation centering"
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

    center_wrist = not args.no_center
    if args.keep_raw:
        print("Flag -k / --keep-raw enabled: Passing through raw coordinates without scale normalization.\n")
    else:
        centering_desc = "wrist-centered (0,0)" if center_wrist else "uncentered"
        print(f"Pure per-frame Scale Normalization enabled (8-distance palm RMS, {centering_desc}).\n")

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Normalizing landmarks: {input_file}")
        try:
            out_file_path = None
            if output_target:
                if os.path.isdir(output_target) or output_target.endswith(os.sep) or output_target.endswith("/"):
                    os.makedirs(output_target, exist_ok=True)
                    base_name = os.path.basename(input_file)
                    if ".raw_landmarks." in base_name:
                        out_name = base_name.replace(".raw_landmarks.", ".normalize_landmarks.")
                    elif ".filtered_landmarks." in base_name:
                        out_name = base_name.replace(".filtered_landmarks.", ".normalize_landmarks.")
                    else:
                        name_no_ext = os.path.splitext(base_name)[0]
                        out_name = f"{name_no_ext}.normalize_landmarks.csv"
                    out_file_path = os.path.join(output_target, out_name)
                elif len(input_files) == 1:
                    out_file_path = output_target
                else:
                    os.makedirs(output_target, exist_ok=True)
                    out_file_path = os.path.join(output_target, os.path.basename(get_default_output_path(input_file)))
            else:
                out_file_path = get_default_output_path(input_file)

            normalize_landmarks_csv(
                input_csv=input_file,
                output_csv=out_file_path,
                keep_raw=args.keep_raw,
                center_wrist=center_wrist
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1
        print()

    # Save comprehensive summary JSON for pipeline audit
    try:
        from summary_utils import save_step_summary
        save_step_summary("step_2_normalize_landmarks.json", {
            "step": 2,
            "name": "normalize_landmarks",
            "total_files": len(input_files),
            "success_count": success_count,
            "fail_count": fail_count,
            "center_wrist": center_wrist,
            "keep_raw": args.keep_raw,
            "normalization_method": "Pure per-frame 8-distance palm RMS scale L_hand",
            "wrist_origin": [0.0, 0.0, 0.0] if center_wrist else "uncentered",
        })
    except Exception as e:
        pass

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
