# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/filter_landmarks.py

Landmark noise filtering script for 12 FPS raw MediaPipe landmark CSVs.
Applies temporal One Euro (1€) filtering across all 21 hand joint 3D coordinates (x, y, z)
over the entire video stream to remove jitter while preserving fast motion.

Supports processing single CSV files, glob wildcard patterns (e.g. '*', '*.raw_landmarks.*', 'videos/*.csv'),
or directory paths. Saves filtered CSV files to a specified output directory or beside the source CSV.
Overrides/prunes target file if it already exists.

Preserves all metadata & extra landmark columns (hand_score, visibility, presence) untouched.
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

# ── 1€ Filter Root Level Parameters ──────────────────────────────────────────
# FILTER_MIN_CUTOFF (Hz): Minimum cutoff frequency when the hand is stationary / still.
# Lower values (e.g., 0.5 - 1.5 Hz) aggressively smooth out jitter and trembling when resting or hovering.
FILTER_MIN_CUTOFF = 1.5

# FILTER_BETA: Speed coefficient (slope) that dynamically increases cutoff frequency
# during fast motion to eliminate latency. Higher values (e.g., 1.0 - 2.0) ensure zero lag
# during quick taps and rapid hand gestures.
FILTER_BETA = 1.0

# FILTER_D_CUTOFF (Hz): Cutoff frequency for filtering signal velocity (derivative).
# This is a STANDARD parameter in the original 1€ Filter paper (Casiez et al., 2012),
# typically fixed at 1.0 Hz to smooth out noisy velocity estimates before computing adaptive cutoff.
FILTER_D_CUTOFF = 1.0

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class OneEuroFilter1D:
    """Filters a single scalar value over time using Casiez et al. 1€ filter."""

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = FILTER_MIN_CUTOFF,
        beta: float = FILTER_BETA,
        d_cutoff: float = FILTER_D_CUTOFF
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_elapsed: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_elapsed
        return r / (r + 1.0)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, t: float, x: float) -> float:
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0:
            return self.x_prev

        a_d = self._smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_elapsed, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class LandmarkEuroFilter:
    """Manages 1€ filters for (x, y, z) coordinates of all 21 hand joints."""

    def __init__(
        self,
        min_cutoff: float = FILTER_MIN_CUTOFF,
        beta: float = FILTER_BETA,
        d_cutoff: float = FILTER_D_CUTOFF
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters: dict[str, OneEuroFilter1D] = {}

    def reset(self):
        """Resets filter states when hand tracking is lost or hand label changes."""
        self.filters.clear()

    def filter_coordinate(self, key: str, t: float, val: float) -> float:
        if key not in self.filters:
            self.filters[key] = OneEuroFilter1D(
                t0=t, x0=val,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff
            )
            return val
        return self.filters[key].filter(t, val)


def get_default_output_path(input_csv_path: str) -> str:
    """Derives default output path <video_name>.filtered_landmarks.<hash>.csv in the same location."""
    dir_name = os.path.dirname(os.path.abspath(input_csv_path))
    base_name = os.path.basename(input_csv_path)

    if ".raw_landmarks." in base_name:
        out_name = base_name.replace(".raw_landmarks.", ".filtered_landmarks.")
    elif ".normalize_landmarks." in base_name:
        out_name = base_name.replace(".normalize_landmarks.", ".filtered_landmarks.")
    else:
        name_no_ext = os.path.splitext(base_name)[0]
        out_name = f"{name_no_ext}.filtered_landmarks.csv"

    return os.path.join(dir_name, out_name)


def filter_landmarks_csv(
    input_csv: str,
    output_csv: str = None,
    min_cutoff: float = FILTER_MIN_CUTOFF,
    beta: float = FILTER_BETA,
    d_cutoff: float = FILTER_D_CUTOFF
) -> str:
    """
    Reads raw landmarks CSV, applies One Euro filter to (x, y, z) coordinates continuously across video frame timestamps,
    and writes filtered CSV while keeping metadata, visibility, presence, and hand_score intact.
    Overwrites existing destination files.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    output_csv = output_csv or get_default_output_path(input_csv)

    print(f"[1/3] Reading raw landmarks from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    print(f"[2/3] Filtering {len(raw_rows)} frames with 1€ Filter on (x, y, z) (min_cutoff={min_cutoff}, beta={beta}, d_cutoff={d_cutoff})...")

    hand_filter = LandmarkEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    filtered_rows = []
    prev_hand_type = None

    for row_idx, row in enumerate(raw_rows):
        out_row = dict(row)
        t_ms = float(row.get("timestamp_ms", "0"))
        t_sec = t_ms / 1000.0
        hand_type = row.get("hand", "None")

        if hand_type == "None" or (prev_hand_type is not None and hand_type != prev_hand_type):
            hand_filter.reset()

        if hand_type != "None":
            for lm_name in ALL_21_LANDMARK_NAMES:
                x_col = f"{lm_name}_x"
                y_col = f"{lm_name}_y"
                z_col = f"{lm_name}_z"

                raw_x = float(row.get(x_col, 0.0))
                raw_y = float(row.get(y_col, 0.0))

                if raw_x == 0.0 and raw_y == 0.0:
                    continue

                fx = hand_filter.filter_coordinate(x_col, t_sec, raw_x)
                fy = hand_filter.filter_coordinate(y_col, t_sec, raw_y)
                out_row[x_col] = f"{fx:.6f}"
                out_row[y_col] = f"{fy:.6f}"

                if z_col in row:
                    raw_z = float(row.get(z_col, 0.0))
                    fz = hand_filter.filter_coordinate(z_col, t_sec, raw_z)
                    out_row[z_col] = f"{fz:.6f}"

        prev_hand_type = hand_type
        filtered_rows.append(out_row)

    print(f"[3/3] Saving filtered landmarks to: {output_csv}")
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

    print(f"[Success] Filtered landmark data saved to: {output_csv}")
    return output_csv


def collect_input_files(input_patterns: list[str]) -> list[str]:
    """Expands glob patterns, directory paths, or file lists into a list of landmark CSV file paths."""
    matched_files = []
    for pattern in input_patterns:
        search_pattern = os.path.join(pattern, "*.csv") if os.path.isdir(pattern) else pattern
        glob_matches = glob.glob(search_pattern, recursive=True)
        if glob_matches:
            for filepath in sorted(glob_matches):
                if os.path.isfile(filepath) and filepath.endswith(".csv"):
                    if ".filtered_landmarks." in filepath and ".filtered_landmarks." not in pattern:
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
        description="Applies One Euro (1€) filtering to 3D (x, y, z) coordinates of MediaPipe hand landmark CSV(s)"
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
    parser.add_argument(
        "-min", "--min-cutoff",
        type=float,
        default=FILTER_MIN_CUTOFF,
        help=f"Minimum cutoff frequency in Hz (default: {FILTER_MIN_CUTOFF})"
    )
    parser.add_argument(
        "-beta", "--beta",
        type=float,
        default=FILTER_BETA,
        help=f"Speed coefficient beta (default: {FILTER_BETA})"
    )
    parser.add_argument(
        "-d", "--d-cutoff",
        type=float,
        default=FILTER_D_CUTOFF,
        help=f"Derivative cutoff frequency in Hz (default: {FILTER_D_CUTOFF})"
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

    print(f"1€ Filter Parameters on (x, y, z): min_cutoff={args.min_cutoff} Hz, beta={args.beta}, d_cutoff={args.d_cutoff} Hz\n")

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Filtering (x, y, z) landmarks: {input_file}")
        try:
            out_file_path = None
            if output_target:
                if os.path.isdir(output_target) or output_target.endswith(os.sep) or output_target.endswith("/"):
                    os.makedirs(output_target, exist_ok=True)
                    base_name = os.path.basename(input_file)
                    if ".raw_landmarks." in base_name:
                        out_name = base_name.replace(".raw_landmarks.", ".filtered_landmarks.")
                    elif ".normalize_landmarks." in base_name:
                        out_name = base_name.replace(".normalize_landmarks.", ".filtered_landmarks.")
                    else:
                        name_no_ext = os.path.splitext(base_name)[0]
                        out_name = f"{name_no_ext}.filtered_landmarks.csv"
                    out_file_path = os.path.join(output_target, out_name)
                elif len(input_files) == 1:
                    out_file_path = output_target
                else:
                    os.makedirs(output_target, exist_ok=True)
                    out_file_path = os.path.join(output_target, os.path.basename(get_default_output_path(input_file)))
            else:
                out_file_path = get_default_output_path(input_file)

            filter_landmarks_csv(
                input_csv=input_file,
                output_csv=out_file_path,
                min_cutoff=args.min_cutoff,
                beta=args.beta,
                d_cutoff=args.d_cutoff
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Landmark Filtering Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
