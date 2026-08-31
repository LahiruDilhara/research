# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
#     "scipy>=1.18.1",
# ]
# ///

"""
datacreator/filter_landmarks.py

Landmark noise filtering script for 12 FPS raw MediaPipe landmark CSVs.
Supports multiple temporal noise filtering strategies across 21 hand joint 3D coordinates (x, y, z):
- 1Euro (1€) Adaptive Low-Pass Filter (Casiez et al., 2012)
- 1D Temporal Median Filter (glitch/impulse spike removal)
- Median + 1Euro Combined Multi-Stage Filtering
- Savitzky-Golay Polynomial Curve Fitting Filter
- Hampel Outlier Rejection Filter
- Velocity Cutoff / Max Step Displacement Clamping

Supports processing single CSV files, glob wildcard patterns (e.g. '*', '*.raw_landmarks.*'),
or directory paths. Overwrites target file if it already exists.

Preserves all metadata & extra landmark columns (hand_score, visibility, presence) untouched.
"""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path
import numpy as np
from scipy.signal import medfilt, savgol_filter

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Defaults ──────────────────────────────────────────────────────────────────
FILTER_MIN_CUTOFF = 3.0
FILTER_BETA = 1.4
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


def apply_hampel_filter(arr: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Hampel filter for temporal outlier rejection."""
    n = len(arr)
    if n < window_size:
        return arr
    out = arr.copy()
    half_w = window_size // 2
    for i in range(n):
        sub = arr[max(0, i - half_w): min(n, i + half_w + 1)]
        med = np.median(sub)
        mad = np.median(np.abs(sub - med))
        threshold = n_sigmas * 1.4826 * mad
        if np.abs(arr[i] - med) > threshold and threshold > 1e-6:
            out[i] = med
    return out


def apply_savgol_filter(arr: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay polynomial curve fitting filter."""
    n = len(arr)
    if n < 3:
        return arr
    w = window_length if window_length % 2 != 0 else window_length + 1
    if n < w:
        w = n if n % 2 != 0 else n - 1
    if w <= polyorder:
        polyorder = max(1, w - 1)
    if w < 3:
        return arr
    return savgol_filter(arr, window_length=w, polyorder=polyorder)


def apply_median_filter(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """1D Median filter for impulse spike/glitch removal."""
    n = len(arr)
    if n < 3:
        return arr
    k = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    if n < k:
        k = n if n % 2 != 0 else n - 1
    if k < 3:
        return arr
    return medfilt(arr, kernel_size=k)


def get_default_output_path(input_csv_path: str) -> str:
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
    filter_mode: str = "euro",
    min_cutoff: float = FILTER_MIN_CUTOFF,
    beta: float = FILTER_BETA,
    d_cutoff: float = FILTER_D_CUTOFF,
    median_kernel: int = 3,
    savgol_window: int = 5,
    savgol_poly: int = 2,
    hampel_window: int = 5,
    hampel_n_sigmas: float = 3.0,
    max_step_vel: float | None = None,
) -> str:
    """
    Reads landmark CSV, partitions into hand tracking segments, applies target filter pipeline to (x, y, z)
    joint trajectories, and writes output CSV.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    output_csv = output_csv or get_default_output_path(input_csv)

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"Input CSV '{input_csv}' is empty!")

    # Partition rows into contiguous segments by hand presence
    segments = []
    curr_seg = []
    prev_hand = None

    for row in raw_rows:
        h_type = row.get("hand", "None")
        if h_type == "None" or (prev_hand is not None and h_type != prev_hand):
            if curr_seg:
                segments.append(curr_seg)
                curr_seg = []
        if h_type != "None":
            curr_seg.append(row)
        else:
            segments.append([row])
        prev_hand = h_type
    if curr_seg:
        segments.append(curr_seg)

    filtered_rows = []

    for seg in segments:
        if len(seg) == 0:
            continue
        h_type = seg[0].get("hand", "None")
        if h_type == "None" or len(seg) == 1:
            for r in seg:
                filtered_rows.append(dict(r))
            continue

        timestamps = [float(r.get("timestamp_ms", "0")) / 1000.0 for r in seg]
        lm_coords = {}

        for lm_name in ALL_21_LANDMARK_NAMES:
            for axis in ["x", "y", "z"]:
                col = f"{lm_name}_{axis}"
                if col in seg[0]:
                    vals = [float(r.get(col, 0.0)) for r in seg]
                    lm_coords[col] = np.array(vals, dtype=np.float64)

        # Stage 1: Pre-smoothing (Median / Hampel)
        if "median" in filter_mode:
            for col in lm_coords:
                lm_coords[col] = apply_median_filter(lm_coords[col], kernel_size=median_kernel)

        if "hampel" in filter_mode:
            for col in lm_coords:
                lm_coords[col] = apply_hampel_filter(lm_coords[col], window_size=hampel_window, n_sigmas=hampel_n_sigmas)

        # Stage 2: Main Smoothing (1Euro / Savitzky-Golay)
        if "euro" in filter_mode or filter_mode == "default":
            euro_filt = LandmarkEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for idx in range(len(seg)):
                t_sec = timestamps[idx]
                for col, arr in lm_coords.items():
                    if arr[idx] == 0.0:
                        continue
                    arr[idx] = euro_filt.filter_coordinate(col, t_sec, arr[idx])

        elif "savgol" in filter_mode:
            for col in lm_coords:
                lm_coords[col] = apply_savgol_filter(lm_coords[col], window_length=savgol_window, polyorder=savgol_poly)

        # Stage 3: Velocity Cutoff / Displacement Clamping
        if max_step_vel is not None and max_step_vel > 0:
            for lm_name in ALL_21_LANDMARK_NAMES:
                x_col, y_col, z_col = f"{lm_name}_x", f"{lm_name}_y", f"{lm_name}_z"
                if x_col in lm_coords and y_col in lm_coords:
                    xs, ys = lm_coords[x_col], lm_coords[y_col]
                    zs = lm_coords.get(z_col, np.zeros_like(xs))
                    for i in range(1, len(xs)):
                        dx, dy, dz = xs[i] - xs[i-1], ys[i] - ys[i-1], zs[i] - zs[i-1]
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                        if dist > max_step_vel and dist > 1e-6:
                            scale = max_step_vel / dist
                            xs[i] = xs[i-1] + dx * scale
                            ys[i] = ys[i-1] + dy * scale
                            if z_col in lm_coords:
                                zs[i] = zs[i-1] + dz * scale

        # Reconstruct rows
        for idx, r in enumerate(seg):
            out_r = dict(r)
            for col, arr in lm_coords.items():
                out_r[col] = f"{arr[idx]:.6f}"
            filtered_rows.append(out_r)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except OSError:
            pass

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

    return output_csv


def collect_input_files(input_patterns: list[str]) -> list[str]:
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
        description="Applies multi-strategy noise filtering to 3D (x, y, z) coordinates of MediaPipe hand landmark CSV(s)"
    )
    parser.add_argument("pos_args", nargs="*", help="Input CSV file(s), glob pattern(s), or output directory")
    parser.add_argument("-i", "--input", nargs="+", default=None, help="Input CSV file path(s) or glob pattern(s)")
    parser.add_argument("-o", "--output", default=None, help="Output directory path")
    parser.add_argument("--mode", default="euro", choices=["euro", "median", "median_euro", "hampel_euro", "savgol", "median_savgol", "none"], help="Filter mode")
    parser.add_argument("-min", "--min-cutoff", type=float, default=FILTER_MIN_CUTOFF, help=f"1Euro min cutoff Hz (default: {FILTER_MIN_CUTOFF})")
    parser.add_argument("-beta", "--beta", type=float, default=FILTER_BETA, help=f"1Euro beta speed slope (default: {FILTER_BETA})")
    parser.add_argument("-d", "--d-cutoff", type=float, default=FILTER_D_CUTOFF, help=f"1Euro d_cutoff Hz (default: {FILTER_D_CUTOFF})")
    parser.add_argument("--median-kernel", type=int, default=3, help="Median filter kernel size (default: 3)")
    parser.add_argument("--savgol-window", type=int, default=5, help="Savitzky-Golay window length (default: 5)")
    parser.add_argument("--savgol-poly", type=int, default=2, help="Savitzky-Golay polynomial order (default: 2)")
    parser.add_argument("--hampel-window", type=int, default=5, help="Hampel window size (default: 5)")
    parser.add_argument("--hampel-n-sigmas", type=float, default=3.0, help="Hampel n_sigmas threshold (default: 3.0)")
    parser.add_argument("--max-step-vel", type=float, default=None, help="Velocity cutoff / max displacement per frame step")

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
    print(f"Filter Mode: '{args.mode}', min_cutoff={args.min_cutoff}, beta={args.beta}, median_kernel={args.median_kernel}")

    success_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
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
                filter_mode=args.mode,
                min_cutoff=args.min_cutoff,
                beta=args.beta,
                d_cutoff=args.d_cutoff,
                median_kernel=args.median_kernel,
                savgol_window=args.savgol_window,
                savgol_poly=args.savgol_poly,
                hampel_window=args.hampel_window,
                hampel_n_sigmas=args.hampel_n_sigmas,
                max_step_vel=args.max_step_vel,
            )
            success_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1

    print(f"Batch Filtering Complete: {success_count}/{len(input_files)} successful.")


if __name__ == "__main__":
    main()
