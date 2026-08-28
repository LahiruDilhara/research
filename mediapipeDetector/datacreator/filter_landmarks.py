# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/filter_landmarks.py

Landmark noise filtering script for 12 FPS raw MediaPipe landmark CSVs.
Applies temporal One Euro (1€) filtering across all 21 hand joint coordinates (x, y)
over the entire video stream to remove jitter while preserving fast motion.

Saves output CSV to <video_name>.filtered_landmarks.<hash>.csv in the same location as input video/CSV.
Overrides file if it already exists.
"""

import argparse
import csv
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

        # Filtered derivative (velocity) of the signal
        a_d = self._smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff: higher when moving fast, lower when still
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_elapsed, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class LandmarkEuroFilter:
    """Manages 1€ filters for x and y coordinates for all 21 hand landmarks."""

    def __init__(
        self,
        min_cutoff: float = FILTER_MIN_CUTOFF,
        beta: float = FILTER_BETA,
        d_cutoff: float = FILTER_D_CUTOFF
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters: dict[str, tuple[OneEuroFilter1D, OneEuroFilter1D]] = {}

    def reset(self):
        self.filters.clear()

    def filter_landmark(self, name: str, t_sec: float, x: float, y: float) -> tuple[float, float]:
        if name not in self.filters:
            fx = OneEuroFilter1D(t_sec, x, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            fy = OneEuroFilter1D(t_sec, y, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            self.filters[name] = (fx, fy)
            return float(x), float(y)

        fx, fy = self.filters[name]
        return fx.filter(t_sec, x), fy.filter(t_sec, y)


def get_default_output_path(raw_csv_path: str) -> str:
    """Derives default output path <video_name>.filtered_landmarks.<hash>.csv in the same location."""
    dir_name = os.path.dirname(os.path.abspath(raw_csv_path))
    base_name = os.path.basename(raw_csv_path)

    if ".raw_landmarks." in base_name:
        out_name = base_name.replace(".raw_landmarks.", ".filtered_landmarks.")
    else:
        name_no_ext = os.path.splitext(base_name)[0]
        out_name = f"{name_no_ext}.filtered_landmarks.csv"

    return os.path.join(dir_name, out_name)


def filter_raw_landmarks_csv(
    input_csv: str,
    output_csv: str = None,
    min_cutoff: float = FILTER_MIN_CUTOFF,
    beta: float = FILTER_BETA,
    d_cutoff: float = FILTER_D_CUTOFF
) -> str:
    """
    Reads raw landmarks CSV, applies One Euro filter across video frames, and writes filtered CSV.
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

    print(f"[2/3] Filtering {len(raw_rows)} frames with 1€ Filter (min_cutoff={min_cutoff}, beta={beta}, d_cutoff={d_cutoff})...")

    hand_filter = LandmarkEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    filtered_rows = []

    for row_idx, row in enumerate(raw_rows):
        out_row = dict(row)
        t_ms = float(row.get("timestamp_ms", "0"))
        t_sec = t_ms / 1000.0
        hand_type = row.get("hand", "None")

        if hand_type == "None":
            # Hand missing in this frame: reset filter state
            hand_filter.reset()
        else:
            for lm_name in ALL_21_LANDMARK_NAMES:
                x_col = f"{lm_name}_x"
                y_col = f"{lm_name}_y"

                raw_x = float(row.get(x_col, 0.0))
                raw_y = float(row.get(y_col, 0.0))

                if raw_x == 0.0 and raw_y == 0.0:
                    continue

                fx, fy = hand_filter.filter_landmark(lm_name, t_sec, raw_x, raw_y)
                out_row[x_col] = f"{fx:.6f}"
                out_row[y_col] = f"{fy:.6f}"

        filtered_rows.append(out_row)

    print(f"[3/3] Saving filtered landmarks to: {output_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[Success] Filtered landmark data saved to: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Applies 1€ temporal filter to raw MediaPipe hand landmark CSV")
    parser.add_argument("-i", "--input", required=True, help="Path to raw landmarks CSV file")
    parser.add_argument("-o", "--output", default="", help="Optional output filtered CSV path")
    parser.add_argument("-min", "--min-cutoff", "--min_cutoff", type=float, default=None, help=f"1€ Filter min_cutoff in Hz (default: {FILTER_MIN_CUTOFF})")
    parser.add_argument("-beta", "--beta", type=float, default=None, help=f"1€ Filter beta parameter (default: {FILTER_BETA})")
    parser.add_argument("-d", "-dcutoff", "--d-cutoff", "--d_cutoff", type=float, default=None, help=f"1€ Filter derivative cutoff frequency (d_cutoff) in Hz (default: {FILTER_D_CUTOFF})")

    args = parser.parse_args()

    min_cutoff = args.min_cutoff if args.min_cutoff is not None else FILTER_MIN_CUTOFF
    beta = args.beta if args.beta is not None else FILTER_BETA
    d_cutoff = args.d_cutoff if args.d_cutoff is not None else FILTER_D_CUTOFF

    try:
        filter_raw_landmarks_csv(
            input_csv=args.input,
            output_csv=args.output,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff
        )
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

