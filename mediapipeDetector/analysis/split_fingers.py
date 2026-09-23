#!/usr/bin/env python3
"""
split_fingers.py — Unrolls 5-finger combined window rows into 5 individual per-finger dataset rows.

Usage:
    python3 analysis/split_fingers.py -i ./analysis/merged_dataset.csv -o ./analysis/per_finger_dataset.csv
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SplitFingers")

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINTS = ["mcp", "pip", "dip"]

# ── Target per-finger CSV header ─────────────────────────────────────────────
TARGET_HEADER = [
    # Metadata
    "video_file", "video_hash", "duration_ms", "start_ms", "end_ms", "start_frame", "end_frame",
    "finger_name",
    # Coordinates across 5 frames (1..5)
    "wrist1_x", "wrist1_y", "mcp1_x", "mcp1_y", "pip1_x", "pip1_y", "dip1_x", "dip1_y",
    "wrist2_x", "wrist2_y", "mcp2_x", "mcp2_y", "pip2_x", "pip2_y", "dip2_x", "dip2_y",
    "wrist3_x", "wrist3_y", "mcp3_x", "mcp3_y", "pip3_x", "pip3_y", "dip3_x", "dip3_y",
    "wrist4_x", "wrist4_y", "mcp4_x", "mcp4_y", "pip4_x", "pip4_y", "dip4_x", "dip4_y",
    "wrist5_x", "wrist5_y", "mcp5_x", "mcp5_y", "pip5_x", "pip5_y", "dip5_x", "dip5_y",
    # Velocities across 4 transitions (1..4)
    "wrist1_vx", "wrist1_vy", "mcp1_vx", "mcp1_vy", "pip1_vx", "pip1_vy", "dip1_vx", "dip1_vy",
    "wrist2_vx", "wrist2_vy", "mcp2_vx", "mcp2_vy", "pip2_vx", "pip2_vy", "dip2_vx", "dip2_vy",
    "wrist3_vx", "wrist3_vy", "mcp3_vx", "mcp3_vy", "pip3_vx", "pip3_vy", "dip3_vx", "dip3_vy",
    "wrist4_vx", "wrist4_vy", "mcp4_vx", "mcp4_vy", "pip4_vx", "pip4_vy", "dip4_vx", "dip4_vy",
    # Target label and context flags
    "touch_finger",
    "hand_move", "hand_point_of_view", "hand_closer", "hovering", "daylight", "hand_visible",
    "out_of_sync", "rightHand", "any_difference"
]

METADATA_COLS = [
    "video_file", "video_hash", "duration_ms", "start_ms", "end_ms", "start_frame", "end_frame"
]

CONTEXT_FLAGS = [
    "hand_move", "hand_point_of_view", "hand_closer", "hovering", "daylight", "hand_visible",
    "out_of_sync", "rightHand", "any_difference"
]

def split_finger_data(input_path: str, output_path: str) -> None:
    in_file = Path(input_path).resolve()
    out_file = Path(output_path).resolve()

    if not in_file.exists():
        logger.error(f"Input file does not exist: {in_file}")
        sys.exit(1)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input file: {in_file}")

    with open(in_file, mode="r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        
        # Verify required columns exist
        missing_meta = set(METADATA_COLS) - set(reader.fieldnames or [])
        missing_flags = set(CONTEXT_FLAGS) - set(reader.fieldnames or [])
        if missing_meta or missing_flags:
            logger.error(f"Input CSV is missing required columns. Missing meta: {missing_meta}, Missing flags: {missing_flags}")
            sys.exit(1)

        input_rows = 0
        output_rows = 0

        with open(out_file, mode="w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(TARGET_HEADER)

            for row in reader:
                input_rows += 1
                meta_vals = [row[col] for col in METADATA_COLS]
                flags_vals = [row[col] for col in CONTEXT_FLAGS]

                # Unroll row into 5 separate finger records
                for finger in FINGERS:
                    finger_row = []
                    
                    # 1. Metadata
                    finger_row.extend(meta_vals)
                    
                    # 2. Finger name
                    finger_row.append(finger)

                    # 3. Coordinates (5 frames: f_step 1..5)
                    for f in range(1, 6):
                        # Wrist coords
                        finger_row.append(row.get(f"wrist{f}_x", ""))
                        finger_row.append(row.get(f"wrist{f}_y", ""))
                        # Finger joint coords (mcp, pip, dip)
                        for joint in JOINTS:
                            finger_row.append(row.get(f"{finger}{f}_{joint}_x", ""))
                            finger_row.append(row.get(f"{finger}{f}_{joint}_y", ""))

                    # 4. Velocities (4 transition steps: v_step 1..4)
                    for v in range(1, 5):
                        # Wrist velocities
                        finger_row.append(row.get(f"wrist{v}_vx", ""))
                        finger_row.append(row.get(f"wrist{v}_vy", ""))
                        # Finger joint velocities (mcp, pip, dip)
                        for joint in JOINTS:
                            finger_row.append(row.get(f"{finger}{v}_{joint}_vx", ""))
                            finger_row.append(row.get(f"{finger}{v}_{joint}_vy", ""))

                    # 5. Finger touch label
                    finger_touch_val = row.get(f"{finger}_touch", "0")
                    finger_row.append(finger_touch_val)

                    # 6. Context flags
                    finger_row.extend(flags_vals)

                    writer.writerow(finger_row)
                    output_rows += 1

    logger.info(f"Splitting complete: {input_rows} input rows -> {output_rows} per-finger rows written to '{out_file}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Unroll 5-finger combined window rows into 5 individual per-finger dataset rows."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input merged CSV file path"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output per-finger CSV file path"
    )

    args = parser.parse_args()
    split_finger_data(args.input, args.output)

if __name__ == "__main__":
    main()
