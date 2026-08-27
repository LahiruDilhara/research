#!/usr/bin/env python3
"""
filter_dataset.py — Filters unrolled per-finger dataset CSV rows based on configurable flags.

Usage:
    python3 analysis/filter_dataset.py -i ./analysis/per_finger_dataset.csv -o ./analysis/filtered_dataset.csv --remove-zero-vel-touch --remove-right-hand --remove-left-hand --remove-out-of-sync
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
logger = logging.getLogger("FilterDataset")


def parse_bool_flag(val: str) -> bool:
    """Parse common string boolean values from CSV."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def filter_dataset(
    input_path: str,
    output_path: str,
    remove_zero_vel_touch: bool = False,
    remove_right_hand: bool = False,
    remove_left_hand: bool = False,
    remove_out_of_sync: bool = False,
) -> None:
    in_file = Path(input_path).resolve()
    out_file = Path(output_path).resolve()

    if not in_file.exists():
        logger.error(f"Input file does not exist: {in_file}")
        sys.exit(1)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input file: {in_file}")

    with open(in_file, mode="r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            logger.error(f"Input CSV '{in_file}' has no headers or is empty.")
            sys.exit(1)

        total_rows = 0
        retained_rows = 0
        removed_zero_vel_touch_cnt = 0
        removed_right_hand_cnt = 0
        removed_left_hand_cnt = 0
        removed_out_of_sync_cnt = 0

        with open(out_file, mode="w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                keep = True

                # 1. Filter zero velocity touch
                if keep and remove_zero_vel_touch:
                    touch = parse_bool_flag(row.get("touch_finger", "0"))
                    if touch:
                        # Check fingertip (dip joint) velocities across all 4 transition points
                        all_zero = True
                        for v in range(1, 5):
                            vx_str = row.get(f"dip{v}_vx", "0")
                            vy_str = row.get(f"dip{v}_vy", "0")
                            try:
                                vx = float(vx_str) if vx_str != "" else 0.0
                                vy = float(vy_str) if vy_str != "" else 0.0
                            except ValueError:
                                vx, vy = 0.0, 0.0

                            if vx != 0.0 or vy != 0.0:
                                all_zero = False
                                break

                        if all_zero:
                            keep = False
                            removed_zero_vel_touch_cnt += 1

                # 2. Filter right hand rows
                if keep and remove_right_hand:
                    is_right = parse_bool_flag(row.get("rightHand", "0"))
                    if is_right:
                        keep = False
                        removed_right_hand_cnt += 1

                # 3. Filter left hand rows
                if keep and remove_left_hand:
                    is_right = parse_bool_flag(row.get("rightHand", "0"))
                    if not is_right:
                        keep = False
                        removed_left_hand_cnt += 1

                # 4. Filter out of sync rows
                if keep and remove_out_of_sync:
                    is_out_of_sync = parse_bool_flag(row.get("out_of_sync", "0"))
                    if is_out_of_sync:
                        keep = False
                        removed_out_of_sync_cnt += 1

                if keep:
                    writer.writerow(row)
                    retained_rows += 1

    logger.info("Filtering complete summary:")
    logger.info(f"  Total input rows: {total_rows}")
    logger.info(f"  Retained rows:    {retained_rows}")
    if remove_zero_vel_touch:
        logger.info(f"  Removed zero-vel touch rows: {removed_zero_vel_touch_cnt}")
    if remove_right_hand:
        logger.info(f"  Removed right-hand rows:     {removed_right_hand_cnt}")
    if remove_left_hand:
        logger.info(f"  Removed left-hand rows:      {removed_left_hand_cnt}")
    if remove_out_of_sync:
        logger.info(f"  Removed out-of-sync rows:    {removed_out_of_sync_cnt}")
    logger.info(f"Filtered dataset successfully saved to '{out_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Filter per-finger dataset CSV rows based on configurable flags."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input per-finger CSV file path"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output filtered CSV file path"
    )
    parser.add_argument(
        "--remove-zero-vel-touch",
        action="store_true",
        help="Remove rows where touch occurs (touch_finger=1) but all velocity components (mcp, pip, dip vx & vy across transitions) are 0",
    )
    parser.add_argument(
        "--remove-right-hand",
        action="store_true",
        help="Remove right hand rows (rightHand=1)",
    )
    parser.add_argument(
        "--remove-left-hand",
        action="store_true",
        help="Remove left hand rows (rightHand=0)",
    )
    parser.add_argument(
        "--remove-out-of-sync",
        action="store_true",
        help="Remove out-of-sync rows (out_of_sync=1)",
    )

    args = parser.parse_args()

    filter_dataset(
        input_path=args.input,
        output_path=args.output,
        remove_zero_vel_touch=args.remove_zero_vel_touch,
        remove_right_hand=args.remove_right_hand,
        remove_left_hand=args.remove_left_hand,
        remove_out_of_sync=args.remove_out_of_sync,
    )


if __name__ == "__main__":
    main()
