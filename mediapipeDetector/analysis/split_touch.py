#!/usr/bin/env python3
"""
split_touch.py — Separates dataset CSV rows into touch and untouch CSV files based on the touch label.

Usage:
    python3 analysis/split_touch.py -i ./data/cleaned_data.csv --touch-out ./data/touch_dataset.csv --untouch-out ./data/untouch_dataset.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SplitTouch")


def parse_bool_flag(val: str) -> bool:
    """Parse common string boolean values from CSV."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def split_touch_untouch(input_path: str, touch_out_path: str, untouch_out_path: str) -> None:
    in_file = Path(input_path).resolve()
    touch_file = Path(touch_out_path).resolve()
    untouch_file = Path(untouch_out_path).resolve()

    if not in_file.exists():
        logger.error(f"Input file does not exist: {in_file}")
        sys.exit(1)

    touch_file.parent.mkdir(parents=True, exist_ok=True)
    untouch_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input CSV: {in_file}")

    with open(in_file, mode="r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            logger.error(f"Input CSV '{in_file}' has no headers or is empty.")
            sys.exit(1)

        # Check for touch label column
        touch_col = None
        if "touch_finger" in fieldnames:
            touch_col = "touch_finger"
        elif "touch" in fieldnames:
            touch_col = "touch"
        else:
            logger.error("Could not find touch column ('touch_finger' or 'touch') in input CSV.")
            sys.exit(1)

        total_rows = 0
        touch_rows_cnt = 0
        untouch_rows_cnt = 0

        with open(touch_file, mode="w", newline="", encoding="utf-8") as ftouch, \
             open(untouch_file, mode="w", newline="", encoding="utf-8") as funtouch:

            touch_writer = csv.DictWriter(ftouch, fieldnames=fieldnames)
            untouch_writer = csv.DictWriter(funtouch, fieldnames=fieldnames)

            touch_writer.writeheader()
            untouch_writer.writeheader()

            for row in reader:
                total_rows += 1
                is_touch = parse_bool_flag(row.get(touch_col, "0"))

                if is_touch:
                    touch_writer.writerow(row)
                    touch_rows_cnt += 1
                else:
                    untouch_writer.writerow(row)
                    untouch_rows_cnt += 1

    logger.info("Separation complete summary:")
    logger.info(f"  Total input rows: {total_rows}")
    logger.info(f"  Touch rows:       {touch_rows_cnt} -> '{touch_file}'")
    logger.info(f"  Untouch rows:     {untouch_rows_cnt} -> '{untouch_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Separate dataset CSV rows into touch and untouch CSV files."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV dataset path"
    )
    parser.add_argument(
        "--touch-out", required=True, help="Output CSV path for touch rows"
    )
    parser.add_argument(
        "--untouch-out", required=True, help="Output CSV path for untouch rows"
    )

    args = parser.parse_args()

    split_touch_untouch(
        input_path=args.input,
        touch_out_path=args.touch_out,
        untouch_out_path=args.untouch_out,
    )


if __name__ == "__main__":
    main()
