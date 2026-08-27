#!/usr/bin/env python3
"""
create_train_test_split.py — Creates balanced training and testing datasets from touch and untouch CSV files.

Pipeline Logic:
  1. Touch Test / Train Division:
     - Select `touch_test_pct`% (e.g., 20%) of rows randomly from the touch dataset for touch_test.
     - The remaining touch rows form touch_train.
  2. Untouch Train / Test Division:
     - Sample `untouch_train_ratio_pct`% (e.g., 120%) relative to the size of touch_train from the untouch dataset for untouch_train.
     - Optionally cap untouch_test size using `--max-untouch-test` flag.
     - Remaining (or capped) untouch rows form untouch_test.
  3. Combination & Ordering:
     - Combine touch_train + untouch_train into train dataset.
     - Combine touch_test + untouch_test into test dataset.
     - Apply sorting (by video_file, video_hash, start_frame, finger_name) or shuffling based on CLI flags.

Usage:
    python3 analysis/create_train_test_split.py \
        --touch-in ./data/touch_dataset.csv \
        --untouch-in ./data/untouch_dataset.csv \
        --train-out ./data/training_data.csv \
        --test-out ./data/testing_data.csv \
        --touch-test-pct 20 \
        --untouch-train-ratio-pct 120 \
        --max-untouch-test 2000 \
        --shuffle \
        --seed 42
"""

# python3 analysis/create_train_test_split.py --touch-in ./data/toutch_dataset.csv --untouch-in ./data/untouch_dataset.csv --train-out ./data/training_data.csv --test-out ./data/test_data.csv --touch-test-pct 15 --untouch-train-ratio-pct 125 --seed 50

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainTestSplitter")


def load_csv(file_path: Path) -> tuple[list[str], list[dict]]:
    if not file_path.exists():
        logger.error(f"Input file does not exist: {file_path}")
        sys.exit(1)

    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def save_csv(file_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict]) -> list[dict]:
    """Sort rows by video_file, video_hash, start_frame, and finger_name if available."""
    def sort_key(r: dict):
        video_file = r.get("video_file", "")
        video_hash = r.get("video_hash", "")
        try:
            start_frame = int(r.get("start_frame", 0))
        except ValueError:
            start_frame = 0
        finger_name = r.get("finger_name", "")
        return (video_file, video_hash, start_frame, finger_name)

    return sorted(rows, key=sort_key)


def split_and_balance(
    touch_in: str,
    untouch_in: str,
    train_out: str,
    test_out: str,
    touch_test_pct: float,
    untouch_train_ratio_pct: float,
    untouch_test_ratio_pct: float | None,
    max_untouch_test: int | None,
    sort_output: bool,
    shuffle_output: bool,
    seed: int | None,
) -> None:
    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    touch_path = Path(touch_in).resolve()
    untouch_path = Path(untouch_in).resolve()
    train_path = Path(train_out).resolve()
    test_path = Path(test_out).resolve()

    touch_fields, touch_rows = load_csv(touch_path)
    untouch_fields, untouch_rows = load_csv(untouch_path)

    if touch_fields != untouch_fields:
        logger.warning("Field names differ between touch and untouch CSV files! Using touch CSV field structure.")

    fieldnames = touch_fields

    logger.info(f"Loaded {len(touch_rows)} touch rows and {len(untouch_rows)} untouch rows.")

    # 1. Separate touch dataset into test and train
    touch_count = len(touch_rows)
    touch_test_count = int(round(touch_count * (touch_test_pct / 100.0)))
    touch_test_count = max(0, min(touch_count, touch_test_count))

    shuffled_touch = list(touch_rows)
    random.shuffle(shuffled_touch)

    touch_test = shuffled_touch[:touch_test_count]
    touch_train = shuffled_touch[touch_test_count:]

    logger.info(f"Touch split ({touch_test_pct}% test): {len(touch_train)} train, {len(touch_test)} test.")

    # 2. Separate untouch dataset into train and test
    # untouch_train count = untouch_train_ratio_pct % of touch_train count
    target_untouch_train_count = int(round(len(touch_train) * (untouch_train_ratio_pct / 100.0)))
    untouch_count = len(untouch_rows)

    if target_untouch_train_count > untouch_count:
        logger.warning(
            f"Requested untouch train size ({target_untouch_train_count}) exceeds available untouch rows ({untouch_count}). "
            f"Using all {untouch_count} rows for untouch train."
        )
        target_untouch_train_count = untouch_count

    shuffled_untouch = list(untouch_rows)
    random.shuffle(shuffled_untouch)

    untouch_train = shuffled_untouch[:target_untouch_train_count]
    remaining_untouch_test = shuffled_untouch[target_untouch_train_count:]

    # Calculate untouch test count based on --untouch-test-ratio-pct if provided, else use remaining
    if untouch_test_ratio_pct is not None and untouch_test_ratio_pct > 0:
        target_untouch_test_count = int(round(len(touch_test) * (untouch_test_ratio_pct / 100.0)))
        if target_untouch_test_count > len(remaining_untouch_test):
            logger.warning(
                f"Requested untouch test size ({target_untouch_test_count}) exceeds remaining untouch rows ({len(remaining_untouch_test)}). "
                f"Using all {len(remaining_untouch_test)} remaining rows."
            )
            untouch_test = remaining_untouch_test
        else:
            logger.info(f"Sampling balanced untouch test dataset ({untouch_test_ratio_pct}% of touch_test = {target_untouch_test_count} rows)")
            untouch_test = remaining_untouch_test[:target_untouch_test_count]
    elif max_untouch_test is not None and max_untouch_test >= 0:
        if len(remaining_untouch_test) > max_untouch_test:
            logger.info(f"Capping untouch test rows from {len(remaining_untouch_test)} to max allowed: {max_untouch_test}")
            untouch_test = remaining_untouch_test[:max_untouch_test]
        else:
            untouch_test = remaining_untouch_test
    else:
        untouch_test = remaining_untouch_test

    # 3. Combine datasets
    train_rows = touch_train + untouch_train
    test_rows = touch_test + untouch_test

    # 4. Ordering (Sort by default unless shuffle is explicitly requested)
    if shuffle_output:
        logger.info("Shuffling final train and test datasets...")
        random.shuffle(train_rows)
        random.shuffle(test_rows)
    else:
        logger.info("Sorting train and test datasets by video name and frame...")
        train_rows = sort_rows(train_rows)
        test_rows = sort_rows(test_rows)

    save_csv(train_path, fieldnames, train_rows)
    save_csv(test_path, fieldnames, test_rows)

    logger.info("Train/Test dataset splitting complete:")
    logger.info(f"  Training dataset saved to '{train_path}' ({len(train_rows)} rows: {len(touch_train)} touch, {len(untouch_train)} untouch)")
    logger.info(f"  Testing dataset saved to  '{test_path}' ({len(test_rows)} rows: {len(touch_test)} touch, {len(untouch_test)} untouch)")


def main():
    parser = argparse.ArgumentParser(
        description="Create balanced training and testing datasets from touch and untouch CSV files."
    )
    parser.add_argument(
        "--touch-in", required=True, help="Input CSV path for touch dataset"
    )
    parser.add_argument(
        "--untouch-in", required=True, help="Input CSV path for untouch dataset"
    )
    parser.add_argument(
        "--train-out", required=True, help="Output CSV path for final training dataset"
    )
    parser.add_argument(
        "--test-out", required=True, help="Output CSV path for final testing dataset"
    )
    parser.add_argument(
        "--touch-test-pct",
        type=float,
        default=20.0,
        help="Percentage of touch dataset rows to allocate to testing (default: 20.0)",
    )
    parser.add_argument(
        "--untouch-train-ratio-pct",
        type=float,
        default=120.0,
        help="Ratio percentage of untouch rows to sample for training relative to touch_train count (default: 120.0)",
    )
    parser.add_argument(
        "--untouch-test-ratio-pct",
        type=float,
        default=None,
        help="Ratio percentage of untouch rows to sample for testing relative to touch_test count (e.g. 100 for equal ratio, 125 for 125%%)",
    )
    parser.add_argument(
        "--max-untouch-test",
        type=int,
        default=None,
        help="Maximum limit on number of untouch rows in the test dataset",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle final combined datasets (default behavior is to sort by video_file, video_hash, start_frame, finger_name)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Explicitly request sorting by video_file, video_hash, start_frame, finger_name (enabled by default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible splits",
    )

    args = parser.parse_args()

    # Sort is default unless --shuffle is provided
    sort_output = not args.shuffle

    split_and_balance(
        touch_in=args.touch_in,
        untouch_in=args.untouch_in,
        train_out=args.train_out,
        test_out=args.test_out,
        touch_test_pct=args.touch_test_pct,
        untouch_train_ratio_pct=args.untouch_train_ratio_pct,
        untouch_test_ratio_pct=args.untouch_test_ratio_pct,
        max_untouch_test=args.max_untouch_test,
        sort_output=sort_output,
        shuffle_output=args.shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
