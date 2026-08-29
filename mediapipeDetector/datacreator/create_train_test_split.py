# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
# ]
# ///

"""
datacreator/create_train_test_split.py

Creates balanced training and testing datasets from touch and untouch CSV files.

Supports CLI flags:
- --touch-in: Path to touch CSV dataset
- --untouch-in: Path to untouch CSV dataset
- --train-out: Output path for final training dataset CSV
- --test-out: Output path for final testing dataset CSV
- --touch-test-pct: Percentage of touch dataset allocated for testing (default: 20.0)
- --untouch-train-ratio-pct: Ratio of untouch rows to sample for training relative to touch_train (default: 120.0)
- --untouch-test-ratio-pct: Ratio of untouch rows to sample for testing relative to touch_test
- --max-untouch-test: Maximum limit on untouch rows in testing dataset
- --no-video-leak: Strict video-level partitioning (no video shared between train and test)
- --shuffle: Shuffle final output rows
- --sort: Sort by video_file, video_hash, start_frame, finger_name
- --seed: Random seed for reproducible splits

Overwrites target output files if they already exist.
Prints rich analytics reports for train and test splits.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def load_csv(file_path: str) -> tuple[list[str], list[dict]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def save_csv(file_path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if os.path.exists(file_path):
        print(f"[Info] Overwriting existing file: {file_path}")
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"[Warning] Could not remove existing file '{file_path}': {e}")

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


def group_by_video(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        v_key = f"{r.get('video_file', 'unknown')}___{r.get('video_hash', 'unknown')}"
        groups.setdefault(v_key, []).append(r)
    return groups


def print_dataset_analytics(dataset_name: str, rows: list[dict], touch_rows: list[dict], untouch_rows: list[dict]) -> None:
    total_cnt = len(rows)
    t_cnt = len(touch_rows)
    u_cnt = len(untouch_rows)
    t_pct = (t_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0
    u_pct = (u_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0

    print(f"  {dataset_name} Analytics Summary:")
    print(f"    - Total Rows   : {total_cnt}")
    print(f"    - Touch Rows   : {t_cnt} ({t_pct:.2f}%)")
    print(f"    - Untouch Rows : {u_cnt} ({u_pct:.2f}%)")

    # Per finger breakdown
    t_fg_cnts = {fg: 0 for fg in FINGERS}
    u_fg_cnts = {fg: 0 for fg in FINGERS}

    for r in touch_rows:
        fg = r.get("finger_name", "").lower()
        if fg in t_fg_cnts:
            t_fg_cnts[fg] += 1

    for r in untouch_rows:
        fg = r.get("finger_name", "").lower()
        if fg in u_fg_cnts:
            u_fg_cnts[fg] += 1

    print(f"    - Finger Touches  : " + ", ".join([f"{fg.capitalize()}:{t_fg_cnts[fg]}" for fg in FINGERS]))
    print(f"    - Finger Untouches: " + ", ".join([f"{fg.capitalize()}:{u_fg_cnts[fg]}" for fg in FINGERS]))


def create_train_test_split(
    touch_in: str,
    untouch_in: str,
    train_out: str,
    test_out: str,
    touch_test_pct: float = 20.0,
    untouch_train_ratio_pct: float = 120.0,
    untouch_test_ratio_pct: float | None = None,
    max_untouch_test: int | None = None,
    no_video_leak: bool = False,
    sort_output: bool = True,
    shuffle_output: bool = False,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        print(f"[Info] Random seed set to: {seed}")

    print(f"[1/4] Loading touch and untouch datasets...")
    touch_fields, touch_rows = load_csv(touch_in)
    untouch_fields, untouch_rows = load_csv(untouch_in)

    if touch_fields != untouch_fields:
        print("[Warning] Field names differ between touch and untouch CSV files! Using touch CSV schema.")

    fieldnames = touch_fields
    print(f"      Loaded {len(touch_rows)} touch rows and {len(untouch_rows)} untouch rows.")

    if no_video_leak:
        print("[2/4] Enforcing video-level partitioning (--no-video-leak enabled)...")
        touch_vgroups = group_by_video(touch_rows)
        v_keys = list(touch_vgroups.keys())
        random.shuffle(v_keys)

        total_touch = len(touch_rows)
        target_touch_test = int(round(total_touch * (touch_test_pct / 100.0)))

        touch_test_vkeys = set()
        current_test_cnt = 0

        for vk in v_keys:
            vk_cnt = len(touch_vgroups[vk])
            if current_test_cnt == 0 or abs((current_test_cnt + vk_cnt) - target_touch_test) < abs(current_test_cnt - target_touch_test):
                touch_test_vkeys.add(vk)
                current_test_cnt += vk_cnt
            else:
                break

        touch_test = []
        touch_train = []
        for vk, rows in touch_vgroups.items():
            if vk in touch_test_vkeys:
                touch_test.extend(rows)
            else:
                touch_train.extend(rows)

        print(f"      Video-partitioned touch split ({len(touch_test_vkeys)} test videos, {len(touch_vgroups) - len(touch_test_vkeys)} train videos): {len(touch_train)} train, {len(touch_test)} test.")

        untouch_vgroups = group_by_video(untouch_rows)
        untouch_test_candidate = []
        untouch_train_candidate = []

        for vk, rows in untouch_vgroups.items():
            if vk in touch_test_vkeys:
                untouch_test_candidate.extend(rows)
            else:
                untouch_train_candidate.extend(rows)

        remaining_vkeys = set(untouch_vgroups.keys()) - set(touch_vgroups.keys())
        for vk in remaining_vkeys:
            untouch_train_candidate.extend(untouch_vgroups[vk])

        target_untouch_train_count = int(round(len(touch_train) * (untouch_train_ratio_pct / 100.0)))
        random.shuffle(untouch_train_candidate)
        untouch_train = untouch_train_candidate[:target_untouch_train_count]

        random.shuffle(untouch_test_candidate)
        if untouch_test_ratio_pct is not None and untouch_test_ratio_pct > 0:
            target_untouch_test_count = int(round(len(touch_test) * (untouch_test_ratio_pct / 100.0)))
            untouch_test = untouch_test_candidate[:target_untouch_test_count]
        elif max_untouch_test is not None and max_untouch_test >= 0:
            untouch_test = untouch_test_candidate[:max_untouch_test]
        else:
            untouch_test = untouch_test_candidate

    else:
        print("[2/4] Performing row-level randomized sampling...")
        touch_count = len(touch_rows)
        touch_test_count = int(round(touch_count * (touch_test_pct / 100.0)))
        touch_test_count = max(0, min(touch_count, touch_test_count))

        shuffled_touch = list(touch_rows)
        random.shuffle(shuffled_touch)

        touch_test = shuffled_touch[:touch_test_count]
        touch_train = shuffled_touch[touch_test_count:]

        print(f"      Row-sampled touch split ({touch_test_pct}% test): {len(touch_train)} train, {len(touch_test)} test.")

        target_untouch_train_count = int(round(len(touch_train) * (untouch_train_ratio_pct / 100.0)))
        untouch_count = len(untouch_rows)

        if target_untouch_train_count > untouch_count:
            print(f"      [Warning] Requested untouch train size ({target_untouch_train_count}) exceeds available untouch rows ({untouch_count}). Using all available.")
            target_untouch_train_count = untouch_count

        shuffled_untouch = list(untouch_rows)
        random.shuffle(shuffled_untouch)

        untouch_train = shuffled_untouch[:target_untouch_train_count]
        remaining_untouch_test = shuffled_untouch[target_untouch_train_count:]

        if untouch_test_ratio_pct is not None and untouch_test_ratio_pct > 0:
            target_untouch_test_count = int(round(len(touch_test) * (untouch_test_ratio_pct / 100.0)))
            if target_untouch_test_count > len(remaining_untouch_test):
                print(f"      [Warning] Requested untouch test size ({target_untouch_test_count}) exceeds remaining untouch rows ({len(remaining_untouch_test)}).")
                untouch_test = remaining_untouch_test
            else:
                untouch_test = remaining_untouch_test[:target_untouch_test_count]
        elif max_untouch_test is not None and max_untouch_test >= 0:
            if len(remaining_untouch_test) > max_untouch_test:
                print(f"      [Info] Capping untouch test rows to max: {max_untouch_test}")
                untouch_test = remaining_untouch_test[:max_untouch_test]
            else:
                untouch_test = remaining_untouch_test
        else:
            untouch_test = remaining_untouch_test

    print("[3/4] Combining and ordering train and test datasets...")
    train_rows = touch_train + untouch_train
    test_rows = touch_test + untouch_test

    if shuffle_output:
        print("      Shuffling final train and test datasets...")
        random.shuffle(train_rows)
        random.shuffle(test_rows)
    else:
        print("      Sorting final train and test datasets by video_file, video_hash, start_frame, finger_name...")
        train_rows = sort_rows(train_rows)
        test_rows = sort_rows(test_rows)

    print(f"\n==========================================")
    print(f"   TRAIN / TEST DATASET ANALYTICS REPORT")
    print(f"==========================================")
    print_dataset_analytics("Training Dataset", train_rows, touch_train, untouch_train)
    print()
    print_dataset_analytics("Testing Dataset ", test_rows, touch_test, untouch_test)
    print(f"==========================================\n")

    print("[4/4] Saving final training and testing datasets...")
    save_csv(train_out, fieldnames, train_rows)
    save_csv(test_out, fieldnames, test_rows)

    print(f"[Success] Train/Test datasets saved to:\n  - {train_out}\n  - {test_out}")

    # Save summary JSON for pipeline audit
    try:
        from summary_utils import save_step_summary
        save_step_summary("step_12_train_test_split.json", {
            "step": 12,
            "name": "create_train_test_split",
            "train_records": len(train_rows),
            "test_records": len(test_rows),
            "train_touch": len(touch_train),
            "train_untouch": len(untouch_train),
            "test_touch": len(touch_test),
            "test_untouch": len(untouch_test),
            "no_video_leak": no_video_leak
        })
    except Exception as e:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Create balanced training and testing datasets from touch and untouch CSV files"
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
        help="Ratio percentage of untouch rows to sample for testing relative to touch_test count",
    )
    parser.add_argument(
        "--max-untouch-test",
        type=int,
        default=None,
        help="Maximum limit on number of untouch rows in the test dataset",
    )
    parser.add_argument(
        "--no-video-leak",
        action="store_true",
        help="Ensure no video file/hash source is shared between train and test datasets to prevent video frame data leakage",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle final combined datasets",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort final combined datasets by video_file, video_hash, start_frame, finger_name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible splits",
    )

    args = parser.parse_args()

    sort_output = not args.shuffle

    create_train_test_split(
        touch_in=args.touch_in,
        untouch_in=args.untouch_in,
        train_out=args.train_out,
        test_out=args.test_out,
        touch_test_pct=args.touch_test_pct,
        untouch_train_ratio_pct=args.untouch_train_ratio_pct,
        untouch_test_ratio_pct=args.untouch_test_ratio_pct,
        max_untouch_test=args.max_untouch_test,
        no_video_leak=args.no_video_leak,
        sort_output=sort_output,
        shuffle_output=args.shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
