#!/usr/bin/env python3
"""
merge_csvs.py — Merges multiple dataset CSV files into a single consolidated CSV file.

Usage:
    python3 merge_csvs.py -i file1.csv file2.csv ... -o merged.csv
    python3 merge_csvs.py -i ./videos/*.csv -o ./analysis/merged_dataset.csv
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
logger = logging.getLogger("MergeCSVs")

EXPECTED_COLUMNS_STR = "video_file,video_hash,duration_ms,start_ms,end_ms,start_frame,end_frame,wrist1_x,wrist1_y,thumb1_mcp_x,thumb1_mcp_y,thumb1_pip_x,thumb1_pip_y,thumb1_dip_x,thumb1_dip_y,index1_mcp_x,index1_mcp_y,index1_pip_x,index1_pip_y,index1_dip_x,index1_dip_y,middle1_mcp_x,middle1_mcp_y,middle1_pip_x,middle1_pip_y,middle1_dip_x,middle1_dip_y,ring1_mcp_x,ring1_mcp_y,ring1_pip_x,ring1_pip_y,ring1_dip_x,ring1_dip_y,pinky1_mcp_x,pinky1_mcp_y,pinky1_pip_x,pinky1_pip_y,pinky1_dip_x,pinky1_dip_y,wrist2_x,wrist2_y,thumb2_mcp_x,thumb2_mcp_y,thumb2_pip_x,thumb2_pip_y,thumb2_dip_x,thumb2_dip_y,index2_mcp_x,index2_mcp_y,index2_pip_x,index2_pip_y,index2_dip_x,index2_dip_y,middle2_mcp_x,middle2_mcp_y,middle2_pip_x,middle2_pip_y,middle2_dip_x,middle2_dip_y,ring2_mcp_x,ring2_mcp_y,ring2_pip_x,ring2_pip_y,ring2_dip_x,ring2_dip_y,pinky2_mcp_x,pinky2_mcp_y,pinky2_pip_x,pinky2_pip_y,pinky2_dip_x,pinky2_dip_y,wrist3_x,wrist3_y,thumb3_mcp_x,thumb3_mcp_y,thumb3_pip_x,thumb3_pip_y,thumb3_dip_x,thumb3_dip_y,index3_mcp_x,index3_mcp_y,index3_pip_x,index3_pip_y,index3_dip_x,index3_dip_y,middle3_mcp_x,middle3_mcp_y,middle3_pip_x,middle3_pip_y,middle3_dip_x,middle3_dip_y,ring3_mcp_x,ring3_mcp_y,ring3_pip_x,ring3_pip_y,ring3_dip_x,ring3_dip_y,pinky3_mcp_x,pinky3_mcp_y,pinky3_pip_x,pinky3_pip_y,pinky3_dip_x,pinky3_dip_y,wrist4_x,wrist4_y,thumb4_mcp_x,thumb4_mcp_y,thumb4_pip_x,thumb4_pip_y,thumb4_dip_x,thumb4_dip_y,index4_mcp_x,index4_mcp_y,index4_pip_x,index4_pip_y,index4_dip_x,index4_dip_y,middle4_mcp_x,middle4_mcp_y,middle4_pip_x,middle4_pip_y,middle4_dip_x,middle4_dip_y,ring4_mcp_x,ring4_mcp_y,ring4_pip_x,ring4_pip_y,ring4_dip_x,ring4_dip_y,pinky4_mcp_x,pinky4_mcp_y,pinky4_pip_x,pinky4_pip_y,pinky4_dip_x,pinky4_dip_y,wrist5_x,wrist5_y,thumb5_mcp_x,thumb5_mcp_y,thumb5_pip_x,thumb5_pip_y,thumb5_dip_x,thumb5_dip_y,index5_mcp_x,index5_mcp_y,index5_pip_x,index5_pip_y,index5_dip_x,index5_dip_y,middle5_mcp_x,middle5_mcp_y,middle5_pip_x,middle5_pip_y,middle5_dip_x,middle5_dip_y,ring5_mcp_x,ring5_mcp_y,ring5_pip_x,ring5_pip_y,ring5_dip_x,ring5_dip_y,pinky5_mcp_x,pinky5_mcp_y,pinky5_pip_x,pinky5_pip_y,pinky5_dip_x,pinky5_dip_y,wrist1_vx,wrist1_vy,thumb1_mcp_vx,thumb1_mcp_vy,thumb1_pip_vx,thumb1_pip_vy,thumb1_dip_vx,thumb1_dip_vy,index1_mcp_vx,index1_mcp_vy,index1_pip_vx,index1_pip_vy,index1_dip_vx,index1_dip_vy,middle1_mcp_vx,middle1_mcp_vy,middle1_pip_vx,middle1_pip_vy,middle1_dip_vx,middle1_dip_vy,ring1_mcp_vx,ring1_mcp_vy,ring1_pip_vx,ring1_pip_vy,ring1_dip_vx,ring1_dip_vy,pinky1_mcp_vx,pinky1_mcp_vy,pinky1_pip_vx,pinky1_pip_vy,pinky1_dip_vx,pinky1_dip_vy,wrist2_vx,wrist2_vy,thumb2_mcp_vx,thumb2_mcp_vy,thumb2_pip_vx,thumb2_pip_vy,thumb2_dip_vx,thumb2_dip_vy,index2_mcp_vx,index2_mcp_vy,index2_pip_vx,index2_pip_vy,index2_dip_vx,index2_dip_vy,middle2_mcp_vx,middle2_mcp_vy,middle2_pip_vx,middle2_pip_vy,middle2_dip_vx,middle2_dip_vy,ring2_mcp_vx,ring2_mcp_vy,ring2_pip_vx,ring2_pip_vy,ring2_dip_vx,ring2_dip_vy,pinky2_mcp_vx,pinky2_mcp_vy,pinky2_pip_vx,pinky2_pip_vy,pinky2_dip_vx,pinky2_dip_vy,wrist3_vx,wrist3_vy,thumb3_mcp_vx,thumb3_mcp_vy,thumb3_pip_vx,thumb3_pip_vy,thumb3_dip_vx,thumb3_dip_vy,index3_mcp_vx,index3_mcp_vy,index3_pip_vx,index3_pip_vy,index3_dip_vx,index3_dip_vy,middle3_mcp_vx,middle3_mcp_vy,middle3_pip_vx,middle3_pip_vy,middle3_dip_vx,middle3_dip_vy,ring3_mcp_vx,ring3_mcp_vy,ring3_pip_vx,ring3_pip_vy,ring3_dip_vx,ring3_dip_vy,pinky3_mcp_vx,pinky3_mcp_vy,pinky3_pip_vx,pinky3_pip_vy,pinky3_dip_vx,pinky3_dip_vy,wrist4_vx,wrist4_vy,thumb4_mcp_vx,thumb4_mcp_vy,thumb4_pip_vx,thumb4_pip_vy,thumb4_dip_vx,thumb4_dip_vy,index4_mcp_vx,index4_mcp_vy,index4_pip_vx,index4_pip_vy,index4_dip_vx,index4_dip_vy,middle4_mcp_vx,middle4_mcp_vy,middle4_pip_vx,middle4_pip_vy,middle4_dip_vx,middle4_dip_vy,ring4_mcp_vx,ring4_mcp_vy,ring4_pip_vx,ring4_pip_vy,ring4_dip_vx,ring4_dip_vy,pinky4_mcp_vx,pinky4_mcp_vy,pinky4_pip_vx,pinky4_pip_vy,pinky4_dip_vx,pinky4_dip_vy,thumb_touch,index_touch,middle_touch,ring_touch,pinky_touch,hand_move,hand_point_of_view,hand_closer,hovering,daylight,hand_visible,out_of_sync,rightHand,any_difference"
EXPECTED_COLUMNS = EXPECTED_COLUMNS_STR.split(",")

def merge_csv_files(input_paths: list[str], output_path: str) -> None:
    """
    Validates CSV headers against explicit schema and concatenates them into output_path.
    """
    if not input_paths:
        logger.error("No input CSV files provided.")
        sys.exit(1)

    resolved_inputs: list[Path] = []
    for path_str in input_paths:
        p = Path(path_str).resolve()
        if not p.exists():
            logger.error(f"Input file does not exist: {p}")
            sys.exit(1)
        if not p.is_file():
            logger.error(f"Input path is not a file: {p}")
            sys.exit(1)
        resolved_inputs.append(p)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Validating {len(resolved_inputs)} CSV files against target schema ({len(EXPECTED_COLUMNS)} columns)...")

    total_rows = 0

    with open(out_file, mode="w", newline="", encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(EXPECTED_COLUMNS)

        for idx, csv_path in enumerate(resolved_inputs, start=1):
            try:
                with open(csv_path, mode="r", newline="", encoding="utf-8") as in_csv:
                    reader = csv.reader(in_csv)
                    header = next(reader, None)

                    if header is None:
                        logger.warning(f"File '{csv_path.name}' is empty. Skipping.")
                        continue

                    if header != EXPECTED_COLUMNS:
                        logger.error(f"Header mismatch in file: {csv_path}")
                        diff_missing = set(EXPECTED_COLUMNS) - set(header)
                        diff_extra = set(header) - set(EXPECTED_COLUMNS)
                        if diff_missing:
                            logger.error(f"Missing columns ({len(diff_missing)}): {diff_missing}")
                        if diff_extra:
                            logger.error(f"Unexpected extra columns ({len(diff_extra)}): {diff_extra}")
                        sys.exit(1)

                    file_rows = 0
                    for row in reader:
                        writer.writerow(row)
                        file_rows += 1

                    total_rows += file_rows
                    logger.info(f"  [{idx}/{len(resolved_inputs)}] Validated & merged '{csv_path.name}' ({file_rows} rows)")

            except Exception as e:
                logger.error(f"Failed to process CSV file {csv_path}: {e}")
                sys.exit(1)

    logger.info(f"Successfully merged {len(resolved_inputs)} files into '{out_file}' (Total rows: {total_rows}).")

def main():
    parser = argparse.ArgumentParser(
        description="Validate and merge multiple CSV files with identical schemas."
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="List of input CSV files to merge"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path for the output merged CSV file"
    )

    args = parser.parse_args()
    merge_csv_files(args.input, args.output)

if __name__ == "__main__":
    main()
