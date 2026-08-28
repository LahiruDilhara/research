# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mediapipe>=1.0.0",
#     "numpy>=2.5.2",
#     "opencv-python>=5.0.0.93",
# ]
# ///

"""
datacreator/mediapipe_analyze_video.py

Video analyzer script for 12.0 FPS videos:
1. Validates that input video(s) are 12.0 FPS (skips non-12.0 FPS files).
2. Calculates SHA-256 fingerprint hash of each video file.
3. Uses MediaPipe HandLandmarker to extract raw landmarks on every frame.
4. Uses reusable hand_selection logic (datacreator.hand_selection) prioritizing 'Right' hand over 'Left' hand.
5. Annotates detected hand label ('Right' / 'Left') and exports frame-by-frame landmarks to a CSV file.

Supports processing single video files, glob patterns (e.g. '*', 'videos/*', '*.mp4'), or directory paths.
CSV files are saved in the same directory as the source video file (or an optional output directory override)
and automatically overwrite existing CSV files of the same name.
"""

import argparse
import csv
import glob
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# Add project root to sys.path for robust module imports when running standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from datacreator.hand_selection import select_single_hand
except ModuleNotFoundError:
    from hand_selection import select_single_hand

import cv2
import mediapipe as mp

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

TARGET_FPS = 12.0
FPS_TOLERANCE = 0.5  # Allowed deviation for 12 FPS check (11.5 to 12.5 FPS)
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

ALL_21_LANDMARK_NAMES = [
    "wrist",         # 0
    "thumb_cmc",     # 1
    "thumb_mcp",     # 2
    "thumb_ip",      # 3
    "thumb_tip",     # 4
    "index_mcp",     # 5
    "index_pip",     # 6
    "index_dip",     # 7
    "index_tip",     # 8
    "middle_mcp",    # 9
    "middle_pip",    # 10
    "middle_dip",    # 11
    "middle_tip",    # 12
    "ring_mcp",      # 13
    "ring_pip",      # 14
    "ring_dip",      # 15
    "ring_tip",      # 16
    "pinky_mcp",     # 17
    "pinky_pip",     # 18
    "pinky_dip",     # 19
    "pinky_tip",     # 20
]
WRIST_INDEX = 0


def ensure_model_downloaded() -> str:
    """Downloads the hand_landmarker.task model file if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Info] Downloading model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Info] Model download complete.")
    return MODEL_PATH


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_hand_scale(pts_px: list) -> float:
    """Calculates hand scale L_hand = sqrt(palm_length^2 + palm_width^2) from pixel points."""
    w_x, w_y = pts_px[WRIST_INDEX]
    m_x, m_y = pts_px[9]
    i_x, i_y = pts_px[5]
    p_x, p_y = pts_px[17]

    palm_length = ((m_x - w_x) ** 2 + (m_y - w_y) ** 2) ** 0.5
    palm_width = ((p_x - i_x) ** 2 + (p_y - i_y) ** 2) ** 0.5
    l_hand = (palm_length ** 2 + palm_width ** 2) ** 0.5
    return l_hand if l_hand > 0 else 1.0


def build_csv_headers() -> list[str]:
    """Builds CSV headers for video metadata and all 21 raw unfiltered landmark features."""
    headers = [
        "video_file", "video_hash", "video_width", "video_height",
        "video_fps", "total_video_frames", "video_duration_sec",
        "frame_idx", "timestamp_ms", "hand"
    ]
    for lm_name in ALL_21_LANDMARK_NAMES:
        headers.append(f"{lm_name}_x")
        headers.append(f"{lm_name}_y")
    return headers


def analyze_video(input_path: str, output_dir: str | None = None) -> str | None:
    """
    Validates strictly 12.0 FPS, computes SHA-256 hash, runs MediaPipe frame-by-frame
    with Right-hand prioritization, and outputs 100% raw (unfiltered) landmarks CSV
    with full video metadata in the video's directory (or optional output_dir). Overwrites existing CSV files.
    Returns the output CSV path on success, or None if skipped.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    if os.path.getsize(input_path) == 0:
        print(f"[Warning] Video file '{input_path}' is empty (0 bytes). Skipping.")
        return None

    # 1. Compute SHA-256 Hash and automatic output path in video's directory (or output_dir)
    video_hash_full = compute_file_hash(input_path)
    video_hash_short = video_hash_full[:16]
    abs_input_path = os.path.abspath(input_path)
    video_dir = os.path.dirname(abs_input_path)
    video_basename = os.path.basename(abs_input_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]

    target_dir = output_dir if output_dir else video_dir
    os.makedirs(target_dir, exist_ok=True)
    output_csv_path = os.path.join(target_dir, f"{video_name_no_ext}.raw_landmarks.{video_hash_short}.csv")

    print("=== Analyzing Video File ===")
    print(f" Input Video File : {input_path}")
    print(f" SHA-256 Hash     : {video_hash_full}")
    print(f" Output CSV Path  : {output_csv_path}")

    # 2. Strict 12.0 FPS Validation
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Warning] Cannot open video file '{input_path}' (file may be corrupted or unsupported format). Skipping.")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    header_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    all_raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        all_raw_frames.append((frame, msec))

    cap.release()

    if not all_raw_frames:
        print(f"[Warning] Video file '{input_path}' has no readable frames. Skipping.")
        return None

    total_frames = len(all_raw_frames)
    first_msec = all_raw_frames[0][1]
    last_msec = all_raw_frames[-1][1]
    duration_sec = (last_msec - first_msec) / 1000.0 if last_msec > first_msec else (total_frames / (header_fps or 12.0))
    actual_fps = total_frames / duration_sec if duration_sec > 0 else header_fps

    print(f" Video Resolution : {w}x{h}")
    print(f" Total Frames     : {total_frames} frames")
    print(f" Video Duration   : {duration_sec:.3f} seconds")
    print(f" Measured FPS     : {actual_fps:.2f} FPS")

    if abs(actual_fps - TARGET_FPS) > FPS_TOLERANCE:
        msg = (f"[Warning] Input video '{input_path}' is NOT 12.0 FPS! "
               f"Measured FPS: {actual_fps:.2f} (Header: {header_fps:.2f}). "
               f"Skipping. (Please run resample_12fps.py first).")
        print(f"\n{msg}\n")
        return None

    print(f"[Check Passed] Video is strictly 12.0 FPS (Measured: {actual_fps:.2f} FPS).")

    # 3. MediaPipe Landmarker Setup
    model_path = ensure_model_downloaded()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # 4. Extract Per-Frame Landmark Features (100% RAW UNFILTERED)
    csv_rows = []
    headers = build_csv_headers()

    for frame_idx, (frame, ts_ms) in enumerate(all_raw_frames):
        timestamp_int = int(ts_ms)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_int)

        raw_detected_hands = []
        if result.hand_landmarks and result.handedness:
            for idx, landmarks in enumerate(result.hand_landmarks):
                hand_label = result.handedness[idx][0].category_name  # "Right" or "Left"
                pts_px = [(lm.x * w, lm.y * h) for lm in landmarks]
                pts_raw = [(lm.x, lm.y) for lm in landmarks]
                l_hand = calculate_hand_scale(pts_px)
                raw_detected_hands.append({
                    "hand": hand_label,
                    "all_pts_raw": pts_raw,
                    "all_pts_px": pts_px,
                    "l_hand": l_hand
                })

        # Apply shared single-hand selection (Prioritizing "Right" hand over "Left" hand)
        selected_hand = select_single_hand(raw_detected_hands, primary_preference="Right")

        row = {
            "video_file": video_basename,
            "video_hash": video_hash_full,
            "video_width": w,
            "video_height": h,
            "video_fps": round(actual_fps, 2),
            "total_video_frames": total_frames,
            "video_duration_sec": round(duration_sec, 3),
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_int,
        }

        if selected_hand:
            hd = selected_hand[0]
            hand_label = hd["hand"]
            pts_raw = hd["all_pts_raw"]

            row["hand"] = hand_label

            for lm_idx, lm_name in enumerate(ALL_21_LANDMARK_NAMES):
                rx, ry = pts_raw[lm_idx]
                row[f"{lm_name}_x"] = round(rx, 6)
                row[f"{lm_name}_y"] = round(ry, 6)
        else:
            row["hand"] = "None"
            for lm_name in ALL_21_LANDMARK_NAMES:
                row[f"{lm_name}_x"] = 0.0
                row[f"{lm_name}_y"] = 0.0

        csv_rows.append(row)

    landmarker.close()

    # 5. Write to CSV File (Overwriting if file exists)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[Success] Extracted {len(csv_rows)} frame records.")
    print(f"          Saved raw landmark CSV to: {output_csv_path}")
    return output_csv_path


def collect_input_files(input_patterns: list[str]) -> list[str]:
    """
    Expands glob patterns or directory paths into a list of file paths.
    Only includes existing non-empty video files.
    """
    matched_files = []
    for pattern in input_patterns:
        search_pattern = os.path.join(pattern, "*") if os.path.isdir(pattern) else pattern
        glob_matches = glob.glob(search_pattern, recursive=True)
        if glob_matches:
            for filepath in sorted(glob_matches):
                ext = os.path.splitext(filepath)[1].lower()
                if os.path.isfile(filepath) and ext in VIDEO_EXTENSIONS and filepath not in matched_files:
                    matched_files.append(filepath)
        elif os.path.isfile(pattern) and pattern not in matched_files:
            ext = os.path.splitext(pattern)[1].lower()
            if ext in VIDEO_EXTENSIONS or not ext:
                matched_files.append(pattern)
        else:
            print(f"[Warning] No files found for input pattern/path: '{pattern}'")

    return matched_files


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 12.0 FPS Video(s) & Export Raw Hand Landmarks CSV"
    )
    parser.add_argument(
        "pos_args",
        nargs="*",
        help="Input video file(s), glob pattern(s), or directory path(s)"
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        default=None,
        help="Input video file path(s) or glob pattern(s) (e.g. video.mp4, '*.mp4', 'videos/*')"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output directory path (defaults to each video's own directory)"
    )

    args = parser.parse_args()

    input_patterns = []
    if args.input:
        input_patterns.extend(args.input)
    if args.pos_args:
        input_patterns.extend(args.pos_args)

    if not input_patterns:
        parser.print_help()
        sys.exit(1)

    input_files = collect_input_files(input_patterns)
    if not input_files:
        print("[Error] No valid input video files found. Exiting.")
        sys.exit(1)

    output_dir = args.output
    if output_dir:
        if os.path.exists(output_dir) and not os.path.isdir(output_dir):
            print(f"[Error] Output path '{output_dir}' exists but is not a directory.")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(input_files)} video file(s) to analyze:")
    for f in input_files:
        print(f"  - {f}")
    if output_dir:
        print(f"Output directory override: {output_dir}")
    print()

    success_count = 0
    skipped_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] Analyzing: {input_file}")
        try:
            csv_path = analyze_video(input_file, output_dir=output_dir)
            if csv_path:
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"[Failed] Could not analyze '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Analysis Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
