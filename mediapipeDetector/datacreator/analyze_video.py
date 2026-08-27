"""
datacreator/analyze_video.py

Video analyzer script for 12.0 FPS videos:
1. Strictly validates that the input video is 12.0 FPS (aborts if not).
2. Calculates SHA-256 fingerprint hash of the video file.
3. Uses MediaPipe HandLandmarker to extract raw landmarks on every frame.
4. Uses reusable hand_selection logic (datacreator.hand_selection) prioritizing 'Right' hand over 'Left' hand.
5. Annotates detected hand label ('Right' / 'Left') and exports frame-by-frame landmarks to a CSV file.
"""

import argparse
import csv
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


def analyze_video(input_path: str) -> str:
    """
    Validates strictly 12.0 FPS, computes SHA-256 hash, runs MediaPipe frame-by-frame
    with Right-hand prioritization, and outputs 100% raw (unfiltered) landmarks CSV
    with full video metadata in the video's directory. Overwrites the file if it already exists.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    # 1. Compute SHA-256 Hash and automatic output path in same directory
    video_hash_full = compute_file_hash(input_path)
    video_hash_short = video_hash_full[:16]
    abs_input_path = os.path.abspath(input_path)
    video_dir = os.path.dirname(abs_input_path)
    video_basename = os.path.basename(abs_input_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]

    output_csv_path = os.path.join(video_dir, f"{video_name_no_ext}.raw_landmarks.{video_hash_short}.csv")

    print("=== Analyzing Video File ===")
    print(f" Input Video File : {input_path}")
    print(f" SHA-256 Hash     : {video_hash_full}")
    print(f" Output CSV Path  : {output_csv_path}")

    # 2. Strict 12.0 FPS Validation
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {input_path}")

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
        raise ValueError(f"Video file is empty: {input_path}")

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
        msg = (f"[STRICT ERROR] Input video is NOT 12.0 FPS! "
               f"Measured FPS: {actual_fps:.2f} (Header: {header_fps:.2f}). "
               f"Please run 'python3 datacreator/resample_12fps.py -i {input_path} -o <output_path>' first.")
        print(f"\n{msg}\n")
        raise ValueError(msg)

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
                l_hand = calculate_hand_scale(pts_px)
                pts_norm = [(px / l_hand, py / l_hand) for px, py in pts_px]
                raw_detected_hands.append({
                    "hand": hand_label,
                    "all_pts_norm": pts_norm,
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
            pts_norm = hd["all_pts_norm"]

            row["hand"] = hand_label

            for lm_idx, lm_name in enumerate(ALL_21_LANDMARK_NAMES):
                fx, fy = pts_norm[lm_idx]
                row[f"{lm_name}_x"] = round(fx, 4)
                row[f"{lm_name}_y"] = round(fy, 4)
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


def main():
    parser = argparse.ArgumentParser(description="Analyze 12.0 FPS Video & Export Raw Hand Landmarks CSV")
    parser.add_argument("-i", "--input", required=True, help="Input 12 FPS video file path")

    args = parser.parse_args()
    try:
        analyze_video(args.input)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
