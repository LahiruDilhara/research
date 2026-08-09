"""
live_camera.py

MediaPipe hand-landmark detection script running solely on live camera feed.
Includes camera FPS validation (requires >= 30 FPS, caps to 30 FPS if > 30 FPS).
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

# =============================================================
# CONSTANTS
# =============================================================

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

FINGER_LANDMARK_GROUPS = {
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Pinky":  [17, 18, 19, 20],
}
FINGERTIP_INDEX = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
WRIST_INDEX = 0

FINGER_COLORS = {
    "Thumb":  (255, 140, 0),
    "Index":  (0, 200, 255),
    "Middle": (0, 255, 0),
    "Ring":   (255, 0, 255),
    "Pinky":  (0, 0, 255),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


# =============================================================
# MODEL SETUP
# =============================================================

def ensure_model_downloaded(path=MODEL_PATH, url=MODEL_URL):
    """Download the hand_landmarker.task model if it isn't present locally."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path

    print(f"Model not found locally — downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the MediaPipe hand landmarker model "
            f"automatically ({e}). Download it manually from:\n  {url}\n"
            f"and place it at:\n  {os.path.abspath(path)}"
        )
    return path


def create_landmarker(model_path, num_hands=2):
    """Create a HandLandmarker in VIDEO running mode for stream processing."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


# =============================================================
# DRAWING & REPORTING
# =============================================================

def draw_and_report(frame, result, frame_index):
    """Draw skeleton + labeled fingertips onto the frame and print coordinates."""
    h, w = frame.shape[:2]

    if not result.hand_landmarks:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    for hand_idx, landmarks in enumerate(result.hand_landmarks):
        handedness = "Unknown"
        if result.handedness and len(result.handedness) > hand_idx:
            handedness = result.handedness[hand_idx][0].category_name

        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Skeleton
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1)

        # All landmarks
        for p in pts:
            cv2.circle(frame, p, 2, (150, 150, 150), -1)

        # Fingertips
        report_line = f"[frame {frame_index}] hand {hand_idx} ({handedness}):"
        for finger_name, tip_idx in FINGERTIP_INDEX.items():
            x, y = pts[tip_idx]
            color = FINGER_COLORS[finger_name]
            cv2.circle(frame, (x, y), 7, color, -1)
            cv2.putText(frame, finger_name, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            report_line += f"  {finger_name}=({x},{y})"

        print(report_line)

    return frame


# =============================================================
# MAIN LIVE CAMERA LOOP
# =============================================================

def run_live_camera(camera_index=0):
    model_path = ensure_model_downloaded()
    landmarker = create_landmarker(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        landmarker.close()
        print(f"Error: Could not open camera {camera_index}.")
        sys.exit(1)

    # Read and validate FPS spec
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback if camera spec isn't reported directly

    print(f"Camera FPS: {fps:.2f}")

    if fps < 30.0:
        print(f"Error: Camera FPS ({fps:.2f}) is less than 30 FPS.")
        cap.release()
        landmarker.close()
        sys.exit(1)

    if fps > 30.0:
        cap.set(cv2.CAP_PROP_FPS, 30.0)
        fps = 30.0

    target_frame_duration = 1.0 / 30.0
    frame_index = 0
    start_time = time.time()
    window_name = "Live Camera MediaPipe Finger Detector (q/ESC to quit)"

    print("Starting live camera feed. Press 'q' or 'ESC' to quit.")

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        timestamp_ms = int((time.time() - start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated_frame = draw_and_report(frame, result, frame_index)
        cv2.imshow(window_name, annotated_frame)

        frame_index += 1

        # Enforce 30 FPS timing pacing
        elapsed = time.time() - loop_start
        wait_time = target_frame_duration - elapsed
        if wait_time > 0:
            time.sleep(wait_time)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera MediaPipe Hand Landmarker")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    run_live_camera(camera_index=args.camera)
