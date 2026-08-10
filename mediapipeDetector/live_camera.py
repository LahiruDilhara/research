"""
live_camera.py

MediaPipe hand-landmark detection modularized pipeline running solely on live camera feed.
Includes FPS validation (>= 30 FPS, capped/paced at 30 FPS).

Functions breakdown:
1. capture_frame(cap): Captures a frame from VideoCapture.
2. analyze_landmarks(frame, landmarker, timestamp_ms): Analyzes hand landmarks and extracts hands + 3 joint coordinates per finger (with fingertip last).
3. render_frame(frame, hands_data): Renders hand skeletons, joint dots, and labels.
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
# CONSTANTS & LANDMARK MAPS
# =============================================================

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Each finger mapped to 3 landmark indices leading to the fingertip (fingertip is last index)
# Thumb: [2 (MCP), 3 (IP), 4 (TIP)]
# Index: [6 (PIP), 7 (DIP), 8 (TIP)]
# Middle: [10 (PIP), 11 (DIP), 12 (TIP)]
# Ring: [14 (PIP), 15 (DIP), 16 (TIP)]
# Pinky: [18 (PIP), 19 (DIP), 20 (TIP)]
FINGER_THREE_LANDMARKS = {
    "Thumb":  [2, 3, 4],
    "Index":  [6, 7, 8],
    "Middle": [10, 11, 12],
    "Ring":   [14, 15, 16],
    "Pinky":  [18, 19, 20],
}

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
# MODEL SETUP & FPS VALIDATION
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


def validate_camera_fps(cap):
    """Check camera FPS specification. Must be >= 30, capped at 30 if > 30."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback if spec isn't reported directly

    print(f"Camera FPS: {fps:.2f}")

    if fps < 30.0:
        print(f"Error: Camera FPS ({fps:.2f}) is less than 30 FPS.")
        return False

    if fps > 30.0:
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    return True


# =============================================================
# STEP 1: CAPTURE FRAME
# =============================================================

def capture_frame(cap):
    """Captures and returns a single frame from the camera."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


# =============================================================
# STEP 2: ANALYZE LANDMARKS
# =============================================================

def analyze_landmarks(frame, landmarker, timestamp_ms):
    """
    Analyzes the frame and returns a list of hand dictionary objects:
    [
        {
            "hand": "Left" or "Right",
            "all_pts": [(x0, y0), (x1, y1), ...],
            "fingers": {
                "Thumb": [(x_mcp, y_mcp), (x_ip, y_ip), (x_tip, y_tip)],
                "Index": [(x_pip, y_pip), (x_dip, y_dip), (x_tip, y_tip)],
                ...
            }
        },
        ...
    ]
    """
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    hands_data = []

    if not result.hand_landmarks:
        return hands_data

    for hand_idx, landmarks in enumerate(result.hand_landmarks):
        handedness = "Unknown"
        if result.handedness and len(result.handedness) > hand_idx:
            handedness = result.handedness[hand_idx][0].category_name

        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        fingers_dict = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            # List of exact 3 (x, y) coordinates with fingertip last
            finger_coords = [pts[idx] for idx in indices]
            fingers_dict[finger_name] = finger_coords

        hand_info = {
            "hand": handedness,
            "all_pts": pts,
            "fingers": fingers_dict
        }
        hands_data.append(hand_info)

    return hands_data


# =============================================================
# STEP 3: RENDER FRAME & REPORT
# =============================================================

def render_frame(frame, hands_data, frame_index):
    """
    Renders hand skeletons and finger landmarks onto the frame, prints details to console,
    and returns the annotated frame.
    """
    if not hands_data:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    for hand_idx, hand_info in enumerate(hands_data):
        handedness = hand_info["hand"]
        pts = hand_info["all_pts"]
        fingers = hand_info["fingers"]

        # Draw skeleton
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1)

        # Draw all landmark points
        for p in pts:
            cv2.circle(frame, p, 2, (150, 150, 150), -1)

        report_line = f"[frame {frame_index}] hand {hand_idx} ({handedness}):"

        # Draw finger points & labels (fingertip is last coordinate in finger list)
        for finger_name, coords in fingers.items():
            color = FINGER_COLORS[finger_name]
            
            # Highlight all 3 finger joints
            for idx, pt in enumerate(coords):
                radius = 7 if idx == 2 else 4  # Larger circle for fingertip
                cv2.circle(frame, pt, radius, color, -1)

            # Fingertip (last coordinate)
            tip_x, tip_y = coords[-1]
            cv2.putText(frame, finger_name, (tip_x + 8, tip_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            report_line += f"  {finger_name}_tip=({tip_x},{tip_y})"

        # print(report_line)

    return frame


# =============================================================
# MAIN LIVE PIPELINE
# =============================================================

def run_live_camera(camera_index=0):
    model_path = ensure_model_downloaded()
    landmarker = create_landmarker(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        landmarker.close()
        print(f"Error: Could not open camera {camera_index}.")
        sys.exit(1)

    if not validate_camera_fps(cap):
        cap.release()
        landmarker.close()
        sys.exit(1)

    target_frame_duration = 1.0 / 30.0
    frame_index = 0
    start_time = time.time()
    window_name = "Live Camera MediaPipe Finger Detector (q/ESC to quit)"

    print("Starting live camera feed. Press 'q' or 'ESC' to quit.")

    while True:
        loop_start = time.time()

        # 1. Capture Frame
        frame = capture_frame(cap)
        if frame is None:
            print("Failed to grab frame from camera.")
            break

        timestamp_ms = int((time.time() - start_time) * 1000)

        # 2. Analyze Landmarks
        hands_data = analyze_landmarks(frame, landmarker, timestamp_ms)
        print(hands_data)

        # 3. Render Frame & Report
        annotated_frame = render_frame(frame, hands_data, frame_index)

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
