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

from one_euro_filter import FingertipFilter

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
WRIST_INDEX = 0

# 1€ Filter Smoothing Constants (adjust to tune smoothing intensity)
# - FILTER_MIN_CUTOFF: Lower value = MORE smoothing / less jitter when still (e.g. 0.05 - 1.0)
# - FILTER_BETA: Lower value = MORE overall smoothing during motion (e.g. 0.0 - 2.0)
FILTER_MIN_CUTOFF = 0.01
FILTER_BETA = 0.04

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
            "wrist": (x_wrist, y_wrist),
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

        wrist_coord = pts[WRIST_INDEX]

        fingers_dict = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            # List of exact 3 (x, y) coordinates with fingertip last
            finger_coords = [pts[idx] for idx in indices]
            fingers_dict[finger_name] = finger_coords

        hand_info = {
            "hand": handedness,
            "wrist": wrist_coord,
            "all_pts": pts,
            "fingers": fingers_dict
        }
        hands_data.append(hand_info)

    return hands_data


# =============================================================
# STEP 2.3: FILTER LANDMARKS (1€ FILTER)
# =============================================================

def filter_landmarks(hands_data, hand_filters_tracker, t_seconds):
    """
    Applies the 1€ Filter (FingertipFilter) to smooth ALL 21 MediaPipe hand landmarks
    so that skeleton drawing, wrist, and finger joint tracking use denoised coordinates.
    """
    filtered_hands_data = []
    current_hand_labels = set()

    for hand_info in hands_data:
        hand_label = hand_info["hand"]
        current_hand_labels.add(hand_label)

        if hand_label not in hand_filters_tracker:
            hand_filters_tracker[hand_label] = [
                FingertipFilter(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA) for _ in range(21)
            ]

        filters = hand_filters_tracker[hand_label]

        # Filter all 21 raw landmarks
        filtered_all_pts = []
        for idx, (rx, ry) in enumerate(hand_info["all_pts"]):
            fx, fy = filters[idx].update(t_seconds, rx, ry)
            filtered_all_pts.append((round(fx, 2), round(fy, 2)))

        # Wrist: index 0
        filtered_wrist = filtered_all_pts[WRIST_INDEX]

        # Fingers: 3 coordinates per finger leading to tip
        filtered_fingers = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            filtered_fingers[finger_name] = [filtered_all_pts[i] for i in indices]

        filtered_hand_info = {
            "hand": hand_label,
            "wrist": filtered_wrist,
            "all_pts": filtered_all_pts,
            "fingers": filtered_fingers
        }
        filtered_hands_data.append(filtered_hand_info)

    # Clean up filters for lost hands
    stale_labels = set(hand_filters_tracker.keys()) - current_hand_labels
    for label in stale_labels:
        del hand_filters_tracker[label]

    return filtered_hands_data


# =============================================================
# STEP 2.5: CALCULATE VELOCITIES
# =============================================================

def calculate_velocities(hands_data, prev_hands_tracker):
    """
    Calculates velocity (dx, dy) for wrist and 3 joint coordinates per finger using ONLY filtered data.
    Tracks previous frame state in prev_hands_tracker.
    Returns list of velocity dictionaries:
    [
        {
            "hand": "Left" or "Right",
            "wrist_velocity": (vx, vy) or None,
            "finger_velocities": {
                "Thumb": [(vx0, vy0), (vx1, vy1), (vx2, vy2)],
                ...
            }
        },
        ...
    ]
    """
    hands_velocity_data = []
    current_hand_labels = set()

    for hand_info in hands_data:
        hand_label = hand_info["hand"]
        current_hand_labels.add(hand_label)

        curr_wrist = hand_info["wrist"]
        curr_fingers = hand_info["fingers"]

        wrist_vel = None
        finger_vels = {}

        if hand_label in prev_hands_tracker:
            prev_hand = prev_hands_tracker[hand_label]
            prev_wrist = prev_hand.get("wrist")
            prev_fingers = prev_hand.get("fingers", {})

            # Wrist velocity from filtered coordinates
            if prev_wrist is not None:
                wrist_vel = (round(curr_wrist[0] - prev_wrist[0], 2), round(curr_wrist[1] - prev_wrist[1], 2))

            # Finger velocities (3 coordinates per finger) from filtered coordinates
            for finger_name, curr_coords in curr_fingers.items():
                if finger_name in prev_fingers and prev_fingers[finger_name] is not None:
                    prev_coords = prev_fingers[finger_name]
                    f_vels = [(round(c[0] - p[0], 2), round(c[1] - p[1], 2)) for c, p in zip(curr_coords, prev_coords)]
                    finger_vels[finger_name] = f_vels
                else:
                    finger_vels[finger_name] = [None, None, None]
        else:
            # First frame for this hand: velocities are None
            wrist_vel = None
            finger_vels = {f: [None, None, None] for f in curr_fingers.keys()}

        # Update previous tracker state with filtered coordinates
        prev_hands_tracker[hand_label] = {
            "wrist": curr_wrist,
            "fingers": curr_fingers
        }

        hand_vel_info = {
            "hand": hand_label,
            "wrist_velocity": wrist_vel,
            "finger_velocities": finger_vels
        }
        hands_velocity_data.append(hand_vel_info)

    # Clean up stale hands from tracker if lost
    stale_labels = set(prev_hands_tracker.keys()) - current_hand_labels
    for label in stale_labels:
        del prev_hands_tracker[label]

    return hands_velocity_data


# =============================================================
# STEP 3: RENDER FRAME & REPORT
# =============================================================

def render_frame(frame, hands_data, frame_index):
    """
    Renders hand skeletons, wrist, and finger landmarks onto the frame, prints details to console,
    and returns the annotated frame.
    """
    if not hands_data:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    for hand_idx, hand_info in enumerate(hands_data):
        handedness = hand_info["hand"]
        wrist_coord = hand_info["wrist"]
        pts = hand_info["all_pts"]
        fingers = hand_info["fingers"]

        # Draw skeleton
        for a, b in HAND_CONNECTIONS:
            pt_a = (int(round(pts[a][0])), int(round(pts[a][1])))
            pt_b = (int(round(pts[b][0])), int(round(pts[b][1])))
            cv2.line(frame, pt_a, pt_b, (200, 200, 200), 1)

        # Draw all landmark points
        for p in pts:
            pt_int = (int(round(p[0])), int(round(p[1])))
            cv2.circle(frame, pt_int, 2, (150, 150, 150), -1)

        # Highlight Wrist
        wx, wy = int(round(wrist_coord[0])), int(round(wrist_coord[1]))
        cv2.circle(frame, (wx, wy), 6, (0, 255, 255), -1)
        cv2.putText(frame, "Wrist", (wx + 8, wy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        report_line = f"[frame {frame_index}] hand {hand_idx} ({handedness}): Wrist=({wx},{wy})"

        # Draw finger points & labels (fingertip is last coordinate in finger list)
        for finger_name, coords in fingers.items():
            color = FINGER_COLORS[finger_name]
            
            # Highlight all 3 finger joints
            for idx, pt in enumerate(coords):
                radius = 7 if idx == 2 else 4  # Larger circle for fingertip
                pt_int = (int(round(pt[0])), int(round(pt[1])))
                cv2.circle(frame, pt_int, radius, color, -1)

            # Fingertip (last coordinate)
            tip_x, tip_y = int(round(coords[-1][0])), int(round(coords[-1][1]))
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
    hand_filters_tracker = {}
    prev_hands_tracker = {}
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
        t_seconds = timestamp_ms / 1000.0

        # 2. Analyze Raw Landmarks
        raw_hands_data = analyze_landmarks(frame, landmarker, timestamp_ms)

        # 2.3 Filter Landmarks with 1€ Filter (reduces noise)
        hands_data = filter_landmarks(raw_hands_data, hand_filters_tracker, t_seconds)

        # 2.5 Calculate Velocities from Filtered Coordinates
        hands_velocity_data = calculate_velocities(hands_data, prev_hands_tracker)

        # Print coordinates then velocities on a new line
        print(hands_data)
        print(hands_velocity_data)

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
