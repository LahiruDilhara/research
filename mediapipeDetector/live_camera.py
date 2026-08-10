"""
live_camera.py

Modularized MediaPipe Hand Landmarker pipeline running solely on live camera feed.

Step-by-step pipeline execution per frame:
  Step 1: capture_frame(cap)
  Step 2: analyze_landmarks(frame, landmarker, timestamp_ms)
  Step 3: filter_landmarks(raw_hands_data, hand_filters_tracker, t_seconds)
  Step 4: calculate_velocities(hands_data, prev_hands_tracker, t_seconds)
  Step 5: render_frame(frame, hands_data, hands_velocity_data, frame_index)
  Step 6: update_velocity_queue(hands_velocity_data, velocity_queue, queue_state)
"""

import argparse
import collections
import math
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
# CONSTANTS & CONFIGURATION
# =============================================================

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# 15 Location Mappings (Wrist + 14 Finger Joints)
# Locations:
#  1. Wrist (Landmark 0)
#  2. Thumb:  Landmarks [2, 3, 4]  (3 joints)
#  3. Index:  Landmarks [6, 7, 8]  (3 joints)
#  4. Middle: Landmarks [10, 11, 12] (3 joints)
#  5. Ring:   Landmarks [14, 15, 16] (3 joints)
#  6. Pinky:  Landmarks [18, 19]   (2 joints)
# Total = 1 + 3 + 3 + 3 + 3 + 2 = 15 locations -> 30 velocity values (vx, vy)
FINGER_THREE_LANDMARKS = {
    "Thumb":  [2, 3, 4],
    "Index":  [6, 7, 8],
    "Middle": [10, 11, 12],
    "Ring":   [14, 15, 16],
    "Pinky":  [18, 19, 20],
}
WRIST_INDEX = 0

# 1€ Filter & Deadband Threshold Constants
FILTER_MIN_CUTOFF = 0.001
FILTER_BETA = 0.04
DEADBAND_THRESHOLD_PIXELS = 5

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
# MODEL INITIALIZATION & SETUP
# =============================================================

def validate_camera_fps(cap):
    """
    Checks if camera supports at least 30.0 FPS.
    Sets target FPS to 30.0 if higher. Returns True if valid, False otherwise.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS: {fps:.2f}")

    if fps < 30.0:
        print(f"Error: Camera FPS ({fps:.2f}) is less than 30 FPS. Program exiting.")
        return False

    if fps > 30.0:
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    return True


def ensure_model_downloaded():
    """Downloads the hand_landmarker.task model file if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


def create_landmarker(model_path):
    """Creates and returns a MediaPipe HandLandmarker instance in VIDEO mode."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


# =============================================================
# STEP 1: CAPTURE FRAME
# =============================================================

def capture_frame(cap):
    """Captures a single BGR frame from the video capture device."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


# =============================================================
# STEP 2: ANALYZE LANDMARKS
# =============================================================

def analyze_landmarks(frame, landmarker, timestamp_ms):
    """
    Passes frame to MediaPipe HandLandmarker and extracts raw landmark coordinates.
    Returns list of raw hand data dictionaries.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks or not result.handedness:
        return []

    h, w, _ = frame.shape
    raw_hands_data = []

    for idx, raw_landmarks in enumerate(result.hand_landmarks):
        hand_label = result.handedness[idx][0].category_name  # "Left" or "Right"

        # 21 landmarks converted to pixel floats
        pts = [(lm.x * w, lm.y * h) for lm in raw_landmarks]

        wrist_coord = pts[WRIST_INDEX]

        fingers_dict = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            fingers_dict[finger_name] = [pts[i] for i in indices]

        hand_info = {
            "hand": hand_label,
            "wrist": wrist_coord,
            "all_pts": pts,
            "fingers": fingers_dict
        }
        raw_hands_data.append(hand_info)

    return raw_hands_data


# =============================================================
# STEP 2.1: SELECT SINGLE HAND (PRIORITIZE LEFT HAND)
# =============================================================

def select_single_hand(raw_hands_data):
    """
    Selects at most ONE hand to pass through the pipeline before filtration.
    If both Left and Right hands are visible, always chooses 'Left' hand.
    Otherwise returns the single visible hand or [] if none.
    """
    if not raw_hands_data:
        return []

    # If both hands visible, always select 'Left' hand
    for hand_info in raw_hands_data:
        if hand_info["hand"] == "Left":
            return [hand_info]

    # Fallback to the single visible hand (e.g. 'Right')
    return [raw_hands_data[0]]


# =============================================================
# STEP 3: FILTER LANDMARKS (1€ FILTER)
# =============================================================

def filter_landmarks(raw_hands_data, hand_filters_tracker, t_seconds):
    """
    Filters raw MediaPipe landmarks using the 1€ Filter.
    Returns filtered hand landmark dictionaries.
    """
    if not raw_hands_data:
        return []

    filtered_hands_data = []
    current_hand_labels = set()

    for hand_info in raw_hands_data:
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

        filtered_wrist = filtered_all_pts[WRIST_INDEX]

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

    # Clean up stale filters
    stale_labels = set(hand_filters_tracker.keys()) - current_hand_labels
    for label in stale_labels:
        del hand_filters_tracker[label]

    return filtered_hands_data


# =============================================================
# STEP 4: CALCULATE VELOCITIES
# =============================================================

def calculate_velocities(hands_data, prev_hands_tracker, t_seconds):
    """
    Calculates hand-scale normalized velocity (vx, vy in hand_lengths / sec) for wrist and finger joints
    using ONLY 1€-filtered coordinates. Applies deadband hysteresis gating to zero out jitter.
    """
    if not hands_data:
        return []

    hands_velocity_data = []
    current_hand_labels = set()

    for hand_info in hands_data:
        hand_label = hand_info["hand"]
        current_hand_labels.add(hand_label)

        curr_wrist = hand_info["wrist"]
        curr_fingers = hand_info["fingers"]
        all_pts = hand_info["all_pts"]

        # Combined hand scale L_hand = sqrt(palm_length^2 + palm_width^2)
        w_x, w_y = all_pts[WRIST_INDEX]
        m_x, m_y = all_pts[9]
        i_x, i_y = all_pts[5]
        p_x, p_y = all_pts[17]

        palm_length = math.hypot(m_x - w_x, m_y - w_y)
        palm_width = math.hypot(p_x - i_x, p_y - i_y)
        l_hand = math.hypot(palm_length, palm_width)

        if l_hand <= 0:
            l_hand = 1.0

        wrist_vel = None
        finger_vels = {}

        if hand_label in prev_hands_tracker:
            prev_hand = prev_hands_tracker[hand_label]
            prev_wrist = prev_hand.get("wrist")
            prev_fingers = prev_hand.get("fingers", {})
            prev_t = prev_hand.get("timestamp", t_seconds)

            dt = t_seconds - prev_t
            if dt <= 0:
                dt = 1.0 / 30.0

            scale_dt = l_hand * dt

            # Wrist velocity with deadband hysteresis
            if prev_wrist is not None:
                dx = curr_wrist[0] - prev_wrist[0]
                dy = curr_wrist[1] - prev_wrist[1]
                if math.hypot(dx, dy) < DEADBAND_THRESHOLD_PIXELS:
                    wrist_vel = (0.0, 0.0)
                    curr_wrist = prev_wrist
                else:
                    vx = round(dx / scale_dt, 4)
                    vy = round(dy / scale_dt, 4)
                    wrist_vel = (vx, vy)

            # Finger velocities with deadband hysteresis
            updated_curr_fingers = {}
            for finger_name, curr_coords in curr_fingers.items():
                if finger_name in prev_fingers and prev_fingers[finger_name] is not None:
                    prev_coords = prev_fingers[finger_name]
                    f_vels = []
                    f_coords_gated = []
                    for c, p in zip(curr_coords, prev_coords):
                        fdx = c[0] - p[0]
                        fdy = c[1] - p[1]
                        if math.hypot(fdx, fdy) < DEADBAND_THRESHOLD_PIXELS:
                            f_vels.append((0.0, 0.0))
                            f_coords_gated.append(p)
                        else:
                            f_vx = round(fdx / scale_dt, 4)
                            f_vy = round(fdy / scale_dt, 4)
                            f_vels.append((f_vx, f_vy))
                            f_coords_gated.append(c)
                    finger_vels[finger_name] = f_vels
                    updated_curr_fingers[finger_name] = f_coords_gated
                else:
                    finger_vels[finger_name] = [None, None, None]
                    updated_curr_fingers[finger_name] = curr_coords

            curr_fingers = updated_curr_fingers
        else:
            wrist_vel = None
            finger_vels = {f: [None, None, None] for f in curr_fingers.keys()}

        # Update previous tracker state
        prev_hands_tracker[hand_label] = {
            "wrist": curr_wrist,
            "fingers": curr_fingers,
            "timestamp": t_seconds
        }

        hand_vel_info = {
            "hand": hand_label,
            "wrist_velocity": wrist_vel,
            "finger_velocities": finger_vels
        }
        hands_velocity_data.append(hand_vel_info)

    # Clean up lost hands
    stale_labels = set(prev_hands_tracker.keys()) - current_hand_labels
    for label in stale_labels:
        del prev_hands_tracker[label]

    return hands_velocity_data


# =============================================================
# STEP 5: RENDER FRAME & VELOCITIES ON IMAGE
# =============================================================

def render_frame(frame, hands_data, hands_velocity_data, frame_index):
    """
    Draws hand skeletons and overlays velocity labels (vx, vy) onto the frame.
    """
    if not hands_data:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    vel_map = {v["hand"]: v for v in (hands_velocity_data or [])}

    for hand_idx, hand_info in enumerate(hands_data):
        handedness = hand_info["hand"]
        wrist_coord = hand_info["wrist"]
        pts = hand_info["all_pts"]
        fingers = hand_info["fingers"]

        hand_vels = vel_map.get(handedness, {})
        wrist_vel = hand_vels.get("wrist_velocity")
        finger_vels = hand_vels.get("finger_velocities", {})

        # Draw skeleton connections
        for a, b in HAND_CONNECTIONS:
            pt_a = (int(round(pts[a][0])), int(round(pts[a][1])))
            pt_b = (int(round(pts[b][0])), int(round(pts[b][1])))
            cv2.line(frame, pt_a, pt_b, (200, 200, 200), 1)

        # Draw Wrist
        wx, wy = int(round(wrist_coord[0])), int(round(wrist_coord[1]))
        cv2.circle(frame, (wx, wy), 6, (0, 255, 255), -1)

        wrist_str = f"Wrist v=({wrist_vel[0]:.2f},{wrist_vel[1]:.2f})" if wrist_vel else "Wrist v=(0.00,0.00)"
        cv2.putText(frame, wrist_str, (wx + 8, wy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw Finger joints & velocity labels
        for finger_name, coords in fingers.items():
            color = FINGER_COLORS[finger_name]
            f_v_list = finger_vels.get(finger_name, [None, None, None])

            for idx, pt in enumerate(coords):
                radius = 6 if idx == 2 else 4
                pt_int = (int(round(pt[0])), int(round(pt[1])))
                cv2.circle(frame, pt_int, radius, color, -1)

                joint_vel = f_v_list[idx] if idx < len(f_v_list) else None
                if joint_vel:
                    v_txt = f"({joint_vel[0]:.2f},{joint_vel[1]:.2f})"
                else:
                    v_txt = "(0.00,0.00)"

                if idx == 2:
                    v_txt = f"{finger_name} v={v_txt}"

                cv2.putText(frame, v_txt, (pt_int[0] + 6, pt_int[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return frame


# =============================================================
# STEP 6: UPDATE VELOCITY QUEUE (30-ELEMENT FLAT ARRAY)
# =============================================================

def pack_flat_velocity_vector(hand_vel_info):
    """
    Packs 15 location velocity pairs (vx, vy) into a flat 1D Python list of 30 float elements:
      - Elements 0..1: Wrist (vx, vy)
      - Elements 2..7: Thumb 3 joints (vx, vy)
      - Elements 8..13: Index 3 joints (vx, vy)
      - Elements 14..19: Middle 3 joints (vx, vy)
      - Elements 20..25: Ring 3 joints (vx, vy)
      - Elements 26..29: Pinky 2 joints (vx, vy)
    Total = 30 float elements.
    """
    flat = []
    wrist_vel = hand_vel_info.get("wrist_velocity")
    if wrist_vel:
        flat.extend([float(wrist_vel[0]), float(wrist_vel[1])])
    else:
        flat.extend([0.0, 0.0])

    finger_vels = hand_vel_info.get("finger_velocities", {})
    finger_layout = [
        ("Thumb", 3),
        ("Index", 3),
        ("Middle", 3),
        ("Ring", 3),
        ("Pinky", 2)
    ]

    for finger_name, count in finger_layout:
        j_vels = finger_vels.get(finger_name, [])
        for idx in range(count):
            joint_vel = j_vels[idx] if idx < len(j_vels) else None
            if joint_vel:
                flat.extend([float(joint_vel[0]), float(joint_vel[1])])
            else:
                flat.extend([0.0, 0.0])

    return flat


def update_velocity_queue(hands_velocity_data, velocity_queue, queue_state):
    """
    Pushes 30-element flat velocity vectors into a 5-element sliding window queue.
    Clears queue immediately if hand is not visible or if active hand changes.
    """
    if not hands_velocity_data:
        # Hand not visible -> clear queue immediately
        if velocity_queue or queue_state["active_hand"] is not None:
            velocity_queue.clear()
            queue_state["active_hand"] = None
            print("No hand visible — Velocity Queue cleared.")
        return velocity_queue

    primary_hand = hands_velocity_data[0]
    current_hand = primary_hand["hand"]

    # Hand changed (e.g. Left -> Right or vice-versa) -> clear queue immediately
    if queue_state["active_hand"] != current_hand:
        velocity_queue.clear()
        queue_state["active_hand"] = current_hand
        print(f"Hand changed to {current_hand} — Velocity Queue cleared.")

    # Pack 30-element flat list and append to queue
    flat_vector = pack_flat_velocity_vector(primary_hand)
    velocity_queue.append(flat_vector)

    # Print current 5-element queue state
    print(f"[{current_hand} Hand] Queue len={len(velocity_queue)}: {list(velocity_queue)}")

    return velocity_queue


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

    # Queue of max size 5 holding 30-element flat velocity lists
    velocity_queue = collections.deque(maxlen=5)
    queue_state = {"active_hand": None}

    window_name = "Live Camera MediaPipe Finger Detector (q/ESC to quit)"

    print("Starting live camera feed. Show hand to track. Press 'q' or 'ESC' to quit.")

    while True:
        loop_start = time.time()

        # Step 1: Capture Frame
        frame = capture_frame(cap)
        if frame is None:
            print("Failed to grab frame from camera.")
            break

        timestamp_ms = int((time.time() - start_time) * 1000)
        t_seconds = timestamp_ms / 1000.0

        # Step 2: MediaPipe Landmark Detection
        raw_hands_data = analyze_landmarks(frame, landmarker, timestamp_ms)

        # Step 2.1: Single Hand Selection (Prioritize 'Left' hand before filtration)
        single_hand_raw = select_single_hand(raw_hands_data)

        # Step 3: Denoise Landmark Coordinates (1€ Filter)
        hands_data = filter_landmarks(single_hand_raw, hand_filters_tracker, t_seconds)

        # Step 4: Calculate Normalized Velocities
        hands_velocity_data = calculate_velocities(hands_data, prev_hands_tracker, t_seconds)

        # Step 5: Render Frame & Velocities on Image
        annotated_frame = render_frame(frame, hands_data, hands_velocity_data, frame_index)

        # Step 6: Update 5-Element Velocity Queue
        velocity_queue = update_velocity_queue(hands_velocity_data, velocity_queue, queue_state)

        # Display window
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
