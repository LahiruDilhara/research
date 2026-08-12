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
import threading
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
# FILTER_MIN_CUTOFF and FILTER_BETA operate in scale-invariant hand-relative units (hand_lengths / sec).
# min_cutoff controls responsiveness when moving slowly (e.g., 0.5 - 1.0)
# beta controls responsiveness during fast motion (e.g., 0.5 - 2.0)
FILTER_MIN_CUTOFF = 1.5
FILTER_BETA = 5.0
DEADBAND_VELOCITY_THRESHOLD = 0.4  # Velocity deadband threshold in hand_lengths / sec
MISSING_FRAMES_TOLERANCE = 2  # Require 2 consecutive missing frames before wiping tracker states

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

# Global 5-Frame Velocity Queue (holds 32-element flat vectors)
VELOCITY_QUEUE = collections.deque(maxlen=5)
ACTIVE_HAND_LABEL = None


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
# THREADED WEBCAM STREAM (PRODUCER-CONSUMER)
# =============================================================

class WebcamStreamThread:
    """
    Thread-safe background camera reader that continuously fetches camera frames.
    Eliminates camera I/O wait latency without thread race conditions.
    """
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.lock = threading.Lock()
        self.stopped = False
        self.frame = None

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            sys.exit(1)

        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

        self.thread = threading.Thread(target=self._update, args=(), daemon=True)

    def start(self):
        self.stopped = False
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# =============================================================
# STEP 1: CAPTURE FRAME
# =============================================================

def capture_frame(camera_stream):
    """Retrieves the most recent BGR frame from the background camera thread."""
    return camera_stream.read()


# =============================================================
# STEP 2: ANALYZE LANDMARKS
# =============================================================

def calculate_hand_scale(pts, prev_l_hand=None, alpha=0.2):
    """
    Calculates combined hand scale L_hand = sqrt(palm_length^2 + palm_width^2) from landmark pixel coordinates.
    Applies exponential smoothing across frames to eliminate scale noise and skeleton flickering.
    """
    w_x, w_y = pts[WRIST_INDEX]
    m_x, m_y = pts[9]
    i_x, i_y = pts[5]
    p_x, p_y = pts[17]

    palm_length = math.hypot(m_x - w_x, m_y - w_y)
    palm_width = math.hypot(p_x - i_x, p_y - i_y)
    l_hand_raw = math.hypot(palm_length, palm_width)

    if l_hand_raw <= 0:
        l_hand_raw = 1.0

    if prev_l_hand is not None:
        l_hand = alpha * l_hand_raw + (1.0 - alpha) * prev_l_hand
    else:
        l_hand = l_hand_raw

    return l_hand

# Global scale memory & missing frame trackers per hand label to ensure smooth scale transitions & drop tolerance
PREV_HAND_SCALES = {}
STALE_FRAME_COUNTERS = {}


def analyze_landmarks(frame, landmarker, timestamp_ms):
    """
    Passes frame to MediaPipe HandLandmarker, computes smoothly-filtered hand scale (l_hand),
    and normalizes raw landmark coordinates immediately (divide x, y by l_hand).
    Returns list of scale-normalized raw hand data dictionaries.
    """
    global PREV_HAND_SCALES, STALE_FRAME_COUNTERS

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks or not result.handedness:
        # Increment missing frame count for all tracked hands
        for label in list(PREV_HAND_SCALES.keys()):
            STALE_FRAME_COUNTERS[label] = STALE_FRAME_COUNTERS.get(label, 0) + 1
            if STALE_FRAME_COUNTERS[label] >= MISSING_FRAMES_TOLERANCE:
                del PREV_HAND_SCALES[label]
                del STALE_FRAME_COUNTERS[label]
        return []

    h, w, _ = frame.shape
    raw_hands_data = []
    current_hand_labels = set()

    for idx, raw_landmarks in enumerate(result.hand_landmarks):
        hand_label = result.handedness[idx][0].category_name  # "Left" or "Right"
        current_hand_labels.add(hand_label)
        STALE_FRAME_COUNTERS[hand_label] = 0  # Reset missing frame count on active detection

        # 21 landmarks converted to pixel floats
        pts_pixel = [(lm.x * w, lm.y * h) for lm in raw_landmarks]
        
        prev_scale = PREV_HAND_SCALES.get(hand_label)
        l_hand = calculate_hand_scale(pts_pixel, prev_l_hand=prev_scale, alpha=0.2)
        PREV_HAND_SCALES[hand_label] = l_hand

        # Scale-normalized landmarks (hand_lengths unit)
        pts_norm = [(px / l_hand, py / l_hand) for px, py in pts_pixel]

        wrist_coord_pixel = pts_pixel[WRIST_INDEX]
        wrist_coord_norm = pts_norm[WRIST_INDEX]

        fingers_dict_pixel = {}
        fingers_dict_norm = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            fingers_dict_pixel[finger_name] = [pts_pixel[i] for i in indices]
            fingers_dict_norm[finger_name] = [pts_norm[i] for i in indices]

        hand_info = {
            "hand": hand_label,
            "wrist": wrist_coord_norm,
            "wrist_pixel": wrist_coord_pixel,
            "all_pts": pts_norm,
            "all_pts_pixel": pts_pixel,
            "fingers": fingers_dict_norm,
            "fingers_pixel": fingers_dict_pixel,
            "l_hand": l_hand
        }
        raw_hands_data.append(hand_info)

    # Increment missing frame count for hands not detected in current frame
    stale_labels = set(PREV_HAND_SCALES.keys()) - current_hand_labels
    for label in stale_labels:
        STALE_FRAME_COUNTERS[label] = STALE_FRAME_COUNTERS.get(label, 0) + 1
        if STALE_FRAME_COUNTERS[label] >= MISSING_FRAMES_TOLERANCE:
            del PREV_HAND_SCALES[label]
            del STALE_FRAME_COUNTERS[label]

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


# Global missing frame trackers for filters and velocity calculation
FILTER_STALE_COUNTERS = {}
PREV_HANDS_STALE_COUNTERS = {}


# =============================================================
# STEP 3: FILTER LANDMARKS (1€ FILTER IN NORMALIZED UNITS)
# =============================================================

def filter_landmarks(raw_hands_data, hand_filters_tracker, t_seconds):
    """
    Filters normalized landmark coordinates (in hand_lengths) using the 1€ Filter.
    Returns filtered scale-normalized hand landmark dictionaries.
    Tolerates transient detection drops (< 2 consecutive missing frames).
    """
    global FILTER_STALE_COUNTERS

    if not raw_hands_data:
        for label in list(hand_filters_tracker.keys()):
            FILTER_STALE_COUNTERS[label] = FILTER_STALE_COUNTERS.get(label, 0) + 1
            if FILTER_STALE_COUNTERS[label] >= MISSING_FRAMES_TOLERANCE:
                del hand_filters_tracker[label]
                del FILTER_STALE_COUNTERS[label]
        return []

    filtered_hands_data = []
    current_hand_labels = set()

    for hand_info in raw_hands_data:
        hand_label = hand_info["hand"]
        l_hand = hand_info["l_hand"]
        current_hand_labels.add(hand_label)
        FILTER_STALE_COUNTERS[hand_label] = 0  # Reset missing frame counter

        if hand_label not in hand_filters_tracker:
            hand_filters_tracker[hand_label] = [
                FingertipFilter(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA) for _ in range(21)
            ]

        filters = hand_filters_tracker[hand_label]

        # Filter all 21 scale-normalized raw landmarks
        filtered_all_pts = []
        for idx, (nx, ny) in enumerate(hand_info["all_pts"]):
            fx, fy = filters[idx].update(t_seconds, nx, ny)
            filtered_all_pts.append((fx, fy))

        filtered_wrist = filtered_all_pts[WRIST_INDEX]

        filtered_fingers = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            filtered_fingers[finger_name] = [filtered_all_pts[i] for i in indices]

        # Re-derive pixel coordinates for rendering visualization
        filtered_all_pts_pixel = [(round(fx * l_hand, 2), round(fy * l_hand, 2)) for fx, fy in filtered_all_pts]
        filtered_wrist_pixel = filtered_all_pts_pixel[WRIST_INDEX]
        filtered_fingers_pixel = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            filtered_fingers_pixel[finger_name] = [filtered_all_pts_pixel[i] for i in indices]

        filtered_hand_info = {
            "hand": hand_label,
            "wrist": filtered_wrist,
            "wrist_pixel": filtered_wrist_pixel,
            "all_pts": filtered_all_pts,
            "all_pts_pixel": filtered_all_pts_pixel,
            "fingers": filtered_fingers,
            "fingers_pixel": filtered_fingers_pixel,
            "l_hand": l_hand
        }
        filtered_hands_data.append(filtered_hand_info)

    # Increment missing frame count for stale filters
    stale_labels = set(hand_filters_tracker.keys()) - current_hand_labels
    for label in stale_labels:
        FILTER_STALE_COUNTERS[label] = FILTER_STALE_COUNTERS.get(label, 0) + 1
        if FILTER_STALE_COUNTERS[label] >= MISSING_FRAMES_TOLERANCE:
            del hand_filters_tracker[label]
            del FILTER_STALE_COUNTERS[label]

    return filtered_hands_data


# =============================================================
# STEP 4: CALCULATE VELOCITIES
# =============================================================

def calculate_velocities(hands_data, prev_hands_tracker, t_seconds):
    """
    Calculates resolution-invariant normalized velocity (vx, vy in hand_lengths / sec)
    from 1€-filtered scale-normalized landmarks. Applies scale-invariant deadband gating.
    Tolerates transient detection drops (< 2 consecutive missing frames).
    """
    global PREV_HANDS_STALE_COUNTERS

    if not hands_data:
        for label in list(prev_hands_tracker.keys()):
            PREV_HANDS_STALE_COUNTERS[label] = PREV_HANDS_STALE_COUNTERS.get(label, 0) + 1
            if PREV_HANDS_STALE_COUNTERS[label] >= MISSING_FRAMES_TOLERANCE:
                del prev_hands_tracker[label]
                del PREV_HANDS_STALE_COUNTERS[label]
        return []

    hands_velocity_data = []
    current_hand_labels = set()

    for hand_info in hands_data:
        hand_label = hand_info["hand"]
        current_hand_labels.add(hand_label)
        PREV_HANDS_STALE_COUNTERS[hand_label] = 0  # Reset missing frame counter

        curr_wrist = hand_info["wrist"]
        curr_fingers = hand_info["fingers"]

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

            # Wrist velocity with hand-length scale velocity deadband hysteresis
            if prev_wrist is not None:
                dnx = curr_wrist[0] - prev_wrist[0]
                dny = curr_wrist[1] - prev_wrist[1]
                vx = round(dnx / dt, 4)
                vy = round(dny / dt, 4)
                if math.hypot(vx, vy) < DEADBAND_VELOCITY_THRESHOLD:
                    wrist_vel = (0.0, 0.0)
                else:
                    wrist_vel = (vx, vy)

            # Finger velocities with hand-length scale velocity deadband hysteresis
            for finger_name, curr_coords in curr_fingers.items():
                if finger_name in prev_fingers and prev_fingers[finger_name] is not None:
                    prev_coords = prev_fingers[finger_name]
                    f_vels = []
                    for c, p in zip(curr_coords, prev_coords):
                        fdnx = c[0] - p[0]
                        fdny = c[1] - p[1]
                        f_vx = round(fdnx / dt, 4)
                        f_vy = round(fdny / dt, 4)
                        if math.hypot(f_vx, f_vy) < DEADBAND_VELOCITY_THRESHOLD:
                            f_vels.append((0.0, 0.0))
                        else:
                            f_vels.append((f_vx, f_vy))
                    finger_vels[finger_name] = f_vels
                else:
                    finger_vels[finger_name] = [None, None, None]
        else:
            wrist_vel = None
            finger_vels = {f: [None, None, None] for f in curr_fingers.keys()}

        # Update previous tracker state with real current filtered landmarks
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

    # Increment missing frame count for stale tracker labels
    stale_labels = set(prev_hands_tracker.keys()) - current_hand_labels
    for label in stale_labels:
        PREV_HANDS_STALE_COUNTERS[label] = PREV_HANDS_STALE_COUNTERS.get(label, 0) + 1
        if PREV_HANDS_STALE_COUNTERS[label] >= MISSING_FRAMES_TOLERANCE:
            del prev_hands_tracker[label]
            del PREV_HANDS_STALE_COUNTERS[label]

    return hands_velocity_data



# =============================================================
# STEP 5: RENDER FRAME & VELOCITIES ON IMAGE
# =============================================================

def render_frame(frame, hands_data, hands_velocity_data, frame_index, fps=0.0):
    """
    Draws real-time FPS overlay, hand skeletons, and velocity labels (vx, vy) onto the frame.
    """
    # Draw FPS overlay in top-left corner
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not hands_data:
        cv2.putText(frame, "No hand detected", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    vel_map = {v["hand"]: v for v in (hands_velocity_data or [])}

    for hand_idx, hand_info in enumerate(hands_data):
        handedness = hand_info["hand"]
        wrist_coord_px = hand_info.get("wrist_pixel", hand_info["wrist"])
        pts_px = hand_info.get("all_pts_pixel", hand_info["all_pts"])
        fingers_px = hand_info.get("fingers_pixel", hand_info["fingers"])

        hand_vels = vel_map.get(handedness, {})
        wrist_vel = hand_vels.get("wrist_velocity")
        finger_vels = hand_vels.get("finger_velocities", {})

        # Draw skeleton connections
        for a, b in HAND_CONNECTIONS:
            pt_a = (int(round(pts_px[a][0])), int(round(pts_px[a][1])))
            pt_b = (int(round(pts_px[b][0])), int(round(pts_px[b][1])))
            cv2.line(frame, pt_a, pt_b, (200, 200, 200), 1)

        # Draw Wrist
        wx, wy = int(round(wrist_coord_px[0])), int(round(wrist_coord_px[1]))
        cv2.circle(frame, (wx, wy), 6, (0, 255, 255), -1)

        wrist_str = f"Wrist v=({wrist_vel[0]:.2f},{wrist_vel[1]:.2f})" if wrist_vel else "Wrist v=(0.00,0.00)"
        cv2.putText(frame, wrist_str, (wx + 8, wy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw Finger joints & velocity labels
        for finger_name, coords in fingers_px.items():
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
# STEP 6: UPDATE GLOBAL VELOCITY QUEUE (32-ELEMENT FLAT LISTS)
# =============================================================

def pack_flat_velocity_vector(hand_vel_info):
    """
    Packs 16 location velocity pairs (vx, vy) into a flat 1D Python list of 32 float elements:
      - Elements 0..1: Wrist (vx, vy)
      - Elements 2..7: Thumb 3 joints (vx, vy)
      - Elements 8..13: Index 3 joints (vx, vy)
      - Elements 14..19: Middle 3 joints (vx, vy)
      - Elements 20..25: Ring 3 joints (vx, vy)
      - Elements 26..31: Pinky 3 joints (vx, vy)
    Total = 32 float elements.
    """
    flat = []
    wrist_vel = hand_vel_info.get("wrist_velocity")
    if wrist_vel:
        flat.extend([float(wrist_vel[0]), float(wrist_vel[1])])
    else:
        flat.extend([0.0, 0.0])

    finger_vels = hand_vel_info.get("finger_velocities", {})
    finger_layout = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    for finger_name in finger_layout:
        j_vels = finger_vels.get(finger_name, [])
        for idx in range(3):
            joint_vel = j_vels[idx] if idx < len(j_vels) else None
            if joint_vel:
                flat.extend([float(joint_vel[0]), float(joint_vel[1])])
            else:
                flat.extend([0.0, 0.0])

    return flat  # Exactly 32 float items


# Global missing frame counter for velocity queue
QUEUE_STALE_COUNTER = 0


def update_velocity_queue(hands_velocity_data):
    """
    Updates the global 5-element velocity queue (VELOCITY_QUEUE).
    Enqueues flat 32-element velocity lists. When 6th item is pushed, 1st item is dropped.
    Clears global queue only after 2 consecutive missing-detection frames or if active hand changes.
    """
    global VELOCITY_QUEUE, ACTIVE_HAND_LABEL, QUEUE_STALE_COUNTER

    if not hands_velocity_data:
        QUEUE_STALE_COUNTER += 1
        if QUEUE_STALE_COUNTER >= MISSING_FRAMES_TOLERANCE:
            if VELOCITY_QUEUE or ACTIVE_HAND_LABEL is not None:
                VELOCITY_QUEUE.clear()
                ACTIVE_HAND_LABEL = None
                print("No hand visible for 2 consecutive frames — Global Velocity Queue cleared.")
        return

    QUEUE_STALE_COUNTER = 0  # Reset missing frame counter on active detection

    primary_hand = hands_velocity_data[0]
    current_hand = primary_hand["hand"]

    # Hand changed (e.g. Left -> Right or vice-versa) -> clear queue immediately
    if ACTIVE_HAND_LABEL != current_hand:
        VELOCITY_QUEUE.clear()
        ACTIVE_HAND_LABEL = current_hand
        print(f"Hand changed to {current_hand} — Global Velocity Queue cleared.")

    # Pack 32-element flat list and enqueue
    flat_vector = pack_flat_velocity_vector(primary_hand)
    VELOCITY_QUEUE.append(flat_vector)

    # Print current 5-element global queue state
    print(f"Queue len={len(VELOCITY_QUEUE)} (item len={len(VELOCITY_QUEUE[-1])}): {list(VELOCITY_QUEUE)}")


# =============================================================
# MAIN LIVE PIPELINE
# =============================================================

def run_live_camera(camera_index=0):
    model_path = ensure_model_downloaded()
    landmarker = create_landmarker(model_path)

    # Initialize threaded camera stream (Producer-Consumer architecture)
    camera_stream = WebcamStreamThread(camera_index=camera_index)
    if not validate_camera_fps(camera_stream.cap):
        camera_stream.stop()
        landmarker.close()
        sys.exit(1)

    camera_stream.start()

    target_frame_duration = 1.0 / 30.0
    frame_index = 0
    start_time = time.time()
    prev_frame_time = time.time()
    hand_filters_tracker = {}
    prev_hands_tracker = {}

    window_name = "Live Camera MediaPipe Finger Detector (q/ESC to quit)"

    print("Starting live camera feed. Show hand to track. Press 'q' or 'ESC' to quit.")

    while True:
        loop_start = time.time()

        # Step 1: Capture Frame from Background Thread
        frame = capture_frame(camera_stream)
        if frame is None:
            print("Failed to grab frame from camera thread.")
            break

        timestamp_ms = int((time.time() - start_time) * 1000)
        t_seconds = timestamp_ms / 1000.0

        # Calculate real-time FPS
        curr_time = time.time()
        fps_dt = curr_time - prev_frame_time
        fps = (1.0 / fps_dt) if fps_dt > 0 else 30.0
        prev_frame_time = curr_time

        # Step 2: MediaPipe Landmark Detection
        raw_hands_data = analyze_landmarks(frame, landmarker, timestamp_ms)

        # Step 2.1: Single Hand Selection (Prioritize 'Left' hand before filtration)
        single_hand_raw = select_single_hand(raw_hands_data)

        # Step 3: Denoise Landmark Coordinates (1€ Filter)
        hands_data = filter_landmarks(single_hand_raw, hand_filters_tracker, t_seconds)

        # Step 4: Calculate Normalized Velocities
        hands_velocity_data = calculate_velocities(hands_data, prev_hands_tracker, t_seconds)

        # Step 5: Render Frame, Velocities, and FPS on Image
        annotated_frame = render_frame(frame, hands_data, hands_velocity_data, frame_index, fps)

        # Step 6: Update Global 5-Element Velocity Queue
        update_velocity_queue(hands_velocity_data)

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

    camera_stream.stop()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera MediaPipe Hand Landmarker")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    run_live_camera(camera_index=args.camera)
