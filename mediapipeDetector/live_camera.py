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

# =============================================================
# 1€ FILTER & DEADBAND CONFIGURATION PARAMETERS
# =============================================================

# FILTER_MIN_CUTOFF (Minimum Cutoff Frequency in Hz, default: 1.5):
#   Controls smoothing responsiveness when the hand is moving SLOWLY or STATIONARY.
#   - INCREASE (e.g. 2.5 - 5.0): Reduces lag during slow movement, but allows static camera jitter/noise through.
#   - DECREASE (e.g. 0.1 - 0.8): Removes static jitter completely when hand is still, but introduces slight lag/latency when starting slow movements.
FILTER_MIN_CUTOFF = 0.1

# FILTER_BETA (Speed Adaptation Coefficient, default: 1.0):
#   Controls how aggressively the filter opens up cutoff frequency during FAST movements.
#   - INCREASE (e.g. 3.0 - 10.0): Ultra-fast response to fast motion, preserving sharp impact/deceleration spikes (great for touch contact), but allows high-speed jitter through.
#   - DECREASE (e.g. 0.1 - 0.5): Heavily smooths fast movements, eliminating fast motion jitter, but rounds off sharp deceleration peaks.
FILTER_BETA = 0.1

# DEADBAND_VELOCITY_THRESHOLD (Velocity Noise Gate in hand_lengths / sec, default: 0.0):
#   Acts as a velocity noise gate. Velocities with magnitude < threshold are force-zeroed to (0.0, 0.0).
#   - INCREASE (e.g. 0.5 - 1.0): Eliminates residual landmark jitter when hand is completely still, but WIPES OUT subtle/slow touch approach movements.
#   - DECREASE (0.0): Keeps all true micro-velocities intact for detecting light finger touches.
DEADBAND_VELOCITY_THRESHOLD = 0.4

# MISSING_FRAMES_TOLERANCE (Frame Drop Buffer, default: 2):
#   Number of consecutive MediaPipe landmark drop frames tolerated before resetting 1€ filter and velocity memory.
#   - INCREASE (e.g. 4 - 5): Prevents filter resets during temporary hand occlusion, but delays resetting when hand disappears.
#   - DECREASE (1): Resets filter memory immediately on any single dropped detection frame.
MISSING_FRAMES_TOLERANCE = 2

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

# Dynamic Sliding Window Config (Default: 5 frames window, 2 frames overlap)
DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_OVERLAP = 2
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
WINDOW_OVERLAP = DEFAULT_WINDOW_OVERLAP
FINGER_VELOCITY_QUEUES = {}


def init_velocity_queues(window_size=5):
    """Re-initializes global velocity queues with dynamic window_size capacity."""
    global FINGER_VELOCITY_QUEUES, WINDOW_SIZE
    WINDOW_SIZE = max(1, int(window_size))
    FINGER_VELOCITY_QUEUES = {
        "Thumb":  collections.deque(maxlen=WINDOW_SIZE),
        "Index":  collections.deque(maxlen=WINDOW_SIZE),
        "Middle": collections.deque(maxlen=WINDOW_SIZE),
        "Ring":   collections.deque(maxlen=WINDOW_SIZE),
        "Pinky":  collections.deque(maxlen=WINDOW_SIZE),
    }


ACTIVE_HAND_LABEL = None


import glob

def find_available_camera(requested_index: int = -1) -> int:
    """
    Auto-detect available camera index.
    Checks requested_index first (if >= 0), then scans /dev/video* devices,
    and returns the first index that successfully opens and reads a frame.
    """
    candidates = []
    if requested_index >= 0:
        candidates.append(requested_index)

    # Scan /dev/video* devices on Linux
    v4l_indices = []
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(dev.replace("/dev/video", ""))
            v4l_indices.append(idx)
        except ValueError:
            pass

    for idx in v4l_indices + [0, 1, 2, 3, 4]:
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"[Info] Successfully auto-detected camera at index {idx} (/dev/video{idx})")
                return idx

    fallback = requested_index if requested_index >= 0 else (v4l_indices[0] if v4l_indices else 0)
    print(f"[Warning] Could not auto-detect active camera stream; using index {fallback}")
    return fallback


# =============================================================
# MODEL INITIALIZATION & SETUP
# =============================================================

def validate_camera_fps(cap):
    """
    Retrieves camera FPS safely. Many Linux V4L2 drivers return 0.0/30.0.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        print(f"Camera reported FPS: {fps:.2f}")
    else:
        print("Camera reported FPS: Dynamic / V4L2 default")
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
    def __init__(self, camera_index=-1):
        actual_index = find_available_camera(camera_index)
        self.cap = cv2.VideoCapture(actual_index)
        self.lock = threading.Lock()
        self.stopped = False
        self.frame = None
        self.actual_index = actual_index

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {actual_index}.")
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

def calculate_hand_scale(pts, prev_l_hand=None, alpha=1.0):
    """
    Calculates combined hand scale L_hand = sqrt(palm_length^2 + palm_width^2) from landmark pixel coordinates.
    Calculated instantly per frame (alpha=1.0) to avoid scale lag during hand movement.
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

    if prev_l_hand is not None and alpha < 1.0:
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
        l_hand = calculate_hand_scale(pts_pixel, prev_l_hand=prev_scale, alpha=1.0)
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
    If both hands appear, selects the last processed hand.
    If ambiguous, selects any of them.
    """
    global ACTIVE_HAND_LABEL
    
    if not raw_hands_data:
        return []

    # If the previously active hand is currently visible, stick with it
    if ACTIVE_HAND_LABEL is not None:
        for hand_info in raw_hands_data:
            if hand_info["hand"] == ACTIVE_HAND_LABEL:
                return [hand_info]

    # Otherwise, fallback to the single visible hand (or any of them)
    return [raw_hands_data[0]]


# Global missing frame trackers for filters and velocity calculation
FILTER_STALE_COUNTERS = {}
PREV_HANDS_STALE_COUNTERS = {}


# =============================================================
# STEP 3: FILTER LANDMARKS (1€ FILTER IN NORMALIZED UNITS)
# =============================================================

def filter_landmarks(raw_hands_data, hand_filters_tracker, t_seconds):
    """
    Filters normalized landmark coordinates (in hand_lengths) and pixel coordinates using the 1€ Filter.
    Returns filtered scale-normalized and pixel-space hand landmark dictionaries.
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

        if hand_label not in hand_filters_tracker or not isinstance(hand_filters_tracker[hand_label], dict):
            hand_filters_tracker[hand_label] = {
                "norm": [FingertipFilter(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA) for _ in range(21)],
                "pixel": [FingertipFilter(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA) for _ in range(21)],
            }

        tracker = hand_filters_tracker[hand_label]
        norm_filters = tracker["norm"]
        pixel_filters = tracker["pixel"]

        # 1. Filter scale-normalized landmarks (for velocity & model inference)
        filtered_all_pts = []
        for idx, (nx, ny) in enumerate(hand_info["all_pts"]):
            fx, fy = norm_filters[idx].update(t_seconds, nx, ny)
            filtered_all_pts.append((fx, fy))

        # 2. Filter pixel coordinates directly (for rock-solid, jitter-free Video HUD rendering)
        filtered_all_pts_pixel = []
        for idx, (px, py) in enumerate(hand_info["all_pts_pixel"]):
            fpx, fpy = pixel_filters[idx].update(t_seconds, px, py)
            filtered_all_pts_pixel.append((round(fpx, 2), round(fpy, 2)))

        filtered_wrist = filtered_all_pts[WRIST_INDEX]
        filtered_wrist_pixel = filtered_all_pts_pixel[WRIST_INDEX]

        filtered_fingers = {}
        filtered_fingers_pixel = {}
        for finger_name, indices in FINGER_THREE_LANDMARKS.items():
            filtered_fingers[finger_name] = [filtered_all_pts[i] for i in indices]
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
# STEP 6: UPDATE PER-FINGER VELOCITY QUEUES (5 QUEUES × 8-ELEMENT VECTORS)
# =============================================================

def pack_per_finger_vectors(hand_vel_info):
    """
    Packs velocity data into 5 per-finger 8-element vectors.
    Each vector layout:
      [wrist_vx, wrist_vy, joint0_vx, joint0_vy, joint1_vx, joint1_vy, joint2_vx, joint2_vy]
    Wrist velocity is included in every finger vector so each finger stream is self-contained.
    Returns dict: {"Thumb": [...8 floats], "Index": [...8 floats], ...}
    """
    wrist_vel = hand_vel_info.get("wrist_velocity")
    wvx = float(wrist_vel[0]) if wrist_vel is not None else 0.0
    wvy = float(wrist_vel[1]) if wrist_vel is not None else 0.0

    finger_vels = hand_vel_info.get("finger_velocities", {})
    finger_layout = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    result = {}
    for finger_name in finger_layout:
        j_vels = finger_vels.get(finger_name, [])
        vec = [wvx, wvy]  # Elements 0-1: wrist velocity shared across all fingers
        for idx in range(3):  # Elements 2-7: 3 finger joints × (vx, vy)
            joint_vel = j_vels[idx] if idx < len(j_vels) else None
            if joint_vel is not None:
                vec.extend([float(joint_vel[0]), float(joint_vel[1])])
            else:
                vec.extend([0.0, 0.0])
        result[finger_name] = vec  # Exactly 8 float elements

    return result


# Global missing frame counter for velocity queues
QUEUE_STALE_COUNTER = 0


def update_velocity_queue(hands_velocity_data):
    """
    Updates 5 global per-finger velocity queues (FINGER_VELOCITY_QUEUES).
    Each queue holds up to 5 frames of 8-element vectors:
      [wrist_vx, wrist_vy, j0_vx, j0_vy, j1_vx, j1_vy, j2_vx, j2_vy]
    When the 6th item is pushed, the oldest is automatically dropped (deque maxlen=5).
    All queues are cleared together after 2 consecutive missing-detection frames
    or immediately if the active hand label changes.
    """
    global FINGER_VELOCITY_QUEUES, ACTIVE_HAND_LABEL, QUEUE_STALE_COUNTER

    if not hands_velocity_data:
        QUEUE_STALE_COUNTER += 1
        if QUEUE_STALE_COUNTER >= MISSING_FRAMES_TOLERANCE:
            if any(len(q) > 0 for q in FINGER_VELOCITY_QUEUES.values()) or ACTIVE_HAND_LABEL is not None:
                for q in FINGER_VELOCITY_QUEUES.values():
                    q.clear()
                ACTIVE_HAND_LABEL = None
                print("No hand visible for 2 consecutive frames — All Finger Velocity Queues cleared.")
        return

    QUEUE_STALE_COUNTER = 0  # Reset missing frame counter on active detection

    primary_hand = hands_velocity_data[0]
    current_hand = primary_hand["hand"]

    # Hand changed (e.g. Left -> Right or vice-versa) -> clear all queues immediately
    if ACTIVE_HAND_LABEL != current_hand:
        for q in FINGER_VELOCITY_QUEUES.values():
            q.clear()
        ACTIVE_HAND_LABEL = current_hand
        print(f"Hand changed to {current_hand} — All Finger Velocity Queues cleared.")

    # Pack 5 × 8-element vectors and enqueue into each per-finger queue
    per_finger_vecs = pack_per_finger_vectors(primary_hand)
    for finger_name, vec in per_finger_vecs.items():
        FINGER_VELOCITY_QUEUES[finger_name].append(vec)

    # Print current state of all 5 per-finger queues
    for finger_name, q in FINGER_VELOCITY_QUEUES.items():
        print(f"{finger_name} Queue len={len(q)} (vec len={len(q[-1]) if q else 0}): {list(q)}")


# =============================================================
# MAIN LIVE PIPELINE
# =============================================================

def run_live_camera(camera_index=0, window_size=5, window_overlap=2):
    init_velocity_queues(window_size)
    print(f"Configured Sliding Window: size={window_size} frames, overlap={window_overlap} frames")
    model_path = ensure_model_downloaded()
    landmarker = create_landmarker(model_path)

    # Initialize threaded camera stream (Producer-Consumer architecture)
    camera_stream = WebcamStreamThread(camera_index=camera_index)
    if not validate_camera_fps(camera_stream.cap):
        camera_stream.stop()
        landmarker.close()
        sys.exit(1)

    camera_stream.start()

    target_frame_duration = 1.0 / 12.0
    frame_index = 0
    start_time = time.time()
    next_frame_time = start_time
    prev_frame_time = start_time
    hand_filters_tracker = {}
    prev_hands_tracker = {}

    window_name = "Live Camera MediaPipe Finger Detector (q/ESC to quit)"

    print("Starting live camera feed. Show hand to track. Press 'q' or 'ESC' to quit.")

    while True:

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
        fps = (1.0 / fps_dt) if fps_dt > 0 else 12.0
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

        # Step 6: Update Global Velocity Queue
        update_velocity_queue(hands_velocity_data)

        # Display window
        cv2.imshow(window_name, annotated_frame)
        frame_index += 1

        # Enforce target pacing
        next_frame_time += target_frame_duration
        wait_time = next_frame_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)
        else:
            # We fell behind, reset the clock to avoid burst catch-up
            next_frame_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    camera_stream.stop()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera MediaPipe Hand Landmarker")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index (default: -1 for auto-detect)")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size in frames (default: 5)")
    parser.add_argument("--window-overlap", type=int, default=2, help="Window overlap in frames (default: 2)")
    args = parser.parse_args()

    run_live_camera(camera_index=args.camera, window_size=args.window_size, window_overlap=args.window_overlap)
