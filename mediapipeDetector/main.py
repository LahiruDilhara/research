"""
mediapipe_finger_viewer.py

A test/debug tool to visually verify MediaPipe hand-landmark accuracy
before building the touch-detection pipeline on top of it.

Usage:
    python3 mediapipe_finger_viewer.py --video path/to/clip.mp4
    python3 mediapipe_finger_viewer.py --live
    python3 mediapipe_finger_viewer.py --live --camera 1

Controls:
    SPACE       play / pause
    d or ->     step forward one frame (works while paused; video files only
                for going frame-by-frame deliberately, also works live)
    a or <-     step backward one frame (VIDEO FILES ONLY - can't rewind a
                live camera stream)
    q or ESC    quit

Every processed frame prints each detected fingertip's name and pixel
(x, y) coordinate to the console, in addition to drawing them on screen.
"""

import argparse
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions as BO

# =============================================================
# CONSTANTS
# =============================================================

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# MediaPipe's 21 hand landmarks, grouped by finger.
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

# Hand connections, grouped the same way, for drawing the skeleton.
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
    """
    Create a HandLandmarker in VIDEO running mode. VIDEO mode (rather than
    IMAGE mode) lets MediaPipe use temporal context between frames you feed
    it in order, which matches how you'll actually run this at runtime.
    """
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
# DRAWING + PRINTING
# =============================================================

def draw_and_report(frame, result, frame_index):
    """
    Draw skeleton + labeled fingertips onto the frame, and print each
    fingertip's name and pixel coordinate to the console.
    Returns the annotated frame.
    """
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

        # All landmarks, small dots
        for p in pts:
            cv2.circle(frame, p, 2, (150, 150, 150), -1)

        # Fingertips: bigger, colored, labeled, and printed to console
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
# MAIN LOOP
# =============================================================

def run(video_path=None, camera_index=0):
    is_live = video_path is None

    model_path = ensure_model_downloaded()
    landmarker = create_landmarker(model_path)

    cap = cv2.VideoCapture(camera_index if is_live else video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {'camera ' + str(camera_index) if is_live else video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default fallback if camera spec isn't explicitly reported

    print(f"Camera FPS: {fps:.2f}")

    if fps < 30.0:
        print(f"Error: Camera FPS ({fps:.2f}) is less than 30 FPS.")
        cap.release()
        landmarker.close()
        return

    if fps > 30.0:
        cap.set(cv2.CAP_PROP_FPS, 30.0)
        fps = 30.0

    ms_per_frame = 1000.0 / fps

    paused = not is_live  # start paused on a video file so you can step through it deliberately
    frame_index = 0
    window_name = "MediaPipe Finger Viewer  (SPACE=play/pause, d/→=step, a/←=back, q=quit)"

    start_time = time.time()
    last_timestamp_ms = -1

    def read_and_process(idx):
        nonlocal landmarker, last_timestamp_ms
        ret, frame = cap.read()
        if not ret:
            return None
        timestamp_ms = int(idx * ms_per_frame) if not is_live else int((time.time() - start_time) * 1000)

        if timestamp_ms <= last_timestamp_ms:
            landmarker.close()
            landmarker = create_landmarker(model_path)

        last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        annotated = draw_and_report(frame, result, idx)
        return annotated

    frame = read_and_process(frame_index)

    while frame is not None:
        loop_start = time.time()

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break

        elif key == ord(' '):
            paused = not paused

        elif key in (ord('a'), 81) and not is_live:  # 'a' or left-arrow, video only
            frame_index = max(0, frame_index - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frame = read_and_process(frame_index)
            paused = True
            continue

        elif key in (ord('d'), 83):  # 'd' or right-arrow
            frame_index += 1
            frame = read_and_process(frame_index)
            paused = True
            continue

        if not paused:
            # Enforce 30 FPS timing
            elapsed = time.time() - loop_start
            wait_time = (1.0 / 30.0) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

            frame_index += 1
            frame = read_and_process(frame_index)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe hand-landmark viewer/tester")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file")
    parser.add_argument("--live", action="store_true", help="Use a live webcam instead of a video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for --live mode")
    args = parser.parse_args()

    if not args.live and not args.video:
        parser.error("Provide either --video <path> or --live")

    run(video_path=args.video, camera_index=args.camera)