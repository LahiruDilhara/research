"""
realtimeprocess/camera_thread.py

Async 12 FPS Camera Capture & MediaPipe Landmark Extraction Worker Thread.

Captures live video frames at target 12 FPS rate (~83.3 ms interval),
extracts MediaPipe hand joint 3D landmarks [(x,y,z)*21], maintains a 5-frame
sliding ring buffer, and fires triggers whenever a 2-frame shift (3-frame overlay) occurs.
"""

import time
import threading
from collections import deque
import cv2
import mediapipe as mp


class CameraThread(threading.Thread):
    """Background worker thread capturing 12 FPS video frames and emitting 5-frame sliding windows (2-frame shift)."""

    def __init__(self, src=0, target_fps: float = 12.0, callback=None):
        super().__init__(daemon=True)
        self.src = src
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # ~0.0833s for 12 FPS
        self.callback = callback

        self.running = False
        self.lock = threading.Lock()

        # Ring buffer holding 5 raw 21-landmark frames [(x,y,z)*21]
        self.landmark_buffer = deque(maxlen=5)
        self.timestamp_buffer = deque(maxlen=5)
        self.shift_counter = 0

        # Latest video frame & annotated frame for UI rendering
        self.latest_frame = None
        self.latest_annotated = None
        self.hand_detected = False
        self.latest_landmarks = None
        self.frame_width = 640
        self.frame_height = 480

        # FPS meter
        self.actual_fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.perf_counter()

    def run(self):
        self.running = True

        # Check and load MediaPipe HandLandmarker task model
        from pathlib import Path
        import urllib.request
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )

        project_root = Path(__file__).resolve().parent.parent
        model_path = str(project_root / "hand_landmarker.task")

        if not Path(model_path).exists():
            print(f"[CameraThread] Downloading hand_landmarker.task model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1
        )
        landmarker = HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print(f"[CameraThread] Error: Could not open video capture source: {self.src}")
            self.running = False
            return

        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
        ]

        from realtimeprocess.realtime_pipeline import (
            HandScaleNormalizer,
            process_streaming_frame,
        )
        normalizer = HandScaleNormalizer()
        score_buffer = deque(maxlen=5)

        last_capture_time = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            elapsed = now - last_capture_time

            # Enforce 12 FPS capture rate
            if elapsed < self.frame_interval:
                time.sleep(max(0.001, self.frame_interval - elapsed))
                continue

            last_capture_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Update dimensions
            h, w, _ = frame.shape
            self.frame_width = w
            self.frame_height = h

            # Flip horizontal for natural user mirror view
            frame_mirror = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            annotated = frame_mirror.copy()

            detected = False
            raw_pts = []
            hand_score = 0.0
            norm_f_dict = None

            if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                detected = True
                landmarks = result.hand_landmarks[0]
                for lm in landmarks:
                    raw_pts.append((lm.x, lm.y, lm.z))

                # Extract hand score
                if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                    hand_score = float(result.handedness[0][0].score)
                else:
                    hand_score = 0.85

                # Exact process.sh pipeline: Scale normalization on raw normalized coordinates
                norm_f_dict, smooth_pts_px = process_streaming_frame(
                    raw_pts, w, h, now, normalizer
                )

                # Draw raw skeleton connections directly on mirrored GUI preview canvas
                for s_idx, e_idx in HAND_CONNECTIONS:
                    x1, y1 = int(smooth_pts_px[s_idx][0]), int(smooth_pts_px[s_idx][1])
                    x2, y2 = int(smooth_pts_px[e_idx][0]), int(smooth_pts_px[e_idx][1])
                    cv2.line(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)

                # Draw joint circles directly on mirrored GUI preview canvas
                for idx, (sx, sy) in enumerate(smooth_pts_px):
                    px, py = int(sx), int(sy)
                    if idx in [4, 8, 12, 16, 20]:
                        cv2.circle(annotated, (px, py), 6, (0, 255, 255), -1)
                    else:
                        cv2.circle(annotated, (px, py), 4, (0, 165, 255), -1)

            with self.lock:
                self.latest_frame = frame_mirror
                self.latest_annotated = annotated
                self.hand_detected = detected
                self.latest_landmarks = raw_pts if detected else None

                # Calculate actual FPS
                self.frame_count += 1
                fps_dur = now - self.fps_start_time
                if fps_dur >= 1.0:
                    self.actual_fps = self.frame_count / fps_dur
                    self.frame_count = 0
                    self.fps_start_time = now

                # If hand detected, update 5-frame ring buffer & shift counter
                if detected and norm_f_dict is not None:
                    self.landmark_buffer.append(norm_f_dict)
                    self.timestamp_buffer.append(now)
                    score_buffer.append(hand_score)
                    self.shift_counter += 1

                    # Trigger processing when buffer is full (5 frames) AND 2 shifts have occurred
                    if len(self.landmark_buffer) == 5 and self.shift_counter >= 2:
                        self.shift_counter = 0
                        norm_window_5 = list(self.landmark_buffer)
                        scores_5 = list(score_buffer)
                        if self.callback:
                            try:
                                self.callback(norm_window_5, scores_5, w, h)
                            except Exception as e:
                                print(f"[CameraThread] Callback error: {e}")
                else:
                    # Hand lost or tracking interrupted: Clear ring buffers
                    self.landmark_buffer.clear()
                    self.timestamp_buffer.clear()
                    score_buffer.clear()
                    self.shift_counter = 0

        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass
        print("[CameraThread] Stopped camera capture loop.")

    def stop(self):
        self.running = False

    def reset_buffers(self):
        """Clears sequence ring buffers to reset state when switching models or settings."""
        with self.lock:
            self.landmark_buffer.clear()
            self.timestamp_buffer.clear()
            self.shift_counter = 0

    def get_latest_frame_data(self):
        with self.lock:
            return (
                self.latest_annotated.copy() if self.latest_annotated is not None else None,
                self.hand_detected,
                self.actual_fps,
                self.frame_width,
                self.frame_height
            )
