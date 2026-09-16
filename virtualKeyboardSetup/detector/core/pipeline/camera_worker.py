"""
core/pipeline/camera_worker.py

Background QThread that runs the complete 12 FPS live capture and processing loop.

Exact pipeline (matches mediapipeDetector/realtimeprocess/camera_thread.py):
──────────────────────────────────────────────────────────────────────────────
Per loop iteration (~83.3 ms):
  1. Enforce 12 FPS with a monotonic perf_counter timer.
  2. Read one BGR frame from cv2.VideoCapture.
  3. Flip frame horizontally (natural mirror view).
  4. Increment the monotonic VIDEO-mode timestamp (ms).
  5. Run MediaPipe HandLandmarker.detect_for_video().
  6. If hand detected:
       a. Extract 21 raw [(x,y,z)] normalised landmarks.
       b. Compute pixel coords: x_px = x*W, y_px = y*H, z_px = z*W.
       c. Apply HandScaleNormalizer → wrist-centred, L_hand-scaled coords.
       d. Build norm_dict: {"wrist_x": ..., ..., "pinky_tip_z": ...}.
       e. Build pixel_list: [(px, py), ...] for all 21 landmarks.
       f. Push norm_dict → landmark_buffer (deque, maxlen=5).
       g. Push pixel_list → pixel_buffer   (deque, maxlen=5).
       h. Increment shift_counter.
       i. If len(buffer)==5 AND shift_counter>=2:
              shift_counter = 0
              emit window_ready(norm_window, pixel_window, W, H)
  7. If hand NOT detected: clear all buffers and reset shift_counter.
  8. Draw hand skeleton on frame.
  9. Run AprilTag tracker on frame (detect + update H + draw key outlines).
 10. Emit frame_ready(annotated_frame, fps, hand_detected, H_matrix_or_None, layout_found).

Signals
───────
  frame_ready(frame: np.ndarray, fps: float,
              hand_detected: bool, H: np.ndarray | None, layout_found: bool)

  window_ready(norm_window_5: list[dict],
               pixel_window_5: list[list[tuple]],
               frame_w: int, frame_h: int)

  error(message: str)
"""

import os
import sys
import time
import urllib.request
import zipfile
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from PySide6.QtCore import QThread, Signal


from config.constants import (
    FINGER_COLORS_BGR,
    FINGERTIP_INDICES,
    HAND_CONNECTIONS,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MODEL_FILENAME,
    MEDIAPIPE_MODEL_URL,
    MEDIAPIPE_NUM_HANDS,
    SHIFT_SIZE,
    TARGET_FPS,
    WINDOW_SIZE,
)
from config.app_config import AppConfig
from core.layout.layout_parser import LayoutData
from core.pipeline.apriltag_tracker import AprilTagTracker
from core.pipeline.normalizer import HandScaleNormalizer
from utils.logger import setup_logger

logger = setup_logger("CameraWorker")

# Landmark drawing helpers
_WRIST_COLOR = (0, 255, 255)
_SKELETON_COLOR = (180, 180, 180)

# Fingertip landmark indices for coloured dot drawing
_TIP_INDICES = set(FINGERTIP_INDICES.values())

# Map flat landmark index → finger name for colour lookup
_IDX_TO_FINGER: dict[int, str] = {}
_FINGER_JOINT_RANGES = {
    "Thumb":  range(1, 5),
    "Index":  range(5, 9),
    "Middle": range(9, 13),
    "Ring":   range(13, 17),
    "Pinky":  range(17, 21),
}
for _f, _r in _FINGER_JOINT_RANGES.items():
    for _i in _r:
        _IDX_TO_FINGER[_i] = _f


@contextmanager
def _suppress_c_stderr():
    """Suppress C-level stderr output to silence noisy C++ drivers."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
    except Exception:
        yield


class CameraWorker(QThread):
    """
    Background QThread: 12 FPS camera capture, MediaPipe, AprilTag, window emission.
    All heavy CV work runs in this thread; results are emitted via Qt signals.
    """

    frame_ready = Signal(object, float, bool, object, bool)
    # (annotated_frame: np.ndarray, fps: float,
    #  hand_detected: bool, H: np.ndarray | None, layout_found: bool)

    window_ready = Signal(list, list, int, int)
    # (norm_window_5: list[dict], pixel_window_5: list[list[tuple]],
    #  frame_w: int, frame_h: int)

    error = Signal(str)

    def __init__(
        self,
        camera_index: int,
        layout: LayoutData,
        config: AppConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._layout = layout
        self._config = config
        self._running = False
        self.setObjectName("CameraWorker")

    # ── QThread lifecycle ──────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        logger.info("CameraWorker started (camera=%d, fps=%s)", self._camera_index, TARGET_FPS)

        # 1. Ensure MediaPipe model file exists
        model_path = self._ensure_mediapipe_model()
        if model_path is None:
            self._running = False
            return

        # 2. Open camera
        with _suppress_c_stderr():
            cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._camera_index)

        if not cap.isOpened():
            self.error.emit(f"Could not open camera index {self._camera_index}.")
            self._running = False
            return

        # 3. Create MediaPipe HandLandmarker (VIDEO mode)
        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=MEDIAPIPE_NUM_HANDS,
                min_hand_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )
            landmarker = HandLandmarker.create_from_options(options)
        except Exception as exc:
            logger.error("Failed to initialize HandLandmarker: %s", exc)
            self.error.emit(f"Failed to initialize HandLandmarker: {exc}")
            cap.release()
            self._running = False
            return

        # 4. Create AprilTag tracker
        apriltag = AprilTagTracker(
            self._layout,
            min_markers=self._config.apriltag_min_markers,
            smoothing_alpha=self._config.apriltag_smoothing,
        )

        # 5. Pipeline state
        normalizer = HandScaleNormalizer()
        landmark_buffer: deque[dict[str, float]] = deque(maxlen=WINDOW_SIZE)
        pixel_buffer: deque[list[tuple[float, float]]] = deque(maxlen=WINDOW_SIZE)
        shift_counter = 0

        frame_interval = 1.0 / TARGET_FPS
        last_capture_t = time.perf_counter()
        frame_timestamp_ms = 0   # monotonic VIDEO-mode counter
        fps_counter = 0
        fps_start = time.perf_counter()
        actual_fps = TARGET_FPS

        # 6. Main loop
        try:
            while self._running:
                now = time.perf_counter()
                elapsed = now - last_capture_t

                # Enforce 12 FPS — sleep the remaining slice if ahead of schedule
                if elapsed < frame_interval:
                    time.sleep(max(0.001, frame_interval - elapsed))
                    continue

                last_capture_t = time.perf_counter()
                ret, raw_frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_h, frame_w = raw_frame.shape[:2]

                # Monotonic timestamp for MediaPipe VIDEO mode
                frame_timestamp_ms += int(frame_interval * 1000)

                # Mirror flip (natural user view)
                frame = cv2.flip(raw_frame, 1)

                # ── FPS meter ──────────────────────────────────────────────
                fps_counter += 1
                dur = time.perf_counter() - fps_start
                if dur >= 1.0:
                    actual_fps = fps_counter / dur
                    fps_counter = 0
                    fps_start = time.perf_counter()

                # ── MediaPipe landmark detection ───────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                hand_detected = bool(
                    result and result.hand_landmarks and len(result.hand_landmarks) > 0
                )

                if hand_detected:
                    raw_lm = result.hand_landmarks[0]   # single-hand mode

                    # Pixel coordinates (mirrored frame space)
                    pts_pixel = [
                        (lm.x * frame_w, lm.y * frame_h, lm.z * frame_w)
                        for lm in raw_lm
                    ]

                    # Scale-normalise (matches stage1_normalizer.py exactly)
                    norm_pts = normalizer.normalize(pts_pixel, center_wrist=True)
                    norm_dict = normalizer.build_norm_dict(norm_pts)

                    # Pixel list for touch resolver (only x, y needed)
                    pixel_list: list[tuple[float, float]] = [
                        (px, py) for px, py, _ in pts_pixel
                    ]

                    # Ring buffer update
                    landmark_buffer.append(norm_dict)
                    pixel_buffer.append(pixel_list)
                    shift_counter += 1

                    # Trigger: buffer full AND shift_size reached
                    if len(landmark_buffer) == WINDOW_SIZE and shift_counter >= SHIFT_SIZE:
                        shift_counter = 0
                        self.window_ready.emit(
                            list(landmark_buffer),
                            list(pixel_buffer),
                            frame_w,
                            frame_h,
                        )

                    # Draw skeleton on frame
                    self._draw_skeleton(frame, pts_pixel)
                else:
                    # Hand lost — reset all buffers immediately
                    landmark_buffer.clear()
                    pixel_buffer.clear()
                    shift_counter = 0

                # ── AprilTag tracking + key overlay ────────────────────────
                apriltag.update(frame)
                apriltag.annotate_frame(frame, self._layout)

                # ── Emit annotated frame ────────────────────────────────────
                self.frame_ready.emit(
                    frame.copy(),
                    actual_fps,
                    hand_detected,
                    apriltag.H.copy() if apriltag.H is not None else None,
                    apriltag.is_valid,
                )

        except Exception as exc:
            logger.exception("CameraWorker loop error: %s", exc)
            self.error.emit(str(exc))
        finally:
            cap.release()
            try:
                landmarker.close()
            except Exception:
                pass
            logger.info("CameraWorker stopped.")

    def stop(self) -> None:
        """Request the worker loop to exit cleanly."""
        self._running = False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pts_pixel: list[tuple[float, float, float]],
    ) -> None:
        """Draw hand skeleton connections and colour-coded joint dots."""
        # Skeleton connections
        for a, b in HAND_CONNECTIONS:
            ax, ay = int(pts_pixel[a][0]), int(pts_pixel[a][1])
            bx, by = int(pts_pixel[b][0]), int(pts_pixel[b][1])
            cv2.line(frame, (ax, ay), (bx, by), _SKELETON_COLOR, 1)

        # Joint dots
        for idx, (px, py, _) in enumerate(pts_pixel):
            x, y = int(px), int(py)
            if idx == 0:
                cv2.circle(frame, (x, y), 6, _WRIST_COLOR, -1)
            else:
                finger = _IDX_TO_FINGER.get(idx)
                color = FINGER_COLORS_BGR.get(finger, (200, 200, 200)) if finger else (200, 200, 200)
                radius = 6 if idx in _TIP_INDICES else 4
                cv2.circle(frame, (x, y), radius, color, -1)

    def _ensure_mediapipe_model(self) -> str | None:
        """Download hand_landmarker.task if it is not already present and valid."""
        # Look next to main.py
        root = Path(__file__).resolve().parent.parent.parent
        model_path = root / MEDIAPIPE_MODEL_FILENAME
        tmp_path = root / f"{MEDIAPIPE_MODEL_FILENAME}.tmp"

        if model_path.exists():
            if zipfile.is_zipfile(str(model_path)):
                return str(model_path)
            logger.warning(
                "Existing %s is incomplete or corrupted. Re-downloading...",
                MEDIAPIPE_MODEL_FILENAME,
            )
            try:
                model_path.unlink(missing_ok=True)
            except Exception:
                pass

        logger.info("Downloading MediaPipe hand_landmarker.task ...")
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(tmp_path))
            if not zipfile.is_zipfile(str(tmp_path)):
                raise ValueError("Downloaded task file is not a valid zip archive.")
            os.replace(str(tmp_path), str(model_path))
            logger.info("Download complete: %s", model_path)
        except Exception as exc:
            logger.error("Failed to download hand_landmarker.task: %s", exc)
            self.error.emit(
                f"Failed to download hand_landmarker.task:\n{exc}\n\n"
                f"Please download it manually from:\n{MEDIAPIPE_MODEL_URL}\n"
                f"and place it at: {model_path}"
            )
            return None

        return str(model_path)

