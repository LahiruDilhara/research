"""
core/pipeline/camera_worker.py

Background QThread that runs the complete 12 FPS live capture and processing loop.

Exact pipeline (matches mediapipeDetector/realtimeprocess/camera_thread.py):
──────────────────────────────────────────────────────────────────────────────
Per loop iteration (~83.3 ms):
  1. Enforce 12 FPS with a monotonic perf_counter timer.
  2. Read one BGR frame from cv2.VideoCapture.
  3. Process raw frame directly (unflipped) so AprilTags and layout match physical space.
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

import math
import os
import sys
import threading
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
from services.touch_pipeline_service import TouchPipelineService
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


class _FrameGrabber(threading.Thread):
    """
    Dedicated background thread continuously reading from the camera.
    Drains the driver buffer so CameraWorker always samples the freshest real-time frame
    without backlog lag.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self._cap = cap
        self._running = True
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

    def run(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def stop(self) -> None:
        self._running = False


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
        pipeline_service: TouchPipelineService | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._layout = layout
        self._config = config
        self._pipeline_service = pipeline_service
        self._render_video = True
        self._running = False
        self._show_overlay = True
        self._active_button_ids: set[str] = set()
        self._active_button_clear_t: float = 0.0
        self._contact_points: dict[str, tuple[float, float, float, float]] = {}
        self._contact_points_clear_t: float = 0.0
        self.setObjectName("CameraWorker")

    def set_show_overlay(self, show: bool) -> None:
        """Toggle displaying the keyboard overlay on the live camera feed."""
        self._show_overlay = bool(show)

    @property
    def show_overlay(self) -> bool:
        return self._show_overlay

    def set_active_buttons(self, button_ids: list[str] | set[str] | None) -> None:
        """Flash active pressed buttons in green on live camera overlay (supports multi-touch)."""
        self._active_button_ids = set(button_ids) if button_ids else set()
        self._active_button_clear_t = time.perf_counter() + 0.35

    def set_active_button(self, button_id: str | None) -> None:
        """Single-button compatibility method."""
        self.set_active_buttons([button_id] if button_id else [])

    def set_contact_points(self, points: dict[str, tuple[float, float, float, float]]) -> None:
        """Stores extrapolated contact points for visual overlay rendering."""
        self._contact_points = dict(points)
        self._contact_points_clear_t = time.perf_counter() + 0.35

    @property
    def render_video(self) -> bool:
        """True if video frame rendering and skeleton drawing are enabled."""
        return self._render_video

    @render_video.setter
    def render_video(self, value: bool) -> None:
        self._render_video = bool(value)

    # ── QThread lifecycle ──────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        logger.info("CameraWorker started (camera=%d, fps=%s)", self._camera_index, TARGET_FPS)

        # 1. Ensure MediaPipe model file exists
        model_path = self._ensure_mediapipe_model()
        if model_path is None:
            self._running = False
            return

        # 2. Open camera with automatic fallback to any discovered working camera
        cap = None
        with _suppress_c_stderr():
            candidate_cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
            if not candidate_cap.isOpened():
                candidate_cap = cv2.VideoCapture(self._camera_index)
            if candidate_cap.isOpened():
                ret, _ = candidate_cap.read()
                if ret:
                    cap = candidate_cap
                else:
                    candidate_cap.release()

        if cap is None:
            from services.camera_discovery import discover_cameras
            discovered = discover_cameras(max_index=6)
            for c in discovered:
                with _suppress_c_stderr():
                    fallback_cap = cv2.VideoCapture(c.index, cv2.CAP_V4L2)
                    if not fallback_cap.isOpened():
                        fallback_cap = cv2.VideoCapture(c.index)
                    if fallback_cap.isOpened():
                        ret, _ = fallback_cap.read()
                        if ret:
                            logger.info(
                                "Camera %d was unavailable. Auto-recovered with camera %d (%s).",
                                self._camera_index,
                                c.index,
                                c.name,
                            )
                            self._camera_index = c.index
                            cap = fallback_cap
                            break
                        fallback_cap.release()

        if cap is None or not cap.isOpened():
            self.error.emit(f"Could not open camera {self._camera_index} and no other working camera found.")
            self._running = False
            return

        # Start low-latency frame grabber
        grabber = _FrameGrabber(cap)
        grabber.start()

        # 3. Create MediaPipe HandLandmarker (VIDEO mode)
        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=MEDIAPIPE_NUM_HANDS,
                min_hand_detection_confidence=self._config.mediapipe_min_detection_confidence,
                min_hand_presence_confidence=self._config.mediapipe_min_presence_confidence,
                min_tracking_confidence=self._config.mediapipe_min_tracking_confidence,
            )
            landmarker = HandLandmarker.create_from_options(options)
        except Exception as exc:
            logger.error("Failed to initialize HandLandmarker: %s", exc)
            self.error.emit(f"Failed to initialize HandLandmarker: {exc}")
            grabber.stop()
            grabber.join(timeout=0.5)
            cap.release()
            self._running = False
            return

        # 4. Create AprilTag tracker
        apriltag = AprilTagTracker(
            self._layout,
            min_markers=self._config.apriltag_min_markers,
            smoothing_alpha=self._config.apriltag_smoothing,
        )

        # 5. Service Layer Pipeline (Manages 5 dedicated finger queues and hand identity)
        if self._pipeline_service is not None:
            pipeline_service = self._pipeline_service
        else:
            pipeline_service = TouchPipelineService(
                window_size=self._config.window_size,
                shift_size=self._config.shift_size,
                one_euro_enabled=self._config.one_euro_enabled,
                one_euro_min_cutoff=self._config.one_euro_min_cutoff,
                one_euro_beta=self._config.one_euro_beta,
                one_euro_d_cutoff=self._config.one_euro_d_cutoff,
            )
            self._pipeline_service = pipeline_service

        frame_interval = 1.0 / TARGET_FPS
        last_capture_t = time.perf_counter()
        frame_timestamp_ms = 0   # monotonic VIDEO-mode counter
        fps_counter = 0
        fps_start = time.perf_counter()
        actual_fps = TARGET_FPS
        prev_hand_detected = False
        prev_layout_valid = False

        # 6. Main loop
        try:
            while self._running:
                now = time.perf_counter()
                elapsed = now - last_capture_t

                # Enforce 12 FPS interval precisely
                if elapsed < frame_interval:
                    time.sleep(max(0.001, frame_interval - elapsed))
                    continue

                last_capture_t = time.perf_counter()
                raw_frame = grabber.get_latest_frame()
                if raw_frame is None:
                    time.sleep(0.005)
                    continue

                frame_h, frame_w = raw_frame.shape[:2]

                # Monotonic timestamp for MediaPipe VIDEO mode
                frame_timestamp_ms += int(frame_interval * 1000)

                # ── FPS meter ──────────────────────────────────────────────
                fps_counter += 1
                dur = time.perf_counter() - fps_start
                if dur >= 1.0:
                    actual_fps = fps_counter / dur
                    fps_counter = 0
                    fps_start = time.perf_counter()

                # ── 1. AprilTag fiducial homography (runs on raw camera frame) ──
                apriltag.update(raw_frame)
                if apriltag.is_valid and not prev_layout_valid:
                    logger.info("AprilTag: Tracking locked (%d markers visible, homography valid).", apriltag.markers_used)
                elif not apriltag.is_valid and prev_layout_valid:
                    logger.warning("AprilTag: Tracking lost (detected %d markers, min required=%d). Searching for layout markers...", apriltag.markers_used, self._config.apriltag_min_markers)
                prev_layout_valid = apriltag.is_valid

                # ── 2. MediaPipe landmark detection (runs on raw camera frame) ───
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                hand_detected = bool(
                    result and result.hand_landmarks and len(result.hand_landmarks) > 0
                )

                raw_lm = result.hand_landmarks[0] if hand_detected else None
                has_handedness = bool(
                    result and result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0
                )
                hand_label = result.handedness[0][0].category_name if has_handedness else None
                hand_score = float(result.handedness[0][0].score) if has_handedness else 0.85

                # ── Service Layer Queue & Hand Ingestion ───────────────────
                win_ready, norm_window, pixel_window = pipeline_service.process_frame(
                    raw_landmarks=raw_lm,
                    hand_label=hand_label,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    hand_score=hand_score,
                    timestamp=now,
                )

                if win_ready and norm_window and pixel_window:
                    logger.info("CameraWorker: Emitting window_ready event to detection viewmodel (%dx%d frame)", frame_w, frame_h)
                    self.window_ready.emit(
                        norm_window,
                        pixel_window,
                        frame_w,
                        frame_h,
                    )

                if hand_detected and not prev_hand_detected:
                    logger.info("MediaPipe: Hand detected in frame (%s, confidence=%.2f, landmarks=%d)", hand_label or "Active", hand_score, len(raw_lm) if raw_lm else 0)
                elif not hand_detected and prev_hand_detected:
                    logger.info("MediaPipe: Hand exited camera frame.")
                prev_hand_detected = hand_detected

                # ── Visual rendering (Play Mode only) ────────────────────────
                if self._render_video:
                    display_frame = raw_frame.copy()
                    if self._active_button_ids and time.perf_counter() > self._active_button_clear_t:
                        self._active_button_ids.clear()
                    if self._show_overlay and apriltag.is_valid:
                        display_frame = apriltag.annotate_frame(
                            display_frame, self._layout, active_button_ids=self._active_button_ids
                        )
                    if hand_detected and pipeline_service.last_pixel_coords is not None:
                        self._draw_skeleton(display_frame, pipeline_service.last_pixel_coords)
                    out_frame = display_frame
                else:
                    out_frame = None

                # ── Emit frame / telemetry ──────────────────────────────────
                self.frame_ready.emit(
                    out_frame,
                    actual_fps,
                    hand_detected,
                    apriltag.H.copy() if apriltag.is_valid else None,
                    apriltag.is_valid,
                )

        except Exception as exc:
            logger.exception("CameraWorker loop error: %s", exc)
            self.error.emit(str(exc))
        finally:
            grabber.stop()
            grabber.join(timeout=0.5)
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

        # Draw forward-extrapolated contact point indicators if offset is enabled
        if self._config and self._config.fingertip_offset_enabled:
            from config.constants import DIP_INDICES
            offset_mm = self._config.fingertip_forward_offset_mm
            for finger, tip_idx in FINGERTIP_INDICES.items():
                dip_idx = DIP_INDICES.get(finger)
                if dip_idx is not None and tip_idx < len(pts_pixel) and dip_idx < len(pts_pixel):
                    t_px, t_py = pts_pixel[tip_idx][:2]
                    d_px, d_py = pts_pixel[dip_idx][:2]
                    vx, vy = t_px - d_px, t_py - d_py
                    v_len = math.hypot(vx, vy)
                    if v_len > 1e-3:
                        # Extrapolate along distal segment in pixels
                        scale = (offset_mm / 25.0) * v_len
                        c_px = int(round(t_px + (vx / v_len) * scale))
                        c_py = int(round(t_py + (vy / v_len) * scale))

                        # Draw guide line and contact point ring
                        cv2.line(frame, (int(t_px), int(t_py)), (c_px, c_py), (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(frame, (c_px, c_py), 4, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (c_px, c_py), 6, (255, 255, 255), 1, cv2.LINE_AA)

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

