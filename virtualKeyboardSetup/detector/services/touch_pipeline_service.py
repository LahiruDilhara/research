"""
services/touch_pipeline_service.py

Touch Pipeline Service & Temporal Queue Manager.

Architecture & Responsibilities:
─────────────────────────────────
1. 12 FPS Ingestion & Timestamp Synchronization:
   - Receives raw landmark stream and regulates temporal ingestion.
2. 5 Dedicated Per-Finger Queues:
   - Maintains 5 separate deques (maxlen=5) for: Thumb, Index, Middle, Ring, Pinky.
   - Preserves continuous spatial-temporal history per finger.
3. Hand State & Identity Tracking:
   - If no hand detected: immediately purges all 5 queues and enters idle/standby state.
   - If hand identity/handedness switches (e.g., Left -> Right): clears all queues and continues.
4. Window Completion & Stride Control:
   - Only triggers model inference when all 5 queues are filled to capacity (len == 5).
5. Batched 5-Finger Parallel Inference:
   - Dispatches the 5-finger temporal window to the active neural network simultaneously
     via batched tensor processing.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from config.constants import FINGERS, SHIFT_SIZE, WINDOW_SIZE
from core.interfaces.touch_model import ITouchModel
from core.pipeline.normalizer import HandScaleNormalizer
from utils.logger import setup_logger

logger = setup_logger("TouchPipelineService")


class TouchPipelineService:
    """Service layer managing per-finger temporal sliding queues and parallel model inference."""

    def __init__(self, window_size: int = WINDOW_SIZE, shift_size: int = SHIFT_SIZE) -> None:
        self._window_size = window_size
        self._shift_size = shift_size
        self._normalizer = HandScaleNormalizer()

        # 5 separate queues for each finger
        self._finger_queues: dict[str, deque[dict[str, float]]] = {
            f: deque(maxlen=self._window_size) for f in FINGERS
        }
        # Corresponding pixel history queue for touch coordinate resolution
        self._pixel_queue: deque[list[tuple[float, float]]] = deque(maxlen=self._window_size)

        self._shift_counter = 0
        self._last_hand_label: str | None = None
        self._hand_detected = False

    # ── Queue State Management ─────────────────────────────────────────────────

    def clear_queues(self) -> None:
        """Clear all 5 finger queues and coordinate history."""
        for f in FINGERS:
            self._finger_queues[f].clear()
        self._pixel_queue.clear()
        self._shift_counter = 0

    def reset(self) -> None:
        """Fully reset the pipeline service state."""
        self.clear_queues()
        self._last_hand_label = None
        self._hand_detected = False

    @property
    def is_queue_full(self) -> bool:
        """Returns True if all 5 finger queues are filled to 5 frames."""
        return all(len(q) == self._window_size for q in self._finger_queues.values())

    @property
    def queue_lengths(self) -> dict[str, int]:
        """Returns the current element count in each finger queue."""
        return {f: len(q) for f, q in self._finger_queues.items()}

    # ── Frame Processing & Ingestion ───────────────────────────────────────────

    def process_frame(
        self,
        raw_landmarks: list[Any] | None,
        hand_label: str | None,
        frame_w: int,
        frame_h: int,
    ) -> tuple[bool, list[dict[str, float]] | None, list[list[tuple[float, float]]] | None]:
        """
        Process a single 12 FPS frame from MediaPipe.

        Returns:
            (window_ready, norm_window_5, pixel_window_5)
            If window_ready is True, both windows contain 5 frames of history.
        """
        if not raw_landmarks or len(raw_landmarks) == 0:
            # No hand detected: clear queues and stay/idle
            if self._hand_detected or any(len(q) > 0 for q in self._finger_queues.values()):
                self.clear_queues()
            self._hand_detected = False
            return False, None, None

        self._hand_detected = True

        # Extract pixel coordinates
        pts_pixel: list[tuple[float, float, float]] = [
            (lm.x * frame_w, lm.y * frame_h, lm.z * frame_w)
            for lm in raw_landmarks
        ]

        # Unitless scale normalization relative to L_hand
        norm_pts = self._normalizer.normalize(pts_pixel, center_wrist=True)
        norm_dict = self._normalizer.build_norm_dict(norm_pts)

        # 2D pixel coordinates for touch resolution
        pixel_list: list[tuple[float, float]] = [(px, py) for px, py, _ in pts_pixel]

        # Ingest into the 5 separate finger queues
        for f in FINGERS:
            self._finger_queues[f].append(norm_dict)

        self._pixel_queue.append(pixel_list)
        self._shift_counter += 1

        # Only trigger when all 5 queues are filled to 5 frames AND 2 shifts have occurred
        if self.is_queue_full and self._shift_counter >= self._shift_size:
            self._shift_counter = 0
            # Retrieve 5-frame window from queues
            norm_window_5 = list(self._finger_queues["Thumb"])
            pixel_window_5 = list(self._pixel_queue)
            return True, norm_window_5, pixel_window_5

        return False, None, None

        return False, None, None

    # ── Parallel Model Inference ───────────────────────────────────────────────

    def run_parallel_inference(
        self,
        model: ITouchModel | None,
        norm_window_5: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Runs batched inference across all 5 finger channels in parallel.
        Returns a probability dictionary: {"Thumb": p1, "Index": p2, "Middle": p3, "Ring": p4, "Pinky": p5}.
        """
        if model is None or len(norm_window_5) < self._window_size:
            return {f: 0.0 for f in FINGERS}

        try:
            probs = model.predict(norm_window_5)
            return {f: float(probs.get(f, 0.0)) for f in FINGERS}
        except Exception as exc:
            logger.error("Parallel model inference failed: %s", exc)
            return {f: 0.0 for f in FINGERS}
