"""
services/touch_pipeline_service.py

Touch Pipeline Service & Temporal Queue Manager.

Architecture & Responsibilities (SOLID & KISS):
─────────────────────────────────────────────────
1. 12 FPS Ingestion & Timestamp Synchronization:
   - Receives raw landmark stream and regulates temporal ingestion.
2. 5 Dedicated Per-Finger Queues:
   - Maintains 5 separate deques (maxlen=5) for: Thumb, Index, Middle, Ring, Pinky.
   - Preserves continuous spatial-temporal history per finger.
3. Hand State & Identity Tracking:
   - If no hand detected: purges all 5 queues and enters idle/standby state.
   - If hand identity/handedness switches: clears all queues.
4. Window Completion & Stride Control:
   - Triggers model inference when all 5 queues are filled (len == 5) and stride reached.
5. Process.sh Filtration Replication:
   - Step 7: HandMovementFilter (transit displacement <= 0.155)
   - Step 8: compute_window_velocities (4 velocity steps + speeds)
   - Step 9: KineticMotionFilter (zero-velocity touch suppression >= 0.008)
   - Step 10: WindowQualityFilter (avg score >= 0.65, min score >= 0.45, drop <= 0.35)
6. Dual-Threshold Hysteresis Debouncing:
   - Evaluates touch onset (>= 0.50) vs touch release (< 0.40).
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any

import numpy as np

from config.constants import (
    FINGERS,
    HAND_MOVEMENT_THRESHOLD,
    MIN_KINETIC_SPEED_THRESHOLD,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_ENABLED,
    ONE_EURO_MIN_CUTOFF,
    QUALITY_MAX_SCORE_DROP,
    QUALITY_MIN_AVG_SCORE,
    QUALITY_MIN_FRAME_SCORE,
    SHIFT_SIZE,
    TOUCH_ONSET_THRESHOLD,
    TOUCH_RELEASE_THRESHOLD,
    WINDOW_SIZE,
)
from core.interfaces.touch_model import ITouchModel
from core.pipeline.feature_extractor import compute_window_velocities
from core.pipeline.filters import (
    HandMovementFilter,
    KineticMotionFilter,
    OneEuroFilter,
    WindowQualityFilter,
)
from core.pipeline.normalizer import HandScaleNormalizer
from utils.logger import setup_logger

logger = setup_logger("TouchPipelineService")


class TouchPipelineService:
    """Service layer managing per-finger temporal sliding queues and parallel model inference."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        shift_size: int = SHIFT_SIZE,
        hand_movement_threshold: float = HAND_MOVEMENT_THRESHOLD,
        quality_min_avg_score: float = QUALITY_MIN_AVG_SCORE,
        quality_min_frame_score: float = QUALITY_MIN_FRAME_SCORE,
        quality_max_score_drop: float = QUALITY_MAX_SCORE_DROP,
        min_kinetic_speed: float = MIN_KINETIC_SPEED_THRESHOLD,
        touch_onset_threshold: float = TOUCH_ONSET_THRESHOLD,
        touch_release_threshold: float = TOUCH_RELEASE_THRESHOLD,
        velocity_threshold: float | None = None,
        one_euro_enabled: bool = ONE_EURO_ENABLED,
        one_euro_min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        one_euro_beta: float = ONE_EURO_BETA,
        one_euro_d_cutoff: float = ONE_EURO_D_CUTOFF,
    ) -> None:
        self._window_size = window_size
        self._shift_size = shift_size
        self._normalizer = HandScaleNormalizer()

        # Modular filters replicating process.sh and signal smoothing
        self._one_euro_filter = OneEuroFilter(
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
            enabled=one_euro_enabled,
        )
        self._hand_movement_filter = HandMovementFilter(threshold=hand_movement_threshold)
        self._window_quality_filter = WindowQualityFilter(
            min_avg_score=quality_min_avg_score,
            min_frame_score=quality_min_frame_score,
            max_score_drop=quality_max_score_drop,
        )
        self._velocity_threshold = (
            float(velocity_threshold) if velocity_threshold is not None else float(min_kinetic_speed)
        )
        self._kinetic_motion_filter = KineticMotionFilter(threshold=self._velocity_threshold)

        # Debouncing thresholds
        self._t_on = touch_onset_threshold
        self._t_off = touch_release_threshold

        # 5 separate queues for each finger
        self._finger_queues: dict[str, deque[dict[str, Any]]] = {
            f: deque(maxlen=self._window_size) for f in FINGERS
        }
        # Corresponding pixel history queue for touch coordinate resolution
        self._pixel_queue: deque[list[tuple[float, float]]] = deque(maxlen=self._window_size)

        # Per-finger touch state tracker for hysteresis debouncing
        self._finger_touch_state: dict[str, bool] = {f: False for f in FINGERS}

        self._shift_counter = 0
        self._last_hand_label: str | None = None
        self._hand_detected = False
        self._last_status = "IDLE"
        self._last_pixel_coords: list[tuple[float, float, float]] | None = None

    # ── Queue State Management ─────────────────────────────────────────────────

    def clear_queues(self) -> None:
        """Clear all 5 finger queues, coordinate history, and debounced touch states."""
        for f in FINGERS:
            self._finger_queues[f].clear()
            self._finger_touch_state[f] = False
        self._pixel_queue.clear()
        self._shift_counter = 0
        self._one_euro_filter.reset()
        self._last_pixel_coords = None

    def reset(self) -> None:
        """Fully reset the pipeline service state."""
        self.clear_queues()
        self._one_euro_filter.reset()
        self._last_hand_label = None
        self._hand_detected = False
        self._last_status = "IDLE"
        self._last_pixel_coords = None

    @property
    def is_queue_full(self) -> bool:
        """Returns True if all 5 finger queues are filled to window_size frames."""
        return all(len(q) == self._window_size for q in self._finger_queues.values())

    @property
    def queue_lengths(self) -> dict[str, int]:
        """Returns the current element count in each finger queue."""
        return {f: len(q) for f, q in self._finger_queues.items()}

    @property
    def last_status(self) -> str:
        """Diagnostic description of last processed window."""
        return self._last_status

    @property
    def last_pixel_coords(self) -> list[tuple[float, float, float]] | None:
        """Last filtered pixel coordinates of the active hand."""
        return self._last_pixel_coords

    # ── Frame Processing & Ingestion ───────────────────────────────────────────

    def process_frame(
        self,
        raw_landmarks: list[Any] | None,
        hand_label: str | None,
        frame_w: int,
        frame_h: int,
        hand_score: float = 0.85,
        timestamp: float | None = None,
    ) -> tuple[bool, list[dict[str, Any]] | None, list[list[tuple[float, float]]] | None]:
        """
        Process a single 12 FPS frame from MediaPipe.

        Returns:
            (window_ready, norm_window_5, pixel_window_5)
            If window_ready is True, both windows contain 5 frames of history.
        """
        if not raw_landmarks or len(raw_landmarks) == 0:
            if self._hand_detected or any(len(q) > 0 for q in self._finger_queues.values()):
                self.clear_queues()
                logger.info("Hand tracking lost, purged all 5 finger queues.")
            self._one_euro_filter.reset()
            self._hand_detected = False
            self._last_status = "NO_HAND"
            return False, None, None

        # Reset queues if hand identity / handedness changes
        if self._last_hand_label is not None and hand_label is not None and self._last_hand_label != hand_label:
            self.clear_queues()
            self._one_euro_filter.reset()
            logger.info("Handedness switched from %s to %s, queues reset.", self._last_hand_label, hand_label)
        self._last_hand_label = hand_label
        self._hand_detected = True

        # Extract raw pixel coordinates
        pts_pixel_raw: list[tuple[float, float, float]] = [
            (lm.x * frame_w, lm.y * frame_h, lm.z * frame_w)
            for lm in raw_landmarks
        ]

        # Apply One Euro adaptive speed filter on coordinates
        t_now = timestamp if timestamp is not None else time.time()
        pts_pixel = self._one_euro_filter.filter_points(pts_pixel_raw, t_now)
        self._last_pixel_coords = pts_pixel

        # Unitless scale normalization relative to L_hand
        norm_pts = self._normalizer.normalize(pts_pixel, center_wrist=True)
        # Build normalized dict with stationary anchors attached for transit displacement filtering
        norm_dict = self._normalizer.build_norm_dict(norm_pts, pts_px=pts_pixel)
        norm_dict["_hand_score"] = float(hand_score)

        # 2D pixel coordinates for touch coordinate resolution
        pixel_list: list[tuple[float, float]] = [(px, py) for px, py, _ in pts_pixel]

        # Ingest into the 5 separate finger queues
        for f in FINGERS:
            self._finger_queues[f].append(norm_dict)

        self._pixel_queue.append(pixel_list)
        self._shift_counter += 1

        # Trigger when all 5 queues are filled to window_size AND stride shifts have occurred
        if self.is_queue_full and self._shift_counter >= self._shift_size:
            self._shift_counter = 0
            norm_window_5 = list(self._finger_queues["Thumb"])
            pixel_window_5 = list(self._pixel_queue)
            logger.info(
                "5-frame temporal window assembled (stride=%d, hand=%s, confidence=%.2f)",
                self._shift_size, hand_label or "Active", hand_score,
            )
            return True, norm_window_5, pixel_window_5

        return False, None, None

    def reset_touch_states(self) -> None:
        """Reset all per-finger debouncing states to False."""
        self._finger_touch_state = {f: False for f in FINGERS}

    @property
    def velocity_threshold(self) -> float:
        """Current fingertip velocity threshold in L_hand per step."""
        return self._velocity_threshold

    def set_velocity_threshold(self, threshold: float) -> None:
        """Update fingertip velocity threshold dynamically."""
        self._velocity_threshold = float(threshold)
        self._kinetic_motion_filter.threshold = float(threshold)
        logger.info("Updated TouchPipelineService velocity threshold: %.4f", self._velocity_threshold)

    @property
    def touch_threshold(self) -> float:
        """Current touch onset probability threshold."""
        return self._t_on

    def set_touch_threshold(self, threshold: float) -> None:
        """Update touch probability onset and release thresholds dynamically."""
        self._t_on = float(threshold)
        # Keep release threshold strictly lower than onset threshold (approx 70% of onset)
        self._t_off = max(0.01, min(float(threshold) * 0.70, float(threshold) - 0.02))
        logger.info(
            "Updated TouchPipelineService touch thresholds: onset=%.2f, release=%.2f",
            self._t_on,
            self._t_off,
        )

    @property
    def hand_movement_threshold(self) -> float:
        """Current whole-hand movement displacement threshold."""
        return self._hand_movement_filter.threshold

    def set_hand_movement_threshold(self, threshold: float) -> None:
        """Update whole-hand movement displacement threshold dynamically."""
        self._hand_movement_filter.threshold = float(threshold)
        logger.info(
            "Updated TouchPipelineService hand movement threshold: %.4f",
            self._hand_movement_filter.threshold,
        )

    def set_quality_filter_thresholds(
        self,
        min_avg: float | None = None,
        min_frame: float | None = None,
        max_drop: float | None = None,
    ) -> None:
        """Update 5-frame sequence window confidence quality filter thresholds dynamically."""
        if min_avg is not None:
            self._window_quality_filter.min_avg_score = float(min_avg)
        if min_frame is not None:
            self._window_quality_filter.min_frame_score = float(min_frame)
        if max_drop is not None:
            self._window_quality_filter.max_score_drop = float(max_drop)
        logger.info(
            "Updated TouchPipelineService window quality thresholds: min_avg=%.2f, min_frame=%.2f, max_drop=%.2f",
            self._window_quality_filter.min_avg_score,
            self._window_quality_filter.min_frame_score,
            self._window_quality_filter.max_score_drop,
        )

    @property
    def one_euro_filter(self) -> OneEuroFilter:
        """One Euro coordinate filter instance."""
        return self._one_euro_filter

    def update_one_euro_settings(
        self,
        enabled: bool,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ) -> None:
        """Update One Euro coordinate filter parameters dynamically."""
        self._one_euro_filter.enabled = bool(enabled)
        self._one_euro_filter.min_cutoff = float(min_cutoff)
        self._one_euro_filter.beta = float(beta)
        self._one_euro_filter.d_cutoff = float(d_cutoff)
        logger.info(
            "Updated TouchPipelineService One Euro filter: enabled=%s, min_cutoff=%.2f, beta=%.2f, d_cutoff=%.2f",
            self._one_euro_filter.enabled,
            self._one_euro_filter.min_cutoff,
            self._one_euro_filter.beta,
            self._one_euro_filter.d_cutoff,
        )

    def run_parallel_inference(
        self,
        model: ITouchModel | None,
        norm_window_5: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Executes complete production filtration pipeline and model inference:
        1. Validates whole-hand transit movement (Step 7: displacement <= 0.155 L_hand).
        2. Validates window confidence and quality (Step 10: avg >= 0.65, min >= 0.45, drop <= 0.35).
        3. Computes 4 velocity steps across 5 frames (Step 8).
        4. Validates fingertip velocity threshold: if no fingertip exceeds the threshold,
           the window is ignored and neural network inference is bypassed.
        5. Runs active touch model inference forward-pass.
        6. Validates finger kinetic motion (Step 9: zero-velocity filter >= threshold).
        7. Applies dual-threshold debouncing hysteresis (onset >= 0.55, release < 0.40).

        Returns:
            Dict mapping each finger to {
                "touch": bool,
                "prob": float,
                "reason": str,
                "hand_moving": bool,
                "disp": float,
            }
        """
        if model is None:
            logger.warning("Pipeline: Active AI model is None. Cannot run inference.")
            self._last_status = "No Model Loaded"
            return {
                f: {
                    "touch": False,
                    "prob": 0.0,
                    "reason": "No Model Loaded",
                    "hand_moving": False,
                    "disp": 0.0,
                }
                for f in FINGERS
            }
        if len(norm_window_5) < self._window_size:
            logger.warning("Pipeline: Incomplete window (%d < %d frames).", len(norm_window_5), self._window_size)
            self._last_status = "Incomplete Window"
            return {
                f: {
                    "touch": False,
                    "prob": 0.0,
                    "reason": "Incomplete Window",
                    "hand_moving": False,
                    "disp": 0.0,
                }
                for f in FINGERS
            }

        # Step 7 Filtration: Whole-hand transit movement displacement filter
        is_stationary, max_disp, hm_reason = self._hand_movement_filter.validate(norm_window_5)
        if not is_stationary:
            self.reset_touch_states()
            self._last_status = hm_reason
            logger.info(
                "Pipeline Filter [Step 7 Hand Movement]: BLOCKED window - %s (stationary displacement=%.4f > limit=%.4f L_hand)",
                hm_reason, max_disp, self._hand_movement_filter.threshold,
            )
            return {
                f: {
                    "touch": False,
                    "prob": 0.0,
                    "reason": hm_reason,
                    "hand_moving": True,
                    "disp": max_disp,
                }
                for f in FINGERS
            }

        logger.info(
            "Pipeline Filter [Step 7 Hand Movement]: PASSED (%s)",
            hm_reason,
        )

        # Step 10 Filtration: Window tracking confidence quality checks
        if isinstance(norm_window_5, np.ndarray):
            scores_5 = [0.85] * len(norm_window_5)
        else:
            scores_5 = [float(f.get("_hand_score", 0.85)) if isinstance(f, dict) else 0.85 for f in norm_window_5]
        is_valid_quality, q_reason = self._window_quality_filter.validate(scores_5)
        scores_str = ", ".join(f"{s:.2f}" for s in scores_5)
        if not is_valid_quality:
            self.reset_touch_states()
            self._last_status = q_reason
            logger.info(
                "Pipeline Filter [Step 10 Window Quality]: BLOCKED window - %s (scores=[%s] vs min_avg=%.2f, min_frame=%.2f, max_drop=%.2f)",
                q_reason, scores_str, self._window_quality_filter.min_avg_score,
                self._window_quality_filter.min_frame_score, self._window_quality_filter.max_score_drop,
            )
            return {
                f: {
                    "touch": False,
                    "prob": 0.0,
                    "reason": q_reason,
                    "hand_moving": False,
                    "disp": max_disp,
                }
                for f in FINGERS
            }

        valid_scores = [s for s in scores_5 if s > 0.0] or [0.85]
        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        score_drop = max(valid_scores) - min(valid_scores)
        logger.info(
            "Pipeline Filter [Step 10 Window Quality]: PASSED (scores=[%s], avg=%.2f >= min_avg=%.2f, min=%.2f >= min_frame=%.2f, drop=%.2f <= max_drop=%.2f)",
            scores_str, avg_score, self._window_quality_filter.min_avg_score,
            min_score, self._window_quality_filter.min_frame_score,
            score_drop, self._window_quality_filter.max_score_drop,
        )

        # Step 8: Compute 4 velocity steps from normalized 5 frames
        v_steps_4 = compute_window_velocities(norm_window_5)

        # Fingertip Velocity Pre-Check:
        # Check whether any fingertip speed exceeds the velocity threshold across the 4 steps.
        # If no fingertip exceeds the speed threshold, the hand is idling or resting.
        # The entire window is ignored to bypass model inference and prevent false touches.
        tip_speeds = {
            f: self._kinetic_motion_filter.get_max_tip_speed(v_steps_4, f)
            for f in FINGERS
        }
        max_any_tip_speed = max(tip_speeds.values(), default=0.0)
        speeds_str = ", ".join(f"{f}: {tip_speeds[f]:.4f}" for f in FINGERS)

        if max_any_tip_speed < self._velocity_threshold:
            self.reset_touch_states()
            reason_str = (
                f"Fingertip speed below threshold (max={max_any_tip_speed:.4f} < limit={self._velocity_threshold:.4f})"
            )
            self._last_status = (
                f"Window Ignored: Max tip speed ({max_any_tip_speed:.4f}) "
                f"below velocity threshold ({self._velocity_threshold:.4f})"
            )
            logger.info(
                "Pipeline Filter [Step 8/9 Velocity Pre-Check]: BLOCKED window - %s | speeds=[%s]",
                reason_str, speeds_str,
            )
            return {
                f: {
                    "touch": False,
                    "prob": 0.0,
                    "reason": (
                        f"Fingertip speed ({tip_speeds[f]:.4f}) "
                        f"below velocity threshold ({self._velocity_threshold:.4f})"
                    ),
                    "hand_moving": False,
                    "disp": max_disp,
                }
                for f in FINGERS
            }

        logger.info(
            "Pipeline Filter [Step 8/9 Velocity Pre-Check]: PASSED (max_tip_speed=%.4f >= limit=%.4f L_hand | speeds=[%s])",
            max_any_tip_speed, self._velocity_threshold, speeds_str,
        )

        # Step 5: Execute active PyTorch neural network model prediction
        model_name = getattr(model, "name", type(model).__name__)
        logger.info("Pipeline AI Model: Evaluating 5-frame window with '%s'...", model_name)
        try:
            raw_probs = model.predict(norm_window_5)
        except Exception as exc:
            logger.error("Parallel model inference failed on '%s': %s", model_name, exc)
            self._last_status = f"Model Error: {exc}"
            return {
                f: {
                    "touch": False,
                    "prob": 0.0,
                    "reason": f"Model Error: {exc}",
                    "hand_moving": False,
                    "disp": max_disp,
                }
                for f in FINGERS
            }

        probs_str = ", ".join(f"{f}: {raw_probs.get(f, 0.0):.2f}" for f in FINGERS)
        logger.info(
            "Model Raw Predictions: [%s] (touch_onset_threshold=%.2f, release_threshold=%.2f)",
            probs_str, self._t_on, self._t_off,
        )

        # Step 9 Filtration + Dual-Threshold Hysteresis Debouncing
        results: dict[str, dict[str, Any]] = {}
        for f in FINGERS:
            p = float(raw_probs.get(f, 0.0))

            was_touch = self._finger_touch_state.get(f, False)
            if was_touch:
                # Retain active touch until probability falls below release cutoff
                is_touch = bool(p >= self._t_off)
                reason = "Touch Maintained (Debounce)" if is_touch else "Touch Released"
            else:
                # Trigger touch directly when raw model probability meets or exceeds onset threshold
                is_touch = bool(p >= self._t_on)
                reason = "Touch Detected" if is_touch else "Below Onset Threshold"

            self._finger_touch_state[f] = is_touch
            results[f] = {
                "touch": is_touch,
                "prob": p,
                "reason": reason,
                "hand_moving": False,
                "disp": max_disp,
            }

        active_touches = [f"{f} ({results[f]['prob']:.2f}, {results[f]['reason']})" for f in FINGERS if results[f]["touch"]]
        if active_touches:
            logger.info("Touch Hysteresis: Active touches: %s", ", ".join(active_touches))
        else:
            max_f = max(FINGERS, key=lambda f: results[f]["prob"])
            logger.info(
                "Touch Hysteresis: No finger reached onset threshold (highest=%s at %.2f, needed >= %.2f)",
                max_f, results[max_f]["prob"], self._t_on,
            )

        self._last_status = "OK"
        return results

