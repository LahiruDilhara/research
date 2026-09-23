"""
core/pipeline/touch_resolver.py

Resolves a model touch prediction into an exact key press.

Algorithm (per window trigger)
──────────────────────────────
1. Collect touch-positive fingers (prob >= threshold).
2. Sort by probability descending (highest confidence first).
3. For each finger in order:
   a. Find the IMPACT FRAME: the velocity step v in [0..3] where the fingertip's
      downward pixel speed is maximum (approximates the moment of surface contact).
   b. Get the fingertip pixel at frame (v+1), position right after impact.
   c. Apply H to map pixel → mm-space.
   d. Hit-test against all key bounding boxes.
   e. On first hit: return (key_id, finger, prob).
4. Return None if no hit for any touch-positive finger.

FIRST TOUCH WINS: Once a valid key hit is found, the remaining fingers are ignored.
"""

from __future__ import annotations

import math

import numpy as np

from config.constants import FINGERTIP_INDICES
from core.layout.layout_parser import ButtonData, LayoutData
from utils.logger import setup_logger

logger = setup_logger("TouchResolver")


class TouchResolver:
    """Resolves per-finger touch predictions into key press events."""

    def __init__(self, layout: LayoutData) -> None:
        self._buttons = layout.buttons

    # ── Public API ─────────────────────────────────────────────────────────────

    def resolve(
        self,
        touch_fingers: list[str],
        probs: dict[str, float],
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        """
        Parameters
        ----------
        touch_fingers   : Finger names confirmed as touch-positive by the model.
        probs           : Full per-finger probability dict from the model.
        pixel_window_5  : 5 frames × 21 landmarks as raw camera pixel coords.
                          Shape: [frame_idx][landmark_idx] → (px, py).
        H               : Homography matrix (camera pixel → mm-space).

        Returns
        -------
        (key_id, finger_name, probability) on first hit, or None.
        """
        if len(pixel_window_5) < 5:
            return None

        # Sort by probability, highest confidence is evaluated first
        sorted_fingers = sorted(touch_fingers, key=lambda f: probs.get(f, 0.0), reverse=True)

        for finger in sorted_fingers:
            result = self._resolve_finger(finger, probs[finger], pixel_window_5, H)
            if result is not None:
                return result

        return None

    # ── Private ────────────────────────────────────────────────────────────────

    def _resolve_finger(
        self,
        finger: str,
        prob: float,
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        tip_idx = FINGERTIP_INDICES[finger]

        # ── Step 1: find the impact frame (max downward speed on fingertip) ──
        impact_step = self._find_impact_step(tip_idx, pixel_window_5)

        # ── Step 2: get fingertip pixel at impact ─────────────────────────────
        # Use the position *after* the impact step (where the tip is at the surface)
        tip_px, tip_py = pixel_window_5[impact_step + 1][tip_idx]

        # ── Step 3: map pixel → mm-space via H ───────────────────────────────
        mm_point = self._pixel_to_mm(tip_px, tip_py, H)
        if mm_point is None:
            return None
        mm_x, mm_y = mm_point

        # ── Step 4: hit-test against key bounding boxes ───────────────────────
        hit = self._hit_test(mm_x, mm_y)
        if hit is None:
            return None

        logger.debug(
            "Touch resolved: finger=%s key=%s prob=%.2f tip=(%.1f,%.1f)px → (%.1f,%.1f)mm",
            finger, hit.id, prob, tip_px, tip_py, mm_x, mm_y,
        )
        return (hit.id, finger, prob)

    @staticmethod
    def _find_impact_step(
        tip_idx: int,
        pixel_window_5: list[list[tuple[float, float]]],
    ) -> int:
        """
        Among the 4 velocity steps (v=0..3), return the step index with the
        highest downward (positive dy) fingertip speed.
        Falls back to the last step if no downward motion is detected.
        """
        best_step = 3   # default to last step
        best_score = -1.0

        for v in range(4):
            prev_px, prev_py = pixel_window_5[v][tip_idx]
            curr_px, curr_py = pixel_window_5[v + 1][tip_idx]
            dy = curr_py - prev_py          # positive = downward on screen
            speed = math.hypot(curr_px - prev_px, curr_py - prev_py)

            # Only consider steps where the fingertip is moving downward
            score = speed if dy > 0 else 0.0
            if score > best_score:
                best_score = score
                best_step = v

        return best_step

    @staticmethod
    def _pixel_to_mm(
        px: float, py: float, H: np.ndarray
    ) -> tuple[float, float] | None:
        p_h = H @ np.array([px, py, 1.0], dtype=np.float64)
        if abs(p_h[2]) < 1e-9:
            return None
        return (p_h[0] / p_h[2], p_h[1] / p_h[2])

    def _hit_test(self, mm_x: float, mm_y: float) -> ButtonData | None:
        for btn in self._buttons:
            if btn.contains_mm(mm_x, mm_y):
                return btn
        return None
