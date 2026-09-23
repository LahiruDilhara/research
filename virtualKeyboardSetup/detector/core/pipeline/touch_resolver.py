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

        # ── Step 1: find the impact frame (kinematic deceleration / turnaround) ──
        impact_frame = self._find_impact_frame(tip_idx, pixel_window_5)

        # ── Step 2: get fingertip pixel at contact frame ──────────────────────
        tip_px, tip_py = pixel_window_5[impact_frame][tip_idx]

        # ── Step 3: map pixel → mm-space via H ───────────────────────────────
        mm_point = self._pixel_to_mm(tip_px, tip_py, H)
        if mm_point is None:
            return None
        mm_x, mm_y = mm_point

        # ── Step 4: hit-test against key bounding boxes ───────────────────────
        hit = self._hit_test(mm_x, mm_y)
        if hit is None:
            logger.info(
                "Touch candidate outside keys: finger=%s prob=%.2f tip=(%.1f, %.1f)px mapped to (%.1f, %.1f)mm (no key intersected)",
                finger, prob, tip_px, tip_py, mm_x, mm_y,
            )
            return None

        logger.info(
            "Touch resolved: finger=%s key=%s prob=%.2f tip=(%.1f, %.1f)px mapped to (%.1f, %.1f)mm (impact frame=%d)",
            finger, hit.id, prob, tip_px, tip_py, mm_x, mm_y, impact_frame,
        )
        return (hit.id, finger, prob)

    @staticmethod
    def _find_impact_frame(
        tip_idx: int,
        pixel_window_5: list[list[tuple[float, float]]],
    ) -> int:
        """
        Identifies the exact physical contact frame k in [1..4] using kinematic
        deceleration and trajectory reversal:
        - Downward strike followed by upward rebound (direction reversal: V_{k-1} . V_k < 0)
        - Downward strike followed by resting still on surface (sharp deceleration: s_in >> s_out)
        - Invariant to camera tilt angle because scalar speed and 2D collinear reversal
          do not depend on screen-axis orientation.
        """
        # Compute step velocities V_0, V_1, V_2, V_3 and scalar speeds
        steps: list[tuple[float, float, float]] = []
        for v in range(4):
            p0 = pixel_window_5[v][tip_idx]
            p1 = pixel_window_5[v + 1][tip_idx]
            vx = p1[0] - p0[0]
            vy = p1[1] - p0[1]
            spd = math.hypot(vx, vy)
            steps.append((vx, vy, spd))

        best_frame = 3
        best_score = -1e9

        # Evaluate candidate contact frames k in {1, 2, 3}
        for k in range(1, 4):
            v_in_x, v_in_y, s_in = steps[k - 1]
            v_out_x, v_out_y, s_out = steps[k]

            if s_in < 1e-4:
                continue

            dot = v_in_x * v_out_x + v_in_y * v_out_y
            cos_theta = dot / (s_in * s_out) if (s_in * s_out) > 1e-6 else 0.0

            # 1. Trajectory reversal (rebound / bounce off key)
            if dot < 0.0:
                # Strong reversal: incoming approach reverses, frame k is the turnaround point
                score = s_in + (s_in - s_out) + s_in * (1.0 - cos_theta)
            else:
                # 2. Kinematic deceleration (landing and staying still on key)
                decel = s_in - s_out
                score = decel

            if score > best_score:
                best_score = score
                best_frame = k

        # Check Frame 4: if the strike occurred on step 3 (F3 -> F4) and landed at the window end
        s3 = steps[3][2]
        if s3 > 1.0 and s3 > best_score:
            best_frame = 4

        return best_frame

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
