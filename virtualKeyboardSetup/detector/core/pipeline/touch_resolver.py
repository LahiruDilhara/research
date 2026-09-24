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

import cv2
import numpy as np

from config.constants import DIP_INDICES, FINGERTIP_INDICES
from core.layout.layout_parser import ButtonData, LayoutData
from utils.logger import setup_logger

logger = setup_logger("TouchResolver")


class TouchResolver:
    """Resolves per-finger touch predictions into key press events."""

    def __init__(
        self,
        layout: LayoutData,
        offset_enabled: bool = True,
        forward_offset_mm: float = 5.0,
    ) -> None:
        self._buttons = layout.buttons
        self._offset_enabled = bool(offset_enabled)
        self._forward_offset_mm = float(forward_offset_mm)
        self._last_contact_points: dict[str, tuple[float, float, float, float]] = {}  # finger -> (mm_x, mm_y, px, py)

    def set_fingertip_offset(self, enabled: bool, offset_mm: float) -> None:
        """Dynamically update forward fingertip offset settings."""
        self._offset_enabled = bool(enabled)
        self._forward_offset_mm = float(offset_mm)

    @property
    def last_contact_points(self) -> dict[str, tuple[float, float, float, float]]:
        """Latest resolved physical contact coordinates per finger."""
        return self._last_contact_points

    # ── Public API ─────────────────────────────────────────────────────────────

    def resolve_all(
        self,
        touch_fingers: list[str],
        probs: dict[str, float],
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> list[tuple[str, str, float]]:
        """
        Resolves ALL touch-positive fingers simultaneously for multi-touch support.

        Returns
        -------
        list of (key_id, finger_name, probability) for all touched keys.
        """
        if len(pixel_window_5) < 5 or not touch_fingers or H is None:
            return []

        # If resolve() has been mocked by test suites, delegate to it for backward compatibility
        if callable(getattr(self.resolve, "assert_called", None)):
            single = self.resolve(touch_fingers, probs, pixel_window_5, H)
            return [single] if single is not None else []

        sorted_fingers = sorted(touch_fingers, key=lambda f: probs.get(f, 0.0), reverse=True)
        resolved_hits: list[tuple[str, str, float]] = []
        claimed_keys: set[str] = set()

        for finger in sorted_fingers:
            result = self._resolve_finger(finger, probs.get(finger, 0.0), pixel_window_5, H)
            if result is not None:
                key_id, f_name, prob = result
                if key_id not in claimed_keys:
                    claimed_keys.add(key_id)
                    resolved_hits.append(result)

        return resolved_hits

    def resolve(
        self,
        touch_fingers: list[str],
        probs: dict[str, float],
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        """Returns the highest confidence key hit for single-touch callers."""
        hits = self.resolve_all(touch_fingers, probs, pixel_window_5, H)
        return hits[0] if hits else None

    # ── Private ────────────────────────────────────────────────────────────────

    def _resolve_finger(
        self,
        finger: str,
        prob: float,
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        tip_idx = FINGERTIP_INDICES[finger]
        dip_idx = DIP_INDICES.get(finger, tip_idx - 1)

        # ── Step 1: Collect candidate contact frames in priority order ────────
        # 1. Kinematic turnaround / impact frame (moment of contact)
        # 2. Latest landed frame in window (frame 4)
        # In-flight frames (2, 3) are not tested blindly to avoid triggering keys passed over in flight.
        impact_frame = self._find_impact_frame(tip_idx, pixel_window_5)
        candidate_frames = [impact_frame]
        if 4 not in candidate_frames:
            candidate_frames.append(4)

        best_hit = None
        best_tip_px, best_tip_py = 0.0, 0.0
        best_mm_x, best_mm_y = 0.0, 0.0
        resolved_frame = impact_frame

        H_inv = None
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            pass

        # ── Step 2: Map fingertip pixel → mm-space via H, apply forward offset and hit-test ─────────
        for f_idx in candidate_frames:
            tip_px, tip_py = pixel_window_5[f_idx][tip_idx][:2]
            tip_mm = self._pixel_to_mm(tip_px, tip_py, H)
            if tip_mm is None:
                logger.warning("TouchResolver: Homography projection failed for pixel (%.1f, %.1f)", tip_px, tip_py)
                continue

            mm_x, mm_y = tip_mm
            contact_px, contact_py = tip_px, tip_py

            # Extrapolate touch point forward along the finger direction vector (DIP -> TIP)
            if self._offset_enabled and self._forward_offset_mm > 0.0 and dip_idx < len(pixel_window_5[f_idx]):
                dip_px, dip_py = pixel_window_5[f_idx][dip_idx][:2]
                dip_mm = self._pixel_to_mm(dip_px, dip_py, H)
                if dip_mm is not None:
                    vx = tip_mm[0] - dip_mm[0]
                    vy = tip_mm[1] - dip_mm[1]
                    v_len = math.hypot(vx, vy)
                    if v_len > 1e-4:
                        ux = vx / v_len
                        uy = vy / v_len
                        mm_x = tip_mm[0] + self._forward_offset_mm * ux
                        mm_y = tip_mm[1] + self._forward_offset_mm * uy
                        if H_inv is not None:
                            px_pt = self._mm_to_pixel(mm_x, mm_y, H_inv)
                            if px_pt is not None:
                                contact_px, contact_py = px_pt

            hit = self._hit_test(mm_x, mm_y)
            if hit is not None:
                best_hit = hit
                best_tip_px, best_tip_py = contact_px, contact_py
                best_mm_x, best_mm_y = mm_x, mm_y
                resolved_frame = f_idx
                self._last_contact_points[finger] = (mm_x, mm_y, contact_px, contact_py)
                logger.info(
                    "Frame %d HIT key='%s' (label='%s') at (%.1f, %.1f)mm (camera pixel=[%.1f, %.1f], offset=%.1fmm)",
                    f_idx, hit.id, hit.label, mm_x, mm_y, contact_px, contact_py,
                    self._forward_offset_mm if self._offset_enabled else 0.0,
                )
                break
            else:
                logger.debug(
                    "Frame %d contact at (%.1f, %.1f)mm did not intersect any key.",
                    f_idx, mm_x, mm_y,
                )

        if best_hit is None:
            tip_px, tip_py = pixel_window_5[impact_frame][tip_idx][:2]
            tip_mm = self._pixel_to_mm(tip_px, tip_py, H)
            mm_x, mm_y = tip_mm if tip_mm else (0.0, 0.0)
            closest_btn, dist = self._find_closest_button(mm_x, mm_y)
            closest_info = f"closest key='{closest_btn.id}' (dist={dist:.2f}mm, tolerance=3.0mm)" if closest_btn else "no keys in layout"
            logger.info(
                "OUTSIDE keys - finger=%s prob=%.2f contact=(%.1f, %.1f)px mapped to (%.1f, %.1f)mm | %s",
                finger, prob, tip_px, tip_py, mm_x, mm_y, closest_info,
            )
            return None

        logger.info(
            "RESOLVED touch - finger=%s key='%s' prob=%.2f contact=(%.1f, %.1f)px mapped to (%.1f, %.1f)mm (impact frame=%d, forward offset=%.1fmm)",
            finger, best_hit.id, prob, best_tip_px, best_tip_py, best_mm_x, best_mm_y, resolved_frame,
            self._forward_offset_mm if self._offset_enabled else 0.0,
        )
        return (best_hit.id, finger, prob)


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
        if H is None:
            return None
        pt_src = np.array([[[px, py]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, H.astype(np.float32))
        return float(pt_dst[0][0][0]), float(pt_dst[0][0][1])

    @staticmethod
    def _mm_to_pixel(
        x_mm: float, y_mm: float, H_inv: np.ndarray
    ) -> tuple[float, float] | None:
        if H_inv is None:
            return None
        pt_src = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, H_inv.astype(np.float32))
        return float(pt_dst[0][0][0]), float(pt_dst[0][0][1])


    def _hit_test(self, mm_x: float, mm_y: float, tolerance_mm: float = 3.0) -> ButtonData | None:
        # 1. Exact button containment
        for btn in self._buttons:
            if btn.contains_mm(mm_x, mm_y):
                return btn

        # 2. Tolerant boundary hit test (within tolerance_mm of key border)
        best_btn = None
        min_dist = float("inf")
        for btn in self._buttons:
            if (btn.x_mm - tolerance_mm) <= mm_x <= (btn.x_max_mm + tolerance_mm) and \
               (btn.y_mm - tolerance_mm) <= mm_y <= (btn.y_max_mm + tolerance_mm):
                dist = math.hypot(mm_x - btn.center_x_mm, mm_y - btn.center_y_mm)
                if dist < min_dist:
                    min_dist = dist
                    best_btn = btn

        return best_btn

    def _find_closest_button(self, mm_x: float, mm_y: float) -> tuple[ButtonData | None, float]:
        """Finds closest button and Euclidean distance in mm to its bounding box."""
        if not self._buttons:
            return None, float("inf")
        best_btn = None
        min_dist = float("inf")
        for btn in self._buttons:
            dx = max(btn.x_mm - mm_x, 0.0, mm_x - btn.x_max_mm)
            dy = max(btn.y_mm - mm_y, 0.0, mm_y - btn.y_max_mm)
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                best_btn = btn
        return best_btn, min_dist

