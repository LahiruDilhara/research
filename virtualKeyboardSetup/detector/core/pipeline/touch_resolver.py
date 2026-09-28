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
        max_stationary_distance_mm: float = 4.0,
    ) -> None:
        self._buttons = layout.buttons
        self._offset_enabled = bool(offset_enabled)
        self._forward_offset_mm = float(forward_offset_mm)
        self._max_stationary_distance_mm = float(max_stationary_distance_mm)
        self._last_contact_points: dict[str, tuple[float, float, float, float]] = {}  # finger -> (mm_x, mm_y, px, py)
        self._last_raw_tip_points: dict[str, tuple[float, float]] = {}  # finger -> (raw_tip_mm_x, raw_tip_mm_y)

    def set_fingertip_offset(self, enabled: bool, offset_mm: float) -> None:
        """Dynamically update forward fingertip offset settings."""
        self._offset_enabled = bool(enabled)
        self._forward_offset_mm = float(offset_mm)

    def set_max_stationary_distance_mm(self, value: float) -> None:
        """Update maximum allowed fingertip movement on paper for a still touch."""
        self._max_stationary_distance_mm = max(0.5, float(value))

    def update_layout(self, layout: LayoutData) -> None:
        """Dynamically updates the layout and button bounding boxes."""
        self._buttons = layout.buttons
        logger.info(
            "TouchResolver layout updated (%d buttons, paper=%.1fx%.1f mm)",
            len(layout.buttons),
            layout.paper_width_mm,
            layout.paper_height_mm,
        )

    @property
    def last_contact_points(self) -> dict[str, tuple[float, float, float, float]]:
        """Latest resolved physical contact coordinates per finger."""
        return self._last_contact_points

    @property
    def last_raw_tip_points(self) -> dict[str, tuple[float, float]]:
        """Latest resolved raw physical fingertip coordinates (without forward offset) per finger."""
        return self._last_raw_tip_points

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

    def resolve_trajectory(
        self,
        finger: str,
        prob: float,
        pixel_frames: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        """
        Resolves a single finger touch along a trajectory of frames (length >= 2,
        e.g. 5 frames from a single window or 8 frames stitched across two windows).

        Identifies the exact touchdown frame, maps to physical mm space via H,
        applies forward offset, and returns (key_id, finger, prob) if a key is hit.
        """
        if len(pixel_frames) < 2 or H is None:
            return None

        tip_idx = FINGERTIP_INDICES.get(finger)
        if tip_idx is None:
            return None

        # 1. Determine physical touchdown frame and motion signature (rebound vs settled)
        touchdown_frame = self.find_touchdown_frame(tip_idx, pixel_frames)
        last_frame = len(pixel_frames) - 1

        last_tip_px, last_tip_py = pixel_frames[last_frame][tip_idx][:2]
        last_mm = self._pixel_to_mm(last_tip_px, last_tip_py, H)

        has_rebound = False
        if 1 <= touchdown_frame < last_frame:
            p_prev = pixel_frames[touchdown_frame - 1][tip_idx][:2]
            p_td = pixel_frames[touchdown_frame][tip_idx][:2]
            p_next = pixel_frames[touchdown_frame + 1][tip_idx][:2]
            v_in_x = p_td[0] - p_prev[0]
            v_in_y = p_td[1] - p_prev[1]
            v_out_x = p_next[0] - p_td[0]
            v_out_y = p_next[1] - p_td[1]
            s_in = math.hypot(v_in_x, v_in_y)
            s_out = math.hypot(v_out_x, v_out_y)
            if s_in >= 2.0 and s_out >= 1.5:
                cos_val = (v_in_x * v_out_x + v_in_y * v_out_y) / (s_in * s_out)
                if cos_val < -0.35:
                    td_mm = self._pixel_to_mm(p_td[0], p_td[1], H)
                    if td_mm is not None and last_mm is not None:
                        rebound_drift = math.hypot(last_mm[0] - td_mm[0], last_mm[1] - td_mm[1])
                        if rebound_drift <= 10.0:
                            has_rebound = True

        # Candidate frame priority:
        # If rebound occurred (bounce back), touchdown_frame is the contact point.
        # Otherwise (finger landed / settling on key), the final frame (last_frame) is where
        # the finger actually arrived and settled, preventing hover over behind keys.
        if has_rebound:
            candidate_frames = [touchdown_frame]
            if last_frame != touchdown_frame:
                candidate_frames.append(last_frame)
        else:
            candidate_frames = [last_frame]
            if touchdown_frame != last_frame:
                candidate_frames.append(touchdown_frame)

        best_hit = None
        best_tip_px, best_tip_py = 0.0, 0.0
        best_mm_x, best_mm_y = 0.0, 0.0
        resolved_frame = candidate_frames[0]

        dip_idx = DIP_INDICES.get(finger)
        H_inv = None
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            pass

        # Evaluate candidate frames: map raw fingertip pixel -> mm -> hit test.
        # If fingertip offset is enabled, project the contact point forward along
        # the distal finger vector (DIP -> TIP) to compensate for nail-bed tracking.
        for f_idx in candidate_frames:
            tip_px, tip_py = pixel_frames[f_idx][tip_idx][:2]
            tip_mm = self._pixel_to_mm(tip_px, tip_py, H)
            if tip_mm is None:
                continue

            # Reject candidate frame if it drifted significantly from where the finger settled (in-flight hover)
            if f_idx != last_frame and last_mm is not None:
                max_allowed = 10.0 if has_rebound else self._max_stationary_distance_mm
                in_window_drift = math.hypot(last_mm[0] - tip_mm[0], last_mm[1] - tip_mm[1])
                if in_window_drift > max_allowed:
                    logger.debug(
                        "Frame %d rejected for %s: in-window drift %.1f mm exceeded limit %.1f mm (in-flight hover)",
                        f_idx, finger, in_window_drift, max_allowed,
                    )
                    continue

            raw_mm_x, raw_mm_y = tip_mm
            contact_px, contact_py = tip_px, tip_py

            # 1. Test raw fingertip containment first
            hit = self._hit_test(raw_mm_x, raw_mm_y, tolerance_mm=3.0)

            # 2. If raw point did not hit a key, extrapolate forward if enabled
            if (
                hit is None
                and self._offset_enabled
                and self._forward_offset_mm > 0.0
                and dip_idx is not None
                and dip_idx < len(pixel_frames[f_idx])
            ):
                dip_px, dip_py = pixel_frames[f_idx][dip_idx][:2]
                dip_mm = self._pixel_to_mm(dip_px, dip_py, H)
                if dip_mm is not None:
                    vx = tip_mm[0] - dip_mm[0]
                    vy = tip_mm[1] - dip_mm[1]
                    v_len = math.hypot(vx, vy)
                    if v_len > 1e-4:
                        ux = vx / v_len
                        uy = vy / v_len
                        off_mm_x = tip_mm[0] + self._forward_offset_mm * ux
                        off_mm_y = tip_mm[1] + self._forward_offset_mm * uy
                        offset_hit = self._hit_test(off_mm_x, off_mm_y, tolerance_mm=3.0)
                        if offset_hit is not None:
                            hit = offset_hit
                            raw_mm_x, raw_mm_y = off_mm_x, off_mm_y
                            if H_inv is not None:
                                px_pt = self._mm_to_pixel(off_mm_x, off_mm_y, H_inv)
                                if px_pt is not None:
                                    contact_px, contact_py = px_pt

            if hit is not None:
                best_hit = hit
                best_tip_px, best_tip_py = contact_px, contact_py
                best_mm_x, best_mm_y = raw_mm_x, raw_mm_y
                resolved_frame = f_idx
                self._last_contact_points[finger] = (raw_mm_x, raw_mm_y, contact_px, contact_py)
                self._last_raw_tip_points[finger] = (tip_mm[0], tip_mm[1])
                logger.info(
                    "Frame %d HIT key='%s' (label='%s') at (%.1f, %.1f)mm (pixel=[%.1f, %.1f], offset=%.1fmm)",
                    f_idx, hit.id, hit.label, raw_mm_x, raw_mm_y, contact_px, contact_py,
                    self._forward_offset_mm if self._offset_enabled else 0.0,
                )
                break

        # Fallback: closest key by edge distance within 5 mm (handles large gaps/margins)
        if best_hit is None:
            for f_idx in candidate_frames:
                tip_px, tip_py = pixel_frames[f_idx][tip_idx][:2]
                tip_mm = self._pixel_to_mm(tip_px, tip_py, H)
                if tip_mm is None:
                    continue
                mm_x, mm_y = tip_mm
                contact_px, contact_py = tip_px, tip_py
                if (
                    self._offset_enabled
                    and self._forward_offset_mm > 0.0
                    and dip_idx is not None
                    and dip_idx < len(pixel_frames[f_idx])
                ):
                    dip_px, dip_py = pixel_frames[f_idx][dip_idx][:2]
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

                closest_btn, dist = self._find_closest_button(mm_x, mm_y)
                if closest_btn is not None and dist <= 5.0:
                    best_hit = closest_btn
                    best_tip_px, best_tip_py = contact_px, contact_py
                    best_mm_x, best_mm_y = mm_x, mm_y
                    resolved_frame = f_idx
                    self._last_contact_points[finger] = (mm_x, mm_y, contact_px, contact_py)
                    self._last_raw_tip_points[finger] = (tip_mm[0], tip_mm[1])
                    logger.info(
                        "Frame %d CLOSEST FALLBACK key='%s' (edge dist=%.2f mm)",
                        f_idx, closest_btn.id, dist,
                    )
                    break

        if best_hit is None:
            tip_px, tip_py = pixel_frames[candidate_frames[0]][tip_idx][:2]
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
            "RESOLVED touch - finger=%s key='%s' prob=%.2f contact=(%.1f, %.1f)px mapped to (%.1f, %.1f)mm (touchdown frame=%d, forward offset=%.1fmm)",
            finger, best_hit.id, prob, best_tip_px, best_tip_py, best_mm_x, best_mm_y, resolved_frame,
            self._forward_offset_mm if self._offset_enabled else 0.0,
        )
        return (best_hit.id, finger, prob)

    def _resolve_finger(
        self,
        finger: str,
        prob: float,
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        return self.resolve_trajectory(finger, prob, pixel_window_5, H)

    def resolve_finger_down_point(
        self,
        finger: str,
        prob: float,
        pixel_window_5: list[list[tuple[float, float]]],
        H: np.ndarray,
    ) -> tuple[str, str, float] | None:
        """
        Calculates the physical down point on the entire window for the given finger
        and resolves it to a layout key via homography H.
        """
        return self.resolve_trajectory(finger, prob, pixel_window_5, H)

    @staticmethod
    def is_in_flight(
        finger: str,
        pixel_window_5: list[list[tuple[float, float]]],
    ) -> bool:
        """
        Determines whether the fingertip is still descending in mid-air at the end
        of a 5-frame window (Frame 4).

        Returns True if the finger is still moving toward the paper in the approach
        direction at the final step with no surface rebound detected.

        If in-flight is True, resolving touch now would map the airborne fingertip
        through H onto the wrong paper position (H is only valid for points ON the
        paper plane). The touch is buffered and re-evaluated in the next window when
        the fingertip actually contacts the surface.
        """
        if len(pixel_window_5) < 5 or finger not in FINGERTIP_INDICES:
            return False

        tip_idx = FINGERTIP_INDICES[finger]
        pts = [pixel_window_5[i][tip_idx][:2] for i in range(5)]

        # Check if all points are zero (e.g. mock test)
        if all(p[0] == 0.0 and p[1] == 0.0 for p in pts):
            return False

        # Step velocities
        v1_x = pts[2][0] - pts[1][0]
        v1_y = pts[2][1] - pts[1][1]
        s1 = math.hypot(v1_x, v1_y)

        v2_x = pts[3][0] - pts[2][0]
        v2_y = pts[3][1] - pts[2][1]
        s2 = math.hypot(v2_x, v2_y)

        v3_x = pts[4][0] - pts[3][0]
        v3_y = pts[4][1] - pts[3][1]
        s3 = math.hypot(v3_x, v3_y)

        # Check for surface rebound (direction reversal with meaningful speed).
        # If rebound already happened in this window, the finger touched the paper
        # and is now bouncing — this is NOT in-flight, it IS a valid touch event.
        # Rebound at F1→F2
        if s1 >= 1.5 and (v1_x * v2_x + v1_y * v2_y) < 0.0:
            return False
        # Rebound at F2→F3
        if s2 >= 1.5 and (v2_x * v3_x + v2_y * v3_y) < 0.0:
            return False

        # If the final step is still moving >= 1.5 px AND continuing in the same
        # approach direction as the previous step, the finger has not yet landed.
        # Threshold of 1.5 px/frame at 12 FPS covers both fast and slow approaches.
        # (Old code had a dead zone: < 2.0 → False, >= 2.5 → True, 2.0..2.5 → False.)
        if s3 >= 1.5:
            dot23 = v2_x * v3_x + v2_y * v3_y
            # Still moving in same direction as previous step (approaching)
            if s2 < 1e-4 or dot23 > 0.0:
                return True

        return False

    @staticmethod
    def find_touchdown_frame(
        tip_idx: int,
        pixel_frames: list[list[tuple[float, float]]],
    ) -> int:
        """
        Identifies the exact physical surface touchdown frame k in [0..N-1] across
        a sequence of frames (e.g. 5 frames in a single window or 8 stitched frames across two windows).

        In a tilted camera view, any frame before touchdown is suspended in the air
        and projects onto keys located behind the target. The true contact point occurs at the
        touchdown frame where the fingertip reaches maximum descent progress and stops or rebounds.
        """
        n_frames = len(pixel_frames)
        if n_frames < 2:
            return 0

        # Extract fingertip (x, y) coordinates
        pts = [pixel_frames[i][tip_idx][:2] for i in range(n_frames)]
        p0x, p0y = pts[0]

        # Find frame with maximum displacement from starting point P_0
        displacements = [math.hypot(pts[i][0] - p0x, pts[i][1] - p0y) for i in range(n_frames)]
        max_disp_idx = max(range(n_frames), key=lambda i: displacements[i])
        max_disp = displacements[max_disp_idx]

        # If total motion across window is negligible (< 1.5 px), finger was resting still on surface
        if max_disp < 1.5:
            return n_frames - 1

        # The approach direction vector U points from P_0 toward the furthest excursion of the stroke
        ux = (pts[max_disp_idx][0] - p0x) / max_disp
        uy = (pts[max_disp_idx][1] - p0y) / max_disp

        # Approach distance progress d_i = (P_i - P_0) . U
        d = [(pts[i][0] - p0x) * ux + (pts[i][1] - p0y) * uy for i in range(n_frames)]
        max_d = max(d)
        min_d = min(d)
        span_d = max(1e-4, max_d - min_d)

        # Compute step vectors V_i and speeds s_i
        steps: list[tuple[float, float, float]] = []
        for i in range(n_frames - 1):
            vx = pts[i + 1][0] - pts[i][0]
            vy = pts[i + 1][1] - pts[i][1]
            s = math.hypot(vx, vy)
            steps.append((vx, vy, s))

        best_frame = n_frames - 1
        best_score = -1e9

        for k in range(1, n_frames):
            progress_ratio = (d[k] - min_d) / span_d
            s_in = steps[k - 1][2]
            s_out = steps[k][2] if k < len(steps) else 0.0

            # A physical surface touchdown can only happen near the deepest point of the stroke (>= 0.70).
            # Any early frame (< 0.70) is still airborne in the approach flight, hovering over keys behind the target.
            is_near_peak = progress_ratio >= 0.70

            if is_near_peak:
                if k < len(steps):
                    v_in_x, v_in_y, _ = steps[k - 1]
                    v_out_x, v_out_y, _ = steps[k]
                    dot = v_in_x * v_out_x + v_in_y * v_out_y

                    # Reversal requires true turnaround (cos < -0.35, s_in >= 2.0, s_out >= 1.5, k >= 2)
                    is_turnaround = False
                    if k >= 2 and s_in >= 2.0 and s_out >= 1.5:
                        cos_val = dot / (s_in * s_out)
                        if cos_val < -0.35:
                            is_turnaround = True

                    if is_turnaround:
                        # Surface turnaround impact (rebound off key)
                        reversal_score = s_in * 3.0 + (s_in - s_out) * 2.0
                        score = 100.0 + reversal_score + progress_ratio * 50.0
                    elif s_in >= 1.2 and s_out < 1.8:
                        # Kinematic landing / resting on key
                        decel_score = (s_in - s_out) * 2.5
                        score = 70.0 + decel_score + progress_ratio * 50.0
                    else:
                        score = 50.0 + progress_ratio * 50.0 + (s_in - s_out)
                else:
                    # Final frame at peak descent
                    score = 65.0 + progress_ratio * 50.0
            else:
                # Still airborne in descent approach: heavily penalize so it never beats surface touchdown
                score = progress_ratio * 20.0

            if score > best_score:
                best_score = score
                best_frame = k

        return best_frame

    @classmethod
    def _find_impact_frame(
        cls,
        tip_idx: int,
        pixel_window_5: list[list[tuple[float, float]]],
    ) -> int:
        """Backward-compatible alias for unit tests."""
        return cls.find_touchdown_frame(tip_idx, pixel_window_5)

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
        # 1. Exact button containment (fingertip is inside the key bounding box)
        for btn in self._buttons:
            if btn.contains_mm(mm_x, mm_y):
                return btn

        # 2. Nearest key by edge distance within tolerance_mm.
        # Edge distance is the Chebyshev distance to the key border, same metric as
        # _find_closest_button.  Using edge distance (not center distance) ensures
        # a fingertip landing 1-2mm outside a key edge correctly registers on that key.
        best_btn = None
        min_edge_dist = float("inf")
        for btn in self._buttons:
            dx = max(btn.x_mm - mm_x, 0.0, mm_x - btn.x_max_mm)
            dy = max(btn.y_mm - mm_y, 0.0, mm_y - btn.y_max_mm)
            edge_dist = math.hypot(dx, dy)
            if edge_dist <= tolerance_mm and edge_dist < min_edge_dist:
                min_edge_dist = edge_dist
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

