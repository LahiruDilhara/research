"""
core/pipeline/homography_engine.py

High-performance computer vision & homography engine replicating designer/analyzer/homography_engine.py.
Optimized for real-time CPU execution:
- Uses OpenCV ArUcoDetector (cv2.aruco.ArucoDetector) with sub-pixel corner refinement.
- Computes planar homography H mapping camera pixels to paper coordinates in mm.
- Employs temporal homography smoothing to eliminate frame-to-frame coordinate jitter.
- Uses cv2.perspectiveTransform for sub-millisecond button and paper boundary projections.
- Supports multi-button active highlighting and spatial tolerance hit testing.
"""

from __future__ import annotations

import math
import cv2
import numpy as np


def get_aruco_dictionary(family_str: str):
    """Maps a marker family string to OpenCV ArUco dictionary constant."""
    mapping = {
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    }
    dict_const = mapping.get(family_str, cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.getPredefinedDictionary(dict_const)


class HomographyEngine:
    def __init__(self, layout_data, ransac_thresh_mm: float = 5.0, smoothing_alpha: float = 0.65):
        """
        layout_data: LayoutData instance (supports both object attributes and dict structures)
        ransac_thresh_mm: RANSAC threshold in destination (paper mm) units
        smoothing_alpha: Temporal smoothing factor (0.0 to 1.0) to stabilize H against sensor noise
        """
        self.layout_data = layout_data
        self.ransac_thresh_mm = ransac_thresh_mm
        self.smoothing_alpha = smoothing_alpha

        family = getattr(layout_data, "marker_family", "DICT_APRILTAG_36h11")
        dictionary = get_aruco_dictionary(family)
        detector_params = cv2.aruco.DetectorParameters()
        # CORNER_REFINE_CONTOUR is more stable than SUBPIX at low resolutions (360p/480p/720p).
        # SUBPIX iterates in a tiny window and amplifies noise when marker pixels are small.
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        self._smoothed_H: np.ndarray | None = None

        # Build fast marker lookup table
        self._build_marker_lookup()

    def _build_marker_lookup(self) -> None:
        """Standardizes marker access so both dicts and list of MarkerData work uniformly."""
        self._marker_dict = {}
        raw_markers = getattr(self.layout_data, "markers", [])
        if isinstance(raw_markers, dict):
            for m_id, m_info in raw_markers.items():
                corners = m_info["corners_mm"] if isinstance(m_info, dict) else m_info.corners_mm
                self._marker_dict[int(m_id)] = corners
        else:
            for m in raw_markers:
                corners = m.corners_mm if hasattr(m, "corners_mm") else m["corners_mm"]
                m_id = m.id if hasattr(m, "id") else m["id"]
                self._marker_dict[int(m_id)] = corners

    def reset_smoothing(self) -> None:
        """Resets temporal homography history."""
        self._smoothed_H = None

    def update_layout(self, layout_data) -> None:
        """Updates layout data and refreshes fiducial marker corner lookup."""
        self.layout_data = layout_data
        self._build_marker_lookup()
        self.reset_smoothing()

    def detect_markers(self, frame: np.ndarray):
        """
        Detects AprilTags in the given frame.

        Returns:
            corners: list of detected corner arrays
            ids: array of marker IDs or None
            annotated_frame: frame with drawn marker outlines and IDs
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids, rejected = self.detector.detectMarkers(gray)

        annotated_frame = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)

        return corners, ids, annotated_frame

    def _compute_H_from_markers(
        self,
        corners,
        ids,
    ) -> tuple[np.ndarray | None, list[int], int]:
        """
        Compute homography via pooled least-squares DLT (deterministic, no RANSAC).

        All 4 corners from every detected and matched marker are pooled into one global
        overdetermined system:  N markers  →  4N point correspondences  →  one SVD solve.
        cv2.findHomography with method=0 uses ordinary least-squares (DLT via SVD).
        The result is the globally optimal H for ALL visible marker corners simultaneously.

        Why NOT per-marker averaging:
          Homography matrices do NOT live in a linear space.  Averaging two Hs element-wise
          does NOT produce a matrix representing the average transform — especially under
          perspective tilt the result is geometrically wrong and causes keys to shift toward
          the image centre.

        Why NOT RANSAC:
          RANSAC is probabilistic — it picks random 4-point subsets each frame, which
          produces slightly different H values on consecutive frames even when nothing moves,
          causing the jitter we were seeing.

        Pooled DLT gives: deterministic + globally optimal + more markers = better.

        Returns:
            H       : 3×3 homography (pixel → mm), or None on failure
            used_ids: list of marker IDs that contributed points
            n_pts   : total corner points used (4 × len(used_ids))
        """
        if ids is None or len(ids) == 0:
            return None, [], 0

        ids_flat = ids.flatten()
        src_pts: list[list[float]] = []   # camera pixel coordinates
        dst_pts: list[list[float]] = []   # paper mm coordinates
        used_ids: list[int] = []

        for idx, m_id in enumerate(ids_flat):
            m_id = int(m_id)
            if m_id not in self._marker_dict:
                continue

            corners_mm = self._marker_dict[m_id]   # 4 × [x, y] in paper mm
            m_corners_px = corners[idx][0]          # shape (4, 2) — TL, TR, BR, BL

            for i in range(4):
                src_pts.append(m_corners_px[i].tolist())
                dst_pts.append(list(corners_mm[i]))

            used_ids.append(m_id)

        if not used_ids:
            return None, [], 0

        src_np = np.array(src_pts, dtype=np.float32)
        dst_np = np.array(dst_pts, dtype=np.float32)

        # method=0: ordinary least-squares via SVD — fully deterministic, no random sampling.
        # All corners from valid markers are genuine inliers so RANSAC is unnecessary.
        H, _ = cv2.findHomography(src_np, dst_np, 0)

        return H, used_ids, len(src_pts)

    def compute_homography(self, frame: np.ndarray):
        """
        Detects markers and computes homography matrix H (image pixels -> paper mm).
        Uses sub-pixel corner refinement and temporal smoothing for rock-solid stability.

        Returns:
            H: 3x3 Homography matrix or None
            info: dict containing detection details
        """
        corners, ids, annotated_frame = self.detect_markers(frame)

        total_buttons = len(getattr(self.layout_data, "buttons", []))

        if ids is None or len(ids) == 0:
            self._smoothed_H = None
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": "No AprilTag markers detected in frame.",
            }

        # ── Per-marker homography averaging (deterministic, no RANSAC) ──────────
        # Each valid marker contributes one exact H via getPerspectiveTransform.
        # Averaging N independent exact Hs gives a stable, noise-resistant estimate.
        H, used_ids, n_pts = self._compute_H_from_markers(corners, ids)
        inliers_count = n_pts  # all corners from detected markers are genuine inliers

        if not used_ids:
            self._smoothed_H = None
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": f"Detected {len(ids)} markers but none matched known layout markers.",
            }

        if H is not None:
            # Simple exponential moving average for stable H.
            # No adaptive jump-detection: a fixed high alpha suppresses per-frame noise
            # while still tracking intentional paper movement within a couple of frames.
            if self._smoothed_H is None:
                self._smoothed_H = H.copy()
            else:
                self._smoothed_H = (
                    self.smoothing_alpha * self._smoothed_H + (1.0 - self.smoothing_alpha) * H
                )
            H_final = self._smoothed_H
        else:
            self._smoothed_H = None
            H_final = None

        identified_buttons = []
        if H_final is not None:
            identified_buttons, _ = self.detect_internal_buttons(frame, H_final)

        info = {
            "success": H_final is not None,
            "detected_markers_count": len(used_ids),
            "used_marker_ids": used_ids,
            "inliers_count": inliers_count,
            "total_points": n_pts,
            "identified_buttons_count": len(identified_buttons),
            "total_buttons": total_buttons,
            "identified_buttons": identified_buttons,
            "annotated_frame": annotated_frame,
            "corners": corners,
            "ids": ids,
            "message": f"Homography computed ({len(used_ids)} markers, avg). Identified {len(identified_buttons)}/{total_buttons} internal buttons.",
        }

        return H_final, info

    def detect_internal_buttons(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        active_button_ids: set[str] | list[str] | str | None = None,
    ):
        """
        Fast direct button projection via inverse homography H^-1.
        Matches analyzer/homography_engine.py button geometry without CPU-heavy contour extraction.

        Returns:
            identified_buttons: list of dicts with detected button details (px bounds, center, text, status)
            cv_contours: list of CV contours (empty list for high FPS)
        """
        if H is None or self.layout_data is None:
            return [], []

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return [], []

        buttons = getattr(self.layout_data, "buttons", [])

        # Normalize active_button_ids to a set
        if active_button_ids is None:
            active_set = set()
        elif isinstance(active_button_ids, str):
            active_set = {active_button_ids}
        else:
            active_set = set(active_button_ids)

        identified_buttons = []

        for btn in buttons:
            bid = btn.id if hasattr(btn, "id") else btn["id"]
            btext = getattr(btn, "label", getattr(btn, "text", bid)) if hasattr(btn, "label") or hasattr(btn, "text") else btn.get("text", btn.get("label", bid))
            x1 = btn.x_mm if hasattr(btn, "x_mm") else btn["x_mm"]
            y1 = btn.y_mm if hasattr(btn, "y_mm") else btn["y_mm"]
            x2 = btn.x_max_mm if hasattr(btn, "x_max_mm") else btn["x_max_mm"]
            y2 = btn.y_max_mm if hasattr(btn, "y_max_mm") else btn["y_max_mm"]
            w = btn.width_mm if hasattr(btn, "width_mm") else btn["width_mm"]
            h = btn.height_mm if hasattr(btn, "height_mm") else btn["height_mm"]
            cx_mm = btn.center_x_mm if hasattr(btn, "center_x_mm") else btn["center_x_mm"]
            cy_mm = btn.center_y_mm if hasattr(btn, "center_y_mm") else btn["center_y_mm"]

            # 4 paper corner points in mm: [TL, TR, BR, BL]
            btn_pts_mm = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            btn_pts_px = cv2.perspectiveTransform(btn_pts_mm, H_inv)[0].astype(np.int32)

            center_mm = np.array([[[cx_mm, cy_mm]]], dtype=np.float32)
            center_px_arr = cv2.perspectiveTransform(center_mm, H_inv)[0][0].astype(np.int32)
            center_px = (int(center_px_arr[0]), int(center_px_arr[1]))

            is_active = bid in active_set

            identified_buttons.append({
                "id": bid,
                "text": btext if btext else bid,
                "x_mm": x1,
                "y_mm": y1,
                "width_mm": w,
                "height_mm": h,
                "center_x_mm": cx_mm,
                "center_y_mm": cy_mm,
                "corners_px": btn_pts_px,
                "center_px": center_px,
                "is_active": is_active,
                "cv_contour_matched": False,
                "status": "IDENTIFIED",
            })

        return identified_buttons, []

    def process_frame(
        self,
        frame: np.ndarray,
        active_button_ids: set[str] | list[str] | str | None = None,
    ):
        """
        Unified single-pass computer vision frame processor.
        Detects markers, computes homography H, identifies internal buttons,
        and renders live overlay in ONE pass. Matches analyzer/homography_engine.py.
        """
        corners, ids, annotated_frame = self.detect_markers(frame)
        total_buttons = len(getattr(self.layout_data, "buttons", []))

        if ids is None or len(ids) == 0:
            self._smoothed_H = None
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": "No AprilTag markers detected in frame.",
            }

        # ── Per-marker homography averaging (deterministic, no RANSAC) ──────────
        H, used_ids, n_pts = self._compute_H_from_markers(corners, ids)
        inliers_count = n_pts

        if not used_ids:
            self._smoothed_H = None
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": f"Detected {len(ids)} markers but none matched layout.",
            }

        if H is not None:
            # Simple exponential moving average — same as compute_homography path.
            if self._smoothed_H is None:
                self._smoothed_H = H.copy()
            else:
                self._smoothed_H = (
                    self.smoothing_alpha * self._smoothed_H + (1.0 - self.smoothing_alpha) * H
                )
            H_final = self._smoothed_H
        else:
            self._smoothed_H = None
            H_final = None

        identified_buttons = []
        if H_final is not None:
            identified_buttons, _ = self.detect_internal_buttons(frame, H_final, active_button_ids=active_button_ids)
            annotated_frame = self.draw_paper_and_buttons_overlay(
                annotated_frame, H_final, active_button_ids=active_button_ids, precomputed_buttons=identified_buttons
            )

        info = {
            "success": H_final is not None,
            "detected_markers_count": len(used_ids),
            "used_marker_ids": used_ids,
            "inliers_count": inliers_count,
            "total_points": n_pts,
            "identified_buttons_count": len(identified_buttons),
            "total_buttons": total_buttons,
            "identified_buttons": identified_buttons,
            "annotated_frame": annotated_frame,
            "message": f"Homography computed ({len(used_ids)} markers, avg). Identified {len(identified_buttons)}/{total_buttons} buttons.",
        }

        return H_final, info

    def draw_paper_and_buttons_overlay(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        active_button_ids: set[str] | list[str] | str | None = None,
        precomputed_buttons: list[dict] | None = None,
    ) -> np.ndarray:
        """
        Uses inverse homography (H^-1) to project paper boundary and identify internal button boxes
        directly onto the camera frame in real-time with visual indicators.
        Matches analyzer/homography_engine.py overlay renderer.
        """
        if H is None:
            return frame

        overlay_frame = frame.copy()

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return frame

        pw = float(getattr(self.layout_data, "paper_width_mm", 297.0))
        ph = float(getattr(self.layout_data, "paper_height_mm", 210.0))

        # 1. Project Paper Boundary (0,0), (W,0), (W,H), (0,H)
        paper_corners_mm = np.array([[[0, 0], [pw, 0], [pw, ph], [0, ph]]], dtype=np.float32)
        paper_corners_px = cv2.perspectiveTransform(paper_corners_mm, H_inv)[0].astype(np.int32)

        # Draw green paper boundary outline
        cv2.polylines(overlay_frame, [paper_corners_px], isClosed=True, color=(0, 230, 118), thickness=2)
        cv2.putText(
            overlay_frame, "PAPER BOUNDARY", (paper_corners_px[0][0], max(20, paper_corners_px[0][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 2,
        )

        # 2. Detect & Identify Internal Buttons
        if precomputed_buttons is not None:
            identified_buttons = precomputed_buttons
        else:
            identified_buttons, _ = self.detect_internal_buttons(frame, H, active_button_ids=active_button_ids)

        for btn_info in identified_buttons:
            btn_pts_px = btn_info["corners_px"]
            cx_px, cy_px = btn_info["center_px"]
            is_active = btn_info["is_active"]
            text = btn_info["text"]

            if is_active:
                color = (0, 230, 118)      # Bright Green for pressed button
                thickness = 3
            else:
                color = (255, 179, 0)      # Amber for identified layout button
                thickness = 2

            if is_active:
                # Optimized semi-transparent highlight fill (blend only bounding box of the active button)
                bx, by, bw, bh = cv2.boundingRect(btn_pts_px)
                ih, iw = overlay_frame.shape[:2]
                bx1, by1 = max(0, bx), max(0, by)
                bx2, by2 = min(iw, bx + bw), min(ih, by + bh)
                if bx2 > bx1 and by2 > by1:
                    roi = overlay_frame[by1:by2, bx1:bx2]
                    sub_overlay = roi.copy()
                    sub_pts = btn_pts_px - np.array([bx1, by1])
                    cv2.fillPoly(sub_overlay, [sub_pts], color=(0, 230, 118))
                    cv2.addWeighted(sub_overlay, 0.40, roi, 0.60, 0, roi)

            # Draw internal button box
            cv2.polylines(overlay_frame, [btn_pts_px], isClosed=True, color=color, thickness=thickness)

            # Draw corner tick marks (anchors) for button identification tracking
            for corner in btn_pts_px:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(overlay_frame, (x, y), 3, color, -1)

            # Draw button center crosshair
            cv2.circle(overlay_frame, (cx_px, cy_px), 2, color, -1)

            # Draw text label badge inside/center of button
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            txt_size, _ = cv2.getTextSize(text, font, font_scale, 1)
            tx = cx_px - txt_size[0] // 2
            ty = cy_px + txt_size[1] // 2

            # Background text box for readability
            cv2.rectangle(overlay_frame, (tx - 3, ty - txt_size[1] - 3), (tx + txt_size[0] + 3, ty + 3), (0, 0, 0), -1)
            cv2.putText(overlay_frame, text, (tx, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top Status HUD Badge on Camera Frame
        total_b = len(getattr(self.layout_data, "buttons", []))
        hud_text = f"INTERNAL BUTTONS IDENTIFIED: {len(identified_buttons)}/{total_b}"
        cv2.rectangle(overlay_frame, (10, 10), (380, 36), (18, 18, 18), -1)
        cv2.rectangle(overlay_frame, (10, 10), (380, 36), (0, 229, 255), 1)
        cv2.putText(overlay_frame, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

        return overlay_frame

    def process_touch_event(self, frame: np.ndarray, pixel_x: float, pixel_y: float) -> dict:
        """
        Computes homography for the current frame, converts (pixel_x, pixel_y)
        to paper mm coordinates, and checks which button was pressed.
        Matches analyzer/homography_engine.py process_touch_event.
        """
        H, info = self.compute_homography(frame)

        if H is None:
            return {
                "success": False,
                "pixel_point": (pixel_x, pixel_y),
                "paper_point_mm": None,
                "button": None,
                "info": info,
                "error_message": info["message"],
            }

        pt_src = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, H)

        paper_x_mm = float(pt_dst[0][0][0])
        paper_y_mm = float(pt_dst[0][0][1])

        button = self.find_button_at(paper_x_mm, paper_y_mm)

        return {
            "success": True,
            "pixel_point": (pixel_x, pixel_y),
            "paper_point_mm": (paper_x_mm, paper_y_mm),
            "button": button,
            "info": info,
            "error_message": None,
        }

    def pixel_to_mm(self, px: float, py: float, H: np.ndarray | None = None) -> tuple[float, float] | None:
        """Map camera pixel (px, py) to paper mm using cv2.perspectiveTransform."""
        h_matrix = H if H is not None else self._smoothed_H
        if h_matrix is None:
            return None
        pt_src = np.array([[[px, py]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, h_matrix.astype(np.float32))
        return float(pt_dst[0][0][0]), float(pt_dst[0][0][1])

    def mm_to_pixel(self, x_mm: float, y_mm: float, H_inv: np.ndarray | None = None) -> tuple[float, float] | None:
        """Map paper mm (x_mm, y_mm) to camera pixel using cv2.perspectiveTransform."""
        if H_inv is None:
            if self._smoothed_H is None:
                return None
            try:
                H_inv = np.linalg.inv(self._smoothed_H)
            except np.linalg.LinAlgError:
                return None
        pt_src = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, H_inv.astype(np.float32))
        return float(pt_dst[0][0][0]), float(pt_dst[0][0][1])

    def find_button_at(self, x_mm: float, y_mm: float, tolerance_mm: float = 3.0):
        """
        Returns button if (x_mm, y_mm) falls inside its bounding box, with a 3.0mm edge tolerance
        to accommodate fingertip width and physical typing contact margin.
        """
        buttons = getattr(self.layout_data, "buttons", [])

        # 1. Exact containment test
        for btn in buttons:
            if hasattr(btn, "contains_mm") and btn.contains_mm(x_mm, y_mm):
                return btn
            elif not hasattr(btn, "contains_mm"):
                bx1, by1 = btn["x_mm"], btn["y_mm"]
                bx2, by2 = btn["x_max_mm"], btn["y_max_mm"]
                if bx1 <= x_mm <= bx2 and by1 <= y_mm <= by2:
                    return btn

        # 2. Tolerant margin hit test (select closest button within margin)
        best_btn = None
        min_dist = float("inf")

        for btn in buttons:
            bx1 = btn.x_mm if hasattr(btn, "x_mm") else btn["x_mm"]
            by1 = btn.y_mm if hasattr(btn, "y_mm") else btn["y_mm"]
            bx2 = btn.x_max_mm if hasattr(btn, "x_max_mm") else btn["x_max_mm"]
            by2 = btn.y_max_mm if hasattr(btn, "y_max_mm") else btn["y_max_mm"]
            bcx = btn.center_x_mm if hasattr(btn, "center_x_mm") else btn["center_x_mm"]
            bcy = btn.center_y_mm if hasattr(btn, "center_y_mm") else btn["center_y_mm"]

            # Expanded bounding box by tolerance_mm
            if (bx1 - tolerance_mm) <= x_mm <= (bx2 + tolerance_mm) and (by1 - tolerance_mm) <= y_mm <= (by2 + tolerance_mm):
                dist = math.hypot(x_mm - bcx, y_mm - bcy)
                if dist < min_dist:
                    min_dist = dist
                    best_btn = btn

        return best_btn
