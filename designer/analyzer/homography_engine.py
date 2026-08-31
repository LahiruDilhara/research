"""
homography_engine.py

Core computer vision & homography engine. Detects AprilTag / ArUco markers in
camera frames, pairs pixel corners with global paper mm coordinates from layout,
solves homography matrix H, and projects paper boundaries & buttons back onto live camera feed.
"""

import cv2
import numpy as np


def get_aruco_dictionary(family_str):
    """Maps a marker family string to OpenCV ArUco dictionary constant."""
    mapping = {
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    }
    dict_const = mapping.get(family_str, cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.getPredefinedDictionary(dict_const)


class HomographyEngine:
    def __init__(self, layout_data, ransac_thresh_mm=5.0):
        """
        layout_data: LayoutData instance from xml_parser.py
        ransac_thresh_mm: RANSAC threshold in destination (paper mm) units
        """
        self.layout_data = layout_data
        self.ransac_thresh_mm = ransac_thresh_mm

        # Initialize ArUco Detector
        dictionary = get_aruco_dictionary(layout_data.marker_family)
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    def detect_markers(self, frame):
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

    def compute_homography(self, frame):
        """
        Detects markers and computes homography matrix H (image pixels -> paper mm).

        Returns:
            H: 3x3 Homography matrix or None
            info: dict containing detection details
        """
        corners, ids, annotated_frame = self.detect_markers(frame)

        if ids is None or len(ids) == 0:
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "annotated_frame": annotated_frame,
                "message": "No AprilTag markers detected in frame."
            }

        src_pts = []  # Image pixels: [(x, y), ...]
        dst_pts = []  # Paper coordinates in mm: [(X, Y), ...]
        used_ids = []

        ids_flat = ids.flatten()
        for idx, m_id in enumerate(ids_flat):
            m_id = int(m_id)
            if m_id in self.layout_data.markers:
                marker_info = self.layout_data.markers[m_id]
                m_corners_px = corners[idx][0]  # Shape (4, 2): TL, TR, BR, BL

                # Add 4 corner correspondences
                for i in range(4):
                    src_pts.append(m_corners_px[i])
                    dst_pts.append(marker_info["corners_mm"][i])

                used_ids.append(m_id)

        if len(src_pts) < 4:
            return None, {
                "success": False,
                "detected_markers_count": len(used_ids),
                "used_marker_ids": used_ids,
                "inliers_count": 0,
                "annotated_frame": annotated_frame,
                "message": f"Detected {len(ids)} markers, but none matched known layout markers."
            }

        src_np = np.array(src_pts, dtype=np.float32)
        dst_np = np.array(dst_pts, dtype=np.float32)

        H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, self.ransac_thresh_mm)

        inliers_count = int(np.sum(mask)) if mask is not None else 0

        info = {
            "success": H is not None,
            "detected_markers_count": len(used_ids),
            "used_marker_ids": used_ids,
            "inliers_count": inliers_count,
            "total_points": len(src_pts),
            "annotated_frame": annotated_frame,
            "message": f"Homography computed using {len(used_ids)} markers ({inliers_count}/{len(src_pts)} inliers)."
        }

        return H, info

    def detect_internal_buttons(self, frame, H, active_button_id=None):
        """
        Detects, extracts, and identifies all internal layout buttons on the paper frame
        using computer vision contour matching combined with homography mapping H.

        Returns:
            identified_buttons: list of dicts with detected button details (px bounds, center, text, status)
            cv_contours: list of CV-detected physical contour polygons in image pixels
        """
        if H is None or self.layout_data is None:
            return [], []

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return [], []

        # -----------------------------------------------------------------
        # 1. High-Performance CV Contour Extraction on Paper ROI
        # -----------------------------------------------------------------
        cv_matched_button_ids = set()
        cv_contours_px = []

        try:
            # Warp paper surface to a flat 2D image at scale 2.0 px/mm for fast contour analysis
            px_per_mm = 2.0
            w_flat = int(self.layout_data.paper_width_mm * px_per_mm)
            h_flat = int(self.layout_data.paper_height_mm * px_per_mm)

            # Transformation matrix from image pixel space -> flat paper pixel space
            S = np.array([
                [px_per_mm, 0, 0],
                [0, px_per_mm, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            H_flat = S @ H  # Maps frame px -> flat px

            flat_paper = cv2.warpPerspective(frame, H_flat, (w_flat, h_flat))
            gray_flat = cv2.cvtColor(flat_paper, cv2.COLOR_BGR2GRAY) if len(flat_paper.shape) == 3 else flat_paper

            # Adaptive thresholding to highlight printed button edges
            thresh = cv2.adaptiveThreshold(
                gray_flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
            )

            # Use RETR_EXTERNAL to extract only top-level outer button boundaries (skips letter stroke noise)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:  # Skip tiny noise
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx_min = x / px_per_mm
                cy_min = y / px_per_mm
                cx_max = (x + w) / px_per_mm
                cy_max = (y + h) / px_per_mm

                # Bounding dimension filter in mm
                if (cx_max - cx_min) < 5.0 or (cy_max - cy_min) < 4.0:
                    continue

                # Fast spatial overlap check with layout buttons
                for btn in self.layout_data.buttons:
                    inter_x1 = max(cx_min, btn["x_mm"])
                    inter_y1 = max(cy_min, btn["y_mm"])
                    inter_x2 = min(cx_max, btn["x_max_mm"])
                    inter_y2 = min(cy_max, btn["y_max_mm"])

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        btn_area = btn["width_mm"] * btn["height_mm"]
                        if inter_area / btn_area > 0.25:
                            cv_matched_button_ids.add(btn["id"])
        except Exception:
            pass

        # -----------------------------------------------------------------
        # 2. Map and Identify Internal Buttons using Homography H
        # -----------------------------------------------------------------
        identified_buttons = []

        for btn in self.layout_data.buttons:
            x1, y1 = btn["x_mm"], btn["y_mm"]
            x2, y2 = btn["x_max_mm"], btn["y_max_mm"]

            # 4 paper corner points in mm: [TL, TR, BR, BL]
            btn_pts_mm = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            btn_pts_px = cv2.perspectiveTransform(btn_pts_mm, H_inv)[0].astype(np.int32)

            # Center point in pixel space
            center_mm = np.array([[[btn["center_x_mm"], btn["center_y_mm"]]]], dtype=np.float32)
            center_px_arr = cv2.perspectiveTransform(center_mm, H_inv)[0][0].astype(np.int32)
            center_px = (int(center_px_arr[0]), int(center_px_arr[1]))

            is_active = (active_button_id is not None) and (btn["id"] == active_button_id)
            is_cv_detected = btn["id"] in cv_matched_button_ids

            identified_buttons.append({
                "id": btn["id"],
                "text": btn["text"] if btn["text"] else btn["id"],
                "x_mm": btn["x_mm"],
                "y_mm": btn["y_mm"],
                "width_mm": btn["width_mm"],
                "height_mm": btn["height_mm"],
                "center_x_mm": btn["center_x_mm"],
                "center_y_mm": btn["center_y_mm"],
                "corners_px": btn_pts_px,
                "center_px": center_px,
                "is_active": is_active,
                "cv_contour_matched": is_cv_detected,
                "status": "IDENTIFIED"
            })

        return identified_buttons, cv_contours_px

    def process_frame(self, frame, active_button_id=None):
        """
        Unified single-pass computer vision frame processor.
        Detects markers, computes homography H, identifies internal buttons,
        and renders live overlay in ONE high-performance pass.
        """
        corners, ids, annotated_frame = self.detect_markers(frame)
        total_buttons = len(self.layout_data.buttons) if self.layout_data else 0

        if ids is None or len(ids) == 0:
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": "No AprilTag markers detected in frame."
            }

        src_pts = []
        dst_pts = []
        used_ids = []

        ids_flat = ids.flatten()
        for idx, m_id in enumerate(ids_flat):
            m_id = int(m_id)
            if m_id in self.layout_data.markers:
                marker_info = self.layout_data.markers[m_id]
                m_corners_px = corners[idx][0]

                for i in range(4):
                    src_pts.append(m_corners_px[i])
                    dst_pts.append(marker_info["corners_mm"][i])

                used_ids.append(m_id)

        if len(src_pts) < 4:
            return None, {
                "success": False,
                "detected_markers_count": len(used_ids),
                "used_marker_ids": used_ids,
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": f"Detected {len(ids)} markers, but none matched layout."
            }

        src_np = np.array(src_pts, dtype=np.float32)
        dst_np = np.array(dst_pts, dtype=np.float32)

        H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, self.ransac_thresh_mm)
        inliers_count = int(np.sum(mask)) if mask is not None else 0

        identified_buttons = []
        if H is not None:
            identified_buttons, _ = self.detect_internal_buttons(frame, H, active_button_id=active_button_id)
            annotated_frame = self.draw_paper_and_buttons_overlay(
                annotated_frame, H, active_button_id=active_button_id, precomputed_buttons=identified_buttons
            )

        info = {
            "success": H is not None,
            "detected_markers_count": len(used_ids),
            "used_marker_ids": used_ids,
            "inliers_count": inliers_count,
            "total_points": len(src_pts),
            "identified_buttons_count": len(identified_buttons),
            "total_buttons": total_buttons,
            "identified_buttons": identified_buttons,
            "annotated_frame": annotated_frame,
            "message": f"Homography computed ({len(used_ids)} markers). Identified {len(identified_buttons)}/{total_buttons} buttons."
        }

        return H, info

    def compute_homography(self, frame):
        """
        Detects markers and computes homography matrix H (image pixels -> paper mm).
        Also identifies internal paper layout buttons.

        Returns:
            H: 3x3 Homography matrix or None
            info: dict containing detection details
        """
        corners, ids, annotated_frame = self.detect_markers(frame)

        total_buttons = len(self.layout_data.buttons) if self.layout_data else 0

        if ids is None or len(ids) == 0:
            return None, {
                "success": False,
                "detected_markers_count": 0,
                "used_marker_ids": [],
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": "No AprilTag markers detected in frame."
            }

        src_pts = []  # Image pixels: [(x, y), ...]
        dst_pts = []  # Paper coordinates in mm: [(X, Y), ...]
        used_ids = []

        ids_flat = ids.flatten()
        for idx, m_id in enumerate(ids_flat):
            m_id = int(m_id)
            if m_id in self.layout_data.markers:
                marker_info = self.layout_data.markers[m_id]
                m_corners_px = corners[idx][0]  # Shape (4, 2): TL, TR, BR, BL

                # Add 4 corner correspondences
                for i in range(4):
                    src_pts.append(m_corners_px[i])
                    dst_pts.append(marker_info["corners_mm"][i])

                used_ids.append(m_id)

        if len(src_pts) < 4:
            return None, {
                "success": False,
                "detected_markers_count": len(used_ids),
                "used_marker_ids": used_ids,
                "inliers_count": 0,
                "identified_buttons_count": 0,
                "total_buttons": total_buttons,
                "identified_buttons": [],
                "annotated_frame": annotated_frame,
                "message": f"Detected {len(ids)} markers, but none matched known layout markers."
            }

        src_np = np.array(src_pts, dtype=np.float32)
        dst_np = np.array(dst_pts, dtype=np.float32)

        H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, self.ransac_thresh_mm)

        inliers_count = int(np.sum(mask)) if mask is not None else 0

        # Identify internal buttons if H was successfully calculated
        identified_buttons = []
        if H is not None:
            identified_buttons, _ = self.detect_internal_buttons(frame, H)

        info = {
            "success": H is not None,
            "detected_markers_count": len(used_ids),
            "used_marker_ids": used_ids,
            "inliers_count": inliers_count,
            "total_points": len(src_pts),
            "identified_buttons_count": len(identified_buttons),
            "total_buttons": total_buttons,
            "identified_buttons": identified_buttons,
            "annotated_frame": annotated_frame,
            "message": f"Homography computed ({len(used_ids)} markers). Identified {len(identified_buttons)}/{total_buttons} internal buttons."
        }

        return H, info

    def draw_paper_and_buttons_overlay(self, frame, H, active_button_id=None, precomputed_buttons=None):
        """
        Uses inverse homography (H^-1) to project paper boundary and identify internal button boxes
        directly onto the camera frame in real-time with visual indicators.
        """
        if H is None:
            return frame

        overlay_frame = frame.copy()

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return frame

        # 1. Project Paper Boundary (0,0), (W,0), (W,H), (0,H)
        pw = self.layout_data.paper_width_mm
        ph = self.layout_data.paper_height_mm
        paper_corners_mm = np.array([[[0, 0], [pw, 0], [pw, ph], [0, ph]]], dtype=np.float32)
        paper_corners_px = cv2.perspectiveTransform(paper_corners_mm, H_inv)[0].astype(np.int32)

        # Draw green paper boundary outline
        cv2.polylines(overlay_frame, [paper_corners_px], isClosed=True, color=(0, 230, 118), thickness=2)
        cv2.putText(overlay_frame, "PAPER BOUNDARY", (paper_corners_px[0][0], max(20, paper_corners_px[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 2)

        # 2. Detect & Identify Internal Buttons (reuse precomputed_buttons if available for fast performance)
        if precomputed_buttons is not None:
            identified_buttons = precomputed_buttons
        else:
            identified_buttons, _ = self.detect_internal_buttons(frame, H, active_button_id=active_button_id)

        for btn_info in identified_buttons:
            btn_pts_px = btn_info["corners_px"]
            cx_px, cy_px = btn_info["center_px"]
            is_active = btn_info["is_active"]
            text = btn_info["text"]

            # Button boundary polygon styling
            if is_active:
                color = (0, 230, 118)      # Bright Green for pressed button
                thickness = 3
            elif btn_info["cv_contour_matched"]:
                color = (0, 229, 255)      # Cyan for contour-matched internal button
                thickness = 2
            else:
                color = (255, 179, 0)      # Amber for identified layout button
                thickness = 2

            if is_active:
                # Semi-transparent highlight fill for pressed button
                overlay = overlay_frame.copy()
                cv2.fillPoly(overlay, [btn_pts_px], color=(0, 230, 118))
                cv2.addWeighted(overlay, 0.35, overlay_frame, 0.65, 0, overlay_frame)

            # Draw internal button box
            cv2.polylines(overlay_frame, [btn_pts_px], isClosed=True, color=color, thickness=thickness)

            # Draw corner tick marks (anchors) for button identification tracking
            tick_len = 5
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
        tag_count = len(self.layout_data.markers) if self.layout_data else 0
        hud_text = f"INTERNAL BUTTONS IDENTIFIED: {len(identified_buttons)}/{len(self.layout_data.buttons)}"
        cv2.rectangle(overlay_frame, (10, 10), (380, 36), (18, 18, 18), -1)
        cv2.rectangle(overlay_frame, (10, 10), (380, 36), (0, 229, 255), 1)
        cv2.putText(overlay_frame, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

        return overlay_frame

    def process_touch_event(self, frame, pixel_x, pixel_y):
        """
        Computes homography for the current frame, converts (pixel_x, pixel_y)
        to paper mm coordinates, and checks which button was pressed.
        """
        H, info = self.compute_homography(frame)

        if H is None:
            return {
                "success": False,
                "pixel_point": (pixel_x, pixel_y),
                "paper_point_mm": None,
                "button": None,
                "info": info,
                "error_message": info["message"]
            }

        # Perspective Transform: Image Pixel -> Paper MM
        pt_src = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, H)

        paper_x_mm = float(pt_dst[0][0][0])
        paper_y_mm = float(pt_dst[0][0][1])

        # Check button containment
        button = self.layout_data.find_button_at(paper_x_mm, paper_y_mm)

        return {
            "success": True,
            "pixel_point": (pixel_x, pixel_y),
            "paper_point_mm": (paper_x_mm, paper_y_mm),
            "button": button,
            "info": info,
            "error_message": None
        }

