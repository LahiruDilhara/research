"""
synthetic_generator.py

Generates synthetic camera frames of printed paper layouts with AprilTag markers,
perspective warping, and background desk texture for offline testing.
"""

import cv2
import numpy as np
try:
    from homography_engine import get_aruco_dictionary
except ImportError:
    from analyzer.homography_engine import get_aruco_dictionary


def render_paper_flat_image(layout_data, px_per_mm=3.0):
    """
    Renders a flat 2D image of the printed paper layout with AprilTag markers and buttons.
    """
    w_px = int(layout_data.paper_width_mm * px_per_mm)
    h_px = int(layout_data.paper_height_mm * px_per_mm)

    # White paper background
    paper_img = np.ones((h_px, w_px, 3), dtype=np.uint8) * 245

    # Draw outer border
    cv2.rectangle(paper_img, (0, 0), (w_px - 1, h_px - 1), (180, 180, 180), 2)

    # Get ArUco dictionary
    dictionary = get_aruco_dictionary(layout_data.marker_family)

    # Render AprilTag Markers
    for m_id, m_info in layout_data.markers.items():
        cx_px = int(m_info["center_x_mm"] * px_per_mm)
        cy_px = int(m_info["center_y_mm"] * px_per_mm)
        size_px = int(m_info["size_mm"] * px_per_mm)

        marker_bmp = cv2.aruco.generateImageMarker(dictionary, m_id, size_px)
        marker_bgr = cv2.cvtColor(marker_bmp, cv2.COLOR_GRAY2BGR)

        x1 = cx_px - size_px // 2
        y1 = cy_px - size_px // 2
        x2 = x1 + size_px
        y2 = y1 + size_px

        # Clip bounds safely
        if x1 >= 0 and y1 >= 0 and x2 <= w_px and y2 <= h_px:
            paper_img[y1:y2, x1:x2] = marker_bgr

    # Render Buttons
    for btn in layout_data.buttons:
        bx1 = int(btn["x_mm"] * px_per_mm)
        by1 = int(btn["y_mm"] * px_per_mm)
        bx2 = int(btn["x_max_mm"] * px_per_mm)
        by2 = int(btn["y_max_mm"] * px_per_mm)

        # Draw button bounding box
        cv2.rectangle(paper_img, (bx1, by1), (bx2, by2), (40, 40, 40), 2)

        # Draw text centered inside button
        text = btn["text"]
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * (px_per_mm / 3.0)
            thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = bx1 + (bx2 - bx1 - text_size[0]) // 2
            ty = by1 + (by2 - by1 + text_size[1]) // 2
            cv2.putText(paper_img, text, (max(bx1 + 2, tx), max(by1 + 2, ty)), font, font_scale, (20, 20, 20), thickness)

    return paper_img, px_per_mm


def generate_synthetic_camera_frame(layout_data, frame_w=1280, frame_h=720, skew_amount=0.15):
    """
    Renders the paper layout warped into a camera view with realistic perspective tilt.

    Returns:
        frame: BGR numpy array of size (frame_h, frame_w, 3)
    """
    flat_img, px_per_mm = render_paper_flat_image(layout_data, px_per_mm=3.0)
    paper_h, paper_w = flat_img.shape[:2]

    # Source 4 corners of flat paper image (TL, TR, BR, BL)
    src_corners = np.float32([
        [0, 0],
        [paper_w, 0],
        [paper_w, paper_h],
        [0, paper_h]
    ])

    # Destination 4 corners in target camera frame (simulating a tilted desk view)
    margin_w = frame_w * 0.15
    margin_h = frame_h * 0.15

    dst_corners = np.float32([
        [margin_w + frame_w * skew_amount, margin_h + frame_h * skew_amount],          # TL (pushed inward for perspective)
        [frame_w - margin_w - frame_w * skew_amount, margin_h + frame_h * skew_amount], # TR
        [frame_w - margin_w * 0.8, frame_h - margin_h * 0.8],                          # BR
        [margin_w * 0.8, frame_h - margin_h * 0.8]                                     # BL
    ])

    # Compute warping matrix
    H_warp = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Background canvas (simulating desk surface color)
    frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 50
    # Add subtle desk wood texture / gradient pattern
    for y in range(frame_h):
        frame[y, :, :] = int(45 + 15 * (y / frame_h))

    # Warp paper image onto background
    warped_paper = cv2.warpPerspective(flat_img, H_warp, (frame_w, frame_h))

    # Create mask for smooth blending
    paper_mask = cv2.warpPerspective(np.ones((paper_h, paper_w), dtype=np.uint8) * 255, H_warp, (frame_w, frame_h))
    mask_3ch = cv2.cvtColor(paper_mask, cv2.COLOR_GRAY2BGR)

    frame = np.where(mask_3ch > 0, warped_paper, frame)

    return frame
