"""
tests/test_homography_replication.py

Verification tests ensuring virtualKeyboardSetup/detector perfectly replicates
the homography computation, keymapping, and button hit-testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.layout.layout_parser import LayoutData, ButtonData, MarkerData
from core.pipeline.homography_engine import HomographyEngine, get_aruco_dictionary
from core.pipeline.apriltag_tracker import AprilTagTracker


def _create_test_synthetic_frame(layout: LayoutData, frame_w: int = 1280, frame_h: int = 720) -> np.ndarray:
    """Renders layout markers into a synthetic camera frame with known homography."""
    frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 240
    dictionary = get_aruco_dictionary(layout.marker_family)

    # Place paper in center of frame
    scale_px_per_mm = 2.0
    paper_w_px = int(layout.paper_width_mm * scale_px_per_mm)
    paper_h_px = int(layout.paper_height_mm * scale_px_per_mm)
    off_x = (frame_w - paper_w_px) // 2
    off_y = (frame_h - paper_h_px) // 2

    # Draw white paper sheet
    cv2.rectangle(frame, (off_x, off_y), (off_x + paper_w_px, off_y + paper_h_px), (255, 255, 255), -1)

    for marker in layout.markers:
        size_px = int(marker.size_mm * scale_px_per_mm)
        mx_px = int(off_x + marker.top_left[0] * scale_px_per_mm)
        my_px = int(off_y + marker.top_left[1] * scale_px_per_mm)

        marker_img = cv2.aruco.generateImageMarker(dictionary, marker.id, size_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        frame[my_px:my_px + size_px, mx_px:mx_px + size_px] = marker_bgr

    return frame


def _make_marker(mid: int, cx: float, cy: float, size: float) -> MarkerData:
    hs = size / 2.0
    return MarkerData(
        id=mid,
        center_x_mm=cx,
        center_y_mm=cy,
        size_mm=size,
        top_left=(cx - hs, cy - hs),
        top_right=(cx + hs, cy - hs),
        bottom_right=(cx + hs, cy + hs),
        bottom_left=(cx - hs, cy + hs),
    )


class TestHomographyReplication(unittest.TestCase):
    def setUp(self):
        # Create standard layout fixture
        markers = [
            _make_marker(0, 20.0, 20.0, 20.0),
            _make_marker(1, 180.0, 20.0, 20.0),
            _make_marker(2, 180.0, 120.0, 20.0),
            _make_marker(3, 20.0, 120.0, 20.0),
        ]
        buttons = [
            ButtonData(
                id="btn_1",
                label="Space",
                x_mm=40.0,
                y_mm=40.0,
                width_mm=50.0,
                height_mm=30.0,
                x_max_mm=90.0,
                y_max_mm=70.0,
                center_x_mm=65.0,
                center_y_mm=55.0,
            ),
            ButtonData(
                id="btn_2",
                label="Enter",
                x_mm=100.0,
                y_mm=40.0,
                width_mm=40.0,
                height_mm=30.0,
                x_max_mm=140.0,
                y_max_mm=70.0,
                center_x_mm=120.0,
                center_y_mm=55.0,
            ),
        ]
        self.layout = LayoutData(
            paper_width_mm=200.0,
            paper_height_mm=140.0,
            marker_size_mm=20.0,
            marker_family="tag36h11",
            buttons=buttons,
            markers=markers,
        )
        self.engine = HomographyEngine(self.layout)

    def test_synthetic_homography_and_keymapping(self):
        """Verify HomographyEngine solves H on synthetic frame and maps button center accurately."""
        frame = _create_test_synthetic_frame(self.layout, frame_w=1280, frame_h=720)
        H, info = self.engine.process_frame(frame)

        self.assertTrue(info["success"], "Homography computation failed on synthetic frame")
        self.assertGreaterEqual(info["detected_markers_count"], 4)
        self.assertGreaterEqual(info["identified_buttons_count"], 2)

        # Spacebar center is at (65.0, 55.0) mm
        H_inv = np.linalg.inv(H)
        center_mm = np.array([[[65.0, 55.0]]], dtype=np.float32)
        center_px = cv2.perspectiveTransform(center_mm, H_inv)[0][0]

        # Test process_touch_event at button center pixel
        result = self.engine.process_touch_event(frame, float(center_px[0]), float(center_px[1]))
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["button"])
        self.assertEqual(result["button"]["id"], "btn_1")

        paper_x, paper_y = result["paper_point_mm"]
        self.assertAlmostEqual(paper_x, 65.0, delta=1.0)
        self.assertAlmostEqual(paper_y, 55.0, delta=1.0)

    def test_blank_frame_rejection(self):
        """Verify that blank frames or missing markers correctly result in no homography."""
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        H, info = self.engine.process_frame(blank)
        self.assertIsNone(H)
        self.assertFalse(info["success"])
        self.assertEqual(info["detected_markers_count"], 0)
        self.assertEqual(info["identified_buttons_count"], 0)

    def test_apriltag_tracker_wrapper(self):
        """Verify AprilTagTracker wrapper mirrors HomographyEngine state."""
        tracker = AprilTagTracker(self.layout, min_markers=4)
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertFalse(tracker.update(blank))
        self.assertFalse(tracker.is_valid)
        self.assertIsNone(tracker.H)

        frame = _create_test_synthetic_frame(self.layout, frame_w=1280, frame_h=720)
        self.assertTrue(tracker.update(frame))
        self.assertTrue(tracker.is_valid)
        self.assertIsNotNone(tracker.H)
        self.assertIsNotNone(tracker.H_inv)

        annotated = tracker.annotate_frame(frame.copy(), self.layout, active_button_ids=["btn_1"])
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
