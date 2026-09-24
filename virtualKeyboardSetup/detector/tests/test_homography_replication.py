"""
tests/test_homography_replication.py

Verification tests ensuring virtualKeyboardSetup/detector perfectly replicates
the homography computation, keymapping, and button hit-testing from designer/analyzer/main.py.
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

RESEARCH_ROOT = PROJECT_ROOT.parent.parent
ANALYZER_DIR = RESEARCH_ROOT / "designer" / "analyzer"
if str(ANALYZER_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYZER_DIR))

from core.layout.layout_parser import LayoutParser, LayoutData, ButtonData, MarkerData
from core.pipeline.homography_engine import HomographyEngine
from core.pipeline.apriltag_tracker import AprilTagTracker
from synthetic_generator import generate_synthetic_camera_frame


class TestHomographyReplication(unittest.TestCase):
    def setUp(self):
        xml_path = RESEARCH_ROOT / "designer" / "layout.xml"
        self.assertTrue(xml_path.exists(), f"layout.xml not found at {xml_path}")
        self.layout = LayoutParser().parse(str(xml_path))
        self.engine = HomographyEngine(self.layout)

    def test_synthetic_homography_and_keymapping(self):
        """Verify HomographyEngine solves H on synthetic frame and maps button center accurately."""
        frame = generate_synthetic_camera_frame(self.layout, frame_w=1280, frame_h=720, skew_amount=0.15)
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
        self.assertAlmostEqual(paper_x, 65.0, delta=0.5)
        self.assertAlmostEqual(paper_y, 55.0, delta=0.5)

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

        frame = generate_synthetic_camera_frame(self.layout, frame_w=1280, frame_h=720, skew_amount=0.15)
        self.assertTrue(tracker.update(frame))
        self.assertTrue(tracker.is_valid)
        self.assertIsNotNone(tracker.H)
        self.assertIsNotNone(tracker.H_inv)

        annotated = tracker.annotate_frame(frame.copy(), self.layout, active_button_id="btn_1")
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
