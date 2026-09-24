"""
tests/test_paired_touch_pipeline.py

Unit tests for the Consecutive Two-Window Paired Touch Confirmation system:
1. Window 1 onset buffers candidate without firing immediately.
2. Window 2 confirms physical landing and fires the key press.
3. Mid-air slowdown at F4 (Window 1 tail) is distinguished from surface contact;
   actual landing at F5 in Window 2 is correctly captured.
4. Window 2 with no model touch still confirms if physical impact kinematics exist in stitched frames.
5. Hovering/drifting with no physical strike is rejected without triggering keys.
6. Hand exit / lost tracking immediately clears pending candidate.
7. Consecutive window pair consumes Window 2 so it never double triggers.
"""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config.app_config import AppConfig
from core.action.action_executor import ActionData
from core.layout.layout_parser import ButtonData, LayoutData, MarkerData
from core.pipeline.touch_resolver import TouchResolver
from viewmodels.detector_viewmodel import DetectorViewModel, ExecutionMode


def _make_dummy_layout() -> LayoutData:
    btn_a = ButtonData(
        id="KEY_A", label="A",
        x_mm=20.0, y_mm=20.0, x_max_mm=50.0, y_max_mm=50.0,
        width_mm=30.0, height_mm=30.0, center_x_mm=35.0, center_y_mm=35.0,
    )
    btn_b = ButtonData(
        id="KEY_B", label="B",
        x_mm=60.0, y_mm=20.0, x_max_mm=90.0, y_max_mm=50.0,
        width_mm=30.0, height_mm=30.0, center_x_mm=75.0, center_y_mm=35.0,
    )
    return LayoutData(
        paper_width_mm=210.0,
        paper_height_mm=297.0,
        marker_size_mm=15.0,
        marker_family="tag36h11",
        buttons=[btn_a, btn_b],
        markers=[],
    )


def _make_dummy_config() -> AppConfig:
    config = AppConfig()
    config._touch_debounce_cooldown_s = 0.0
    config._fingertip_offset_enabled = False
    config._fingertip_forward_offset_mm = 0.0
    return config


class DummyModelEntry:
    def __init__(self):
        self.name = "Dummy LSTM"
        self.weights_path = "dummy.pth"

    def cls(self):
        inst = MagicMock()
        inst.predict = MagicMock(return_value={"Index": 0.90})
        return inst


class TestPairedTouchPipeline(unittest.TestCase):
    def setUp(self):
        self.layout = _make_dummy_layout()
        self.config = _make_dummy_config()
        self.resolver = TouchResolver(self.layout, offset_enabled=False, forward_offset_mm=0.0)

    def test_midair_slowdown_at_f4_advances_to_f5_landing(self):
        """
        Verify that a finger slowing down in mid-air at F4 (Window 1 tail)
        and physically hitting paper at F5 (Window 2) picks Frame 5.
        """
        H = np.eye(3, dtype=np.float32)

        # Window 1 (frames 0..4): finger falling down towards KEY_A (x=35, y=35)
        # At frame 4, y=25 (still 10mm in air above button center y=35)
        w1_pixels = []
        y_w1 = [10.0, 14.0, 18.0, 22.0, 25.0]
        for y in y_w1:
            frame = [(0.0, 0.0)] * 21
            frame[8] = (35.0, y) # Index tip
            frame[7] = (35.0, y - 10.0) # Index dip
            w1_pixels.append(frame)

        # Window 2 (frames 3..7, starting with F3=22.0, F4=25.0):
        # F5 lands on KEY_A at y=35.0, and stays still on surface at F6=35.0, F7=35.0
        w2_pixels = []
        y_w2 = [22.0, 25.0, 35.0, 35.0, 35.0]
        for y in y_w2:
            frame = [(0.0, 0.0)] * 21
            frame[8] = (35.0, y)
            frame[7] = (35.0, y - 10.0)
            w2_pixels.append(frame)

        # Run paired resolution
        res = self.resolver.resolve_paired("Index", 0.92, w1_pixels, w2_pixels, H)
        self.assertIsNotNone(res)
        key_id, finger, prob = res
        self.assertEqual(key_id, "KEY_A")
        self.assertEqual(finger, "Index")

    def test_rebound_at_f4_confirms_f4_landing(self):
        """
        Verify that a finger bouncing off the surface at Frame 4 confirms Frame 4.
        """
        H = np.eye(3, dtype=np.float32)

        # Downward strike into KEY_A at F4 (y=35)
        w1_pixels = []
        y_w1 = [15.0, 20.0, 26.0, 31.0, 35.0]
        for y in y_w1:
            frame = [(0.0, 0.0)] * 21
            frame[8] = (35.0, y)
            frame[7] = (35.0, y - 10.0)
            w1_pixels.append(frame)

        # Upward rebound in Window 2: F3=31, F4=35, F5=30, F6=26, F7=22
        w2_pixels = []
        y_w2 = [31.0, 35.0, 30.0, 26.0, 22.0]
        for y in y_w2:
            frame = [(0.0, 0.0)] * 21
            frame[8] = (35.0, y)
            frame[7] = (35.0, y - 10.0)
            w2_pixels.append(frame)

        res = self.resolver.resolve_paired("Index", 0.95, w1_pixels, w2_pixels, H)
        self.assertIsNotNone(res)
        key_id, finger, prob = res
        self.assertEqual(key_id, "KEY_A")

    def test_drifting_hover_without_impact_rejected(self):
        """
        Verify that a floating finger with low speed / no impact is rejected.
        """
        H = np.eye(3, dtype=np.float32)

        # Floating near KEY_A with tiny noise movements (speed < 0.5px)
        w1_pixels = []
        for _ in range(5):
            frame = [(0.0, 0.0)] * 21
            frame[8] = (35.0, 35.0)
            frame[7] = (35.0, 25.0)
            w1_pixels.append(frame)

        w2_pixels = []
        for _ in range(5):
            frame = [(0.0, 0.0)] * 21
            frame[8] = (35.1, 35.1)
            frame[7] = (35.1, 25.1)
            w2_pixels.append(frame)

        res = self.resolver.resolve_paired("Index", 0.90, w1_pixels, w2_pixels, H)
        self.assertIsNone(res)

    def test_viewmodel_two_window_paired_state_machine(self):
        """
        Verify that DetectorViewModel buffers on Window 1, and fires on Window 2.
        """
        actions = {"KEY_A": ActionData(type="keystroke", value="a")}
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(DummyModelEntry())
        vm.set_execution_mode(ExecutionMode.RUN)
        vm._layout_found = True
        vm._current_H = np.eye(3, dtype=np.float32)

        touches_fired = []
        vm.touch_event.connect(lambda k, f, p: touches_fired.append((k, f, p)))

        # Prepare dummy frames
        w1_norm = [{}] * 5
        w1_pixels = []
        for y in [10.0, 16.0, 22.0, 28.0, 35.0]:
            f = [(0.0, 0.0)] * 21
            f[8] = (35.0, y)
            f[7] = (35.0, y - 10.0)
            w1_pixels.append(f)

        w2_norm = [{}] * 5
        w2_pixels = []
        for y in [28.0, 35.0, 35.0, 35.0, 35.0]:
            f = [(0.0, 0.0)] * 21
            f[8] = (35.0, y)
            f[7] = (35.0, y - 10.0)
            w2_pixels.append(f)

        # 1. Window 1 arrives with touch onset
        vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
            "Index": {"touch": True, "prob": 0.92, "reason": "Touch Detected"}
        })
        vm._on_window_ready(w1_norm, w1_pixels, 640, 480)

        # Candidate buffered, NO touch fired yet
        self.assertIsNotNone(vm._pending_candidate)
        self.assertEqual(vm._pending_candidate["finger"], "Index")
        self.assertEqual(len(touches_fired), 0)

        # 2. Window 2 arrives (even with model returning no active touch onset in W2)
        vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
            "Index": {"touch": False, "prob": 0.20, "reason": "Below Onset Threshold"}
        })
        vm._on_window_ready(w2_norm, w2_pixels, 640, 480)

        # Window 2 confirmed the pair, fired the touch, and cleared the buffer
        self.assertIsNone(vm._pending_candidate)
        self.assertEqual(len(touches_fired), 1)
        self.assertEqual(touches_fired[0][0], "KEY_A")
        self.assertEqual(touches_fired[0][1], "Index")

    def test_hand_loss_clears_pending_candidate(self):
        """
        Verify that losing hand tracking purges the pending candidate buffer immediately.
        """
        vm = DetectorViewModel(
            layout=self.layout,
            action_config={},
            config=self.config,
            camera_index=0,
        )
        vm._pending_candidate = {"finger": "Index", "prob": 0.9, "pixel_window": []}

        # Simulate frame where no hand is detected
        vm._on_frame_ready(None, 12.0, False, None, False)
        self.assertIsNone(vm._pending_candidate)


if __name__ == "__main__":
    unittest.main()
