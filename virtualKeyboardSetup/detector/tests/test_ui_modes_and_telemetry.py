"""
tests/test_ui_modes_and_telemetry.py

Unit tests for Play Mode, Run Mode, Telemetry Graph, Paper Canvas,
and MainWindow navigation routing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtWidgets import QApplication

# Ensure QApplication exists for UI tests
app = QApplication.instance() or QApplication(sys.argv)

from config.app_config import AppConfig
from core.action.action_executor import ActionData
from core.interfaces.touch_model import ModelEntry
from core.layout.layout_parser import ButtonData, LayoutData, MarkerData
from ui.components.paper_layout_canvas import PaperLayoutCanvasWidget
from ui.components.telemetry_graph import TelemetryGraphWidget
from ui.main_window import MainWindow
from ui.views.file_landing_view import FileLandingView
from ui.views.play_mode_view import PlayModeView
from ui.views.run_mode_view import RunModeView
from viewmodels.detector_viewmodel import DetectorViewModel, ExecutionMode
from viewmodels.startup_viewmodel import StartupViewModel


from core.interfaces.touch_model import ITouchModel, ModelEntry


class DummyTouchModel(ITouchModel):
    def load(self, weights_path: str) -> None:
        pass

    def predict(self, window_5x21x3: np.ndarray) -> dict[str, float]:
        return {"Thumb": 0.1, "Index": 0.85, "Middle": 0.2, "Ring": 0.05, "Pinky": 0.05}


class TestUiModesAndTelemetry(unittest.TestCase):

    def setUp(self) -> None:
        self.btn = ButtonData(
            id="btn_1",
            label="A",
            x_mm=100.0,
            y_mm=100.0,
            x_max_mm=150.0,
            y_max_mm=150.0,
            width_mm=50.0,
            height_mm=50.0,
            center_x_mm=125.0,
            center_y_mm=125.0,
        )
        self.marker = MarkerData(
            id=0,
            center_x_mm=10.0,
            center_y_mm=10.0,
            size_mm=20.0,
            top_left=(0.0, 0.0),
            top_right=(20.0, 0.0),
            bottom_right=(20.0, 20.0),
            bottom_left=(0.0, 20.0),
        )
        self.layout = LayoutData(
            paper_width_mm=500.0,
            paper_height_mm=300.0,
            marker_size_mm=20.0,
            marker_family="tag36h11",
            buttons=[self.btn],
            markers=[self.marker],
        )
        self.actions = {"btn_1": ActionData(type="key", value="a")}
        self.config = AppConfig()
        self.config.set_printed_marker_side_width_mm(0.0)
        self.config.set_touch_release_windows(2)
        self.config.set_touch_max_stationary_distance_mm(4.0)
        self.model_instance = DummyTouchModel()
        self.model_entry = ModelEntry(
            name="Dummy LSTM",
            description="Mock LSTM model for testing",
            weights_file="dummy.pth",
            weights_path="dummy.pth",
            cls=DummyTouchModel,
            instance=self.model_instance,
        )

    def test_execution_mode_enum(self) -> None:
        self.assertEqual(ExecutionMode.PLAY, "play")
        self.assertEqual(ExecutionMode.RUN, "run")

    def test_detector_viewmodel_mode_switching(self) -> None:
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        self.assertEqual(vm.execution_mode, ExecutionMode.PLAY)

        modes_recorded = []
        vm.mode_changed.connect(modes_recorded.append)

        vm.set_execution_mode(ExecutionMode.RUN)
        self.assertEqual(vm.execution_mode, ExecutionMode.RUN)
        self.assertEqual(modes_recorded, ["run"])

        vm.set_execution_mode(ExecutionMode.PLAY)
        self.assertEqual(vm.execution_mode, ExecutionMode.PLAY)
        self.assertEqual(modes_recorded, ["run", "play"])

    def test_play_mode_does_not_execute_action(self) -> None:
        """In Play Mode, detected touches must NOT invoke ActionExecutor."""
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.PLAY)

        vm._layout_found = True
        vm._current_H = np.eye(3)
        vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
            "Index": {"touch": True, "prob": 0.95}
        })
        vm._resolver.resolve = MagicMock(return_value=("btn_1", "Index", 0.95))

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            actions_executed = []
            vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

            dummy_window = np.zeros((5, 21, 3), dtype=np.float32)
            vm._on_window_ready(dummy_window, dummy_window, 640, 480)

            # In PLAY mode, ActionExecutor.execute should NOT have been called
            mock_exec.assert_not_called()
            self.assertEqual(len(actions_executed), 0)

    def test_run_mode_executes_action(self) -> None:
        """In Run Mode, confirmed touches MUST invoke ActionExecutor."""
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.RUN)

        vm._layout_found = True
        vm._current_H = np.eye(3)
        vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
            "Index": {"touch": True, "prob": 0.95}
        })
        vm._resolver.resolve = MagicMock(return_value=("btn_1", "Index", 0.95))

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            actions_executed = []
            vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

            dummy_window = np.zeros((5, 21, 3), dtype=np.float32)
            vm._on_window_ready(dummy_window, dummy_window, 640, 480)

            # In RUN mode, ActionExecutor.execute SHOULD be called
            mock_exec.assert_called_once()
            self.assertEqual(len(actions_executed), 1)
            self.assertEqual(actions_executed[0], ("key", "a", "btn_1", "Index"))

    def test_telemetry_graph_widget(self) -> None:
        graph = TelemetryGraphWidget(threshold=0.5)
        self.assertEqual(graph._threshold, 0.5)

        # Push sample probabilities
        probs = {"Thumb": 0.1, "Index": 0.9, "Middle": 0.3, "Ring": 0.05, "Pinky": 0.02}
        graph.push_probabilities(probs)
        self.assertAlmostEqual(graph._history["Index"][-1], 0.9)

        graph.set_threshold(0.7)
        self.assertEqual(graph._threshold, 0.7)

    def test_paper_layout_canvas_widget(self) -> None:
        canvas = PaperLayoutCanvasWidget()
        canvas.set_layout(self.layout)
        self.assertIsNotNone(canvas._layout)

        # Test touch highlight
        canvas.highlight_touch("btn_1", "Index", 0.95)
        self.assertEqual(canvas._active_key_id, "btn_1")
        self.assertEqual(canvas._active_finger, "Index")

    def test_views_initialization(self) -> None:
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)

        # Test FileLandingView (file-only selection)
        startup_vm = StartupViewModel(self.config.plugins_dir)
        landing_view = FileLandingView(startup_vm)
        self.assertIsNotNone(landing_view)
        self.assertFalse(landing_view.btn_open.isEnabled())

        # Simulate layout loaded
        landing_view._xml_path = "tests/dummy_layout.xml"
        landing_view._on_loaded(self.layout)
        self.assertTrue(landing_view.btn_open.isEnabled())
        self.assertFalse(landing_view.drop_zone.isVisible())
        self.assertEqual(landing_view.pill_keys.text(), "1 keys")

        # Test workspace entry signal
        workspace_signal_fired = []
        landing_view.workspace_entered.connect(lambda l, p: workspace_signal_fired.append((l, p)))
        landing_view.btn_open.click()
        self.assertEqual(len(workspace_signal_fired), 1)
        self.assertEqual(workspace_signal_fired[0][0], self.layout)

        # Test browse file mock
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=("/tmp/sample.xml", "XML Layout Files (*.xml)")):
            with patch.object(landing_view, "_load") as mock_load:
                landing_view._on_browse()
                mock_load.assert_called_once_with("/tmp/sample.xml")

        # Test PlayModeView
        play_view = PlayModeView(vm)
        play_view.set_layout_data(self.layout)
        self.assertIsNotNone(play_view)

        # Test RunModeView
        run_view = RunModeView(vm)
        run_view.set_layout_data(self.layout)
        self.assertIsNotNone(run_view)

        # Test DetectionDynamicsGraphWidget
        from ui.components.dynamics_graph import DetectionDynamicsGraphWidget
        dyn_graph = DetectionDynamicsGraphWidget()
        dyn_graph.push_metrics(2.5, 12.0)
        self.assertEqual(len(dyn_graph._latency_history), 80)
        self.assertAlmostEqual(dyn_graph._latency_history[-1], 2.5)

    def test_two_window_in_flight_bridging_in_viewmodel(self) -> None:
        """Verify in-flight descent buffering (W1) and instant confirmed touchdown execution (W2)."""
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.RUN)

        vm._layout_found = True
        vm._current_H = np.eye(3)

        actions_executed = []
        vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

        # W1: Finger descending rapidly in mid-air toward btn_1 (125, 125)
        # s1=5, s2=6, s3=9 px/frame -> in_flight is True!
        descending_pts = [(125.0, 100.0), (125.0, 105.0), (125.0, 110.0), (125.0, 116.0), (125.0, 125.0)]
        descending_pixel = [[pt for _ in range(21)] for pt in descending_pts]

        # W2: Finger settled on btn_1 (125, 125) -> in_flight is False!
        settled_pts = [(125.0, 125.0)] * 5
        settled_pixel = [[pt for _ in range(21)] for pt in settled_pts]

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            # Window 1: touch detected while in mid-air -> buffered, NOT fired yet
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": True, "prob": 0.95}
            })
            vm._on_window_ready([], descending_pixel, 640, 480)
            mock_exec.assert_not_called()
            self.assertEqual(len(actions_executed), 0)
            self.assertEqual(vm._touch_streak.get("Index", 0), 1)
            self.assertIn("Index", vm._touch_pending_hit)

            # Window 2: finger lands on paper at btn_1 -> buffered, NOT fired yet while touching
            vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_not_called()
            self.assertEqual(len(actions_executed), 0)
            self.assertEqual(vm._touch_streak.get("Index", 0), 2)

            # Window 3: non-touch window 1/2 -> waiting for release confirmation
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": False, "prob": 0.20, "reason": "Below Threshold"}
            })
            vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_not_called()

            # Window 4: non-touch window 2/2 -> release confirmed -> FIRES action!
            vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_called_once()
            self.assertEqual(len(actions_executed), 1)
            self.assertEqual(actions_executed[0], ("key", "a", "btn_1", "Index"))
            self.assertEqual(vm._touch_streak.get("Index", 0), 0)
            self.assertNotIn("Index", vm._touch_pending_hit)

    def test_simultaneous_multi_finger_independent_actions(self) -> None:
        """Verify per-finger independent confirmed touchdown actions."""
        btn_2 = ButtonData(
            id="btn_2",
            label="B",
            x_mm=200.0,
            y_mm=100.0,
            x_max_mm=250.0,
            y_max_mm=150.0,
            width_mm=50.0,
            height_mm=50.0,
            center_x_mm=225.0,
            center_y_mm=125.0,
        )
        layout_multi = LayoutData(
            paper_width_mm=500.0,
            paper_height_mm=300.0,
            marker_size_mm=20.0,
            marker_family="tag36h11",
            buttons=[self.btn, btn_2],
            markers=[self.marker],
        )
        actions_multi = {
            "btn_1": ActionData(type="key", value="a"),
            "btn_2": ActionData(type="key", value="b"),
        }

        vm = DetectorViewModel(
            layout=layout_multi,
            action_config=actions_multi,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.RUN)

        vm._layout_found = True
        vm._current_H = np.eye(3)

        actions_executed = []
        vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

        # 5 stationary frames: Index at btn_1 (125,125), Middle at btn_2 (225,125)
        pixel_window = []
        for _ in range(5):
            landmarks = [(0.0, 0.0) for _ in range(21)]
            landmarks[8] = (125.0, 125.0)   # Index tip
            landmarks[12] = (225.0, 125.0)  # Middle tip
            pixel_window.append(landmarks)

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            # Window 1: both fingers touch -> buffered while touching
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": True, "prob": 0.95},
                "Middle": {"touch": True, "prob": 0.92},
            })
            vm._on_window_ready([], pixel_window, 640, 480)
            mock_exec.assert_not_called()
            self.assertEqual(len(actions_executed), 0)

            # Window 2: both fingers continue touching -> still buffered
            vm._on_window_ready([], pixel_window, 640, 480)
            mock_exec.assert_not_called()

            # Window 3: non-touch window 1/2 -> waiting
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": False, "prob": 0.20, "reason": "Below Threshold"},
                "Middle": {"touch": False, "prob": 0.18, "reason": "Below Threshold"},
            })
            vm._on_window_ready([], pixel_window, 640, 480)
            mock_exec.assert_not_called()

            # Window 4: non-touch window 2/2 -> release confirmed -> FIRES BOTH!
            vm._on_window_ready([], pixel_window, 640, 480)
            self.assertEqual(mock_exec.call_count, 2)
            self.assertEqual(len(actions_executed), 2)
            executed_keys = {act[2] for act in actions_executed}
            self.assertEqual(executed_keys, {"btn_1", "btn_2"})
            executed_fingers = {act[3] for act in actions_executed}
            self.assertEqual(executed_fingers, {"Index", "Middle"})
            self.assertEqual(vm._touch_streak.get("Index", 0), 0)
            self.assertEqual(vm._touch_streak.get("Middle", 0), 0)

    def test_quick_tap_release_firing(self) -> None:
        """Verify that when touch_release_windows=1, tap fires immediately after 1 non-touch window."""
        self.config.set_touch_release_windows(1)
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.RUN)
        vm._layout_found = True
        vm._current_H = np.eye(3)

        descending_pts = [(125.0, 100.0), (125.0, 105.0), (125.0, 110.0), (125.0, 116.0), (125.0, 125.0)]
        descending_pixel = [[pt for _ in range(21)] for pt in descending_pts]
        actions_executed = []
        vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            # W1: Fast tap (streak = 1) -> buffered, not fired yet
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": True, "prob": 0.95}
            })
            vm._on_window_ready([], descending_pixel, 640, 480)
            mock_exec.assert_not_called()

            # W2: Lifted (untouch window 1) -> with touch_release_windows=1, FIRES on release!
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": False, "prob": 0.15}
            })
            vm._on_window_ready([], descending_pixel, 640, 480)
            mock_exec.assert_called_once()
            self.assertEqual(len(actions_executed), 1)
            self.assertEqual(actions_executed[0], ("key", "a", "btn_1", "Index"))

        # Restore default
        self.config.set_touch_release_windows(2)

    def test_touch_and_hold_immediate_firing_and_no_duplicate(self) -> None:
        """Verify that touching and holding buffers touch and fires once upon confirmed release."""
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.RUN)
        vm._layout_found = True
        vm._current_H = np.eye(3)

        settled_pts = [(125.0, 125.0)] * 5
        settled_pixel = [[pt for _ in range(21)] for pt in settled_pts]
        actions_executed = []
        vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            # W1..W4: Finger held down on paper -> buffered, NO action fired while held
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": True, "prob": 0.95}
            })
            for _ in range(4):
                vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_not_called()
            self.assertEqual(len(actions_executed), 0)

            # W5: Lift begins (non-touch window 1/2) -> waiting for release confirmation
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": False, "prob": 0.15}
            })
            vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_not_called()

            # W6: Non-touch window 2/2 -> release confirmed -> FIRES exactly once!
            vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_called_once()
            self.assertEqual(len(actions_executed), 1)

            # W7: Continued non-touch -> NO duplicate fire!
            vm._on_window_ready([], settled_pixel, 640, 480)
            mock_exec.assert_called_once()
            self.assertEqual(len(actions_executed), 1)

    def test_finger_liftoff_release_confirmation_fires_action(self) -> None:
        """Verify that finger lift-off across non-touch windows confirms release and triggers Run Mode action."""
        vm = DetectorViewModel(
            layout=self.layout,
            action_config=self.actions,
            config=self.config,
            camera_index=0,
        )
        vm.set_model(self.model_entry)
        vm.set_execution_mode(ExecutionMode.RUN)
        vm._layout_found = True
        vm._current_H = np.eye(3)

        # W1: Finger touches btn_1 at (125, 125)
        w1_pixel = [[(125.0, 125.0) for _ in range(21)] for _ in range(5)]
        # W2 & W3: Finger lifts off into the air (projecting upward/away to (150, 125))
        w2_pixel = [[(150.0, 125.0) for _ in range(21)] for _ in range(5)]

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            # W1: Touch detected over btn_1 -> candidate glows, touch buffered
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": True, "prob": 0.95}
            })
            vm._on_window_ready([], w1_pixel, 640, 480)
            mock_exec.assert_not_called()
            self.assertTrue(vm._finger_touch_active.get("Index", False))
            self.assertIn("btn_1", vm._last_pressed_keys)

            # W2: First non-touch window (finger lifting up) -> waiting for release confirmation
            vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
                "Index": {"touch": False, "prob": 0.10}
            })
            vm._on_window_ready([], w2_pixel, 640, 480)
            mock_exec.assert_not_called()
            self.assertTrue(vm._finger_touch_active.get("Index", False))

            # W3: Second non-touch window -> release confirmed! Action executes!
            vm._on_window_ready([], w2_pixel, 640, 480)
            mock_exec.assert_called_once()
            self.assertNotIn("Index", vm._touch_pending_hit)

            # W4: Subsequent window -> key is no longer in last_pressed_keys
            vm._on_window_ready([], w2_pixel, 640, 480)
            self.assertNotIn("btn_1", vm._last_pressed_keys)

    def test_main_window_startup_sidebar_hidden(self) -> None:
        """On startup, MainWindow must have sidebar hidden and show FileLandingView."""
        win = MainWindow(self.config)
        self.assertTrue(win.navigationInterface.isHidden())
        self.assertEqual(win.stackedWidget.currentWidget().objectName(), "fileLandingView")
        win.close()

    def test_primary_model_decorator_and_registry(self) -> None:
        """Verify @primaryModel designates the primary model and ModelRegistry selects it."""
        from core.interfaces.touch_model import primaryModel, ModelRegistry, ITouchModel

        class SecondaryDummy(ITouchModel):
            @classmethod
            def model_name(cls) -> str:
                return "Secondary Model"
            def predict_touch(self, window):
                return {}

        @primaryModel
        class PrimaryDummy(ITouchModel):
            @classmethod
            def model_name(cls) -> str:
                return "Primary Model"
            def predict_touch(self, window):
                return {}

        self.assertTrue(getattr(PrimaryDummy, "_is_primary", False))
        self.assertFalse(getattr(SecondaryDummy, "_is_primary", False))

        primary_entry = ModelRegistry.get_primary()
        self.assertIsNotNone(primary_entry)
        self.assertTrue(primary_entry.is_primary)
        self.assertEqual(primary_entry.name, "LSTM Touch Detector")


if __name__ == "__main__":
    unittest.main()

