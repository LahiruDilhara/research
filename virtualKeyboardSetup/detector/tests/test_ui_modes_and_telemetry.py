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
        """Verify that in-flight candidate in Window 1 delays resolution to Window 2 touchdown."""
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

        actions_executed = []
        vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

        # In Window 1: finger still moving fast (in-flight)
        # F0..(125, 50), F1..(125, 65), F2..(125, 80), F3..(125, 95), F4..(125, 110)
        # Final speed = 15 px/frame -> in_flight is True
        w1_pts = [(125.0, 50.0), (125.0, 65.0), (125.0, 80.0), (125.0, 95.0), (125.0, 110.0)]
        w1_pixel = [[pt for _ in range(21)] for pt in w1_pts]

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            vm._on_window_ready([], w1_pixel, 640, 480)

            # Window 1 was in-flight: no execution yet, candidate buffered
            mock_exec.assert_not_called()
            self.assertEqual(len(actions_executed), 0)
            self.assertIsNotNone(vm._pending_candidate)
            self.assertEqual(vm._pending_candidate["finger"], "Index")

            # Window 2 arrives with landing at (125, 125) (inside btn_1: [100..150, 100..150])
            w2_pts = [(125.0, 95.0), (125.0, 110.0), (125.0, 125.0), (125.0, 125.0), (125.0, 120.0)]
            w2_pixel = [[pt for _ in range(21)] for pt in w2_pts]

            vm._on_window_ready([], w2_pixel, 640, 480)

            # Window 2 resolved the bridged touch!
            mock_exec.assert_called_once()
            self.assertEqual(len(actions_executed), 1)
            self.assertEqual(actions_executed[0], ("key", "a", "btn_1", "Index"))
            self.assertIsNone(vm._pending_candidate)

    def test_simultaneous_multi_finger_independent_actions(self) -> None:
        """Verify that multiple fingers touching different keys in the same window all execute actions."""
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
        vm._pipeline_service.run_parallel_inference = MagicMock(return_value={
            "Index": {"touch": True, "prob": 0.95},
            "Middle": {"touch": True, "prob": 0.92},
        })

        actions_executed = []
        vm.action_executed.connect(lambda t, v, k, f: actions_executed.append((t, v, k, f)))

        # 5 frames where Index lands on btn_1 (125, 125) and Middle lands on btn_2 (225, 125)
        pixel_window = []
        for _ in range(5):
            landmarks = [(0.0, 0.0) for _ in range(21)]
            # Index tip (idx 8) at (125, 125)
            landmarks[8] = (125.0, 125.0)
            # Middle tip (idx 12) at (225, 125)
            landmarks[12] = (225.0, 125.0)
            pixel_window.append(landmarks)

        with patch("viewmodels.detector_viewmodel.ActionExecutor.execute") as mock_exec:
            vm._on_window_ready([], pixel_window, 640, 480)

            # Both independent finger actions must be executed!
            self.assertEqual(mock_exec.call_count, 2)
            self.assertEqual(len(actions_executed), 2)
            executed_keys = {act[2] for act in actions_executed}
            self.assertEqual(executed_keys, {"btn_1", "btn_2"})
            executed_fingers = {act[3] for act in actions_executed}
            self.assertEqual(executed_fingers, {"Index", "Middle"})

    def test_main_window_startup_sidebar_hidden(self) -> None:
        """On startup, MainWindow must have sidebar hidden and show FileLandingView."""
        win = MainWindow(self.config)
        self.assertTrue(win.navigationInterface.isHidden())
        self.assertEqual(win.stackedWidget.currentWidget().objectName(), "fileLandingView")
        win.close()


if __name__ == "__main__":
    unittest.main()

