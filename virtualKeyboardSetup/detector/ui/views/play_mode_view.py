"""
ui/views/play_mode_view.py

Play Mode View: Visual Testing HUD.
Renders the live monocular camera feed with hand skeletal landmarks and AprilTag tracking.
Simulates touch events visually without triggering real operating system keystrokes.
Includes header controls for camera selection, AI model hot-swapping, mode switching,
and a toggleable 2D paper layout canvas that highlights keys in finger-specific colors.
Sidebar features comfortable breathing room and generous padding.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    ComboBox,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    SingleDirectionScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
)

from config.constants import (
    UI_ACCENT,
    UI_BG_CARD,
    UI_BG_DARK,
    UI_TEXT_PRI,
    UI_TEXT_SEC,
)
from core.interfaces.touch_model import ModelEntry, ModelRegistry
from core.layout.layout_parser import LayoutData
from services.camera_discovery import CameraInfo, discover_cameras
from ui.components.camera_feed_widget import CameraFeedWidget
from ui.components.finger_status_bar import FingerStatusBar
from ui.components.paper_layout_canvas import PaperLayoutCanvasWidget
from ui.components.touch_event_log import TouchEventLog
from viewmodels.detector_viewmodel import DetectorViewModel, ExecutionMode

_SIDEBAR_WIDTH = 380


class PlayModeView(QWidget):
    """Interactive visual testing HUD (simulated events only)."""

    mode_switch_requested = Signal(str)  # "play" or "run"
    stop_requested = Signal()

    def __init__(self, vm: DetectorViewModel, parent=None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._layout_data: LayoutData | None = None
        self._models: list[ModelEntry] = []
        self._cameras: list[CameraInfo] = []

        self._setup_ui()
        self._connect_vm()
        self._populate_models()
        self._refresh_cameras()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header Toolbar ────────────────────────────────────────────────────
        header = QWidget(self)
        header.setFixedHeight(58)
        header.setStyleSheet(
            "QWidget { background-color: #16161A; border-bottom: 1px solid rgba(255, 255, 255, 0.06); }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        header_layout.setSpacing(14)

        # Title and badge
        title_box = QHBoxLayout()
        title_box.setSpacing(10)
        title_lbl = SubtitleLabel("Play Mode", header)
        title_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-size: 16px; font-weight: bold;")
        badge = CaptionLabel("TESTING HUD", header)
        badge.setStyleSheet(
            "background: rgba(0, 190, 255, 0.12); color: #00BEFF; "
            "padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;"
        )
        title_box.addWidget(title_lbl)
        title_box.addWidget(badge)
        header_layout.addLayout(title_box)

        header_layout.addStretch(1)

        # Camera Selector
        cam_box = QHBoxLayout()
        cam_box.setSpacing(6)
        cam_lbl = CaptionLabel("Camera:", header)
        cam_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px;")
        self.combo_camera = ComboBox(header)
        self.combo_camera.setFixedWidth(180)
        self.combo_camera.currentIndexChanged.connect(self._on_camera_selection_changed)
        self.btn_refresh_cams = PushButton(FluentIcon.SYNC, "", header)
        self.btn_refresh_cams.setFixedSize(32, 32)
        self.btn_refresh_cams.setToolTip("Refresh connected cameras")
        self.btn_refresh_cams.clicked.connect(self._refresh_cameras)
        cam_box.addWidget(cam_lbl)
        cam_box.addWidget(self.combo_camera)
        cam_box.addWidget(self.btn_refresh_cams)
        header_layout.addLayout(cam_box)

        # Quick Model Selector
        model_box = QHBoxLayout()
        model_box.setSpacing(6)
        model_caption = CaptionLabel("Model:", header)
        model_caption.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px;")
        self.combo_model = ComboBox(header)
        self.combo_model.setFixedWidth(190)
        self.combo_model.currentIndexChanged.connect(self._on_model_selection_changed)
        model_box.addWidget(model_caption)
        model_box.addWidget(self.combo_model)
        header_layout.addLayout(model_box)

        # Sidebar View Switcher (Telemetry Log vs 2D Paper Canvas)
        self.view_switcher = SegmentedWidget(header)
        self.view_switcher.addItem("telemetryView", "Telemetry & Logs")
        self.view_switcher.addItem("canvasView", "2D Paper Layout")
        self.view_switcher.setCurrentItem("telemetryView")
        self.view_switcher.currentItemChanged.connect(self._on_view_switch)
        header_layout.addWidget(self.view_switcher)

        # Switch to Run Mode Primary Button
        self.btn_run_mode = PrimaryPushButton(FluentIcon.SPEED_HIGH, "Switch to Run Mode", header)
        self.btn_run_mode.setFixedHeight(34)
        self.btn_run_mode.clicked.connect(self._on_switch_to_run_mode)
        header_layout.addWidget(self.btn_run_mode)

        root_layout.addWidget(header)

        # ── Main Content Area ─────────────────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        # Left: Live Camera Feed
        self.feed = CameraFeedWidget()
        content_row.addWidget(self.feed, 1)

        # Right: Interactive Sidebar Panel with Generous Padding
        sidebar = QWidget(self)
        sidebar.setFixedWidth(_SIDEBAR_WIDTH)
        sidebar.setStyleSheet(
            "QWidget { background-color: #18181C; border-left: 1px solid rgba(255, 255, 255, 0.06); }"
        )
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(20, 18, 20, 18)
        sb_layout.setSpacing(14)

        # Stacked container for Telemetry vs 2D Paper Canvas
        self.side_stack = QStackedWidget(sidebar)
        self.side_stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")

        # ── Page 1: Telemetry & Logs ──────────────────────────────────────────
        page_telemetry = QWidget(self.side_stack)
        page_tel_layout = QVBoxLayout(page_telemetry)
        page_tel_layout.setContentsMargins(0, 0, 0, 0)
        page_tel_layout.setSpacing(12)

        # Live performance stats card
        perf_card = CardWidget(page_telemetry)
        perf_card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        perf_inner = QHBoxLayout(perf_card)
        perf_inner.setContentsMargins(16, 12, 16, 12)

        self.lbl_fps = StrongBodyLabel("FPS: 0.0", perf_card)
        self.lbl_fps.setStyleSheet("background: transparent; color: #00DC64; font-size: 13px; font-weight: bold;")
        perf_inner.addWidget(self.lbl_fps)
        perf_inner.addStretch(1)

        status_col = QVBoxLayout()
        status_col.setSpacing(3)
        self.lbl_layout_status = CaptionLabel("Layout: Searching...", perf_card)
        self.lbl_layout_status.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px;")
        self.lbl_hand_status = CaptionLabel("Hand: Not visible", perf_card)
        self.lbl_hand_status.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px;")
        status_col.addWidget(self.lbl_layout_status)
        status_col.addWidget(self.lbl_hand_status)
        perf_inner.addLayout(status_col)
        page_tel_layout.addWidget(perf_card)

        # Scrollable container for finger bars and touch event log
        tel_scroll = SingleDirectionScrollArea(orient=Qt.Vertical, parent=page_telemetry)
        tel_scroll.setWidgetResizable(True)
        tel_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        tel_scroll_content = QWidget()
        tel_scroll_content.setStyleSheet("background-color: transparent;")
        tsc_layout = QVBoxLayout(tel_scroll_content)
        tsc_layout.setContentsMargins(0, 0, 6, 0)
        tsc_layout.setSpacing(14)

        # Finger Probabilities Card
        tsc_layout.addWidget(self._make_section_label("Finger Touch Probabilities"))
        self.finger_bar = FingerStatusBar(tel_scroll_content)
        tsc_layout.addWidget(self.finger_bar)

        # Simulated Touch Event Log
        tsc_layout.addWidget(self._make_section_label("Simulated Touch Events"))
        self.event_log = TouchEventLog(tel_scroll_content)
        tsc_layout.addWidget(self.event_log)

        tel_scroll.setWidget(tel_scroll_content)
        page_tel_layout.addWidget(tel_scroll, 1)

        self.side_stack.addWidget(page_telemetry)

        # ── Page 2: 2D Paper Layout Canvas ────────────────────────────────────
        page_canvas = QWidget(self.side_stack)
        page_canvas_layout = QVBoxLayout(page_canvas)
        page_canvas_layout.setContentsMargins(0, 0, 0, 0)
        page_canvas_layout.setSpacing(10)

        canvas_header = QHBoxLayout()
        canvas_title = StrongBodyLabel("2D Paper Layout Map", page_canvas)
        canvas_title.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-size: 13px; font-weight: bold;")
        canvas_hint = CaptionLabel("Active Finger Touch Illumination", page_canvas)
        canvas_hint.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px;")
        canvas_header.addWidget(canvas_title)
        canvas_header.addStretch(1)
        canvas_header.addWidget(canvas_hint)
        page_canvas_layout.addLayout(canvas_header)

        self.paper_canvas = PaperLayoutCanvasWidget(page_canvas)
        page_canvas_layout.addWidget(self.paper_canvas, 1)

        self.side_stack.addWidget(page_canvas)

        sb_layout.addWidget(self.side_stack, 1)
        content_row.addWidget(sidebar)

        root_layout.addLayout(content_row)

    def _connect_vm(self) -> None:
        self._vm.frame_updated.connect(self._on_frame_updated)
        self._vm.finger_probs_updated.connect(self._on_probs_updated)
        self._vm.touch_event.connect(self._on_touch_event)
        self._vm.model_changed.connect(self._on_vm_model_changed)
        self._vm.pipeline_error.connect(self._on_error)

    def _populate_models(self) -> None:
        self._models = ModelRegistry.all_entries()
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        active_idx = 0
        current_name = self._vm.active_model_name

        for i, m in enumerate(self._models):
            self.combo_model.addItem(m.name, userData=m)
            if current_name and m.name == current_name:
                active_idx = i
            elif not current_name and ("lstm" in m.name.lower() or "best" in m.name.lower()):
                active_idx = i

        if self._models:
            self.combo_model.setCurrentIndex(active_idx)
        self.combo_model.blockSignals(False)

    def _refresh_cameras(self) -> None:
        self._cameras = discover_cameras(max_index=6)
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        if not self._cameras:
            self.combo_camera.addItem("Default Camera 0", userData=0)
            self.combo_camera.blockSignals(False)
            return

        selected_idx = 0
        current_cam = self._vm.camera_index
        for i, cam in enumerate(self._cameras):
            label = f"Camera {cam.index}: {cam.name}"
            self.combo_camera.addItem(label, userData=cam.index)
            if cam.index == current_cam:
                selected_idx = i
        self.combo_camera.setCurrentIndex(selected_idx)
        self.combo_camera.blockSignals(False)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_layout_data(self, layout_data: LayoutData) -> None:
        self._layout_data = layout_data
        self.paper_canvas.set_layout(layout_data)

    def refresh_models(self) -> None:
        self._populate_models()

    def sync_camera_index(self, index: int) -> None:
        self.combo_camera.blockSignals(True)
        for i in range(self.combo_camera.count()):
            if self.combo_camera.itemData(i) == index:
                self.combo_camera.setCurrentIndex(i)
                break
        self.combo_camera.blockSignals(False)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_view_switch(self, item_name: str) -> None:
        if item_name == "canvasView":
            self.side_stack.setCurrentIndex(1)
        else:
            self.side_stack.setCurrentIndex(0)

    def _on_camera_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        cam_idx = self.combo_camera.currentData()
        if cam_idx is not None:
            self._vm.set_camera_index(int(cam_idx))
            InfoBar.success(
                title="Camera Switched",
                content=f"Capture device switched to Camera {cam_idx}.",
                position=InfoBarPosition.TOP,
                parent=self,
                duration=2500,
            )

    def _on_model_selection_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._models):
            return
        selected_model = self.combo_model.currentData()
        if selected_model:
            self._vm.set_model(selected_model)
            InfoBar.success(
                title="Model Hot-Swapped",
                content=f"Active touch detector set to {selected_model.name}.",
                position=InfoBarPosition.TOP,
                parent=self,
                duration=3000,
            )

    @Slot(str)
    def _on_vm_model_changed(self, model_name: str) -> None:
        self.combo_model.blockSignals(True)
        for i, m in enumerate(self._models):
            if m.name == model_name:
                self.combo_model.setCurrentIndex(i)
                break
        self.combo_model.blockSignals(False)

    def _on_switch_to_run_mode(self) -> None:
        self.mode_switch_requested.emit("run")

    @Slot(object, float, bool, bool)
    def _on_frame_updated(
        self,
        frame: np.ndarray,
        fps: float,
        hand_detected: bool,
        layout_found: bool,
    ) -> None:
        if self._vm.execution_mode != ExecutionMode.PLAY:
            return

        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            self.feed.update_frame(frame)

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_layout_status.setText(
            "Layout: Tracked" if layout_found else "Layout: Searching..."
        )
        self.lbl_layout_status.setStyleSheet(
            f"background: transparent; color: {'#00DC64' if layout_found else UI_TEXT_SEC}; font-size: 11px;"
        )
        self.lbl_hand_status.setText(
            "Hand: Detected" if hand_detected else "Hand: Not visible"
        )

        if not hand_detected:
            self.finger_bar.set_no_hand()

    @Slot(dict)
    def _on_probs_updated(self, probs: dict) -> None:
        if self._vm.execution_mode == ExecutionMode.PLAY:
            self.finger_bar.set_probs(probs)

    @Slot(str, str, float)
    def _on_touch_event(self, key_id: str, finger: str, prob: float) -> None:
        if self._vm.execution_mode != ExecutionMode.PLAY:
            return

        label = key_id
        if self._layout_data:
            for btn in self._layout_data.buttons:
                if btn.id == key_id:
                    label = btn.label
                    break

        self.event_log.add_event(key_id, label, finger, prob)
        self.paper_canvas.highlight_touch(key_id, finger, prob)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        InfoBar.error(
            title="Pipeline Error",
            content=msg,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=6000,
        )

    @staticmethod
    def _make_section_label(text: str) -> StrongBodyLabel:
        lbl = StrongBodyLabel(text)
        lbl.setStyleSheet(
            f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;"
        )
        return lbl
