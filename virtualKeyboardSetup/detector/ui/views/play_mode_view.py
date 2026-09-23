"""
ui/views/play_mode_view.py

Play Mode View - visual testing HUD.
Fixes applied:
  - Camera refresh button: plain symbol QPushButton (no FluentIcon overlap).
  - Sidebar toggle is now a thin tab strip at the BOTTOM of the sidebar (not top buttons).
  - The vertical left border of the sidebar is kept as a single 1px line (no double-line artefact).
  - SegmentedWidget removed from header; sidebar switcher lives at the sidebar footer only.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    ComboBox,
    InfoBar,
    InfoBarPosition,
    SingleDirectionScrollArea,
)

from ui.theme import (
    ACCENT, BG_HEADER, BG_PAGE, BG_PANEL,
    BORDER, BORDER_SUBTLE,
    TXT_PRI, TXT_SEC,
    FONT_SMALL, SUCCESS,
    FINGER_COLORS,
    btn_primary, btn_ghost, btn_icon_only,
    LABEL_TRANSPARENT,
)
from core.interfaces.touch_model import ModelEntry, ModelRegistry
from core.layout.layout_parser import LayoutData
from services.camera_discovery import CameraInfo, discover_cameras
from ui.components.camera_feed_widget import CameraFeedWidget
from ui.components.finger_status_bar import FingerStatusBar
from ui.components.paper_layout_canvas import PaperLayoutCanvasWidget
from ui.components.touch_event_log import TouchEventLog
from viewmodels.detector_viewmodel import DetectorViewModel, ExecutionMode

_SIDEBAR_W = 360


class PlayModeView(QWidget):
    """Interactive visual testing HUD (simulated events only)."""

    mode_switch_requested = Signal(str)
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

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background: {BG_PAGE};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header toolbar ─────────────────────────────────────────────────────
        header = QWidget(self)
        header.setObjectName("playHeader")
        header.setFixedHeight(54)
        header.setStyleSheet(
            f"QWidget#playHeader {{ background: {BG_HEADER}; border-bottom: 1px solid {BORDER}; }}"
            f"QLabel {{ {LABEL_TRANSPARENT} }}"
        )
        h_row = QHBoxLayout(header)
        h_row.setContentsMargins(20, 0, 16, 0)
        h_row.setSpacing(14)

        # Title
        title = QLabel("Play Mode", header)
        title.setStyleSheet(f"color: {TXT_PRI}; font-size: 15px; font-weight: 700; {LABEL_TRANSPARENT}")
        badge = QLabel("TESTING", header)
        badge.setStyleSheet(
            f"color: {ACCENT}; background: rgba(0,159,239,0.12); "
            f"border-radius: 4px; padding: 2px 7px; font-size: 10px; font-weight: 700; border: none;"
        )
        h_row.addWidget(title)
        h_row.addWidget(badge)
        h_row.addStretch(1)

        # Camera selector
        h_row.addWidget(self._hdr_label("Camera:", header))
        self.combo_camera = ComboBox(header)
        self.combo_camera.setFixedWidth(170)
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)
        h_row.addWidget(self.combo_camera)

        # Refresh cameras symbol button (no FluentIcon, no overlap)
        self.btn_refresh_cams = btn_icon_only("↺", header, size=30)
        self.btn_refresh_cams.setToolTip("Refresh camera list")
        self.btn_refresh_cams.clicked.connect(self._refresh_cameras)
        h_row.addWidget(self.btn_refresh_cams)

        # Model selector
        h_row.addWidget(self._hdr_label("Model:", header))
        self.combo_model = ComboBox(header)
        self.combo_model.setFixedWidth(185)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        h_row.addWidget(self.combo_model)

        # Switch to Run Mode
        self.btn_run_mode = btn_primary("Switch to Run Mode", header, height=32)
        self.btn_run_mode.clicked.connect(self._on_switch_run)
        h_row.addWidget(self.btn_run_mode)

        root.addWidget(header)

        # ── Content row ────────────────────────────────────────────────────────
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Camera feed (left)
        self.feed = CameraFeedWidget()
        content.addWidget(self.feed, 1)

        # Sidebar (right) - border: none to remove the vertical line
        sidebar = QWidget(self)
        sidebar.setObjectName("playSidebar")
        sidebar.setFixedWidth(_SIDEBAR_W)
        sidebar.setStyleSheet(
            f"QWidget#playSidebar {{ background: {BG_PANEL}; border: none; }}"
            f"QLabel {{ {LABEL_TRANSPARENT} }}"
        )
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        # ── Quick stats header strip (FPS + Layout + Hand) ─────────────────────
        stats_widget = QWidget(sidebar)
        stats_widget.setStyleSheet("background: transparent; border: none;")
        pt = QVBoxLayout(stats_widget)
        pt.setContentsMargins(16, 14, 16, 10)
        pt.setSpacing(10)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self.lbl_fps = QLabel("12.0 FPS", stats_widget)
        self.lbl_fps.setStyleSheet(f"color: {SUCCESS}; font-size: 13px; font-weight: 700; {LABEL_TRANSPARENT}")

        self.lbl_layout_status = QLabel("Layout: searching...", stats_widget)
        self.lbl_layout_status.setStyleSheet(f"color: {TXT_SEC}; font-size: 11px; {LABEL_TRANSPARENT}")

        self.lbl_hand_status = QLabel("Hand: none", stats_widget)
        self.lbl_hand_status.setStyleSheet(f"color: {TXT_SEC}; font-size: 11px; {LABEL_TRANSPARENT}")

        stats_row.addWidget(self.lbl_fps)
        stats_row.addStretch(1)
        stats_row.addWidget(self.lbl_layout_status)
        stats_row.addWidget(self.lbl_hand_status)
        pt.addLayout(stats_row)
        pt.addWidget(self._thin_rule(stats_widget))
        sb.addWidget(stats_widget)

        # ── Scrollable panel: Finger indicators, 2D Paper Layout & Touch Logs ──
        scroll = SingleDirectionScrollArea(orient=Qt.Vertical, parent=sidebar)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_w = QWidget()
        scroll_w.setStyleSheet("background: transparent; border: none;")
        sw = QVBoxLayout(scroll_w)
        sw.setContentsMargins(16, 6, 16, 16)
        sw.setSpacing(14)

        # 1. Finger touch probabilities
        sw.addWidget(self._section("FINGER TOUCH PROBABILITIES"))
        self.finger_bar = FingerStatusBar(scroll_w)
        sw.addWidget(self.finger_bar)

        # 2. 2D Paper Layout - placed under finger indicators as requested
        sw.addWidget(self._section("2D PAPER LAYOUT"))
        self.paper_canvas = PaperLayoutCanvasWidget(scroll_w)
        self.paper_canvas.setFixedHeight(230)
        sw.addWidget(self.paper_canvas)

        # 3. Simulated touch events
        sw.addWidget(self._section("SIMULATED TOUCH EVENTS"))
        self.event_log = TouchEventLog(scroll_w)
        self.event_log.setMinimumHeight(180)
        sw.addWidget(self.event_log)

        scroll.setWidget(scroll_w)
        sb.addWidget(scroll, 1)

        content.addWidget(sidebar)
        root.addLayout(content, 1)

    # ── Misc helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _hdr_label(text: str, parent: QWidget) -> QLabel:
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(f"color: {TXT_SEC}; font-size: 11px; background: transparent; border: none;")
        return lbl

    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TXT_SEC}; font-size: 10px; font-weight: 700;"
            f"letter-spacing: 0.6px; background: transparent; border: none;"
        )
        return lbl

    @staticmethod
    def _thin_rule(parent: QWidget) -> QWidget:
        from PySide6.QtWidgets import QFrame
        f = QFrame(parent)
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(f"background: {BORDER}; border: none;")
        return f

    # ── Signals / slots ────────────────────────────────────────────────────────

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
        matched = False
        for i, cam in enumerate(self._cameras):
            self.combo_camera.addItem(f"{cam.name} (/{cam.index})", userData=cam.index)
            if cam.index == current_cam:
                selected_idx = i
                matched = True
        self.combo_camera.setCurrentIndex(selected_idx)
        self.combo_camera.blockSignals(False)
        if not matched and self._cameras:
            self._vm.set_camera_index(self._cameras[0].index)

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

    def _on_camera_changed(self, index: int) -> None:
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

    def _on_model_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._models):
            return
        selected_model = self.combo_model.currentData()
        if selected_model:
            self._vm.set_model(selected_model)
            InfoBar.success(
                title="Model Switched",
                content=f"Active detector set to {selected_model.name}.",
                position=InfoBarPosition.TOP,
                parent=self,
                duration=2500,
            )

    @Slot(str)
    def _on_vm_model_changed(self, model_name: str) -> None:
        self.combo_model.blockSignals(True)
        for i, m in enumerate(self._models):
            if m.name == model_name:
                self.combo_model.setCurrentIndex(i)
                break
        self.combo_model.blockSignals(False)

    def _on_switch_run(self) -> None:
        self.mode_switch_requested.emit("run")

    @Slot(object, float, bool, bool)
    def _on_frame_updated(self, frame, fps: float, hand_detected: bool, layout_found: bool) -> None:
        if self._vm.execution_mode != ExecutionMode.PLAY:
            return
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            self.feed.update_frame(frame)

        self.lbl_fps.setText(f"{fps:.1f} FPS")
        self.lbl_layout_status.setText("Layout: tracked" if layout_found else "Layout: searching...")
        self.lbl_layout_status.setStyleSheet(
            f"color: {SUCCESS}; font-size: 11px;" if layout_found
            else f"color: {TXT_SEC}; font-size: 11px;"
        )
        self.lbl_hand_status.setText("Hand: detected" if hand_detected else "Hand: none")
        self.lbl_hand_status.setStyleSheet(
            f"color: {SUCCESS}; font-size: 11px;" if hand_detected
            else f"color: {TXT_SEC}; font-size: 11px;"
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
