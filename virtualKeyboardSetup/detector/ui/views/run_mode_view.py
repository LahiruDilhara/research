"""
ui/views/run_mode_view.py

Run Mode View: Headless Production HUD.
Optimized for zero video rendering overhead. Video frames and skeletal overlays are not
drawn to the user interface, and frame memory is discarded immediately after landmark extraction.
Features header controls for camera selection, AI model hot-swapping, and mode switching.
Displays multi-finger probability curves, live pipeline dynamics graphs, and an executed actions feed.
"""

from __future__ import annotations

from datetime import datetime
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
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
    PushButton,
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
from ui.components.dynamics_graph import DetectionDynamicsGraphWidget
from ui.components.telemetry_graph import TelemetryGraphWidget
from viewmodels.detector_viewmodel import DetectorViewModel, ExecutionMode


class ExecutedActionItem(QWidget):
    """Custom list item widget displaying an executed action event."""

    def __init__(
        self,
        timestamp: str,
        action_type: str,
        payload: str,
        key_id: str,
        finger: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Time
        lbl_time = CaptionLabel(timestamp, self)
        lbl_time.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 10px; font-family: monospace;")
        layout.addWidget(lbl_time)

        # Key badge
        lbl_key = StrongBodyLabel(key_id, self)
        lbl_key.setStyleSheet(
            f"background-color: rgba(255, 255, 255, 0.08); color: {UI_TEXT_PRI}; "
            "padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;"
        )
        layout.addWidget(lbl_key)

        # Finger badge
        finger_colors = {
            "Thumb": "#FF8C00",
            "Index": "#00BEFF",
            "Middle": "#00E678",
            "Ring": "#E650FF",
            "Pinky": "#FF4B4B",
        }
        color = finger_colors.get(finger, UI_ACCENT)
        lbl_finger = CaptionLabel(finger, self)
        lbl_finger.setStyleSheet(
            f"background-color: rgba(255, 255, 255, 0.05); color: {color}; "
            "padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 500;"
        )
        layout.addWidget(lbl_finger)

        layout.addStretch(1)

        # Action payload
        lbl_payload = CaptionLabel(f"[{action_type.upper()}] {payload}", self)
        lbl_payload.setStyleSheet(f"color: {UI_ACCENT}; font-size: 11px; font-weight: 500;")
        layout.addWidget(lbl_payload)


class RunModeView(QWidget):
    """Headless production telemetry HUD with dual dynamics graphs."""

    mode_switch_requested = Signal(str)  # "play" or "run"

    def __init__(self, vm: DetectorViewModel, parent=None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._layout_data: LayoutData | None = None
        self._models: list[ModelEntry] = []
        self._cameras: list[CameraInfo] = []
        self._latest_fps: float = 12.0
        self._latest_latency: float = 0.0

        self._setup_ui()
        self._connect_vm()
        self._populate_models()
        self._refresh_cameras()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 20, 24, 20)
        root_layout.setSpacing(16)

        # ── Header Toolbar ────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(14)

        title_box = QHBoxLayout()
        title_box.setSpacing(10)
        title_lbl = SubtitleLabel("Run Mode", self)
        title_lbl.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 18px; font-weight: bold;")
        badge = CaptionLabel("PRODUCTION ENGINE (ACTIONS ACTIVE)", self)
        badge.setStyleSheet(
            "background: rgba(0, 220, 100, 0.15); color: #00DC64; "
            "padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 10px;"
        )
        title_box.addWidget(title_lbl)
        title_box.addWidget(badge)
        header_row.addLayout(title_box)

        header_row.addStretch(1)

        # Camera Selector
        cam_box = QHBoxLayout()
        cam_box.setSpacing(6)
        cam_lbl = CaptionLabel("Camera:", self)
        cam_lbl.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        self.combo_camera = ComboBox(self)
        self.combo_camera.setFixedWidth(180)
        self.combo_camera.currentIndexChanged.connect(self._on_camera_selection_changed)
        self.btn_refresh_cams = PushButton(FluentIcon.SYNC, "", self)
        self.btn_refresh_cams.setFixedSize(32, 32)
        self.btn_refresh_cams.setToolTip("Refresh connected cameras")
        self.btn_refresh_cams.clicked.connect(self._refresh_cameras)
        cam_box.addWidget(cam_lbl)
        cam_box.addWidget(self.combo_camera)
        cam_box.addWidget(self.btn_refresh_cams)
        header_row.addLayout(cam_box)

        # Quick Model Selector
        model_box = QHBoxLayout()
        model_box.setSpacing(6)
        model_caption = CaptionLabel("Model:", self)
        model_caption.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        self.combo_model = ComboBox(self)
        self.combo_model.setFixedWidth(190)
        self.combo_model.currentIndexChanged.connect(self._on_model_selection_changed)
        model_box.addWidget(model_caption)
        model_box.addWidget(self.combo_model)
        header_row.addLayout(model_box)

        # Switch to Play Mode Button
        self.btn_play_mode = PushButton(FluentIcon.VIDEO, "Switch to Play Mode", self)
        self.btn_play_mode.setFixedHeight(34)
        self.btn_play_mode.clicked.connect(self._on_switch_to_play_mode)
        header_row.addWidget(self.btn_play_mode)

        root_layout.addLayout(header_row)

        # ── Top Stat Cards Row ────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)

        self.card_fps = self._make_stat_card("PIPELINE FPS", "0.0 FPS", "#00DC64")
        self.card_latency = self._make_stat_card("MODEL LATENCY", "-- ms", "#00BEFF")
        self.card_model = self._make_stat_card("ACTIVE MODEL", self._vm.active_model_name or "Default", UI_TEXT_PRI)
        self.card_layout = self._make_stat_card("APRILTAG TRACKING", "Searching...", UI_TEXT_SEC)

        stats_row.addWidget(self.card_fps, 1)
        stats_row.addWidget(self.card_latency, 1)
        stats_row.addWidget(self.card_model, 1)
        stats_row.addWidget(self.card_layout, 1)

        root_layout.addLayout(stats_row)

        # ── Main Center Area: Dual Graphs + Executed Actions Feed ─────────────
        main_body = QHBoxLayout()
        main_body.setSpacing(16)

        # Left Column: Dual Telemetry & Dynamics Graphs
        graphs_col = QVBoxLayout()
        graphs_col.setSpacing(14)

        # 1. Multi-Finger Probability Telemetry Graph Card
        graph1_card = CardWidget(self)
        graph1_card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        g1_layout = QVBoxLayout(graph1_card)
        g1_layout.setContentsMargins(18, 16, 18, 16)
        g1_layout.setSpacing(8)

        g1_header = QHBoxLayout()
        g1_title = StrongBodyLabel("Multi-Finger Contact Probabilities", graph1_card)
        g1_title.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 13px; font-weight: bold;")
        g1_sub = CaptionLabel("Rolling window touch probability curves (0.0 to 1.0)", graph1_card)
        g1_sub.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        g1_header.addWidget(g1_title)
        g1_header.addStretch(1)
        g1_header.addWidget(g1_sub)
        g1_layout.addLayout(g1_header)

        self.graph_widget = TelemetryGraphWidget(threshold=0.5, parent=graph1_card)
        g1_layout.addWidget(self.graph_widget, 1)
        graphs_col.addWidget(graph1_card, 1)

        # 2. Pipeline Dynamics Graph Card (Latency & FPS)
        graph2_card = CardWidget(self)
        graph2_card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        g2_layout = QVBoxLayout(graph2_card)
        g2_layout.setContentsMargins(18, 16, 18, 16)
        g2_layout.setSpacing(8)

        g2_header = QHBoxLayout()
        g2_title = StrongBodyLabel("Pipeline Latency & Frame Rate Dynamics", graph2_card)
        g2_title.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 13px; font-weight: bold;")
        g2_sub = CaptionLabel("Real-time CPU inference latency (ms) and capture FPS", graph2_card)
        g2_sub.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        g2_header.addWidget(g2_title)
        g2_header.addStretch(1)
        g2_header.addWidget(g2_sub)
        g2_layout.addLayout(g2_header)

        self.dynamics_graph = DetectionDynamicsGraphWidget(parent=graph2_card)
        g2_layout.addWidget(self.dynamics_graph, 1)
        graphs_col.addWidget(graph2_card, 1)

        main_body.addLayout(graphs_col, 6)

        # Right Column: Executed Actions Feed Card
        actions_card = CardWidget(self)
        actions_card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(18, 16, 18, 16)
        actions_layout.setSpacing(10)

        act_header = QHBoxLayout()
        act_title = StrongBodyLabel("Executed Actions Feed", actions_card)
        act_title.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 13px; font-weight: bold;")
        self.lbl_action_count = CaptionLabel("0 executed", actions_card)
        self.lbl_action_count.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        act_header.addWidget(act_title)
        act_header.addStretch(1)
        act_header.addWidget(self.lbl_action_count)
        actions_layout.addLayout(act_header)

        self.actions_list = QListWidget(actions_card)
        self.actions_list.setStyleSheet(
            "QListWidget { background: transparent; border: none; outline: none; } "
            "QListWidget::item { border-bottom: 1px solid rgba(255, 255, 255, 0.04); }"
        )
        actions_layout.addWidget(self.actions_list, 1)

        main_body.addWidget(actions_card, 4)

        root_layout.addLayout(main_body, 1)

    def _connect_vm(self) -> None:
        self._vm.fps_updated.connect(self._on_fps_updated)
        self._vm.latency_updated.connect(self._on_latency_updated)
        self._vm.finger_probs_updated.connect(self._on_probs_updated)
        self._vm.action_executed.connect(self._on_action_executed)
        self._vm.model_changed.connect(self._on_model_changed)
        self._vm.frame_updated.connect(self._on_frame_updated)

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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def set_layout_data(self, layout: LayoutData) -> None:
        self._layout_data = layout

    def set_touch_threshold(self, threshold: float) -> None:
        self.graph_widget.set_threshold(threshold)

    def refresh_models(self) -> None:
        self._populate_models()

    def sync_camera_index(self, index: int) -> None:
        self.combo_camera.blockSignals(True)
        for i in range(self.combo_camera.count()):
            if self.combo_camera.itemData(i) == index:
                self.combo_camera.setCurrentIndex(i)
                break
        self.combo_camera.blockSignals(False)

    def _make_stat_card(self, title: str, initial_val: str, val_color: str) -> CardWidget:
        card = CardWidget(self)
        card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; }} "
            "QLabel { background-color: transparent; border: none; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(4)

        lbl_t = CaptionLabel(title, card)
        lbl_t.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")

        lbl_v = StrongBodyLabel(initial_val, card)
        lbl_v.setObjectName("valueLabel")
        lbl_v.setStyleSheet(f"color: {val_color}; font-size: 16px; font-weight: bold;")

        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)
        return card

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_switch_to_play_mode(self) -> None:
        self.mode_switch_requested.emit("play")

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

    @Slot(float)
    def _on_fps_updated(self, fps: float) -> None:
        self._latest_fps = fps
        lbl = self.card_fps.findChild(StrongBodyLabel, "valueLabel")
        if lbl:
            lbl.setText(f"{fps:.1f} FPS")
        self.dynamics_graph.push_metrics(self._latest_latency, self._latest_fps)

    @Slot(float)
    def _on_latency_updated(self, latency_ms: float) -> None:
        self._latest_latency = latency_ms
        lbl = self.card_latency.findChild(StrongBodyLabel, "valueLabel")
        if lbl:
            lbl.setText(f"{latency_ms:.1f} ms")
        self.dynamics_graph.push_metrics(self._latest_latency, self._latest_fps)

    @Slot(dict)
    def _on_probs_updated(self, probs: dict) -> None:
        if self._vm.execution_mode == ExecutionMode.RUN:
            self.graph_widget.push_probabilities(probs)

    @Slot(str, str, str, str)
    def _on_action_executed(
        self, action_type: str, payload: str, key_id: str, finger: str
    ) -> None:
        now = datetime.now().strftime("%H:%M:%S.%f")[:12]
        item_widget = ExecutedActionItem(now, action_type, payload, key_id, finger)

        item = QListWidgetItem()
        item.setSizeHint(item_widget.sizeHint())
        self.actions_list.insertItem(0, item)
        self.actions_list.setItemWidget(item, item_widget)

        # Cap list to 100 entries
        if self.actions_list.count() > 100:
            self.actions_list.takeItem(self.actions_list.count() - 1)

        self.lbl_action_count.setText(f"{self.actions_list.count()} executed")

    @Slot(str)
    def _on_model_changed(self, model_name: str) -> None:
        self.combo_model.blockSignals(True)
        for i, m in enumerate(self._models):
            if m.name == model_name:
                self.combo_model.setCurrentIndex(i)
                break
        self.combo_model.blockSignals(False)

        lbl = self.card_model.findChild(StrongBodyLabel, "valueLabel")
        if lbl:
            lbl.setText(model_name)

    @Slot(object, float, bool, bool)
    def _on_frame_updated(
        self, frame, fps: float, hand_detected: bool, layout_found: bool
    ) -> None:
        lbl = self.card_layout.findChild(StrongBodyLabel, "valueLabel")
        if lbl:
            if layout_found:
                lbl.setText("Locked")
                lbl.setStyleSheet("color: #00DC64; font-size: 16px; font-weight: bold;")
            else:
                lbl.setText("Searching...")
                lbl.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 16px; font-weight: bold;")
