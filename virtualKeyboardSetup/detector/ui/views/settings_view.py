"""
ui/views/settings_view.py

Settings view following MVVM architecture and SOLID principles.
Provides ergonomic input controls for runtime thresholds, MediaPipe detector confidences,
window quality filters, and layout configuration.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    DoubleSpinBox,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SmoothScrollArea,
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
from viewmodels.settings_viewmodel import SettingsViewModel


class SettingsView(QWidget):
    """View layer for application settings and runtime overrides."""

    settings_applied = Signal()

    def __init__(self, vm: SettingsViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._setup_ui()
        self._connect_signals()

    @property
    def view_model(self) -> SettingsViewModel:
        return self._vm

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Smooth scroll area for seamless viewing across screen sizes
        scroll_area = SmoothScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            f"SmoothScrollArea {{ background-color: {UI_BG_DARK}; border: none; }}"
        )

        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {UI_BG_DARK};")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(32, 28, 32, 28)
        content_layout.setSpacing(20)

        title = SubtitleLabel("Settings")
        title.setStyleSheet(f"color: {UI_ACCENT}; font-size: 20px; font-weight: bold;")
        content_layout.addWidget(title)

        # ── Card 1: Pipeline & Motion Thresholds ───────────────────────────────
        content_layout.addWidget(self._make_card(
            "Pipeline & Motion Thresholds",
            [
                (
                    "Target FPS",
                    "Capture and processing rate (default 12.0 FPS)",
                    self._make_double_spin(
                        value=self._vm.target_fps,
                        mn=1.0,
                        mx=60.0,
                        attr="target_fps_spin",
                        step=1.0,
                        decimals=1,
                    ),
                ),
                (
                    "Touch Threshold",
                    "Minimum model probability to trigger a touch (0.01 to 1.00)",
                    self._make_double_spin(
                        value=self._vm.touch_threshold,
                        mn=0.01,
                        mx=1.00,
                        attr="touch_threshold_spin",
                        step=0.05,
                        decimals=3,
                    ),
                ),
                (
                    "Fingertip Velocity Threshold",
                    "Minimum fingertip speed magnitude to evaluate window (default 0.0080)",
                    self._make_double_spin(
                        value=self._vm.fingertip_velocity_threshold,
                        mn=0.0001,
                        mx=2.0000,
                        attr="fingertip_vel_spin",
                        step=0.001,
                        decimals=4,
                    ),
                ),
                (
                    "Hand Movement Threshold",
                    "Maximum stationary joint displacement before flagging whole-hand motion (default 0.1550)",
                    self._make_double_spin(
                        value=self._vm.hand_movement_threshold,
                        mn=0.0010,
                        mx=1.0000,
                        attr="hand_movement_spin",
                        step=0.005,
                        decimals=4,
                    ),
                ),
            ]
        ))

        # ── Card 2: MediaPipe Detector Confidences ─────────────────────────────
        content_layout.addWidget(self._make_card(
            "MediaPipe Detector Confidences",
            [
                (
                    "Min Detection Confidence",
                    "Minimum confidence score for hand detection to be considered successful (default 0.50)",
                    self._make_double_spin(
                        value=self._vm.mediapipe_min_detection_confidence,
                        mn=0.05,
                        mx=1.00,
                        attr="mp_detection_conf_spin",
                        step=0.05,
                        decimals=2,
                    ),
                ),
                (
                    "Min Presence Confidence",
                    "Minimum confidence score for hand presence in video stream (default 0.50)",
                    self._make_double_spin(
                        value=self._vm.mediapipe_min_presence_confidence,
                        mn=0.05,
                        mx=1.00,
                        attr="mp_presence_conf_spin",
                        step=0.05,
                        decimals=2,
                    ),
                ),
                (
                    "Min Tracking Confidence",
                    "Minimum confidence score for hand landmark tracking to be robust (default 0.50)",
                    self._make_double_spin(
                        value=self._vm.mediapipe_min_tracking_confidence,
                        mn=0.05,
                        mx=1.00,
                        attr="mp_tracking_conf_spin",
                        step=0.05,
                        decimals=2,
                    ),
                ),
            ]
        ))

        # ── Card 3: Window Quality Filters (Step 10) ───────────────────────────
        content_layout.addWidget(self._make_card(
            "Window Quality Filters (Step 10)",
            [
                (
                    "Min Average Score",
                    "Minimum average hand confidence across 5-frame sequence window (default 0.65)",
                    self._make_double_spin(
                        value=self._vm.quality_min_avg_score,
                        mn=0.05,
                        mx=1.00,
                        attr="quality_min_avg_spin",
                        step=0.05,
                        decimals=2,
                    ),
                ),
                (
                    "Min Single Frame Score",
                    "Minimum hand confidence in any individual frame of sequence window (default 0.45)",
                    self._make_double_spin(
                        value=self._vm.quality_min_frame_score,
                        mn=0.05,
                        mx=1.00,
                        attr="quality_min_frame_spin",
                        step=0.05,
                        decimals=2,
                    ),
                ),
                (
                    "Max Score Drop",
                    "Maximum confidence fluctuation drop between frames in sequence window (default 0.35)",
                    self._make_double_spin(
                        value=self._vm.quality_max_score_drop,
                        mn=0.05,
                        mx=1.00,
                        attr="quality_max_drop_spin",
                        step=0.05,
                        decimals=2,
                    ),
                ),
            ]
        ))

        # ── Card 4: AI Model Plugins Directory ─────────────────────────────────
        content_layout.addWidget(self._make_card(
            "AI Model Plugins Directory",
            [
                (
                    "AI Model Plugins Dir",
                    "Relative or absolute path to the AI model plugins directory",
                    self._make_line_edit(
                        value=self._vm.plugins_dir,
                        attr="plugins_dir_edit",
                    ),
                ),
            ]
        ))

        content_layout.addSpacing(10)

        # ── Action Buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_restore = PushButton(FluentIcon.SYNC, "Restore Defaults")
        btn_restore.setFixedHeight(38)
        btn_restore.setMinimumWidth(150)
        btn_restore.clicked.connect(self._on_restore_defaults_clicked)

        btn_save = PrimaryPushButton(FluentIcon.SAVE, "Apply & Save")
        btn_save.setFixedHeight(38)
        btn_save.setMinimumWidth(140)
        btn_save.clicked.connect(self._on_save_clicked)

        btn_row.addStretch(1)
        btn_row.addWidget(btn_restore)
        btn_row.addWidget(btn_save)
        content_layout.addLayout(btn_row)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    def _connect_signals(self) -> None:
        self._vm.settings_saved.connect(self._on_settings_saved)
        self._vm.settings_loaded.connect(self._refresh_fields)
        self._vm.error_occurred.connect(self._on_settings_error)

    def _refresh_fields(self) -> None:
        self.target_fps_spin.setValue(self._vm.target_fps)
        self.touch_threshold_spin.setValue(self._vm.touch_threshold)
        self.fingertip_vel_spin.setValue(self._vm.fingertip_velocity_threshold)
        self.hand_movement_spin.setValue(self._vm.hand_movement_threshold)
        self.mp_detection_conf_spin.setValue(self._vm.mediapipe_min_detection_confidence)
        self.mp_presence_conf_spin.setValue(self._vm.mediapipe_min_presence_confidence)
        self.mp_tracking_conf_spin.setValue(self._vm.mediapipe_min_tracking_confidence)
        self.quality_min_avg_spin.setValue(self._vm.quality_min_avg_score)
        self.quality_min_frame_spin.setValue(self._vm.quality_min_frame_score)
        self.quality_max_drop_spin.setValue(self._vm.quality_max_score_drop)
        self.plugins_dir_edit.setText(self._vm.plugins_dir)

    def _make_card(self, section: str, rows: list[tuple]) -> CardWidget:
        card = CardWidget()
        clean_section = re.sub(r"[^a-zA-Z0-9_]", "", section.replace(" ", "_").lower())
        obj_name = f"settingsCard_{clean_section}"
        card.setObjectName(obj_name)
        card.setStyleSheet(
            f"#{obj_name} {{ background-color: {UI_BG_CARD}; "
            "border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; } "
            "QLabel { background-color: transparent; border: none; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        sec_lbl = StrongBodyLabel(section)
        sec_lbl.setStyleSheet(f"color: {UI_ACCENT}; font-size: 13px;")
        layout.addWidget(sec_lbl)

        for label, hint, widget in rows:
            row_layout = QHBoxLayout()
            col = QVBoxLayout()
            lbl = StrongBodyLabel(label)
            lbl.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 12px;")
            hint_lbl = CaptionLabel(hint)
            hint_lbl.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 10px;")
            col.addWidget(lbl)
            col.addWidget(hint_lbl)
            row_layout.addLayout(col, 1)
            row_layout.addWidget(widget)
            layout.addLayout(row_layout)

        return card

    def _make_double_spin(
        self,
        value: float,
        mn: float,
        mx: float,
        attr: str,
        step: float = 1.0,
        decimals: int = 2,
    ) -> DoubleSpinBox:
        spin = DoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(mn, mx)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFixedWidth(180)
        spin.setFixedHeight(34)
        setattr(self, attr, spin)
        return spin

    def _make_line_edit(self, value: str, attr: str) -> LineEdit:
        edit = LineEdit()
        edit.setText(value)
        edit.setFixedWidth(320)
        edit.setFixedHeight(34)
        setattr(self, attr, edit)
        return edit

    def _on_save_clicked(self) -> None:
        fps_val = self.target_fps_spin.value()
        thresh_val = self.touch_threshold_spin.value()
        vel_val = self.fingertip_vel_spin.value()
        hand_mov_val = self.hand_movement_spin.value()
        mp_det_val = self.mp_detection_conf_spin.value()
        mp_pres_val = self.mp_presence_conf_spin.value()
        mp_track_val = self.mp_tracking_conf_spin.value()
        q_avg_val = self.quality_min_avg_spin.value()
        q_frame_val = self.quality_min_frame_spin.value()
        q_drop_val = self.quality_max_drop_spin.value()
        plugins_val = self.plugins_dir_edit.text()

        self._vm.save_settings(
            fps=fps_val,
            touch_threshold=thresh_val,
            velocity_threshold=vel_val,
            hand_movement_threshold=hand_mov_val,
            detection_confidence=mp_det_val,
            presence_confidence=mp_pres_val,
            tracking_confidence=mp_track_val,
            quality_min_avg=q_avg_val,
            quality_min_frame=q_frame_val,
            quality_max_drop=q_drop_val,
            plugins_dir=plugins_val,
        )

    def _on_settings_saved(self, message: str) -> None:
        InfoBar.success(
            title="Saved",
            content=message,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=3000,
        )
        self.settings_applied.emit()

    def _on_settings_error(self, error_message: str) -> None:
        InfoBar.error(
            title="Error",
            content=error_message,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=4000,
        )

    def _on_restore_defaults_clicked(self) -> None:
        """Restores all input fields and saved configuration to default parameters."""
        self._vm.restore_defaults(persist=True)
