"""
ui/views/settings_view.py

Settings view following MVVM architecture and SOLID principles.
Provides ergonomic input controls for runtime thresholds and configuration.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
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
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(20)

        title = SubtitleLabel("Settings")
        title.setStyleSheet(f"color: {UI_ACCENT}; font-size: 20px; font-weight: bold;")
        root.addWidget(title)

        root.addWidget(self._make_card(
            "Pipeline",
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
                    "Minimum fingertip speed to evaluate window (default 0.0080)",
                    self._make_double_spin(
                        value=self._vm.fingertip_velocity_threshold,
                        mn=0.0001,
                        mx=2.0000,
                        attr="fingertip_vel_spin",
                        step=0.001,
                        decimals=4,
                    ),
                ),
            ]
        ))

        root.addWidget(self._make_card(
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

        root.addStretch(1)

        btn_row = QHBoxLayout()
        btn_save = PrimaryPushButton(FluentIcon.SAVE, "Apply & Save")
        btn_save.setFixedHeight(38)
        btn_save.setMinimumWidth(140)
        btn_save.clicked.connect(self._on_save_clicked)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _connect_signals(self) -> None:
        self._vm.settings_saved.connect(self._on_settings_saved)
        self._vm.settings_loaded.connect(self._refresh_fields)
        self._vm.error_occurred.connect(self._on_settings_error)

    def _refresh_fields(self) -> None:
        self.target_fps_spin.setValue(self._vm.target_fps)
        self.touch_threshold_spin.setValue(self._vm.touch_threshold)
        self.fingertip_vel_spin.setValue(self._vm.fingertip_velocity_threshold)
        self.plugins_dir_edit.setText(self._vm.plugins_dir)

    def _make_card(self, section: str, rows: list[tuple]) -> CardWidget:
        card = CardWidget()
        obj_name = f"settingsCard_{section.replace(' ', '_').lower()}"
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
        plugins_val = self.plugins_dir_edit.text()

        self._vm.save_settings(
            fps=fps_val,
            touch_threshold=thresh_val,
            velocity_threshold=vel_val,
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
