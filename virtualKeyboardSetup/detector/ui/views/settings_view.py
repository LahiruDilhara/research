"""
ui/views/settings_view.py

Simple settings view — FPS override, touch threshold, model plugin directory.
Changes write to the .env file and take effect on next pipeline start.
"""

from __future__ import annotations

from pathlib import Path

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
    PushButton,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
)

from config.constants import (
    UI_ACCENT, UI_BG_CARD, UI_BG_DARK, UI_TEXT_PRI, UI_TEXT_SEC,
    TARGET_FPS, TOUCH_PROBABILITY_THRESHOLD,
)


class SettingsView(QWidget):
    """App settings / overrides view."""

    settings_applied = Signal()

    def __init__(self, env_path: str, parent=None) -> None:
        super().__init__(parent)
        self._env_path = Path(env_path)
        self._setup_ui()

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
                ("Target FPS", "Capture and processing rate (default 12)",
                 self._make_double_spin(TARGET_FPS, 5.0, 30.0, "target_fps_spin")),
                ("Touch Threshold", "Minimum model probability to trigger a touch (0–1)",
                 self._make_double_spin(TOUCH_PROBABILITY_THRESHOLD, 0.1, 1.0, "touch_threshold_spin", step=0.05)),
            ]
        ))

        root.addWidget(self._make_card(
            "Plugin Directory",
            [
                ("Plugins Dir", "Relative path to the model plugins directory",
                 self._make_line_edit("plugins", "plugins_dir_edit")),
            ]
        ))

        root.addStretch(1)

        btn_row = QHBoxLayout()
        btn_save = PrimaryPushButton(FluentIcon.SAVE, "Apply & Save")
        btn_save.setFixedHeight(38)
        btn_save.clicked.connect(self._on_save)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _make_card(self, section: str, rows: list[tuple]) -> CardWidget:
        card = CardWidget()
        card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; "
            "border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; }}"
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

    def _make_double_spin(self, value, mn, mx, attr, step=1.0) -> DoubleSpinBox:
        spin = DoubleSpinBox()
        spin.setRange(mn, mx)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFixedWidth(120)
        setattr(self, attr, spin)
        return spin

    def _make_line_edit(self, value, attr) -> LineEdit:
        edit = LineEdit()
        edit.setText(value)
        edit.setFixedWidth(200)
        setattr(self, attr, edit)
        return edit

    def _on_save(self) -> None:
        fps_val = self.target_fps_spin.value()
        thresh_val = self.touch_threshold_spin.value()
        plugins_val = self.plugins_dir_edit.text().strip() or "plugins"

        env_content = (
            f"TARGET_FPS={fps_val}\n"
            f"TOUCH_THRESHOLD={thresh_val}\n"
            f"PLUGINS_DIR={plugins_val}\n"
        )
        try:
            existing: dict[str, str] = {}
            if self._env_path.exists():
                for line in self._env_path.read_text().splitlines():
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        existing[k.strip()] = v.strip()
            existing["TARGET_FPS"] = str(fps_val)
            existing["TOUCH_THRESHOLD"] = str(thresh_val)
            existing["PLUGINS_DIR"] = plugins_val

            lines = [f"{k}={v}" for k, v in existing.items()]
            self._env_path.write_text("\n".join(lines) + "\n")

            InfoBar.success(
                title="Saved",
                content=f"Settings written to {self._env_path.name}. "
                        "Restart the pipeline to apply.",
                position=InfoBarPosition.TOP,
                parent=self,
                duration=3000,
            )
        except Exception as exc:
            InfoBar.error(
                title="Save Failed",
                content=str(exc),
                position=InfoBarPosition.TOP,
                parent=self,
                duration=4000,
            )
