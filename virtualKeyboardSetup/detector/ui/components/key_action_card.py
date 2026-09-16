"""
ui/components/key_action_card.py

Widget card for a single key in the action configuration view.
Displays key label, action type combo, and interactive KeyCaptureButton / text input.
Emits changed(button_id, type, value) whenever the user edits.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QStackedLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ComboBox,
    LineEdit,
    StrongBodyLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC
from core.action.action_executor import ActionData
from ui.components.key_capture_edit import KeyCaptureButton

_ACTION_TYPES = ["none", "keystroke", "shortcut", "shell", "macro"]
_PLACEHOLDERS = {
    "none":      "—",
    "keystroke": "Click to press key...",
    "shortcut":  "Click to press shortcut...",
    "shell":     "e.g.  xdotool key XF86AudioPlay",
    "macro":     "e.g.  ctrl+c:200:ctrl+v",
}


class KeyActionCard(QWidget):
    """Single key action assignment card with physical key recorder."""

    changed = Signal(str, str, str)   # (button_id, type, value)

    def __init__(self, button_id: str, label: str, action: ActionData, parent=None) -> None:
        super().__init__(parent)
        self._button_id = button_id
        self._building  = True

        card = CardWidget(self)
        card.setObjectName("keyCard")
        card.setBorderRadius(10)
        card.setStyleSheet(
            f"#keyCard {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); }} "
            "QLabel { background-color: transparent; border: none; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(16, 12, 16, 14)
        inner.setSpacing(10)

        # Key label
        key_lbl = StrongBodyLabel(label)
        key_lbl.setStyleSheet(
            f"background: transparent; color: {UI_ACCENT}; font-size: 13px; font-weight: bold;"
        )
        id_lbl = BodyLabel(f"id: {button_id}")
        id_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 10px;")

        header = QHBoxLayout()
        header.addWidget(key_lbl)
        header.addStretch(1)
        header.addWidget(id_lbl)
        inner.addLayout(header)

        # Action type + value row
        row = QHBoxLayout()
        row.setSpacing(10)

        self._combo = ComboBox()
        self._combo.addItems(_ACTION_TYPES)
        self._combo.setFixedWidth(115)
        self._combo.setFixedHeight(34)
        idx = _ACTION_TYPES.index(action.type) if action.type in _ACTION_TYPES else 0
        self._combo.setCurrentIndex(idx)
        self._combo.currentTextChanged.connect(self._on_type_changed)

        # Input container
        self._input_container = QWidget()
        self._input_container.setStyleSheet("background: transparent;")
        self._stack = QStackedLayout(self._input_container)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._capture_btn = KeyCaptureButton(self._input_container)
        self._capture_btn.setFixedHeight(34)
        self._capture_btn.key_captured.connect(self._on_key_captured)
        self._stack.addWidget(self._capture_btn)

        self._value_edit = LineEdit(self._input_container)
        self._value_edit.setFixedHeight(34)
        self._value_edit.textChanged.connect(self._on_value_changed)
        self._stack.addWidget(self._value_edit)

        self._update_input_mode(action.type, action.value)

        row.addWidget(self._combo)
        row.addWidget(self._input_container, 1)
        inner.addLayout(row)

        self._building = False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_input_mode(self, type_str: str, value: str) -> None:
        if type_str in ("keystroke", "shortcut"):
            self._stack.setCurrentIndex(0)
            self._capture_btn.setEnabled(True)
            self._capture_btn.set_mode(type_str)
            self._capture_btn.set_value(value)
        elif type_str in ("shell", "macro"):
            self._stack.setCurrentIndex(1)
            self._value_edit.setEnabled(True)
            self._value_edit.setPlaceholderText(_PLACEHOLDERS.get(type_str, ""))
            self._value_edit.setText(value)
        else:
            self._stack.setCurrentIndex(0)
            self._capture_btn.setEnabled(False)
            self._capture_btn.set_mode("keystroke")
            self._capture_btn.set_value("")
            self._value_edit.setText("")

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_type_changed(self, type_str: str) -> None:
        current_val = self._capture_btn.value if self._stack.currentIndex() == 0 else self._value_edit.text()
        if type_str == "none":
            current_val = ""
        self._update_input_mode(type_str, current_val)
        if not self._building:
            self.changed.emit(self._button_id, type_str, current_val)

    def _on_key_captured(self, key_str: str) -> None:
        if not self._building:
            self.changed.emit(self._button_id, self._combo.currentText(), key_str)

    def _on_value_changed(self, text: str) -> None:
        if not self._building:
            self.changed.emit(self._button_id, self._combo.currentText(), text)

    # ── Public ─────────────────────────────────────────────────────────────────

    def get_action(self) -> ActionData:
        val = self._capture_btn.value if self._stack.currentIndex() == 0 else self._value_edit.text()
        return ActionData(type=self._combo.currentText(), value=val)
