"""
ui/components/key_action_card.py

Widget card for a single key in the action configuration view.
Displays key label, action type combo, and action value input.
Emits changed(button_id, type, value) whenever the user edits.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ComboBox,
    LineEdit,
    StrongBodyLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC
from core.action.action_executor import ActionData

_ACTION_TYPES = ["none", "keystroke", "shortcut", "shell", "macro"]
_PLACEHOLDERS = {
    "none":      "—",
    "keystroke": "e.g.  a   enter   f5   space",
    "shortcut":  "e.g.  ctrl+c   ctrl+shift+t",
    "shell":     "e.g.  xdotool key XF86AudioPlay",
    "macro":     "e.g.  ctrl+c:200:ctrl+v  (ms delays)",
}

_CARD_STYLE = (
    f"CardWidget {{ "
    f"  background-color: {UI_BG_CARD}; "
    "  border: 1px solid rgba(255, 255, 255, 0.06); "
    "  border-radius: 10px; "
    "} "
    "QLabel { "
    "  background-color: transparent; "
    "  border: none; "
    "}"
)


class KeyActionCard(QWidget):
    """Single key action assignment card."""

    changed = Signal(str, str, str)   # (button_id, type, value)

    def __init__(self, button_id: str, label: str, action: ActionData, parent=None) -> None:
        super().__init__(parent)
        self._button_id = button_id
        self._building  = True

        card = CardWidget(self)
        card.setObjectName("keyCard")
        card.setStyleSheet(
            f"#keyCard {{ background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; }} "
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
        self._combo.setFixedWidth(120)
        self._combo.setFixedHeight(34)
        idx = _ACTION_TYPES.index(action.type) if action.type in _ACTION_TYPES else 0
        self._combo.setCurrentIndex(idx)
        self._combo.currentTextChanged.connect(self._on_type_changed)

        self._value_edit = LineEdit()
        self._value_edit.setText(action.value)
        self._value_edit.setFixedHeight(34)
        self._value_edit.setPlaceholderText(_PLACEHOLDERS.get(action.type, ""))
        self._value_edit.setEnabled(action.type != "none")
        self._value_edit.textChanged.connect(self._on_value_changed)

        row.addWidget(self._combo)
        row.addWidget(self._value_edit, 1)
        inner.addLayout(row)

        self._building = False

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_type_changed(self, type_str: str) -> None:
        self._value_edit.setEnabled(type_str != "none")
        self._value_edit.setPlaceholderText(_PLACEHOLDERS.get(type_str, ""))
        if not self._building:
            self.changed.emit(self._button_id, type_str, self._value_edit.text())

    def _on_value_changed(self, text: str) -> None:
        if not self._building:
            self.changed.emit(self._button_id, self._combo.currentText(), text)

    # ── Public ─────────────────────────────────────────────────────────────────

    def get_action(self) -> ActionData:
        return ActionData(type=self._combo.currentText(), value=self._value_edit.text())
