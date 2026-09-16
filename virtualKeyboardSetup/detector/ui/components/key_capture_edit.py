"""
ui/components/key_capture_edit.py

Interactive physical key capture widget.
Allows users to press any key or shortcut combination on their physical keyboard
to record the action automatically without manual text typing.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)
from qfluentwidgets import FluentIcon, PushButton

from config.constants import UI_ACCENT, UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC

_SPECIAL_KEY_MAP = {
    Qt.Key_Return: "enter",
    Qt.Key_Enter: "enter",
    Qt.Key_Tab: "tab",
    Qt.Key_Backtab: "tab",
    Qt.Key_Space: "space",
    Qt.Key_Backspace: "backspace",
    Qt.Key_Delete: "delete",
    Qt.Key_Insert: "insert",
    Qt.Key_Escape: "escape",
    Qt.Key_Home: "home",
    Qt.Key_End: "end",
    Qt.Key_PageUp: "pageup",
    Qt.Key_PageDown: "pagedown",
    Qt.Key_Left: "left",
    Qt.Key_Right: "right",
    Qt.Key_Up: "up",
    Qt.Key_Down: "down",
    Qt.Key_F1: "f1",
    Qt.Key_F2: "f2",
    Qt.Key_F3: "f3",
    Qt.Key_F4: "f4",
    Qt.Key_F5: "f5",
    Qt.Key_F6: "f6",
    Qt.Key_F7: "f7",
    Qt.Key_F8: "f8",
    Qt.Key_F9: "f9",
    Qt.Key_F10: "f10",
    Qt.Key_F11: "f11",
    Qt.Key_F12: "f12",
    Qt.Key_CapsLock: "caps_lock",
    Qt.Key_NumLock: "num_lock",
    Qt.Key_ScrollLock: "scroll_lock",
    Qt.Key_Print: "print_screen",
    Qt.Key_Pause: "pause",
}

_MODIFIER_KEYS = {
    Qt.Key_Control,
    Qt.Key_Shift,
    Qt.Key_Alt,
    Qt.Key_Meta,
    Qt.Key_AltGr,
    Qt.Key_Super_L,
    Qt.Key_Super_R,
}


class KeyCaptureButton(QPushButton):
    """Button that listens for physical key presses and captures the key / shortcut."""

    key_captured = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recording = False
        self._value = ""
        self._mode = "shortcut"  # 'keystroke' or 'shortcut'
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(36)
        self.clicked.connect(self._toggle_recording)
        self._update_style()

    def set_value(self, value: str) -> None:
        self._value = value
        self._recording = False
        self._update_style()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._update_style()

    @property
    def value(self) -> str:
        return self._value

    def _toggle_recording(self) -> None:
        self._recording = not self._recording
        self._update_style()
        if self._recording:
            self.setFocus()

    def focusOutEvent(self, event) -> None:
        if self._recording:
            self._recording = False
            self._update_style()
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._recording:
            super().keyPressEvent(event)
            return

        key = event.key()

        # Cancel on Escape if alone and recording
        if key == Qt.Key_Escape and event.modifiers() == Qt.NoModifier:
            self._recording = False
            self._update_style()
            event.accept()
            return

        # Ignore bare modifier key down
        if key in _MODIFIER_KEYS:
            event.accept()
            return

        parts: list[str] = []
        mods = event.modifiers()

        if self._mode == "shortcut":
            if mods & Qt.ControlModifier:
                parts.append("ctrl")
            if mods & Qt.AltModifier:
                parts.append("alt")
            if mods & Qt.ShiftModifier:
                parts.append("shift")
            if mods & Qt.MetaModifier:
                parts.append("meta")

        # Map key
        key_str = ""
        if key in _SPECIAL_KEY_MAP:
            key_str = _SPECIAL_KEY_MAP[key]
        else:
            text = event.text().strip().lower()
            if text and len(text) == 1 and text.isprintable():
                key_str = text
            else:
                key_name = QKeySequence(key).toString().lower()
                key_str = key_name if key_name else f"key_{key}"

        if key_str:
            parts.append(key_str)
            captured = "+".join(parts) if self._mode == "shortcut" else key_str
            self._value = captured
            self._recording = False
            self._update_style()
            self.key_captured.emit(captured)
            event.accept()
        else:
            super().keyPressEvent(event)

    def _update_style(self) -> None:
        if self._recording:
            self.setText("Listening... Press key or combo")
            self.setStyleSheet(
                "background-color: #1A2433; color: #38BDF8; "
                f"border: 1px solid {UI_ACCENT}; border-radius: 6px; font-weight: 600; font-size: 11px; padding: 4px;"
            )
        elif self._value:
            display_val = self._value.upper() if self._mode == "shortcut" else self._value
            self.setText(f"{display_val} (Click to rebind)")
            self.setStyleSheet(
                "background-color: #1E1E28; color: #F1F5F9; "
                "border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 6px; font-weight: 600; font-size: 12px; padding: 4px;"
            )
        else:
            self.setText("Click to record key...")
            self.setStyleSheet(
                f"background-color: rgba(255, 255, 255, 0.03); color: {UI_TEXT_SEC}; "
                "border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 6px; font-size: 11px; padding: 4px;"
            )
