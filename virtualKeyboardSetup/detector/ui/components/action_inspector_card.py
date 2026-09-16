"""
ui/components/action_inspector_card.py

Sidebar inspector card for editing the selected button's action bindings in real-time.
Features physical keypress / shortcut recording via KeyCaptureButton.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    FluentIcon,
    LineEdit,
    PushButton,
    StrongBodyLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC
from core.action.action_executor import ActionData
from core.layout.layout_parser import ButtonData
from ui.components.key_capture_edit import KeyCaptureButton

_ACTION_TYPES = ["none", "keystroke", "shortcut", "shell", "macro"]
_PLACEHOLDERS = {
    "none":      "No digital action attached",
    "keystroke": "Press any key on keyboard...",
    "shortcut":  "Press key combo (e.g. Ctrl+C)...",
    "shell":     "e.g.  xdotool key XF86AudioPlay",
    "macro":     "e.g.  ctrl+c:200:ctrl+v  (ms delays)",
}

_TYPE_DESCRIPTIONS = {
    "none":      "Button will trigger no keypress or command.",
    "keystroke": "Types a single alphanumeric key or special key (e.g. Return, Space, A).",
    "shortcut":  "Executes a shortcut combination (e.g. Ctrl+C, Alt+Tab).",
    "shell":     "Runs an arbitrary Linux system command or script.",
    "macro":     "Executes timed sequence of keys separated by :delay_ms:.",
}


class ActionInspectorCard(QWidget):
    """Inspector sidebar card showing details and interactive key recorder for the active button."""

    action_changed = Signal(str, str, str)  # (button_id, type, value)
    navigate_key   = Signal(str)            # 'prev' or 'next'

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("actionInspectorCard")
        self._button: ButtonData | None = None
        self._building = True

        self._setup_ui()
        self._building = False

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = CardWidget(self)
        card.setBorderRadius(12)
        card.setStyleSheet(
            f"background-color: {UI_BG_CARD}; border: 1px solid rgba(255, 255, 255, 0.06); "
            "QLabel { background: transparent; border: none; }"
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_title = StrongBodyLabel("Key Inspector", card)
        self.lbl_title.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-size: 15px; font-weight: bold; border: none;")
        
        self.lbl_status_badge = CaptionLabel("Unbound", card)
        self.lbl_status_badge.setStyleSheet(
            f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none;"
        )

        header_row.addWidget(self.lbl_title)
        header_row.addStretch(1)
        header_row.addWidget(self.lbl_status_badge)
        layout.addLayout(header_row)

        # Key details (direct transparent labels)
        self.lbl_id_info = BodyLabel("Select a key from the map", card)
        self.lbl_id_info.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-size: 13px; font-weight: 600; border: none;")
        
        self.lbl_geo_info = CaptionLabel("", card)
        self.lbl_geo_info.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none;")

        layout.addWidget(self.lbl_id_info)
        layout.addWidget(self.lbl_geo_info)

        # ── Action Type ────────────────────────────────────────────────────────
        lbl_type = StrongBodyLabel("Action Type", card)
        lbl_type.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; font-weight: 600; border: none;")
        
        self.combo_type = ComboBox(card)
        self.combo_type.addItems(_ACTION_TYPES)
        self.combo_type.setFixedHeight(36)
        self.combo_type.currentTextChanged.connect(self._on_type_changed)

        layout.addWidget(lbl_type)
        layout.addWidget(self.combo_type)

        # ── Action Value / Input Container ─────────────────────────────────────
        self.lbl_value = StrongBodyLabel("Assigned Action / Key", card)
        self.lbl_value.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; font-weight: 600; border: none;")
        layout.addWidget(self.lbl_value)

        # Stacked inputs: index 0 = KeyCaptureButton (for keystroke/shortcut), index 1 = LineEdit (for shell/macro)
        self._input_stack_widget = QWidget(card)
        self._input_stack_widget.setStyleSheet("background: transparent;")
        self._input_stack = QStackedLayout(self._input_stack_widget)
        self._input_stack.setContentsMargins(0, 0, 0, 0)

        # 1. Key Capture Button
        self.key_capture_btn = KeyCaptureButton(self._input_stack_widget)
        self.key_capture_btn.key_captured.connect(self._on_key_captured)
        self._input_stack.addWidget(self.key_capture_btn)

        # 2. Text LineEdit (for shell / macro)
        self.edit_value = LineEdit(self._input_stack_widget)
        self.edit_value.setFixedHeight(36)
        self.edit_value.textChanged.connect(self._on_text_changed)
        self._input_stack.addWidget(self.edit_value)

        layout.addWidget(self._input_stack_widget)

        self.lbl_type_hint = CaptionLabel(_TYPE_DESCRIPTIONS["none"], card)
        self.lbl_type_hint.setStyleSheet(
            f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none; padding: 0px;"
        )
        self.lbl_type_hint.setWordWrap(True)
        layout.addWidget(self.lbl_type_hint)

        layout.addStretch(1)

        # ── Navigation & Clear ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_prev = PushButton(FluentIcon.LEFT_ARROW, "Prev", card)
        self.btn_prev.setFixedHeight(34)
        self.btn_prev.clicked.connect(lambda: self.navigate_key.emit("prev"))

        self.btn_clear = PushButton(FluentIcon.DELETE, "Clear", card)
        self.btn_clear.setFixedHeight(34)
        self.btn_clear.clicked.connect(self._on_clear_clicked)

        self.btn_next = PushButton(FluentIcon.RIGHT_ARROW, "Next", card)
        self.btn_next.setFixedHeight(34)
        self.btn_next.clicked.connect(lambda: self.navigate_key.emit("next"))

        btn_row.addWidget(self.btn_prev)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_next)
        layout.addLayout(btn_row)

        outer.addWidget(card)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_button(self, button: ButtonData | None, action: ActionData | None) -> None:
        self._button = button
        if button is None:
            self.clear_selection()
            return

        self._building = True

        self.lbl_title.setText(f'Key: "{button.label}"')
        self.lbl_id_info.setText(f"{button.label}  ({button.id})")
        self.lbl_geo_info.setText(
            f"Size: {button.width_mm:.1f} × {button.height_mm:.1f} mm  •  Pos: ({button.x_mm:.1f}, {button.y_mm:.1f})"
        )

        act = action or ActionData(type="none", value="")
        idx = _ACTION_TYPES.index(act.type) if act.type in _ACTION_TYPES else 0
        self.combo_type.setCurrentIndex(idx)
        self._update_input_mode(act.type, act.value)

        self.combo_type.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

        self._building = False

    def clear_selection(self) -> None:
        self._building = True
        self._button = None
        self.lbl_title.setText("Key Inspector")
        self.lbl_id_info.setText("Select a key from the map")
        self.lbl_geo_info.setText("")
        self.lbl_status_badge.setText("Unbound")
        self.lbl_status_badge.setStyleSheet(
            f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none;"
        )
        self.combo_type.setCurrentIndex(0)
        self.key_capture_btn.set_value("")
        self.edit_value.setText("")
        self.combo_type.setEnabled(False)
        self.key_capture_btn.setEnabled(False)
        self.edit_value.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self._building = False

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_type_changed(self, type_str: str) -> None:
        current_val = self.key_capture_btn.value if self._input_stack.currentIndex() == 0 else self.edit_value.text()
        if type_str == "none":
            current_val = ""
        self._update_input_mode(type_str, current_val)
        if not self._building and self._button:
            self.action_changed.emit(self._button.id, type_str, current_val)

    def _on_key_captured(self, captured_key: str) -> None:
        if not self._building and self._button:
            act_type = self.combo_type.currentText()
            self._update_badge(act_type, captured_key)
            self.action_changed.emit(self._button.id, act_type, captured_key)

    def _on_text_changed(self, text: str) -> None:
        if not self._building and self._button:
            act_type = self.combo_type.currentText()
            self._update_badge(act_type, text)
            self.action_changed.emit(self._button.id, act_type, text)

    def _on_clear_clicked(self) -> None:
        if self._button:
            self.combo_type.setCurrentIndex(0)
            self.key_capture_btn.set_value("")
            self.edit_value.setText("")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _update_input_mode(self, type_str: str, value: str) -> None:
        self.lbl_type_hint.setText(_TYPE_DESCRIPTIONS.get(type_str, ""))

        if type_str in ("keystroke", "shortcut"):
            self._input_stack.setCurrentIndex(0)
            self.key_capture_btn.setEnabled(True)
            self.key_capture_btn.set_mode(type_str)
            self.key_capture_btn.set_value(value)
        elif type_str in ("shell", "macro"):
            self._input_stack.setCurrentIndex(1)
            self.edit_value.setEnabled(True)
            self.edit_value.setPlaceholderText(_PLACEHOLDERS.get(type_str, ""))
            self.edit_value.setText(value)
        else:  # none
            self._input_stack.setCurrentIndex(0)
            self.key_capture_btn.setEnabled(False)
            self.key_capture_btn.set_mode("keystroke")
            self.key_capture_btn.set_value("")
            self.edit_value.setText("")

        self._update_badge(type_str, value)

    def _update_badge(self, type_str: str, value_str: str) -> None:
        if type_str != "none" and value_str:
            self.lbl_status_badge.setText(f"{type_str.capitalize()}: {value_str}")
            self.lbl_status_badge.setStyleSheet(
                "background: transparent; color: #00DC64; font-size: 11px; font-weight: 600; border: none;"
            )
        else:
            self.lbl_status_badge.setText("Unbound")
            self.lbl_status_badge.setStyleSheet(
                f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none;"
            )
