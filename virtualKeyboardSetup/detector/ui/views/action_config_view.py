"""
ui/views/action_config_view.py

Scrollable grid of KeyActionCard widgets — one per key in the loaded layout.
Bound to ActionConfigViewModel.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    SingleDirectionScrollArea,
    SubtitleLabel,
    CaptionLabel,
)

from config.constants import UI_ACCENT, UI_BG_DARK, UI_TEXT_SEC
from ui.components.key_action_card import KeyActionCard
from viewmodels.action_config_viewmodel import ActionConfigViewModel


class ActionConfigView(QWidget):
    """Key → Action assignment view."""

    back_requested  = Signal()
    start_requested = Signal()   # proceed to camera selection

    COLS = 3

    def __init__(self, vm: ActionConfigViewModel, parent=None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._cards: dict[str, KeyActionCard] = {}
        self._setup_ui()
        self._connect_vm()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(16)

        # Header
        title = SubtitleLabel("Configure Key Actions")
        title.setStyleSheet(f"color: {UI_ACCENT}; font-size: 20px; font-weight: bold;")
        hint = CaptionLabel(
            "Set an action for each key. Leave type as 'none' to skip."
        )
        hint.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 12px;")
        root.addWidget(title)
        root.addWidget(hint)

        # Scrollable card grid
        scroll = SingleDirectionScrollArea(orient=Qt.Vertical)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        grid_container = QWidget()
        grid_container.setStyleSheet("background-color: transparent;")
        self._grid = QGridLayout(grid_container)
        self._grid.setContentsMargins(0, 4, 0, 4)
        self._grid.setSpacing(12)

        # Populate cards
        for idx, btn in enumerate(self._vm.buttons):
            action = self._vm.get_action(btn.id)
            card = KeyActionCard(btn.id, btn.label, action)
            card.changed.connect(self._on_card_changed)
            self._cards[btn.id] = card
            row, col = divmod(idx, self.COLS)
            self._grid.addWidget(card, row, col)

        scroll.setWidget(grid_container)
        root.addWidget(scroll, 1)

        # Bottom buttons
        btn_row_layout = __import__("PySide6.QtWidgets", fromlist=["QHBoxLayout"]).QHBoxLayout()
        self.btn_back = PushButton(FluentIcon.RETURN, "Back")
        self.btn_back.setFixedHeight(38)
        self.btn_back.clicked.connect(self.back_requested.emit)

        self.btn_save = PushButton(FluentIcon.SAVE, "Save")
        self.btn_save.setFixedHeight(38)
        self.btn_save.clicked.connect(self._on_save)

        self.btn_continue = PrimaryPushButton(FluentIcon.PLAY, "Save & Start Detector")
        self.btn_continue.setFixedHeight(38)
        self.btn_continue.clicked.connect(self._on_save_and_continue)

        btn_row_layout.addWidget(self.btn_back)
        btn_row_layout.addStretch(1)
        btn_row_layout.addWidget(self.btn_save)
        btn_row_layout.addWidget(self.btn_continue)
        root.addLayout(btn_row_layout)

    def _connect_vm(self) -> None:
        self._vm.config_saved.connect(self._on_saved)
        self._vm.error_occurred.connect(self._on_error)

    def _on_card_changed(self, button_id: str, action_type: str, value: str) -> None:
        self._vm.set_action(button_id, action_type, value)

    def _on_save(self) -> None:
        self._vm.save()

    def _on_save_and_continue(self) -> None:
        if self._vm.save():
            self.start_requested.emit()

    def _on_saved(self, path: str) -> None:
        InfoBar.success(
            title="Saved",
            content=f"Actions saved to {path}",
            position=InfoBarPosition.TOP,
            parent=self,
            duration=2500,
        )

    def _on_error(self, msg: str) -> None:
        InfoBar.error(
            title="Save Error",
            content=msg,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=4000,
        )
