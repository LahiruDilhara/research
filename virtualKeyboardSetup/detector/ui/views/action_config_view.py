"""
ui/views/action_config_view.py

Key Action Configuration View.
Provides dual-mode action configuration:
  1. Interactive Visual Layout Map (Default) with Key Action Inspector sidebar.
  2. All Keys Grid View with individual key cards.
Both views stay synchronized in real time with the ActionConfigViewModel and save directly to the XML.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    SingleDirectionScrollArea,
    SubtitleLabel,
)

from config.constants import UI_ACCENT, UI_BG_DARK, UI_TEXT_PRI, UI_TEXT_SEC
from core.action.action_executor import ActionData
from ui.components.action_inspector_card import ActionInspectorCard
from ui.components.interactive_layout_map import InteractiveLayoutMapWidget
from ui.components.key_action_card import KeyActionCard
from viewmodels.action_config_viewmodel import ActionConfigViewModel


class ActionConfigView(QWidget):
    """Dual-mode key action configuration view."""

    back_requested  = Signal()
    start_requested = Signal()   # proceed to camera selection

    COLS = 3

    def __init__(self, vm: ActionConfigViewModel, parent=None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._cards: dict[str, KeyActionCard] = {}
        self._setup_ui()
        self._connect_vm()

        # Initialize with first button selected if available
        if self._vm.buttons:
            self._select_button(self._vm.buttons[0].id)

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # ── Header Toolbar ────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        title = SubtitleLabel("Configure Key Actions", self)
        title.setStyleSheet(f"background: transparent; color: {UI_ACCENT}; font-size: 20px; font-weight: bold;")
        hint = CaptionLabel(
            "Click any key on the visual layout map to assign actions, or switch to the grid list.",
            self,
        )
        hint.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 12px;")
        title_col.addWidget(title)
        title_col.addWidget(hint)
        header_row.addLayout(title_col, 1)

        # Mode Switcher (Segmented Control)
        self.mode_switcher = SegmentedWidget(self)
        self.mode_switcher.addItem("visualMode", "Visual Layout Map")
        self.mode_switcher.addItem("gridMode", "All Keys Grid")
        self.mode_switcher.setCurrentItem("visualMode")
        self.mode_switcher.currentItemChanged.connect(self._on_mode_changed)
        header_row.addWidget(self.mode_switcher, 0, Qt.AlignVCenter)

        root.addLayout(header_row)

        # ── Stacked Mode Container ─────────────────────────────────────────────
        self.stack = QStackedWidget(self)
        self.stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")

        # ── 1. Visual Layout Mode Page ─────────────────────────────────────────
        visual_page = QWidget(self.stack)
        visual_page.setStyleSheet("background: transparent;")
        v_layout = QHBoxLayout(visual_page)
        v_layout.setContentsMargins(0, 4, 0, 4)
        v_layout.setSpacing(16)

        self.layout_map = InteractiveLayoutMapWidget(visual_page)
        self.layout_map.set_layout(self._vm.layout)
        self.layout_map.set_actions(self._vm.config)
        self.layout_map.button_clicked.connect(self._on_map_button_clicked)
        v_layout.addWidget(self.layout_map, 1)

        self.inspector = ActionInspectorCard(visual_page)
        self.inspector.setFixedWidth(320)
        self.inspector.action_changed.connect(self._on_inspector_action_changed)
        self.inspector.navigate_key.connect(self._on_navigate_key)
        v_layout.addWidget(self.inspector)

        self.stack.addWidget(visual_page)

        # ── 2. All Keys Grid Mode Page ─────────────────────────────────────────
        grid_page = QWidget(self.stack)
        grid_page.setStyleSheet("background: transparent;")
        g_layout = QVBoxLayout(grid_page)
        g_layout.setContentsMargins(0, 4, 0, 4)

        scroll = SingleDirectionScrollArea(orient=Qt.Vertical)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        grid_container = QWidget()
        grid_container.setStyleSheet("background-color: transparent;")
        self._grid = QGridLayout(grid_container)
        self._grid.setContentsMargins(4, 8, 12, 16)
        self._grid.setSpacing(14)
        self._grid.setAlignment(Qt.AlignTop)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(2, 1)

        for idx, btn in enumerate(self._vm.buttons):
            action = self._vm.get_action(btn.id)
            card = KeyActionCard(btn.id, btn.label, action)
            card.changed.connect(self._on_card_changed)
            self._cards[btn.id] = card
            row, col = divmod(idx, self.COLS)
            self._grid.addWidget(card, row, col)

        scroll.setWidget(grid_container)
        g_layout.addWidget(scroll)
        self.stack.addWidget(grid_page)

        root.addWidget(self.stack, 1)

        # ── Footer Toolbar ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(12)

        self.btn_back = PushButton(FluentIcon.RETURN, "Back", self)
        self.btn_back.setFixedHeight(38)
        self.btn_back.clicked.connect(self.back_requested.emit)

        self.btn_save = PushButton(FluentIcon.SAVE, "Save XML", self)
        self.btn_save.setFixedHeight(38)
        self.btn_save.clicked.connect(self._on_save)

        self.btn_continue = PrimaryPushButton(FluentIcon.PLAY, "Save & Start Detector", self)
        self.btn_continue.setFixedHeight(38)
        self.btn_continue.clicked.connect(self._on_save_and_continue)

        btn_row.addWidget(self.btn_back)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_continue)
        root.addLayout(btn_row)

    def _connect_vm(self) -> None:
        self._vm.config_saved.connect(self._on_saved)
        self._vm.error_occurred.connect(self._on_error)
        self._vm.duplicate_rejected.connect(self._on_duplicate_rejected)
        self._vm.action_changed.connect(self._on_vm_action_changed)
        self._vm.button_selected.connect(self._on_vm_button_selected)

    # ── Mode Switching ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, item_key: str) -> None:
        if item_key == "visualMode":
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)

    # ── Key Selection & Navigation ─────────────────────────────────────────────

    def _select_button(self, button_id: str) -> None:
        self._vm.select_button(button_id)

    def _on_vm_button_selected(self, button_id: str) -> None:
        self.layout_map.select_button(button_id)
        btn_obj = self._vm.get_button(button_id)
        act = self._vm.get_action(button_id)
        self.inspector.set_button(btn_obj, act)

    def _on_map_button_clicked(self, button_id: str) -> None:
        self._select_button(button_id)

    def _on_navigate_key(self, direction: str) -> None:
        buttons = self._vm.buttons
        if not buttons:
            return
        curr_id = self._vm.selected_button_id
        curr_idx = next((i for i, b in enumerate(buttons) if b.id == curr_id), 0)
        if direction == "prev":
            new_idx = (curr_idx - 1) % len(buttons)
        else:
            new_idx = (curr_idx + 1) % len(buttons)
        self._select_button(buttons[new_idx].id)

    # ── Action Synchronization ─────────────────────────────────────────────────

    def _on_inspector_action_changed(self, button_id: str, action_type: str, value: str) -> None:
        self._vm.set_action(button_id, action_type, value)

    def _on_card_changed(self, button_id: str, action_type: str, value: str) -> None:
        self._vm.set_action(button_id, action_type, value)

    def _on_vm_action_changed(self, button_id: str, action_type: str, value: str) -> None:
        act = ActionData(type=action_type, value=value)
        self.layout_map.set_action_for_button(button_id, act)
        if self._vm.selected_button_id == button_id:
            btn_obj = self._vm.get_button(button_id)
            self.inspector.set_button(btn_obj, act)
        if button_id in self._cards:
            card = self._cards[button_id]
            card._combo.blockSignals(True)
            idx = card._combo.findText(action_type)
            if idx >= 0:
                card._combo.setCurrentIndex(idx)
            card._update_input_mode(action_type, value)
            card._combo.blockSignals(False)

    def _on_duplicate_rejected(
        self,
        button_id: str,
        action_type: str,
        value: str,
        conflict_label: str,
        conflict_id: str,
    ) -> None:
        InfoBar.error(
            title="Duplicate Action Conflict",
            content=f"Cannot bind '{value}' ({action_type}). It is already assigned to '{conflict_label}' ({conflict_id}).",
            position=InfoBarPosition.TOP,
            parent=self,
            duration=5000,
        )

    # ── Save & Continue ────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        self._vm.save()

    def _on_save_and_continue(self) -> None:
        if self._vm.save():
            self.start_requested.emit()

    def _on_saved(self, path: str) -> None:
        InfoBar.success(
            title="Saved to XML",
            content=f"Key actions saved inside layout XML: {path}",
            position=InfoBarPosition.TOP,
            parent=self,
            duration=3000,
        )

    def _on_error(self, msg: str) -> None:
        InfoBar.error(
            title="Action Error",
            content=msg,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=5000,
        )
