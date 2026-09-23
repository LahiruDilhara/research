"""
ui/views/file_landing_view.py

Modern, minimal startup landing view designed for real-world non-technical users.
Features a clean card surface, intuitive file selection with drag and drop,
clear visual confirmation of the chosen keyboard layout, and zero sidebars on launch.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
)

from config.constants import (
    UI_ACCENT,
    UI_BG_CARD,
    UI_BG_DARK,
    UI_TEXT_PRI,
    UI_TEXT_SEC,
)
from core.layout.layout_parser import LayoutData
from viewmodels.startup_viewmodel import StartupViewModel


class ModernDropZone(QFrame):
    """Clean, friendly drop target with dynamic hover state."""

    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._is_hovered = False
        self._update_appearance()

    def _update_appearance(self) -> None:
        if self._is_hovered:
            border = UI_ACCENT
            bg = "rgba(0, 190, 255, 0.06)"
        else:
            border = "rgba(255, 255, 255, 0.12)"
            bg = "rgba(255, 255, 255, 0.02)"

        self.setStyleSheet(
            f"QFrame {{ "
            f"  background-color: {bg}; "
            f"  border: 2px dashed {border}; "
            f"  border-radius: 16px; "
            f"}} "
            f"QLabel {{ background: transparent; border: none; }}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".xml"):
                    event.acceptProposedAction()
                    self._is_hovered = True
                    self._update_appearance()
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._is_hovered = False
        self._update_appearance()

    def dropEvent(self, event: QDropEvent) -> None:
        self._is_hovered = False
        self._update_appearance()
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".xml"):
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()
                return


class FileLandingView(QWidget):
    """Full-window, user-friendly startup landing screen with zero sidebars."""

    workspace_entered = Signal(object, str)  # layout_data, xml_path

    def __init__(
        self,
        startup_vm: StartupViewModel,
        last_xml_path: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._vm = startup_vm
        self._last_xml_path = last_xml_path
        self._current_layout: LayoutData | None = None
        self._current_xml_path: str = ""

        self._setup_ui()
        self._connect_signals()

        # Auto-load previous layout if it exists
        if self._last_xml_path and Path(self._last_xml_path).is_file():
            self._load_xml_path(self._last_xml_path)

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setAlignment(Qt.AlignCenter)

        # Centered modern card
        card = CardWidget(self)
        card.setFixedWidth(660)
        card.setStyleSheet(
            f"CardWidget {{ "
            f"  background-color: #16161A; "
            f"  border: 1px solid rgba(255, 255, 255, 0.08); "
            f"  border-radius: 20px; "
            f"}} "
            f"QLabel {{ background: transparent; border: none; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 36, 40, 36)
        card_layout.setSpacing(22)

        # ── Header Section ────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(16)

        # App visual badge icon
        badge_frame = QFrame(card)
        badge_frame.setFixedSize(52, 52)
        badge_frame.setStyleSheet(
            "background-color: rgba(0, 190, 255, 0.12); "
            "border: 1px solid rgba(0, 190, 255, 0.3); "
            "border-radius: 14px;"
        )
        badge_layout = QVBoxLayout(badge_frame)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setAlignment(Qt.AlignCenter)
        badge_icon = StrongBodyLabel("⌨", badge_frame)
        badge_icon.setStyleSheet("color: #00BEFF; font-size: 24px; font-weight: bold;")
        badge_icon.setAlignment(Qt.AlignCenter)
        badge_layout.addWidget(badge_icon)
        header_row.addWidget(badge_frame)

        # Title & Subtitle
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = TitleLabel("Choose Your Keyboard Layout", card)
        title.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 22px; font-weight: 700;")
        subtitle = CaptionLabel(
            "Select your printed paper keyboard design to begin typing on any flat surface.",
            card,
        )
        subtitle.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 13px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col, 1)

        card_layout.addLayout(header_row)

        # Subtle divider
        divider = QFrame(card)
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: rgba(255, 255, 255, 0.06); min-height: 1px; max-height: 1px;")
        card_layout.addWidget(divider)

        # ── Interactive Selection Area (Stacked: Empty vs Selected) ───────────
        self.selection_stack = QStackedWidget(card)
        self.selection_stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")

        # Page 0: Empty Drop Zone
        page_empty = QWidget(self.selection_stack)
        pe_layout = QVBoxLayout(page_empty)
        pe_layout.setContentsMargins(0, 0, 0, 0)

        self.drop_zone = ModernDropZone(page_empty)
        self.drop_zone.setFixedHeight(140)
        dz_layout = QVBoxLayout(self.drop_zone)
        dz_layout.setContentsMargins(24, 20, 24, 20)
        dz_layout.setSpacing(12)
        dz_layout.setAlignment(Qt.AlignCenter)

        dz_msg = StrongBodyLabel("Drag and drop your layout XML file here", self.drop_zone)
        dz_msg.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 14px; font-weight: 500;")
        dz_msg.setAlignment(Qt.AlignCenter)

        dz_actions = QHBoxLayout()
        dz_actions.setSpacing(10)
        dz_actions.setAlignment(Qt.AlignCenter)
        self.btn_browse = PushButton(FluentIcon.FOLDER, "Browse File", self.drop_zone)
        self.btn_browse.setFixedHeight(34)
        self.btn_browse.setFixedWidth(160)
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        dz_actions.addWidget(self.btn_browse)

        dz_sub = CaptionLabel("Compatible with layout designs exported from the designer (.xml)", self.drop_zone)
        dz_sub.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        dz_sub.setAlignment(Qt.AlignCenter)

        dz_layout.addWidget(dz_msg)
        dz_layout.addLayout(dz_actions)
        dz_layout.addWidget(dz_sub)

        self.drop_zone.file_dropped.connect(self._load_xml_path)
        pe_layout.addWidget(self.drop_zone)
        self.selection_stack.addWidget(page_empty)

        # Page 1: Selected Layout Confirmation Card
        page_selected = QWidget(self.selection_stack)
        ps_layout = QVBoxLayout(page_selected)
        ps_layout.setContentsMargins(0, 0, 0, 0)

        self.selected_card = CardWidget(page_selected)
        self.selected_card.setFixedHeight(140)
        self.selected_card.setStyleSheet(
            f"CardWidget {{ "
            f"  background-color: rgba(0, 220, 100, 0.04); "
            f"  border: 1px solid rgba(0, 220, 100, 0.25); "
            f"  border-radius: 16px; "
            f"}} "
            f"QLabel {{ background: transparent; border: none; }}"
        )
        sc_layout = QVBoxLayout(self.selected_card)
        sc_layout.setContentsMargins(24, 18, 24, 18)
        sc_layout.setSpacing(10)

        # Status badge row
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        dot = StrongBodyLabel("●", self.selected_card)
        dot.setStyleSheet("color: #00DC64; font-size: 12px;")
        status_lbl = StrongBodyLabel("LAYOUT READY", self.selected_card)
        status_lbl.setStyleSheet("color: #00DC64; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;")
        status_row.addWidget(dot)
        status_row.addWidget(status_lbl)
        status_row.addStretch(1)

        self.btn_change = PushButton(FluentIcon.SYNC, "Change File", self.selected_card)
        self.btn_change.setFixedHeight(30)
        self.btn_change.clicked.connect(self._on_browse_clicked)
        status_row.addWidget(self.btn_change)
        sc_layout.addLayout(status_row)

        # File name
        self.lbl_filename = StrongBodyLabel("layout.xml", self.selected_card)
        self.lbl_filename.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 16px; font-weight: 700;")
        sc_layout.addWidget(self.lbl_filename)

        # Stats pills row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self.lbl_keys_pill = CaptionLabel("0 Keys", self.selected_card)
        self.lbl_keys_pill.setStyleSheet(
            f"background-color: rgba(255, 255, 255, 0.06); color: {UI_TEXT_SEC}; "
            "padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500;"
        )
        self.lbl_tags_pill = CaptionLabel("0 AprilTags", self.selected_card)
        self.lbl_tags_pill.setStyleSheet(
            f"background-color: rgba(255, 255, 255, 0.06); color: {UI_TEXT_SEC}; "
            "padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500;"
        )
        stats_row.addWidget(self.lbl_keys_pill)
        stats_row.addWidget(self.lbl_tags_pill)
        stats_row.addStretch(1)
        sc_layout.addLayout(stats_row)

        ps_layout.addWidget(self.selected_card)
        self.selection_stack.addWidget(page_selected)

        card_layout.addWidget(self.selection_stack)

        # ── Recent Layouts Row ────────────────────────────────────────────────
        recent_row = QHBoxLayout()
        recent_row.setContentsMargins(4, 0, 4, 0)
        recent_row.setSpacing(10)

        self.lbl_recent_caption = CaptionLabel("Recent design:", card)
        self.lbl_recent_caption.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        self.lbl_recent_caption.setVisible(False)
        recent_row.addWidget(self.lbl_recent_caption)

        self.btn_recent = PushButton(FluentIcon.HISTORY, "", card)
        self.btn_recent.setFixedHeight(28)
        self.btn_recent.setVisible(False)
        self.btn_recent.clicked.connect(self._on_load_recent)
        recent_row.addWidget(self.btn_recent)
        recent_row.addStretch(1)

        card_layout.addLayout(recent_row)

        if self._last_xml_path and Path(self._last_xml_path).is_file():
            recent_name = Path(self._last_xml_path).name
            self.lbl_recent_caption.setVisible(True)
            self.btn_recent.setText(recent_name)
            self.btn_recent.setVisible(True)

        # ── Open Workspace Action Button ──────────────────────────────────────
        self.btn_start = PrimaryPushButton(FluentIcon.ACCEPT, "Open Workspace", card)
        self.btn_start.setFixedHeight(46)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_enter_workspace)
        card_layout.addWidget(self.btn_start)

        # Friendly footnote
        footnote = CaptionLabel(
            "Print your layout on plain paper, position it in camera view, and begin typing.",
            card,
        )
        footnote.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        footnote.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(footnote)

        root.addWidget(card, 0, Qt.AlignCenter)

    def _connect_signals(self) -> None:
        self._vm.layout_loaded.connect(self._on_layout_loaded)
        self._vm.error_occurred.connect(self._show_error)

    # ── File loading ───────────────────────────────────────────────────────────

    def _on_browse_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Keyboard Layout", "", "XML Layout Files (*.xml)"
        )
        if path:
            self._load_xml_path(path)

    def _on_load_recent(self) -> None:
        if self._last_xml_path and Path(self._last_xml_path).is_file():
            self._load_xml_path(self._last_xml_path)

    def _load_xml_path(self, path: str) -> None:
        self._current_xml_path = path
        self._vm.load_layout(path)

    def _on_layout_loaded(self, layout: LayoutData) -> None:
        self._current_layout = layout
        file_name = Path(self._current_xml_path).name
        btn_count = len(layout.buttons)
        marker_count = len(layout.markers)

        self.lbl_filename.setText(file_name)
        self.lbl_keys_pill.setText(f"{btn_count} Keys")
        self.lbl_tags_pill.setText(f"{marker_count} AprilTags")

        self.selection_stack.setCurrentIndex(1)
        self.btn_start.setEnabled(True)

    # ── Workspace Entry ────────────────────────────────────────────────────────

    def _on_enter_workspace(self) -> None:
        if self._current_layout is None:
            self._show_error("Please select a keyboard layout file first.")
            return

        self.workspace_entered.emit(
            self._current_layout,
            self._current_xml_path,
        )

    def _show_error(self, message: str) -> None:
        InfoBar.error(
            title="Unable to Load Layout",
            content=message,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=5000,
        )
