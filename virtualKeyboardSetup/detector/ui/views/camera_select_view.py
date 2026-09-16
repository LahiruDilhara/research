"""
ui/views/camera_select_view.py

Camera selection view — shows detected cameras as clickable cards.
User picks one, then clicks Start.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    SubtitleLabel,
    StrongBodyLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_BG_DARK, UI_TEXT_PRI, UI_TEXT_SEC
from services.camera_discovery import CameraInfo
from viewmodels.camera_select_viewmodel import CameraSelectViewModel

_CARD_STYLE = (
    f"#cameraCard {{ "
    f"  background-color: {UI_BG_CARD}; "
    "  border: 1px solid rgba(255, 255, 255, 0.06); "
    "  border-radius: 12px; "
    "} "
    "QLabel { "
    "  background-color: transparent; "
    "  border: none; "
    "}"
)
_CARD_SELECTED = (
    f"#cameraCard {{ "
    "  background-color: #0D2A40; "
    f"  border: 2px solid {UI_ACCENT}; "
    "  border-radius: 12px; "
    "} "
    "QLabel { "
    "  background-color: transparent; "
    "  border: none; "
    "}"
)


class CameraSelectView(QWidget):
    """Camera picker cards view."""

    back_requested  = Signal()
    start_requested = Signal(int)  # selected camera index

    def __init__(self, vm: CameraSelectViewModel, parent=None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._card_widgets: dict[int, CardWidget] = {}
        self._selected_index: int | None = None
        self._setup_ui()
        self._connect_vm()
        self._vm.refresh()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 24)
        root.setSpacing(20)

        title = SubtitleLabel("Select Camera")
        title.setStyleSheet(f"background: transparent; color: {UI_ACCENT}; font-size: 20px; font-weight: bold;")
        hint = CaptionLabel("Select the camera to use for real-time detection.")
        hint.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 12px;")
        root.addWidget(title)
        root.addWidget(hint)

        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(16)
        self._cards_row.setAlignment(Qt.AlignLeft)
        root.addLayout(self._cards_row)
        root.addStretch(1)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_back = PushButton(FluentIcon.RETURN, "Back")
        self.btn_back.setFixedHeight(38)
        self.btn_back.clicked.connect(self.back_requested.emit)

        self.btn_refresh = PushButton(FluentIcon.SYNC, "Refresh")
        self.btn_refresh.setFixedHeight(38)
        self.btn_refresh.clicked.connect(self._vm.refresh)

        self.btn_start = PrimaryPushButton(FluentIcon.PLAY, "Start Detector")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)

        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_start)
        root.addLayout(btn_row)

    def _connect_vm(self) -> None:
        self._vm.cameras_ready.connect(self._on_cameras_ready)

    def _on_cameras_ready(self, cameras: list[CameraInfo]) -> None:
        # Clear existing cards
        while self._cards_row.count():
            item = self._cards_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._card_widgets.clear()
        self._selected_index = None
        self.btn_start.setEnabled(False)

        if not cameras:
            lbl = BodyLabel("No cameras detected. Connect a camera and click Refresh.")
            lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC};")
            self._cards_row.addWidget(lbl)
            return

        for cam in cameras:
            card = self._make_camera_card(cam)
            self._card_widgets[cam.index] = card
            self._cards_row.addWidget(card)

        # Auto-select first
        self._select_camera(cameras[0].index)

    def _make_camera_card(self, cam: CameraInfo) -> CardWidget:
        card = CardWidget()
        card.setObjectName("cameraCard")
        card.setBorderRadius(12)
        card.setStyleSheet(_CARD_STYLE)
        card.setFixedSize(220, 145)
        card.setCursor(Qt.PointingHandCursor)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(18, 16, 18, 16)
        inner.setSpacing(6)
        inner.setAlignment(Qt.AlignCenter)

        icon_lbl = StrongBodyLabel("📷")
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("background: transparent; font-size: 26px; border: none;")
        
        name_lbl = StrongBodyLabel(cam.name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-size: 12px; font-weight: bold; border: none;")
        name_lbl.setWordWrap(True)
        
        res_lbl = CaptionLabel(f"{cam.width}×{cam.height}  •  {cam.fps:.0f} fps")
        res_lbl.setAlignment(Qt.AlignCenter)
        res_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC}; font-size: 11px; border: none;")

        inner.addWidget(icon_lbl)
        inner.addWidget(name_lbl)
        inner.addWidget(res_lbl)

        card.mousePressEvent = lambda _e, idx=cam.index: self._select_camera(idx)
        return card

    def _select_camera(self, index: int) -> None:
        self._selected_index = index
        self._vm.select(index)
        # Update card borders
        for idx, card in self._card_widgets.items():
            card.setStyleSheet(_CARD_SELECTED if idx == index else _CARD_STYLE)
        self.btn_start.setEnabled(True)

    def _on_start(self) -> None:
        if self._selected_index is not None:
            self.start_requested.emit(self._selected_index)
