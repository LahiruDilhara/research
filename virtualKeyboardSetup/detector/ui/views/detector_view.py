"""
ui/views/detector_view.py

Live detector HUD view.

Layout
──────
  [Camera feed — fills left area]  |  [Sidebar: model info, layout status,
                                         finger probs, touch log, stop button]
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    PushButton,
    StrongBodyLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_BG_DARK, UI_TEXT_PRI, UI_TEXT_SEC
from core.interfaces.touch_model import ModelEntry
from ui.components.camera_feed_widget import CameraFeedWidget
from ui.components.finger_status_bar import FingerStatusBar
from ui.components.touch_event_log import TouchEventLog
from viewmodels.detector_viewmodel import DetectorViewModel

_SIDEBAR_W = 320


class DetectorView(QWidget):
    """Full live detector HUD."""

    stop_requested = Signal()

    def __init__(self, vm: DetectorViewModel, model_entry: ModelEntry, parent=None) -> None:
        super().__init__(parent)
        self._vm = vm
        self._model_entry = model_entry
        self._layout_data = None   # set via set_layout()
        self._setup_ui()
        self._connect_vm()

    # ── UI setup ───────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {UI_BG_DARK};")
        main_row = QHBoxLayout(self)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)

        # ── Left: camera feed ─────────────────────────────────────────────────
        self.feed = CameraFeedWidget()
        main_row.addWidget(self.feed, 1)

        # ── Right: sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(_SIDEBAR_W)
        sidebar.setStyleSheet(
            f"QWidget {{ background-color: #18181C; border-left: 1px solid rgba(255,255,255,0.06); }}"
        )
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(16, 20, 16, 16)
        sb_layout.setSpacing(14)

        # Title
        title_lbl = StrongBodyLabel("DETECTOR HUD")
        title_lbl.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 15px; letter-spacing: 1px;")
        sb_layout.addWidget(title_lbl)

        # ── Model info card ───────────────────────────────────────────────────
        sb_layout.addWidget(self._make_section_label("Active Model"))
        self._model_card = self._make_info_card(
            self._model_entry.name if self._model_entry else "None",
            self._model_entry.description if self._model_entry else "",
        )
        sb_layout.addWidget(self._model_card)

        # ── Performance stats ─────────────────────────────────────────────────
        sb_layout.addWidget(self._make_section_label("Pipeline"))
        perf = CardWidget()
        perf.setStyleSheet(self._card_style())
        perf_inner = QHBoxLayout(perf)
        perf_inner.setContentsMargins(14, 10, 14, 10)

        self.lbl_fps = StrongBodyLabel("FPS: —")
        self.lbl_fps.setStyleSheet("color: #00DC64; font-size: 12px;")
        self.lbl_layout_status = CaptionLabel("Layout: Searching...")
        self.lbl_layout_status.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")
        self.lbl_hand_status = CaptionLabel("Hand: —")
        self.lbl_hand_status.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 11px;")

        perf_inner.addWidget(self.lbl_fps)
        perf_inner.addStretch(1)
        perf_col = QVBoxLayout()
        perf_col.setSpacing(2)
        perf_col.addWidget(self.lbl_layout_status)
        perf_col.addWidget(self.lbl_hand_status)
        perf_inner.addLayout(perf_col)
        sb_layout.addWidget(perf)

        # ── Finger status bars ────────────────────────────────────────────────
        sb_layout.addWidget(self._make_section_label("Touch Probabilities"))
        self.finger_bar = FingerStatusBar()
        sb_layout.addWidget(self.finger_bar)

        # ── Touch event log ───────────────────────────────────────────────────
        self.event_log = TouchEventLog()
        sb_layout.addWidget(self.event_log)

        sb_layout.addStretch(1)

        # ── Stop button ───────────────────────────────────────────────────────
        self.btn_stop = PushButton(FluentIcon.CLOSE, "Stop Detector")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.clicked.connect(self._on_stop)
        sb_layout.addWidget(self.btn_stop)

        main_row.addWidget(sidebar)

    # ── ViewModel binding ──────────────────────────────────────────────────────

    def _connect_vm(self) -> None:
        self._vm.frame_updated.connect(self._on_frame_updated)
        self._vm.finger_probs_updated.connect(self._on_probs_updated)
        self._vm.touch_event.connect(self._on_touch_event)
        self._vm.pipeline_error.connect(self._on_error)

    # ── Slots ──────────────────────────────────────────────────────────────────

    @Slot(object, float, bool, bool)
    def _on_frame_updated(
        self,
        frame: np.ndarray,
        fps: float,
        hand_detected: bool,
        layout_found: bool,
    ) -> None:
        self.feed.update_frame(frame)
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_layout_status.setText(
            "Layout: ✓ Found" if layout_found else "Layout: Searching..."
        )
        self.lbl_layout_status.setStyleSheet(
            f"color: {'#00DC64' if layout_found else UI_TEXT_SEC}; font-size: 11px;"
        )
        self.lbl_hand_status.setText(
            "Hand: ✓ Detected" if hand_detected else "Hand: Not visible"
        )

        if not hand_detected:
            self.finger_bar.set_no_hand()

    @Slot(dict)
    def _on_probs_updated(self, probs: dict) -> None:
        self.finger_bar.set_probs(probs)

    @Slot(str, str, float)
    def _on_touch_event(self, key_id: str, finger: str, prob: float) -> None:
        # Look up label from layout
        label = key_id
        if self._layout_data:
            for btn in self._layout_data.buttons:
                if btn.id == key_id:
                    label = btn.label
                    break
        self.event_log.add_event(key_id, label, finger, prob)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        from qfluentwidgets import InfoBar, InfoBarPosition
        InfoBar.error(
            title="Pipeline Error",
            content=msg,
            position=InfoBarPosition.TOP,
            parent=self,
            duration=6000,
        )

    def _on_stop(self) -> None:
        self._vm.stop()
        self.stop_requested.emit()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def set_layout_data(self, layout_data) -> None:
        self._layout_data = layout_data

    @staticmethod
    def _card_style() -> str:
        return (
            f"CardWidget {{ background-color: {UI_BG_CARD}; "
            "border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; }}"
        )

    @staticmethod
    def _make_section_label(text: str) -> BodyLabel:
        lbl = BodyLabel(text)
        lbl.setStyleSheet(
            f"color: {UI_TEXT_SEC}; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;"
        )
        return lbl

    @staticmethod
    def _make_info_card(title: str, subtitle: str) -> CardWidget:
        card = CardWidget()
        card.setStyleSheet(
            f"CardWidget {{ background-color: {UI_BG_CARD}; "
            "border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; }}"
        )
        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 10, 14, 10)
        inner.setSpacing(3)
        t = StrongBodyLabel(title)
        t.setStyleSheet(f"color: {UI_ACCENT}; font-size: 12px; font-weight: bold;")
        t.setWordWrap(True)
        s = CaptionLabel(subtitle)
        s.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 10px;")
        s.setWordWrap(True)
        inner.addWidget(t)
        inner.addWidget(s)
        return card
