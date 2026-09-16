"""
ui/components/touch_event_log.py

Scrolling list of recent touch events shown in the detector sidebar.
Each entry shows: timestamp, finger, key label, probability.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    SingleDirectionScrollArea,
    StrongBodyLabel,
)

from config.constants import UI_ACCENT, UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC

_MAX_EVENTS = 30


class TouchEventLog(QWidget):
    """Displays a scrollable list of the most recent touch events."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = StrongBodyLabel("Recent Touch Events")
        title.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 12px;")
        layout.addWidget(title)

        self._scroll = SingleDirectionScrollArea(orient=Qt.Vertical)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background-color: transparent; }}"
        )
        self._scroll.setFixedHeight(200)

        self._container = QWidget()
        self._container.setStyleSheet("background-color: transparent;")
        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

        self._entries: list[QWidget] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_event(self, key_id: str, key_label: str, finger: str, prob: float) -> None:
        """Prepend a new event row at the top of the list."""
        entry = self._make_entry(key_label, finger, prob)
        # Insert before the trailing stretch
        self._list_layout.insertWidget(0, entry)
        self._entries.append(entry)

        # Trim old entries
        while len(self._entries) > _MAX_EVENTS:
            oldest = self._entries.pop(0)
            self._list_layout.removeWidget(oldest)
            oldest.deleteLater()

    def clear(self) -> None:
        for w in self._entries:
            self._list_layout.removeWidget(w)
            w.deleteLater()
        self._entries.clear()

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_entry(key_label: str, finger: str, prob: float) -> QWidget:
        row = QWidget()
        row.setStyleSheet(
            f"QWidget {{ background-color: {UI_BG_CARD}; "
            "border-radius: 6px; }}"
        )
        inner = QVBoxLayout(row)
        inner.setContentsMargins(10, 6, 10, 6)
        inner.setSpacing(2)

        ts = datetime.now().strftime("%H:%M:%S")

        top_row_layout = __import__("PySide6.QtWidgets", fromlist=["QHBoxLayout"]).QHBoxLayout()
        key_lbl = StrongBodyLabel(f'"{key_label}"')
        key_lbl.setStyleSheet(f"color: {UI_ACCENT}; font-size: 12px;")
        ts_lbl = CaptionLabel(ts)
        ts_lbl.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 10px;")
        top_row_layout.addWidget(key_lbl)
        top_row_layout.addStretch(1)
        top_row_layout.addWidget(ts_lbl)

        detail_lbl = CaptionLabel(f"{finger}  •  {prob*100:.1f}%")
        detail_lbl.setStyleSheet(f"color: {UI_TEXT_SEC}; font-size: 10px;")

        inner.addLayout(top_row_layout)
        inner.addWidget(detail_lbl)
        return row
