"""
ui/components/finger_status_bar.py

Widget showing 5 per-finger touch probability progress bars with status labels.
Updates colours based on touch state (green = touch, blue-grey = no touch, dim = no hand).
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ProgressBar,
    StrongBodyLabel,
)

from config.constants import FINGERS, UI_ACCENT, UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC

_CARD_BASE = (
    f"background-color: {UI_BG_CARD}; "
    "border: 1px solid rgba(255,255,255,0.06); "
    "border-radius: 10px;"
)
_TOUCH_BG    = "#183C24"
_NO_TOUCH_BG = "#22223A"
_NO_HAND_BG  = UI_BG_CARD


class FingerStatusBar(QWidget):
    """5 stacked per-finger touch status cards."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = StrongBodyLabel("Per-Finger Touch Status")
        title.setStyleSheet(f"color: {UI_TEXT_PRI}; font-size: 12px;")
        layout.addWidget(title)

        self._cards: dict[str, dict] = {}

        for finger in FINGERS:
            card, widgets = self._make_finger_card(finger)
            layout.addWidget(card)
            self._cards[finger] = {"card": card, **widgets}

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_probs(self, probs: dict[str, float]) -> None:
        """Update all 5 cards from a probability dict."""
        for finger, prob in probs.items():
            if finger not in self._cards:
                continue
            self._update_card(finger, prob=prob, hand_present=True)

    def set_no_hand(self) -> None:
        """Mark all fingers as 'no hand detected'."""
        for finger in FINGERS:
            self._update_card(finger, prob=0.0, hand_present=False)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_card(self, finger: str, prob: float, hand_present: bool) -> None:
        c = self._cards[finger]
        card: CardWidget = c["card"]
        status_lbl: BodyLabel = c["status"]
        pbar: ProgressBar = c["pbar"]

        is_touch = hand_present and prob >= 0.5

        if not hand_present:
            card.setStyleSheet(f"background-color: {_NO_HAND_BG}; border-radius: 10px;")
            status_lbl.setText("NO HAND")
            status_lbl.setStyleSheet("color: #606070;")
            pbar.setValue(0)
        elif is_touch:
            card.setStyleSheet(f"background-color: {_TOUCH_BG}; border-radius: 10px;")
            status_lbl.setText(f"TOUCH  {prob*100:.0f}%")
            status_lbl.setStyleSheet("color: #00DC64; font-weight: bold;")
            pbar.setValue(int(prob * 100))
        else:
            card.setStyleSheet(f"background-color: {_NO_TOUCH_BG}; border-radius: 10px;")
            status_lbl.setText(f"{prob*100:.0f}%")
            status_lbl.setStyleSheet(f"color: {UI_TEXT_SEC};")
            pbar.setValue(int(prob * 100))

    @staticmethod
    def _make_finger_card(finger: str) -> tuple[CardWidget, dict]:
        card = CardWidget()
        card.setStyleSheet(_CARD_BASE)
        card.setFixedHeight(68)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(4)

        header = QHBoxLayout()
        name_lbl = BodyLabel(finger.upper())
        name_lbl.setStyleSheet(f"color: {UI_TEXT_PRI}; font-weight: bold; font-size: 11px;")
        status_lbl = BodyLabel("NO HAND")
        status_lbl.setStyleSheet("color: #606070; font-size: 11px;")
        header.addWidget(name_lbl)
        header.addStretch(1)
        header.addWidget(status_lbl)

        pbar = ProgressBar()
        pbar.setFixedHeight(6)
        pbar.setValue(0)
        pbar.setMinimum(0)
        pbar.setMaximum(100)

        inner.addLayout(header)
        inner.addWidget(pbar)

        return card, {"status": status_lbl, "pbar": pbar}
