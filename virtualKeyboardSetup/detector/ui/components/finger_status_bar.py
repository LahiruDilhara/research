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

_NO_HAND_BG  = UI_BG_CARD
_TOUCH_BG    = "#183C24"
_NO_TOUCH_BG = "#22223A"

def _card_css(bg_color: str, border_color: str = "rgba(255,255,255,0.06)") -> str:
    return (
        f"#fingerCard {{ "
        f"  background-color: {bg_color}; "
        f"  border: 1px solid {border_color}; "
        "  border-radius: 10px; "
        "} "
        "QLabel { "
        "  background-color: transparent; "
        "  border: none; "
        "}"
    )


class FingerStatusBar(QWidget):
    """5 stacked per-finger touch status cards."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = StrongBodyLabel("Per-Finger Touch Status")
        title.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-size: 12px;")
        layout.addWidget(title)

        self._cards: dict[str, dict] = {}

        for finger in FINGERS:
            card, widgets = self._make_finger_card(finger)
            layout.addWidget(card)
            self._cards[finger] = {"card": card, **widgets}

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_probs(self, probs: dict[str, Any]) -> None:
        """Update all 5 cards from a probability dict or structured results dict."""
        for finger, val in probs.items():
            if finger not in self._cards:
                continue
            if isinstance(val, dict):
                prob = float(val.get("prob", 0.0))
                is_touch = bool(val.get("touch", False))
                hand_moving = bool(val.get("hand_moving", False))
            else:
                prob = float(val)
                is_touch = prob >= 0.55
                hand_moving = False

            self._update_card(
                finger,
                prob=prob,
                is_touch=is_touch,
                hand_moving=hand_moving,
                hand_present=True,
            )

    def set_no_hand(self) -> None:
        """Mark all fingers as 'no hand detected'."""
        for finger in FINGERS:
            self._update_card(finger, prob=0.0, is_touch=False, hand_moving=False, hand_present=False)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_card(
        self,
        finger: str,
        prob: float,
        is_touch: bool,
        hand_moving: bool,
        hand_present: bool,
    ) -> None:
        c = self._cards[finger]
        card: CardWidget = c["card"]
        status_lbl: BodyLabel = c["status"]
        pbar: ProgressBar = c["pbar"]

        if not hand_present:
            card.setStyleSheet(_card_css(_NO_HAND_BG))
            status_lbl.setText("NO HAND")
            status_lbl.setStyleSheet("background: transparent; color: #606070;")
            pbar.setValue(0)
        elif hand_moving:
            card.setStyleSheet(_card_css("#3A2810", "#FF9000"))
            status_lbl.setText("HAND MOVING")
            status_lbl.setStyleSheet("background: transparent; color: #FFA500; font-weight: bold;")
            pbar.setValue(0)
        elif is_touch:
            card.setStyleSheet(_card_css(_TOUCH_BG, "#00DC64"))
            status_lbl.setText(f"TOUCH  {prob*100:.0f}%")
            status_lbl.setStyleSheet("background: transparent; color: #00DC64; font-weight: bold;")
            pbar.setValue(int(prob * 100))
        else:
            card.setStyleSheet(_card_css(_NO_TOUCH_BG))
            status_lbl.setText(f"UNTOUCH  {prob*100:.0f}%")
            status_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_SEC};")
            pbar.setValue(int(prob * 100))

    @staticmethod
    def _make_finger_card(finger: str) -> tuple[CardWidget, dict]:
        card = CardWidget()
        card.setObjectName("fingerCard")
        card.setStyleSheet(_card_css(_NO_HAND_BG))
        card.setFixedHeight(68)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(4)

        header = QHBoxLayout()
        name_lbl = BodyLabel(finger.upper())
        name_lbl.setStyleSheet(f"background: transparent; color: {UI_TEXT_PRI}; font-weight: bold; font-size: 11px;")
        status_lbl = BodyLabel("NO HAND")
        status_lbl.setStyleSheet("background: transparent; color: #606070; font-size: 11px;")
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
