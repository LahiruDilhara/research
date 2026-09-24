"""
ui/components/finger_status_bar.py

Widget showing 5 per-finger touch probability progress bars with status labels.
Updates colours based on touch state (green = touch, blue-grey = no touch, dim = no hand).
"""

from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import (
    CardWidget,
    ProgressBar,
)

from config.constants import FINGERS
from ui.theme import (
    ACCENT,
    BG_CARD,
    BORDER,
    FINGER_COLORS,
    LABEL_TRANSPARENT,
    SUCCESS,
    TXT_PRI,
    TXT_SEC,
)

_NO_HAND_BG  = BG_CARD
_TOUCH_BG    = "#183C24"
_NO_TOUCH_BG = "#1A1A28"

def _card_css(bg_color: str, border_color: str = BORDER) -> str:
    return (
        f"#fingerCard {{ "
        f"  background-color: {bg_color}; "
        f"  border: 1px solid {border_color}; "
        "  border-radius: 8px; "
        "} "
        f"QLabel {{ {LABEL_TRANSPARENT} }}"
    )


class FingerStatusBar(QWidget):
    """5 stacked per-finger touch status cards."""

    def __init__(self, touch_threshold: float = 0.55, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        self._touch_threshold: float = float(touch_threshold)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self._cards: dict[str, dict] = {}

        for finger in FINGERS:
            card, widgets = self._make_finger_card(finger)
            layout.addWidget(card)
            self._cards[finger] = {"card": card, **widgets}

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_touch_threshold(self, threshold: float) -> None:
        """Dynamically update touch probability threshold."""
        self._touch_threshold = max(0.01, min(1.00, float(threshold)))

    def set_probs(self, probs: dict[str, Any]) -> None:
        """Update all 5 cards from a probability dict or structured results dict."""
        for finger, val in probs.items():
            if finger not in self._cards:
                continue
            if isinstance(val, dict):
                prob = float(val.get("prob", 0.0))
                if "touch" in val:
                    is_touch = bool(val["touch"])
                else:
                    is_touch = prob >= self._touch_threshold
                hand_moving = bool(val.get("hand_moving", False))
            else:
                prob = float(val)
                is_touch = prob >= self._touch_threshold
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
        status_lbl: QLabel = c["status"]
        pbar: ProgressBar = c["pbar"]

        if not hand_present:
            card.setStyleSheet(_card_css(_NO_HAND_BG))
            status_lbl.setText("NO HAND")
            status_lbl.setStyleSheet(f"color: #555566; font-size: 10px; {LABEL_TRANSPARENT}")
            pbar.setValue(0)
        elif hand_moving:
            card.setStyleSheet(_card_css("#3A2810", "#FF9000"))
            status_lbl.setText("MOVING")
            status_lbl.setStyleSheet(f"color: #FFA500; font-weight: bold; font-size: 10px; {LABEL_TRANSPARENT}")
            pbar.setValue(0)
        elif is_touch:
            card.setStyleSheet(_card_css(_TOUCH_BG, SUCCESS))
            status_lbl.setText(f"TOUCH {prob*100:.0f}%")
            status_lbl.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 10px; {LABEL_TRANSPARENT}")
            pbar.setValue(int(prob * 100))
        else:
            card.setStyleSheet(_card_css(_NO_TOUCH_BG))
            status_lbl.setText(f"{prob*100:.0f}%")
            status_lbl.setStyleSheet(f"color: {TXT_SEC}; font-size: 10px; {LABEL_TRANSPARENT}")
            pbar.setValue(int(prob * 100))

    @staticmethod
    def _make_finger_card(finger: str) -> tuple[CardWidget, dict]:
        card = CardWidget()
        card.setObjectName("fingerCard")
        card.setStyleSheet(_card_css(_NO_HAND_BG))
        card.setFixedHeight(44)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(10, 5, 10, 5)
        inner.setSpacing(3)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        finger_color = FINGER_COLORS.get(finger, TXT_PRI)
        name_lbl = QLabel(finger.upper())
        name_lbl.setStyleSheet(
            f"color: {finger_color}; font-weight: 700; font-size: 11px; {LABEL_TRANSPARENT}"
        )
        status_lbl = QLabel("NO HAND")
        status_lbl.setStyleSheet(f"color: #555566; font-size: 10px; {LABEL_TRANSPARENT}")
        header.addWidget(name_lbl)
        header.addStretch(1)
        header.addWidget(status_lbl)

        pbar = ProgressBar()
        pbar.setFixedHeight(4)
        pbar.setValue(0)
        pbar.setMinimum(0)
        pbar.setMaximum(100)

        inner.addLayout(header)
        inner.addWidget(pbar)

        return card, {"status": status_lbl, "pbar": pbar}
