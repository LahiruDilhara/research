"""
ui/theme.py

Single source of truth for the app's design system.
All colors, font sizes, and shared QPushButton helpers live here.
Every view and component should import from here instead of defining inline colors.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QWidget


# ── Palette ────────────────────────────────────────────────────────────────────

BG_PAGE   = "#18191E"
BG_PANEL  = "#1E1F26"
BG_HEADER = "#16161A"
BG_CARD   = "#1A1A24"

BORDER        = "#2C2E38"
BORDER_SUBTLE = "rgba(255, 255, 255, 0.06)"

ACCENT    = "#009FEF"
ACCENT_DK = "#007FD4"

SUCCESS    = "#10B981"
SUCCESS_BG = "rgba(16, 185, 129, 0.12)"
DANGER     = "#FF4D4D"
WARN       = "#FBBF24"

TXT_PRI = "#F1F3F7"
TXT_SEC = "#7B8192"
TXT_DIM = "#4A5060"

FINGER_COLORS: dict[str, str] = {
    "Thumb":  "#FF8C00",
    "Index":  "#00BEFF",
    "Middle": "#00E678",
    "Ring":   "#E650FF",
    "Pinky":  "#FF4B4B",
}

# ── Typography ─────────────────────────────────────────────────────────────────

FONT_FAMILY   = "Inter, Segoe UI, Arial, sans-serif"
FONT_MONO     = "JetBrains Mono, Consolas, monospace"
FONT_TITLE    = 18
FONT_SUBTITLE = 15
FONT_BODY     = 13
FONT_SMALL    = 11
FONT_TINY     = 10


# ── Button helpers (icon-free, no FluentIcon overlap) ─────────────────────────

def btn_primary(label: str, parent: "QWidget | None" = None, height: int = 36) -> QPushButton:
    """Solid accent-filled primary action button."""
    b = QPushButton(label, parent)
    b.setFixedHeight(height)
    b.setStyleSheet(
        f"QPushButton {{"
        f"  background: {ACCENT}; color: #ffffff; border: none;"
        f"  border-radius: 7px; font-size: 13px; font-weight: 600; padding: 0 18px;"
        f"}}"
        f"QPushButton:hover {{ background: {ACCENT_DK}; }}"
        f"QPushButton:disabled {{ background: #2A2D38; color: {TXT_SEC}; }}"
    )
    return b


def btn_ghost(label: str, parent: "QWidget | None" = None, height: int = 36) -> QPushButton:
    """Bordered ghost / secondary action button."""
    b = QPushButton(label, parent)
    b.setFixedHeight(height)
    b.setStyleSheet(
        f"QPushButton {{"
        f"  background: #252730; color: {TXT_PRI};"
        f"  border: 1px solid {BORDER}; border-radius: 7px;"
        f"  font-size: 13px; padding: 0 16px;"
        f"}}"
        f"QPushButton:hover {{ background: #2D3040; border-color: #3C4055; }}"
        f"QPushButton:disabled {{ color: {TXT_SEC}; }}"
    )
    return b


def btn_icon_only(symbol: str, parent: "QWidget | None" = None, size: int = 32) -> QPushButton:
    """Small square symbol button for toolbar use."""
    b = QPushButton(symbol, parent)
    b.setFixedSize(size, size)
    b.setStyleSheet(
        f"QPushButton {{"
        f"  background: #252730; color: {TXT_PRI};"
        f"  border: 1px solid {BORDER}; border-radius: 6px; font-size: 14px;"
        f"}}"
        f"QPushButton:hover {{ background: #2D3040; }}"
    )
    return b


# ── Shared QSS snippets ────────────────────────────────────────────────────────

CARD_QSS = (
    f"background: {BG_CARD}; "
    f"border: 1px solid {BORDER}; "
    f"border-radius: 10px;"
)

LABEL_TRANSPARENT = "background: transparent; border: none; padding: 0;"
