"""
ui/components/paper_layout_canvas.py

Interactive 2D digital replica of the physical paper keyboard.
Renders a clean white paper sheet with black key outlines and fiducial markers.
Features:
  - Layout aligned to bottom with padding.
  - Touched key is brightly colored in the contacting finger's color.
  - Dedicated configured command strip at the bottom showing active and last-triggered action.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from config.constants import FINGERS
from core.layout.layout_parser import LayoutData
from ui.theme import (
    ACCENT,
    BG_CARD,
    BORDER,
    TXT_PRI,
    TXT_SEC,
)

# High-contrast finger colors for highlighting on the white paper canvas
FINGER_CANVAS_COLORS: dict[str, QColor] = {
    "Thumb":  QColor(230, 120, 0),     # Amber
    "Index":  QColor(0, 150, 240),     # Vibrant Blue
    "Middle": QColor(0, 190, 85),      # Emerald Green
    "Ring":   QColor(190, 45, 215),    # Magenta
    "Pinky":  QColor(235, 50, 50),     # Crimson Red
}


class PaperLayoutCanvasWidget(QWidget):
    """2D white canvas replica of the physical paper keyboard layout."""

    def __init__(
        self,
        layout: LayoutData | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._layout = layout
        self._active_key_id: str | None = None
        self._active_key_label: str = ""
        self._active_finger: str | None = None
        self._active_prob: float = 0.0
        self._active_action: str = ""

        # Persistence for last executed action at the bottom
        self._last_key_label: str = ""
        self._last_finger: str = ""
        self._last_action: str = ""

        # Auto-reset timer to clear highlight after touch impact
        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.setInterval(650)
        self._reset_timer.timeout.connect(self._on_highlight_timeout)

        self.setMinimumSize(280, 220)

    def set_layout(self, layout: LayoutData) -> None:
        """Update active paper layout data."""
        self._layout = layout
        self._active_key_id = None
        self._active_action = ""
        self.update()

    def highlight_touch(
        self,
        key_id: str,
        finger: str,
        prob: float = 1.0,
        action_str: str = "",
        key_label: str = "",
    ) -> None:
        """Highlights the touched key with the triggering finger's color and updates action strip."""
        self._active_key_id = key_id
        self._active_key_label = key_label if key_label else key_id
        self._active_finger = finger
        self._active_prob = float(prob)
        self._active_action = action_str

        self._last_key_label = self._active_key_label
        self._last_finger = finger
        self._last_action = action_str

        self._reset_timer.start()
        self.update()

    def _on_highlight_timeout(self) -> None:
        self._active_key_id = None
        self._active_finger = None
        self._active_action = ""
        self.update()

    # ── Paint Event ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # Outer container background with rounded border
        container_rect = QRectF(0, 0, w, h)
        painter.setBrush(QColor(BG_CARD))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRoundedRect(container_rect, 10, 10)

        if self._layout is None or self._layout.paper_width_mm <= 0:
            painter.setPen(QColor(TXT_SEC))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(
                container_rect,
                Qt.AlignCenter,
                "No Paper Layout Loaded",
            )
            return

        # Dimensions & padding
        pad_x = 12.0
        pad_top = 8.0
        pad_bottom = 8.0
        cmd_strip_h = 32.0
        space_paper_cmd = 8.0

        # Available space for 2D paper
        avail_paper_w = max(20.0, w - pad_x * 2)
        avail_paper_h = max(20.0, h - pad_top - cmd_strip_h - space_paper_cmd - pad_bottom)

        paper_w_mm = float(self._layout.paper_width_mm)
        paper_h_mm = float(self._layout.paper_height_mm)

        scale_x = avail_paper_w / paper_w_mm
        scale_y = avail_paper_h / paper_h_mm
        scale = min(scale_x, scale_y)

        draw_paper_w = paper_w_mm * scale
        draw_paper_h = paper_h_mm * scale

        # Centered horizontally, aligned to bottom of paper area (with padding)
        paper_x = pad_x + (avail_paper_w - draw_paper_w) / 2.0
        paper_y = pad_top + (avail_paper_h - draw_paper_h)

        paper_rect = QRectF(paper_x, paper_y, draw_paper_w, draw_paper_h)

        # 1. Draw Clean White Paper Sheet
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(190, 195, 205), 1))
        painter.drawRoundedRect(paper_rect, 5, 5)

        # 2. Draw AprilTag Markers
        painter.setBrush(QColor(35, 35, 40))
        painter.setPen(Qt.NoPen)
        for m in self._layout.markers:
            m_size_px = max(5.0, m.size_mm * scale)
            mx = paper_x + m.center_x_mm * scale - m_size_px / 2.0
            my = paper_y + m.center_y_mm * scale - m_size_px / 2.0
            painter.drawRect(QRectF(mx, my, m_size_px, m_size_px))

        # 3. Draw Key Boxes
        for btn in self._layout.buttons:
            bx = paper_x + btn.x_mm * scale
            by = paper_y + btn.y_mm * scale
            bw = btn.width_mm * scale
            bh = btn.height_mm * scale
            btn_rect = QRectF(bx, by, bw, bh)

            is_active = (btn.id == self._active_key_id)

            if is_active and self._active_finger:
                f_color = FINGER_CANVAS_COLORS.get(self._active_finger, QColor(0, 160, 255))
                # Fill illuminated key with finger's color
                fill_color = QColor(f_color.red(), f_color.green(), f_color.blue(), 210)
                painter.setBrush(fill_color)
                painter.setPen(QPen(f_color, 2.5))
            else:
                painter.setBrush(QColor(246, 247, 250))
                painter.setPen(QPen(QColor(40, 40, 50), 1.2))

            painter.drawRoundedRect(btn_rect, 3, 3)

            # Key label text
            font_size = max(7, int(min(bw, bh) * 0.32))
            painter.setFont(QFont("Segoe UI", font_size, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(255, 255, 255) if is_active else QColor(15, 15, 20))
            painter.drawText(btn_rect, Qt.AlignCenter, btn.label[:6])

        # 4. Configured Command Strip at the bottom
        cmd_rect = QRectF(pad_x, h - cmd_strip_h - pad_bottom, w - pad_x * 2, cmd_strip_h)

        if self._active_key_id and self._active_finger:
            f_color = FINGER_CANVAS_COLORS.get(self._active_finger, QColor(0, 160, 255))
            painter.setBrush(QColor(22, 26, 36))
            painter.setPen(QPen(f_color, 1.5))
            painter.drawRoundedRect(cmd_rect, 6, 6)

            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            cmd_text = f"Key: {self._active_key_label} [{self._active_finger}]  →  {self._active_action}"
            painter.drawText(cmd_rect.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, cmd_text)
        elif self._last_action:
            painter.setBrush(QColor(18, 18, 26))
            painter.setPen(QPen(QColor(BORDER), 1))
            painter.drawRoundedRect(cmd_rect, 6, 6)

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(190, 195, 210))
            last_text = f"Last: {self._last_key_label} [{self._last_finger}]  →  {self._last_action}"
            painter.drawText(cmd_rect.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, last_text)
        else:
            painter.setBrush(QColor(18, 18, 26))
            painter.setPen(QPen(QColor(BORDER), 1))
            painter.drawRoundedRect(cmd_rect, 6, 6)

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(120, 120, 140))
            painter.drawText(
                cmd_rect.adjusted(10, 0, -10, 0),
                Qt.AlignVCenter | Qt.AlignLeft,
                "Command: Ready (touch any paper key to test)",
            )
