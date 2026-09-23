"""
ui/components/paper_layout_canvas.py

Interactive 2D digital replica of the physical paper keyboard.
Renders a clean white paper sheet with black key outlines and fiducial markers.
Highlights keys dynamically in the touching finger's color when touch events occur.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from config.constants import FINGERS
from core.layout.layout_parser import LayoutData

# High-contrast finger colors for highlighting on the white paper canvas
FINGER_CANVAS_COLORS: dict[str, QColor] = {
    "Thumb":  QColor(230, 115, 0),    # Amber
    "Index":  QColor(0, 140, 220),    # Blue
    "Middle": QColor(0, 180, 80),     # Emerald
    "Ring":   QColor(190, 40, 210),   # Magenta
    "Pinky":  QColor(230, 45, 45),    # Crimson Red
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
        self._active_finger: str | None = None
        self._active_prob: float = 0.0

        # Auto-reset timer to clear highlight after touch impact
        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.setInterval(280)
        self._reset_timer.timeout.connect(self._on_highlight_timeout)

        self.setMinimumSize(280, 200)

    def set_layout(self, layout: LayoutData) -> None:
        """Update active paper layout data."""
        self._layout = layout
        self._active_key_id = None
        self.update()

    def highlight_touch(self, key_id: str, finger: str, prob: float = 1.0) -> None:
        """Highlights the touched key with the triggering finger's color."""
        self._active_key_id = key_id
        self._active_finger = finger
        self._active_prob = float(prob)
        self._reset_timer.start()
        self.update()

    def _on_highlight_timeout(self) -> None:
        self._active_key_id = None
        self._active_finger = None
        self.update()

    # ── Paint Event ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # Outer dark container background
        painter.fillRect(0, 0, int(w), int(h), QColor("#181820"))

        if self._layout is None or self._layout.paper_width_mm <= 0:
            painter.setPen(QColor(160, 160, 175))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(
                QRectF(0, 0, w, h),
                Qt.AlignCenter,
                "No Paper Layout Loaded",
            )
            return

        # Calculate aspect ratio of physical paper (A4 / Letter mm)
        paper_w_mm = float(self._layout.paper_width_mm)
        paper_h_mm = float(self._layout.paper_height_mm)

        margin = 16.0
        avail_w = max(20.0, w - margin * 2)
        avail_h = max(20.0, h - margin * 2)

        scale_x = avail_w / paper_w_mm
        scale_y = avail_h / paper_h_mm
        scale = min(scale_x, scale_y)

        draw_paper_w = paper_w_mm * scale
        draw_paper_h = paper_h_mm * scale
        paper_x = margin + (avail_w - draw_paper_w) / 2.0
        paper_y = margin + (avail_h - draw_paper_h) / 2.0

        paper_rect = QRectF(paper_x, paper_y, draw_paper_w, draw_paper_h)

        # 1. Draw Clean White Paper Sheet
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(200, 200, 210), 1))
        painter.drawRoundedRect(paper_rect, 6, 6)

        # 2. Draw AprilTag Markers
        painter.setBrush(QColor(40, 40, 45))
        painter.setPen(Qt.NoPen)
        for m in self._layout.markers:
            m_size_px = max(6.0, m.size_mm * scale)
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
                # Fill illuminated key
                fill_color = QColor(f_color.red(), f_color.green(), f_color.blue(), 100)
                painter.setBrush(fill_color)
                painter.setPen(QPen(f_color, 2.5))
            else:
                painter.setBrush(QColor(248, 248, 250))
                painter.setPen(QPen(QColor(30, 30, 35), 1.2))

            painter.drawRoundedRect(btn_rect, 3, 3)

            # Key label
            font_size = max(7, int(min(bw, bh) * 0.32))
            painter.setFont(QFont("Segoe UI", font_size, QFont.Bold if is_active else QFont.Normal))
            painter.setPen(QColor(0, 0, 0) if not is_active else QColor(20, 20, 30))
            painter.drawText(btn_rect, Qt.AlignCenter, btn.label[:6])

        # 4. Active Touch Banner on top
        if self._active_key_id and self._active_finger:
            banner_rect = QRectF(paper_x, paper_y + 6, draw_paper_w, 22)
            f_color = FINGER_CANVAS_COLORS.get(self._active_finger, QColor(0, 160, 255))
            painter.setBrush(QColor(f_color.red(), f_color.green(), f_color.blue(), 230))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(banner_rect, 4, 4)

            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            banner_text = f"Key: {self._active_key_id} | Finger: {self._active_finger} ({self._active_prob:.2f})"
            painter.drawText(banner_rect, Qt.AlignCenter, banner_text)
