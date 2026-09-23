"""
ui/components/dynamics_graph.py

Minimalist real-time graph widget for pipeline dynamics.
Plots rolling inference latency (milliseconds) and processing FPS.
Uses hardware-accelerated Qt QPainter with antialiasing and spacious layout.
"""

from __future__ import annotations

from collections import deque
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from config.constants import UI_BG_CARD, UI_TEXT_PRI, UI_TEXT_SEC

_COLOR_LATENCY = QColor(0, 190, 255)   # Cyan
_COLOR_FPS     = QColor(0, 220, 100)   # Emerald Green
_COLOR_GRID    = QColor(255, 255, 255, 16)
_COLOR_AXIS    = QColor(255, 255, 255, 36)


class DetectionDynamicsGraphWidget(QWidget):
    """Minimalist live graph for pipeline latency and frame rate dynamics."""

    def __init__(self, history_len: int = 80, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history_len = history_len
        self._latency_history: deque[float] = deque([0.0] * history_len, maxlen=history_len)
        self._fps_history: deque[float] = deque([12.0] * history_len, maxlen=history_len)

        self.setMinimumHeight(130)
        self.setStyleSheet(f"background-color: {UI_BG_CARD}; border-radius: 10px;")

    def push_metrics(self, latency_ms: float, fps: float) -> None:
        """Push latest latency and FPS samples and request repaint."""
        self._latency_history.append(float(latency_ms))
        self._fps_history.append(float(fps))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        w = self.width()
        h = self.height()

        # Margins with generous breathing room
        pad_left = 40
        pad_right = 24
        pad_top = 26
        pad_bottom = 24

        plot_w = max(10, w - pad_left - pad_right)
        plot_h = max(10, h - pad_top - pad_bottom)

        # ── Background Grid & Reference Lines ──────────────────────────────────
        pen_grid = QPen(_COLOR_GRID, 1, Qt.DashLine)
        painter.setPen(pen_grid)

        # Horizontal grid at 25%, 50%, 75%
        for ratio in [0.25, 0.5, 0.75]:
            gy = pad_top + plot_h * ratio
            painter.drawLine(QPointF(pad_left, gy), QPointF(pad_left + plot_w, gy))

        # Bottom baseline
        pen_axis = QPen(_COLOR_AXIS, 1, Qt.SolidLine)
        painter.setPen(pen_axis)
        painter.drawLine(
            QPointF(pad_left, pad_top + plot_h),
            QPointF(pad_left + plot_w, pad_top + plot_h),
        )

        # ── Axis Labels ────────────────────────────────────────────────────────
        font_lbl = QFont("Sans-Serif", 8)
        painter.setFont(font_lbl)
        painter.setPen(QColor(UI_TEXT_SEC))
        painter.drawText(QRectF(4, pad_top - 6, pad_left - 10, 16), Qt.AlignRight | Qt.AlignVCenter, "10ms")
        painter.drawText(QRectF(4, pad_top + plot_h * 0.5 - 8, pad_left - 10, 16), Qt.AlignRight | Qt.AlignVCenter, "5ms")
        painter.drawText(QRectF(4, pad_top + plot_h - 10, pad_left - 10, 16), Qt.AlignRight | Qt.AlignVCenter, "0ms")

        # ── Curves Drawing ─────────────────────────────────────────────────────
        x_step = plot_w / max(1, self._history_len - 1)

        # 1. Latency curve (max reference: 10 ms)
        lat_path = QPainterPath()
        lat_pts = list(self._latency_history)
        for i, val in enumerate(lat_pts):
            norm_val = min(1.0, max(0.0, val / 10.0))
            px = pad_left + i * x_step
            py = pad_top + plot_h * (1.0 - norm_val)
            if i == 0:
                lat_path.moveTo(px, py)
            else:
                lat_path.lineTo(px, py)

        pen_lat = QPen(_COLOR_LATENCY, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_lat)
        painter.drawPath(lat_path)

        # 2. FPS curve (max reference: 20 FPS)
        fps_path = QPainterPath()
        fps_pts = list(self._fps_history)
        for i, val in enumerate(fps_pts):
            norm_val = min(1.0, max(0.0, val / 20.0))
            px = pad_left + i * x_step
            py = pad_top + plot_h * (1.0 - norm_val)
            if i == 0:
                fps_path.moveTo(px, py)
            else:
                fps_path.lineTo(px, py)

        pen_fps = QPen(_COLOR_FPS, 1.8, Qt.DotLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_fps)
        painter.drawPath(fps_path)

        # ── Minimalist Legend in Top-Right ─────────────────────────────────────
        legend_x = pad_left + plot_w - 180
        legend_y = 6

        # Latency legend item
        painter.setPen(QPen(_COLOR_LATENCY, 2))
        painter.drawLine(QPointF(legend_x, legend_y + 7), QPointF(legend_x + 16, legend_y + 7))
        painter.setPen(QColor(UI_TEXT_PRI))
        cur_lat = lat_pts[-1] if lat_pts else 0.0
        painter.drawText(QRectF(legend_x + 20, legend_y, 70, 14), Qt.AlignLeft | Qt.AlignVCenter, f"Lat: {cur_lat:.1f}ms")

        # FPS legend item
        painter.setPen(QPen(_COLOR_FPS, 2, Qt.DotLine))
        painter.drawLine(QPointF(legend_x + 95, legend_y + 7), QPointF(legend_x + 111, legend_y + 7))
        painter.setPen(QColor(UI_TEXT_PRI))
        cur_fps = fps_pts[-1] if fps_pts else 0.0
        painter.drawText(QRectF(legend_x + 115, legend_y, 65, 14), Qt.AlignLeft | Qt.AlignVCenter, f"FPS: {cur_fps:.1f}")

        painter.end()
