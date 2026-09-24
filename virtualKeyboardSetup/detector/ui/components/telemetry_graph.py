"""
ui/components/telemetry_graph.py

High-performance real-time telemetry graph widget.
Plots live multi-finger touch probability curves and threshold lines using QPainter.
Zero external library dependencies.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from config.constants import FINGERS

# Distinct high-contrast colors for each finger in RGB
FINGER_COLORS_RGB: dict[str, QColor] = {
    "Thumb":  QColor(255, 140, 0),    # Amber
    "Index":  QColor(0, 190, 255),    # Cyan
    "Middle": QColor(0, 230, 120),    # Emerald
    "Ring":   QColor(230, 80, 255),   # Magenta
    "Pinky":  QColor(255, 75, 75),    # Coral Red
}


class TelemetryGraphWidget(QWidget):
    """Real-time scrolling line graph of 5-finger touch probabilities."""

    def __init__(
        self,
        max_points: int = 80,
        threshold: float = 0.55,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_points = max_points
        self._threshold = threshold
        self.setMinimumHeight(180)

        # Buffers for each finger
        self._history: dict[str, deque[float]] = {
            f: deque(maxlen=self._max_points) for f in FINGERS
        }
        # Pre-fill with zeros for smooth start
        for f in FINGERS:
            for _ in range(self._max_points):
                self._history[f].append(0.0)

    def set_threshold(self, threshold: float) -> None:
        """Update horizontal threshold indicator."""
        self._threshold = float(threshold)
        self.update()

    def push_probabilities(self, probs: dict[str, Any]) -> None:
        """Appends a new probability sample for all fingers and schedules a repaint."""
        for f in FINGERS:
            val = probs.get(f, 0.0)
            if isinstance(val, dict):
                p = float(val.get("prob", 0.0))
            else:
                p = float(val)
            p = max(0.0, min(1.0, p))
            self._history[f].append(p)
        self.update()

    def clear(self) -> None:
        """Reset all probability history buffers."""
        for f in FINGERS:
            self._history[f].clear()
            for _ in range(self._max_points):
                self._history[f].append(0.0)
        self.update()

    # ── Paint Event ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        # Margins for axis labels and top legend
        left_m = 40.0
        right_m = 16.0
        top_m = 32.0
        bottom_m = 24.0

        plot_w = max(10.0, w - left_m - right_m)
        plot_h = max(10.0, h - top_m - bottom_m)

        # 1. Background
        painter.fillRect(0, 0, w, h, QColor("#121218"))

        # Plot area background
        plot_rect = QRectF(left_m, top_m, plot_w, plot_h)
        painter.fillRect(plot_rect, QColor("#161622"))

        # Border
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.drawRect(plot_rect)

        # 2. Gridlines and Y-axis labels
        painter.setFont(QFont("Segoe UI", 8))
        grid_vals = [0.0, 0.5, 1.0]
        for g in grid_vals:
            y_pos = top_m + plot_h * (1.0 - g)
            painter.setPen(QPen(QColor(255, 255, 255, 15), 1, Qt.DashLine))
            painter.drawLine(QPointF(left_m, y_pos), QPointF(left_m + plot_w, y_pos))

            # Y label
            painter.setPen(QColor(160, 160, 175))
            painter.drawText(
                QRectF(0, y_pos - 8, left_m - 8, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{g:.1f}",
            )

        # 3. Touch Threshold Cutoff Marker Line (Dashed Horizontal Line)
        thresh_y = top_m + plot_h * (1.0 - self._threshold)
        painter.setPen(QPen(QColor(255, 205, 50, 220), 1.6, Qt.DashLine))
        painter.drawLine(QPointF(left_m, thresh_y), QPointF(left_m + plot_w, thresh_y))

        pct_val = int(round(self._threshold * 100))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(255, 215, 60))
        painter.drawText(
            QRectF(left_m + plot_w - 100, thresh_y - 14, 95, 12),
            Qt.AlignRight,
            f"Cutoff: {pct_val}%",
        )

        # 4. Finger Probability Curves
        x_step = plot_w / float(self._max_points - 1)

        for f in FINGERS:
            color = FINGER_COLORS_RGB[f]
            pts = list(self._history[f])

            path = QPainterPath()
            first = True
            for i, p in enumerate(pts):
                px = left_m + i * x_step
                py = top_m + plot_h * (1.0 - p)
                if first:
                    path.moveTo(px, py)
                    first = False
                else:
                    path.lineTo(px, py)

            painter.setPen(QPen(color, 1.8))
            painter.drawPath(path)

        # 5. Top Legend
        leg_x = left_m
        leg_y = 10.0
        dot_r = 4.0
        for f in FINGERS:
            c = FINGER_COLORS_RGB[f]
            painter.setBrush(c)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(leg_x + dot_r, leg_y + dot_r), dot_r, dot_r)

            painter.setPen(QColor(220, 220, 230))
            painter.drawText(QRectF(leg_x + 14, leg_y - 2, 50, 16), Qt.AlignLeft, f)
            leg_x += 68.0
