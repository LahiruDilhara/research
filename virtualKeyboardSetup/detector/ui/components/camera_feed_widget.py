"""
ui/components/camera_feed_widget.py

QLabel subclass that efficiently renders OpenCV BGR frames as QPixmap.
Maintains aspect ratio and scales to fill available space.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class CameraFeedWidget(QLabel):
    """Displays live camera frames. Call update_frame(bgr_array) from the main thread."""

    frame_clicked = Signal(int, int)  # Emits (pixel_x, pixel_y) in raw frame coordinates

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setText("Waiting for camera...")
        self.setStyleSheet(
            "background-color: #0D0D11; color: #4A4A5A; font-size: 14px;"
        )
        self._current_pixmap: QPixmap | None = None
        self._raw_frame_size: tuple[int, int] | None = None  # (width, height)
        self._rendered_rect: tuple[int, int, int, int] | None = None  # (offset_x, offset_y, w, h)

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        """Convert a BGR numpy array to QPixmap and display it, preserving aspect ratio."""
        if bgr_frame is None or bgr_frame.size == 0:
            return

        h, w, ch = bgr_frame.shape
        self._raw_frame_size = (w, h)
        # Convert BGR → RGB (contiguous buffer for QImage)
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q_img)
        self._render_current()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_current()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self._raw_frame_size and self._rendered_rect:
            click_x = event.position().x()
            click_y = event.position().y()
            off_x, off_y, ren_w, ren_h = self._rendered_rect
            if off_x <= click_x <= off_x + ren_w and off_y <= click_y <= off_y + ren_h:
                raw_w, raw_h = self._raw_frame_size
                px = int((click_x - off_x) / ren_w * raw_w)
                py = int((click_y - off_y) / ren_h * raw_h)
                self.frame_clicked.emit(px, py)

    def _render_current(self) -> None:
        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            target_size = self.size()
            if target_size.width() > 10 and target_size.height() > 10:
                scaled = self._current_pixmap.scaled(
                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(scaled)
                # Compute rendered pixmap rectangle for click mapping
                off_x = (target_size.width() - scaled.width()) // 2
                off_y = (target_size.height() - scaled.height()) // 2
                self._rendered_rect = (off_x, off_y, scaled.width(), scaled.height())
