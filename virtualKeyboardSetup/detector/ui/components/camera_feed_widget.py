"""
ui/components/camera_feed_widget.py

QLabel subclass that efficiently renders OpenCV BGR frames as QPixmap.
Maintains aspect ratio and scales to fill available space.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class CameraFeedWidget(QLabel):
    """Displays live camera frames. Call update_frame(bgr_array) from the main thread."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0D0D11;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setText("Waiting for camera...")
        self.setStyleSheet(
            "background-color: #0D0D11; color: #4A4A5A; font-size: 14px;"
        )

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        """Convert a BGR numpy array to QPixmap and display it, preserving aspect ratio."""
        h, w, ch = bgr_frame.shape
        # Convert BGR → RGB
        rgb = bgr_frame[:, :, ::-1].copy()
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit while keeping aspect ratio
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
