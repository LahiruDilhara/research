"""
viewmodels/camera_select_viewmodel.py

ViewModel for the camera selection view.
Runs camera discovery and tracks the user's selection.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from services.camera_discovery import CameraInfo, discover_cameras
from utils.logger import setup_logger

logger = setup_logger("CameraSelectViewModel")


class CameraSelectViewModel(QObject):
    """Discovers cameras and tracks selection."""

    cameras_ready  = Signal(list)   # list[CameraInfo]
    error_occurred = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cameras: list[CameraInfo] = []
        self._selected_index: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh(self) -> list[CameraInfo]:
        try:
            self._cameras = discover_cameras()
            if self._cameras:
                self._selected_index = self._cameras[0].index
            self.cameras_ready.emit(self._cameras)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        return self._cameras

    def select(self, camera_index: int) -> None:
        self._selected_index = camera_index
        logger.info("Camera selected: %d", camera_index)

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def cameras(self) -> list[CameraInfo]:
        return self._cameras
