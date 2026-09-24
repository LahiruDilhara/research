"""
core/pipeline/apriltag_tracker.py

Fiducial homography tracker and button overlay engine.
Directly wraps HomographyEngine replicating designer/analyzer/homography_engine.py:
- Uses OpenCV ArUcoDetector (cv2.aruco.ArucoDetector).
- Maps detected marker corners [TL, TR, BR, BL] directly to XML marker corners in mm.
- Solves planar homography H using cv2.findHomography(cv2.RANSAC, 5.0).
- Employs cv2.perspectiveTransform for all pixel <-> mm coordinate transformations.
- Projects internal layout buttons, corner anchor tick marks, center crosshairs,
  and paper boundaries directly onto camera frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.constants import APRILTAG_MIN_MARKERS
from core.layout.layout_parser import LayoutData
from core.pipeline.homography_engine import HomographyEngine
from utils.logger import setup_logger

logger = setup_logger("AprilTagTracker")


class AprilTagTracker:
    """
    Per-frame AprilTag detection and H matrix computation.
    Uses the exact engine and rendering from designer/analyzer/homography_engine.py.
    """

    def __init__(
        self,
        layout: LayoutData,
        min_markers: int = APRILTAG_MIN_MARKERS,
        smoothing_alpha: float = 0.75,
    ) -> None:
        self._layout = layout
        self._min_markers = min_markers
        self._alpha = smoothing_alpha

        # Core CV & Homography engine from designer/analyzer
        self._engine = HomographyEngine(layout, ransac_thresh_mm=5.0)

        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._markers_used: int = 0
        self._info: dict = {"success": False, "detected_markers_count": 0, "identified_buttons_count": 0}

        logger.info(
            "AprilTagTracker initialized with HomographyEngine (%d markers, %d buttons, family=%s)",
            len(layout.markers),
            len(layout.buttons),
            layout.marker_family,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def H(self) -> np.ndarray | None:
        """Current homography (camera pixel -> mm-space), or None if not yet valid."""
        return self._H if self.is_valid else None

    @property
    def H_inv(self) -> np.ndarray | None:
        """Inverse H (mm-space -> camera pixel), or None if not yet valid."""
        return self._H_inv if self.is_valid else None

    @property
    def is_valid(self) -> bool:
        """True if tracking is actively locked with sufficient markers and valid H."""
        return self._H is not None and self._markers_used >= self._min_markers

    @property
    def markers_used(self) -> int:
        return self._markers_used

    @property
    def info(self) -> dict:
        return self._info

    def update(self, frame_bgr: np.ndarray) -> bool:
        """
        Detects AprilTags and computes homography matrix H (image pixels -> paper mm).
        Replicates exact algorithm from designer/analyzer/homography_engine.py.
        """
        H, info = self._engine.compute_homography(frame_bgr)
        self._info = info
        detected_count = info.get("detected_markers_count", 0)
        self._markers_used = detected_count

        if H is None or detected_count < self._min_markers:
            self._H = None
            self._H_inv = None
            self._markers_used = 0
            return False

        self._H = H
        try:
            self._H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self._H = None
            self._H_inv = None
            self._markers_used = 0
            return False

        return True

    def pixel_to_mm(self, px: float, py: float) -> tuple[float, float] | None:
        """Map camera pixel (px, py) to paper mm using cv2.perspectiveTransform."""
        if self._H is None:
            return None
        return self._engine.pixel_to_mm(px, py, self._H)

    def mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float] | None:
        """Map paper mm (x_mm, y_mm) to camera pixel using cv2.perspectiveTransform."""
        if self._H_inv is None:
            return None
        return self._engine.mm_to_pixel(x_mm, y_mm, self._H_inv)

    def find_button_at(self, x_mm: float, y_mm: float):
        """Find layout button containing (x_mm, y_mm)."""
        return self._engine.find_button_at(x_mm, y_mm)

    def process_touch_event(self, frame: np.ndarray, pixel_x: float, pixel_y: float) -> dict:
        """Process a direct pixel touch event replicating analyzer/homography_engine.py."""
        return self._engine.process_touch_event(frame, pixel_x, pixel_y)

    def reset_smoothing(self) -> None:
        """Reset temporal homography history."""
        self._engine.reset_smoothing()

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        layout: LayoutData | None = None,
        active_button_id: str | None = None,
        active_button_ids: set[str] | list[str] | str | None = None,
    ) -> np.ndarray:
        """
        Draw detected tag outlines, paper boundary, and key bounding boxes onto frame_bgr in-place.
        Replicates designer/analyzer/homography_engine.py overlay renderer.
        Supports highlighting multiple simultaneous active buttons.
        """
        if active_button_ids is None and active_button_id is not None:
            active_button_ids = active_button_id

        if not self.is_valid or self._H is None:
            return frame_bgr

        # Draw detected marker outlines & IDs
        corners, ids, annotated_frame = self._engine.detect_markers(frame_bgr)
        # Draw paper boundary, buttons, center crosshairs, corner tick marks, and active button highlights
        return self._engine.draw_paper_and_buttons_overlay(
            annotated_frame, self._H, active_button_ids=active_button_ids
        )
