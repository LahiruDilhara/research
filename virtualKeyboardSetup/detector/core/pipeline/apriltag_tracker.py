"""
core/pipeline/apriltag_tracker.py

AprilTag fiducial homography tracker.

Uses dt_apriltags (pupil-apriltags) to detect tag36h11 markers in each camera frame,
then calls cv2.findHomography to compute H that maps camera pixel coordinates to
paper mm-space as defined by the LayoutData markers.

Applies exponential smoothing to H for stability across frames.
Minimum 1 detected marker is required before H is considered valid.
"""

from __future__ import annotations

import cv2
import numpy as np
from dt_apriltags import Detector

from config.constants import (
    APRILTAG_FAMILY,
    APRILTAG_MIN_MARKERS,
    APRILTAG_NTHREADS,
    APRILTAG_SMOOTHING_ALPHA,
)
from core.layout.layout_parser import LayoutData
from utils.logger import setup_logger

logger = setup_logger("AprilTagTracker")


class AprilTagTracker:
    """
    Per-frame AprilTag detection and H matrix computation.

    The resulting H maps camera-pixel coordinates → mm-space coordinates defined
    in the XML layout. To draw layout overlays on the camera frame, use H_inv (inverse).
    """

    def __init__(
        self,
        layout: LayoutData,
        min_markers: int = APRILTAG_MIN_MARKERS,
        smoothing_alpha: float = APRILTAG_SMOOTHING_ALPHA,
    ) -> None:
        self._layout = layout
        self._min_markers = min_markers
        self._alpha = smoothing_alpha
        self._detector = Detector(
            families=APRILTAG_FAMILY,
            nthreads=APRILTAG_NTHREADS,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
        )
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._markers_used: int = 0
        # Build id → MarkerData lookup once
        self._marker_lookup = layout.marker_by_id
        logger.info(
            "AprilTagTracker ready, %d markers in layout, family=%s",
            len(layout.markers),
            APRILTAG_FAMILY,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def H(self) -> np.ndarray | None:
        """Current homography (camera pixel → mm-space), or None if not yet valid."""
        return self._H

    @property
    def H_inv(self) -> np.ndarray | None:
        """Inverse H (mm-space → camera pixel), or None if not yet valid."""
        return self._H_inv

    @property
    def is_valid(self) -> bool:
        return self._H is not None

    @property
    def markers_used(self) -> int:
        return self._markers_used

    def update(self, frame_bgr: np.ndarray) -> bool:
        """
        Detect AprilTags in frame_bgr and update H matrix.

        Returns True if H is valid (at least min_markers detected and H solved).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tags = self._detector.detect(gray, estimate_tag_pose=False)

        image_pts: list[list[float]] = []
        paper_pts_mm: list[list[float]] = []

        for tag in tags:
            marker = self._marker_lookup.get(tag.tag_id)
            if marker is None:
                continue
            # tag.corners shape (4, 2): bottom-left, bottom-right, top-right, top-left
            image_pts.extend(tag.corners.tolist())
            paper_pts_mm.extend(marker.corners_mm)

        self._markers_used = len(image_pts) // 4

        if self._markers_used < self._min_markers:
            return self.is_valid  # keep previous H if any

        img_np = np.array(image_pts, dtype=np.float32)
        pap_np = np.array(paper_pts_mm, dtype=np.float32)

        H_new, mask = cv2.findHomography(img_np, pap_np, cv2.RANSAC, 5.0)
        if H_new is None:
            return self.is_valid

        if self._H is None:
            self._H = H_new
        else:
            self._H = self._alpha * self._H + (1.0 - self._alpha) * H_new

        try:
            self._H_inv = np.linalg.inv(self._H)
        except np.linalg.LinAlgError:
            self._H_inv = None

        return True

    def annotate_frame(self, frame_bgr: np.ndarray, layout: LayoutData) -> None:
        """
        Draw detected tag outlines and key bounding boxes onto frame_bgr in-place.
        Called by CameraWorker before emitting frame_ready.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tags = self._detector.detect(gray, estimate_tag_pose=False)

        # Draw detected tag outlines
        for tag in tags:
            corners = tag.corners.astype(int)
            cv2.polylines(frame_bgr, [corners.reshape(-1, 1, 2)], True, (0, 220, 80), 2)
            cx, cy = tag.center.astype(int)
            cv2.putText(
                frame_bgr, f"#{tag.tag_id}", (cx - 14, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 80), 1,
            )

        if self._H_inv is None:
            return

        # Draw key bounding boxes projected onto camera frame
        for btn in layout.buttons:
            corners_mm = [
                (btn.x_mm,     btn.y_mm),
                (btn.x_max_mm, btn.y_mm),
                (btn.x_max_mm, btn.y_max_mm),
                (btn.x_mm,     btn.y_max_mm),
            ]
            px_pts = [self._mm_to_pixel(x, y) for x, y in corners_mm]
            if any(p is None for p in px_pts):
                continue
            pts = np.array([(int(x), int(y)) for x, y in px_pts], dtype=np.int32)
            cv2.polylines(frame_bgr, [pts.reshape(-1, 1, 2)], True, (0, 159, 239), 1)

            # Label at button centre
            center_px = self._mm_to_pixel(btn.center_x_mm, btn.center_y_mm)
            if center_px:
                cv2.putText(
                    frame_bgr, btn.label[:8],
                    (int(center_px[0]) - 16, int(center_px[1]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 159, 239), 1,
                )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _mm_to_pixel(
        self, x_mm: float, y_mm: float
    ) -> tuple[float, float] | None:
        """Map an mm-space point back to camera-pixel space using H_inv."""
        if self._H_inv is None:
            return None
        p = self._H_inv @ np.array([x_mm, y_mm, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-9:
            return None
        return (p[0] / p[2], p[1] / p[2])

    def pixel_to_mm(
        self, px: float, py: float
    ) -> tuple[float, float] | None:
        """Map a camera-pixel point to mm-space using H."""
        if self._H is None:
            return None
        p = self._H @ np.array([px, py, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-9:
            return None
        return (p[0] / p[2], p[1] / p[2])
