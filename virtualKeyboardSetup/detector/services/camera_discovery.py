"""
services/camera_discovery.py

Probes available video capture devices on the system.
On Linux scans /dev/video* devices cleanly without stderr driver spam.
Returns a list of CameraInfo dicts for the UI camera selection view.
"""

from __future__ import annotations

import glob
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import cv2

from utils.logger import setup_logger

logger = setup_logger("CameraDiscovery")


@contextmanager
def suppress_c_stderr():
    """Suppress C-level stderr output to silence noisy C++ OpenCV/V4L2/OBSENSOR drivers."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
    except Exception:
        yield


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int
    fps: float


def discover_cameras(max_index: int = 10) -> list[CameraInfo]:
    """
    Probes video capture devices and returns those that produce a readable frame.

    Parameters
    ----------
    max_index : Upper bound for index scan (exclusive) when /dev/video* is not available.

    Returns
    -------
    List of CameraInfo objects sorted by device index.
    """
    # Configure OpenCV logging
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass

    candidates: list[int] = []
    dev_videos = sorted(glob.glob("/dev/video*"))
    if dev_videos:
        for dev in dev_videos:
            try:
                idx = int(dev.replace("/dev/video", ""))
                if idx not in candidates:
                    candidates.append(idx)
            except ValueError:
                pass
    else:
        candidates = list(range(max_index))

    found: list[CameraInfo] = []
    seen: set[int] = set()

    for idx in sorted(candidates):
        if idx in seen:
            continue
        seen.add(idx)

        with suppress_c_stderr():
            # On Linux try V4L2 first to avoid FFMPEG/OBSENSOR fallback spam
            if dev_videos:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                cap.release()
                continue

            ret, _ = cap.read()
            if not ret:
                cap.release()
                continue

            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()

        info = CameraInfo(
            index=idx,
            name=f"Camera {idx}  (/dev/video{idx})",
            width=w,
            height=h,
            fps=fps,
        )
        found.append(info)
        logger.info("Found camera: %s  [%dx%d @ %.1f fps]", info.name, w, h, fps)

    if not found:
        logger.warning("No usable cameras detected.")

    return sorted(found, key=lambda c: c.index)

