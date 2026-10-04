"""
services/camera_discovery.py

Probes available video capture devices on the system.
On Linux scans /dev/video* devices cleanly without stderr driver spam.
Uses QtMultimedia QMediaDevices for instant hardware enumeration without locking OpenCV camera handles on the UI thread.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
from PySide6.QtMultimedia import QMediaDevices

from utils.logger import setup_logger

logger = setup_logger("CameraDiscovery")


@contextmanager
def suppress_c_stderr():
    """Suppress C-level stderr output to silence noisy C++ OpenCV/V4L2 drivers."""
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


# Cache discovery results for 5 seconds to prevent GUI thread freezing
_DISCOVERY_CACHE: list[CameraInfo] | None = None
_LAST_DISCOVERY_TIME: float = 0.0


def discover_cameras(max_index: int = 6, force_refresh: bool = False) -> list[CameraInfo]:
    """
    Probes video capture devices and returns those that produce a readable frame.
    Uses Qt QMediaDevices to determine exact camera count instantly without touching hardware pins.

    Parameters
    ----------
    max_index : Upper bound for index scan (exclusive).
    force_refresh : If True, bypasses the 5-second discovery cache.

    Returns
    -------
    List of CameraInfo objects sorted by device index.
    """
    global _DISCOVERY_CACHE, _LAST_DISCOVERY_TIME

    now = time.perf_counter()
    if not force_refresh and _DISCOVERY_CACHE is not None and (now - _LAST_DISCOVERY_TIME) < 5.0:
        return _DISCOVERY_CACHE

    # Configure OpenCV logging
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass

    # 1. Try Qt QMediaDevices for instant hardware enumeration without hardware locks
    try:
        qt_devices = QMediaDevices.videoInputs()
        if qt_devices:
            found: list[CameraInfo] = []
            for idx, dev in enumerate(qt_devices):
                if idx < max_index:
                    desc = dev.description() or f"Camera {idx}"
                    found.append(
                        CameraInfo(
                            index=idx,
                            name=f"{desc} (/{idx})",
                            width=640,
                            height=480,
                            fps=30.0,
                        )
                    )
            if found:
                logger.info("Discovered %d camera(s) via QMediaDevices instantly.", len(found))
                _DISCOVERY_CACHE = found
                _LAST_DISCOVERY_TIME = time.perf_counter()
                return found
    except Exception as exc:
        logger.debug("QMediaDevices discovery fallback: %s", exc)

    # 2. Linux fallback /dev/video*
    candidates: list[int] = []
    dev_videos: list[str] = []
    if sys.platform.startswith("linux"):
        dev_videos = sorted(glob.glob("/dev/video*"))
        for dev in dev_videos:
            try:
                idx = int(dev.replace("/dev/video", ""))
                if idx not in candidates and idx < max_index:
                    candidates.append(idx)
            except ValueError:
                pass

    if not candidates:
        candidates = list(range(max_index))

    found: list[CameraInfo] = []
    seen: set[int] = set()
    consecutive_fails = 0

    for idx in sorted(candidates):
        if idx in seen:
            continue
        seen.add(idx)

        with suppress_c_stderr():
            if sys.platform.startswith("linux") and dev_videos:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            elif sys.platform == "win32":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                cap.release()
                consecutive_fails += 1
                if sys.platform == "win32" and consecutive_fails >= 1:
                    break
                continue

            ret, _ = cap.read()
            if not ret:
                cap.release()
                consecutive_fails += 1
                if sys.platform == "win32" and consecutive_fails >= 1:
                    break
                continue

            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            consecutive_fails = 0

        name_str = f"Camera {idx}  (/dev/video{idx})" if sys.platform.startswith("linux") else f"Camera {idx}"
        info = CameraInfo(
            index=idx,
            name=name_str,
            width=w,
            height=h,
            fps=fps,
        )
        found.append(info)
        logger.info("Found camera: %s  [%dx%d @ %.1f fps]", info.name, w, h, fps)

    if not found:
        logger.warning("No usable cameras detected.")

    result = sorted(found, key=lambda c: c.index)
    _DISCOVERY_CACHE = result
    _LAST_DISCOVERY_TIME = time.perf_counter()
    return result
