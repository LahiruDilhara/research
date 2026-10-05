"""
services/camera_discovery.py

Probes available video capture devices on the system.
On Windows uses DirectShow (CAP_DSHOW) with thread-timeout protection to prevent
driver hangs and FFMPEG/MSMF fallback deadlocks.
On Linux scans /dev/video* devices cleanly using V4L2 without driver spam.
Caches discovered cameras so UI views do not freeze while re-probing hardware.
"""

from __future__ import annotations

import glob
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import cv2

from utils.logger import setup_logger

logger = setup_logger("CameraDiscovery")

# Global cache to avoid repeated hardware probing on the GUI thread
_CACHED_CAMERAS: list[CameraInfo] | None = None


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


def _open_camera_capture(index: int) -> cv2.VideoCapture:
    """Open camera with the appropriate OS-specific native backend without unbacked fallback."""
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    dev_videos = glob.glob("/dev/video*")
    if dev_videos:
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def _probe_single_camera(index: int, timeout_sec: float = 1.0) -> CameraInfo | None:
    """
    Probes a camera index inside a worker thread with timeout protection.
    Prevents buggy or unresponsive drivers from hanging the process.
    """
    result: list[CameraInfo | None] = [None]
    dev_videos = sorted(glob.glob("/dev/video*"))

    def _worker() -> None:
        with suppress_c_stderr():
            cap = _open_camera_capture(index)
            if not cap.isOpened():
                cap.release()
                return

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()

            name = f"Camera {index}" + (f" (/dev/video{index})" if dev_videos else "")
            result[0] = CameraInfo(
                index=index,
                name=name,
                width=w,
                height=h,
                fps=fps,
            )

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    if th.is_alive():
        logger.warning(
            "Camera %d probe timed out after %.1fs (driver unresponsive). Skipping device.",
            index,
            timeout_sec,
        )
        return None

    return result[0]


def is_camera_available(index: int) -> bool:
    """Quickly check if a camera index is available and responsive."""
    global _CACHED_CAMERAS
    if _CACHED_CAMERAS is not None:
        for c in _CACHED_CAMERAS:
            if c.index == index:
                return True

    info = _probe_single_camera(index, timeout_sec=1.0)
    return info is not None


def discover_cameras(
    max_index: int = 4,
    force_refresh: bool = False,
    preferred_index: int | None = None,
) -> list[CameraInfo]:
    """
    Probes video capture devices and returns those that produce a readable frame.
    Results are cached to ensure instant UI rendering without repeated hardware probes.

    Parameters
    ----------
    max_index : Upper bound for index scan (exclusive) when /dev/video* is not available.
    force_refresh : Force re-scanning hardware even if cache exists.
    preferred_index : Prioritize probing this camera index first.

    Returns
    -------
    List of CameraInfo objects sorted by device index.
    """
    global _CACHED_CAMERAS
    if _CACHED_CAMERAS is not None and not force_refresh:
        return list(_CACHED_CAMERAS)

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

    # Prioritize preferred index if specified
    if preferred_index is not None and preferred_index in candidates:
        candidates.remove(preferred_index)
        candidates.insert(0, preferred_index)

    found: list[CameraInfo] = []
    seen: set[int] = set()

    for idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)

        info = _probe_single_camera(idx, timeout_sec=1.2)
        if info is not None:
            found.append(info)
            logger.info("Found camera: %s  [%dx%d @ %.1f fps]", info.name, info.width, info.height, info.fps)
        else:
            # On Windows without /dev/video*, non-existent devices fail immediately with CAP_DSHOW.
            # If an index > 0 fails to open, stop scanning higher indices unless we have specific candidates.
            if not dev_videos and idx >= 2 and preferred_index != idx:
                break

    if not found:
        logger.warning("No usable cameras detected.")

    _CACHED_CAMERAS = sorted(found, key=lambda c: c.index)
    return list(_CACHED_CAMERAS)
