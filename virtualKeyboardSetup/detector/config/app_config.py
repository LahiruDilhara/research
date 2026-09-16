"""
config/app_config.py

Typed configuration loader. Reads values from a .env file (or environment variables)
and exposes them as strongly typed properties consumed by all layers.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .constants import (
    TARGET_FPS,
    WINDOW_SIZE,
    SHIFT_SIZE,
    MEDIAPIPE_MODEL_FILENAME,
    APRILTAG_MIN_MARKERS,
    APRILTAG_SMOOTHING_ALPHA,
    TOUCH_PROBABILITY_THRESHOLD,
)


class AppConfig:
    """Loads .env and provides typed runtime settings."""

    def __init__(self, env_path: str | Path | None = None) -> None:
        if env_path and Path(env_path).exists():
            load_dotenv(env_path)
        else:
            # Try sibling .env next to main.py
            fallback = Path(__file__).resolve().parent.parent / ".env"
            if fallback.exists():
                load_dotenv(fallback)
            else:
                load_dotenv()

        self._app_title   = os.getenv("APP_TITLE", "Virtual Keyboard Detector")
        self._app_theme   = os.getenv("APP_THEME", "Dark")
        self._camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self._target_fps  = float(os.getenv("TARGET_FPS", str(TARGET_FPS)))
        self._window_size = int(os.getenv("WINDOW_SIZE", str(WINDOW_SIZE)))
        self._shift_size  = int(os.getenv("SHIFT_SIZE", str(SHIFT_SIZE)))
        self._mediapipe_model_path = os.getenv(
            "MEDIAPIPE_MODEL_PATH", MEDIAPIPE_MODEL_FILENAME
        )
        self._plugins_dir = os.getenv("PLUGINS_DIR", "plugins")
        self._touch_threshold = float(
            os.getenv("TOUCH_THRESHOLD", str(TOUCH_PROBABILITY_THRESHOLD))
        )
        self._apriltag_min_markers = int(
            os.getenv("APRILTAG_MIN_MARKERS", str(APRILTAG_MIN_MARKERS))
        )
        self._apriltag_smoothing = float(
            os.getenv("APRILTAG_SMOOTHING_ALPHA", str(APRILTAG_SMOOTHING_ALPHA))
        )

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def app_title(self) -> str:
        return self._app_title

    @property
    def app_theme(self) -> str:
        return self._app_theme

    @property
    def camera_index(self) -> int:
        return self._camera_index

    @property
    def target_fps(self) -> float:
        return self._target_fps

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def shift_size(self) -> int:
        return self._shift_size

    @property
    def mediapipe_model_path(self) -> str:
        return self._mediapipe_model_path

    @property
    def plugins_dir(self) -> str:
        return self._plugins_dir

    @property
    def touch_threshold(self) -> float:
        return self._touch_threshold

    @property
    def apriltag_min_markers(self) -> int:
        return self._apriltag_min_markers

    @property
    def apriltag_smoothing(self) -> float:
        return self._apriltag_smoothing
