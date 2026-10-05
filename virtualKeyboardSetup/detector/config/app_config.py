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
    TOUCH_ONSET_THRESHOLD,
    TOUCH_RELEASE_THRESHOLD,
    HAND_MOVEMENT_THRESHOLD,
    QUALITY_MIN_AVG_SCORE,
    QUALITY_MIN_FRAME_SCORE,
    QUALITY_MAX_SCORE_DROP,
    MIN_KINETIC_SPEED_THRESHOLD,
    FINGERTIP_VELOCITY_THRESHOLD,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    ONE_EURO_ENABLED,
    ONE_EURO_MIN_CUTOFF,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    FINGERTIP_OFFSET_ENABLED,
    FINGERTIP_FORWARD_OFFSET_MM,
    FINGERTIP_PHALANX_RATIO,
    FINGERTIP_EXTRA_OFFSET_MM,
    FINGERTIP_PAPER_ANGLE_FACTOR_MM,
    FINGERTIP_EXTRA_FRONT_MM,
    TOUCH_DEBOUNCE_COOLDOWN_S,
    PRINTED_MARKER_SIDE_WIDTH_MM,
    PRINTED_MARKER_WIDTH_MM,
    PRINTED_MARKER_HEIGHT_MM,
)


class AppConfig:
    """Loads .env and provides typed runtime settings."""

    def __init__(self, env_path: str | Path | None = None) -> None:
        if env_path and Path(env_path).exists():
            load_dotenv(env_path, override=True)
        else:
            # Try sibling .env next to main.py
            fallback = Path(__file__).resolve().parent.parent / ".env"
            if fallback.exists():
                load_dotenv(fallback, override=True)
            else:
                load_dotenv(override=True)

        self._app_title   = os.getenv("APP_TITLE", "Virtual Keyboard Detector")
        self._app_theme   = os.getenv("APP_THEME", "Dark")
        self._camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self._target_fps  = float(os.getenv("TARGET_FPS", str(TARGET_FPS)))
        self._window_size = int(os.getenv("WINDOW_SIZE", str(WINDOW_SIZE)))
        self._shift_size  = int(os.getenv("SHIFT_SIZE", str(SHIFT_SIZE)))
        self._mediapipe_model_path = os.getenv(
            "MEDIAPIPE_MODEL_PATH", MEDIAPIPE_MODEL_FILENAME
        )
        self._plugins_dir = os.getenv(
            "AI_MODEL_PLUGINS_DIR",
            os.getenv("PLUGINS_DIR", "ai_model_plugins"),
        )
        self._touch_threshold = float(
            os.getenv("TOUCH_THRESHOLD", str(TOUCH_PROBABILITY_THRESHOLD))
        )
        self._touch_onset_threshold = float(
            os.getenv("TOUCH_ONSET_THRESHOLD", str(TOUCH_ONSET_THRESHOLD))
        )
        self._touch_release_threshold = float(
            os.getenv("TOUCH_RELEASE_THRESHOLD", str(TOUCH_RELEASE_THRESHOLD))
        )
        self._hand_movement_threshold = float(
            os.getenv("HAND_MOVEMENT_THRESHOLD", str(HAND_MOVEMENT_THRESHOLD))
        )
        self._quality_min_avg_score = float(
            os.getenv("QUALITY_MIN_AVG_SCORE", str(QUALITY_MIN_AVG_SCORE))
        )
        self._quality_min_frame_score = float(
            os.getenv("QUALITY_MIN_FRAME_SCORE", str(QUALITY_MIN_FRAME_SCORE))
        )
        self._quality_max_score_drop = float(
            os.getenv("QUALITY_MAX_SCORE_DROP", str(QUALITY_MAX_SCORE_DROP))
        )
        self._min_kinetic_speed = float(
            os.getenv("MIN_KINETIC_SPEED_THRESHOLD", str(MIN_KINETIC_SPEED_THRESHOLD))
        )
        self._fingertip_velocity_threshold = float(
            os.getenv(
                "FINGERTIP_VELOCITY_THRESHOLD",
                os.getenv("MIN_KINETIC_SPEED_THRESHOLD", str(FINGERTIP_VELOCITY_THRESHOLD)),
            )
        )
        self._apriltag_min_markers = int(
            os.getenv("APRILTAG_MIN_MARKERS", str(APRILTAG_MIN_MARKERS))
        )
        self._apriltag_smoothing = float(
            os.getenv("APRILTAG_SMOOTHING_ALPHA", str(APRILTAG_SMOOTHING_ALPHA))
        )
        self._mediapipe_min_detection_confidence = float(
            os.getenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", str(MEDIAPIPE_MIN_DETECTION_CONFIDENCE))
        )
        self._mediapipe_min_presence_confidence = float(
            os.getenv("MEDIAPIPE_MIN_PRESENCE_CONFIDENCE", str(MEDIAPIPE_MIN_PRESENCE_CONFIDENCE))
        )
        self._mediapipe_min_tracking_confidence = float(
            os.getenv("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", str(MEDIAPIPE_MIN_TRACKING_CONFIDENCE))
        )
        self._one_euro_enabled = os.getenv("ONE_EURO_ENABLED", str(ONE_EURO_ENABLED)).lower() in ("true", "1", "yes")
        self._one_euro_min_cutoff = float(os.getenv("ONE_EURO_MIN_CUTOFF", str(ONE_EURO_MIN_CUTOFF)))
        self._one_euro_beta = float(os.getenv("ONE_EURO_BETA", str(ONE_EURO_BETA)))
        self._one_euro_d_cutoff = float(os.getenv("ONE_EURO_D_CUTOFF", str(ONE_EURO_D_CUTOFF)))

        self._fingertip_offset_enabled = (
            os.getenv("FINGERTIP_OFFSET_ENABLED", str(FINGERTIP_OFFSET_ENABLED)).lower()
            in ("true", "1", "yes")
        )
        self._fingertip_forward_offset_mm = float(
            os.getenv("FINGERTIP_FORWARD_OFFSET_MM", str(FINGERTIP_FORWARD_OFFSET_MM))
        )
        self._fingertip_phalanx_ratio = float(
            os.getenv("FINGERTIP_PHALANX_RATIO", str(FINGERTIP_PHALANX_RATIO))
        )
        self._fingertip_extra_offset_mm = float(
            os.getenv("FINGERTIP_EXTRA_OFFSET_MM", str(FINGERTIP_EXTRA_OFFSET_MM))
        )
        self._fingertip_paper_angle_factor_mm = float(
            os.getenv("FINGERTIP_PAPER_ANGLE_FACTOR_MM", str(FINGERTIP_PAPER_ANGLE_FACTOR_MM))
        )
        self._fingertip_extra_front_mm = float(
            os.getenv("FINGERTIP_EXTRA_FRONT_MM", str(FINGERTIP_EXTRA_FRONT_MM))
        )
        self._touch_debounce_cooldown_s = float(
            os.getenv("TOUCH_DEBOUNCE_COOLDOWN_S", str(TOUCH_DEBOUNCE_COOLDOWN_S))
        )
        self._printed_marker_side_width_mm = float(
            os.getenv(
                "PRINTED_MARKER_SIDE_WIDTH_MM",
                os.getenv("PRINTED_MARKER_WIDTH_MM", str(PRINTED_MARKER_SIDE_WIDTH_MM)),
            )
        )
        self._printed_marker_width_mm = self._printed_marker_side_width_mm
        self._printed_marker_height_mm = self._printed_marker_side_width_mm

        self._env_path = env_path or (Path(__file__).resolve().parent.parent / ".env")
        self._last_xml_path = os.getenv("LAST_XML_PATH", "")

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def last_xml_path(self) -> str:
        return self._last_xml_path

    def set_last_xml_path(self, path: str) -> None:
        """Persist the last chosen XML layout path into .env."""
        self._last_xml_path = str(path)
        os.environ["LAST_XML_PATH"] = str(path)
        env_file = Path(self._env_path)
        try:
            lines: list[str] = []
            found = False
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("LAST_XML_PATH="):
                        lines.append(f"LAST_XML_PATH={path}")
                        found = True
                    else:
                        lines.append(line)
            if not found:
                lines.append(f"LAST_XML_PATH={path}")
            env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    @property
    def app_title(self) -> str:
        return self._app_title

    @property
    def app_theme(self) -> str:
        return self._app_theme

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def set_camera_index(self, index: int) -> None:
        """Set the active camera index and save to .env if needed."""
        self._camera_index = int(index)
        os.environ["CAMERA_INDEX"] = str(index)
        env_file = Path(self._env_path)
        try:
            lines: list[str] = []
            found = False
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("CAMERA_INDEX="):
                        lines.append(f"CAMERA_INDEX={index}")
                        found = True
                    else:
                        lines.append(line)
            if not found:
                lines.append(f"CAMERA_INDEX={index}")
            env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

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
    def ai_model_plugins_dir(self) -> str:
        return self._plugins_dir

    @property
    def plugins_dir(self) -> str:
        return self._plugins_dir

    @property
    def touch_threshold(self) -> float:
        return self._touch_threshold

    @property
    def touch_onset_threshold(self) -> float:
        return self._touch_onset_threshold

    @property
    def touch_release_threshold(self) -> float:
        return self._touch_release_threshold

    @property
    def hand_movement_threshold(self) -> float:
        return self._hand_movement_threshold

    @property
    def quality_min_avg_score(self) -> float:
        return self._quality_min_avg_score

    @property
    def quality_min_frame_score(self) -> float:
        return self._quality_min_frame_score

    @property
    def quality_max_score_drop(self) -> float:
        return self._quality_max_score_drop

    @property
    def min_kinetic_speed(self) -> float:
        return self._min_kinetic_speed

    @property
    def fingertip_velocity_threshold(self) -> float:
        return self._fingertip_velocity_threshold

    @property
    def mediapipe_min_detection_confidence(self) -> float:
        return self._mediapipe_min_detection_confidence

    def set_mediapipe_min_detection_confidence(self, value: float) -> None:
        self._mediapipe_min_detection_confidence = float(value)

    @property
    def mediapipe_min_presence_confidence(self) -> float:
        return self._mediapipe_min_presence_confidence

    def set_mediapipe_min_presence_confidence(self, value: float) -> None:
        self._mediapipe_min_presence_confidence = float(value)

    @property
    def mediapipe_min_tracking_confidence(self) -> float:
        return self._mediapipe_min_tracking_confidence

    def set_mediapipe_min_tracking_confidence(self, value: float) -> None:
        self._mediapipe_min_tracking_confidence = float(value)

    def set_hand_movement_threshold(self, value: float) -> None:
        self._hand_movement_threshold = float(value)

    def set_quality_min_avg_score(self, value: float) -> None:
        self._quality_min_avg_score = float(value)

    def set_quality_min_frame_score(self, value: float) -> None:
        self._quality_min_frame_score = float(value)

    def set_quality_max_score_drop(self, value: float) -> None:
        self._quality_max_score_drop = float(value)

    def set_target_fps(self, value: float) -> None:
        self._target_fps = float(value)

    def set_touch_threshold(self, value: float) -> None:
        self._touch_threshold = float(value)
        self._touch_onset_threshold = float(value)
        self._touch_release_threshold = max(0.01, min(float(value) * 0.70, float(value) - 0.02))

    def set_plugins_dir(self, value: str) -> None:
        self._plugins_dir = str(value)

    def set_fingertip_velocity_threshold(self, value: float) -> None:
        self._fingertip_velocity_threshold = float(value)
        self._min_kinetic_speed = float(value)

    @property
    def one_euro_enabled(self) -> bool:
        return self._one_euro_enabled

    def set_one_euro_enabled(self, value: bool) -> None:
        self._one_euro_enabled = bool(value)

    @property
    def one_euro_min_cutoff(self) -> float:
        return self._one_euro_min_cutoff

    def set_one_euro_min_cutoff(self, value: float) -> None:
        self._one_euro_min_cutoff = float(value)

    @property
    def one_euro_beta(self) -> float:
        return self._one_euro_beta

    def set_one_euro_beta(self, value: float) -> None:
        self._one_euro_beta = float(value)

    @property
    def one_euro_d_cutoff(self) -> float:
        return self._one_euro_d_cutoff

    def set_one_euro_d_cutoff(self, value: float) -> None:
        self._one_euro_d_cutoff = float(value)

    def apply_dict(self, settings: dict[str, Any]) -> None:
        """Apply a dictionary of settings directly into in-memory properties."""
        if "TARGET_FPS" in settings:
            try:
                self.set_target_fps(float(settings["TARGET_FPS"]))
            except (ValueError, TypeError):
                pass
        if "TOUCH_THRESHOLD" in settings:
            try:
                self.set_touch_threshold(float(settings["TOUCH_THRESHOLD"]))
            except (ValueError, TypeError):
                pass
        if "FINGERTIP_VELOCITY_THRESHOLD" in settings:
            try:
                self.set_fingertip_velocity_threshold(float(settings["FINGERTIP_VELOCITY_THRESHOLD"]))
            except (ValueError, TypeError):
                pass
        elif "MIN_KINETIC_SPEED_THRESHOLD" in settings:
            try:
                self.set_fingertip_velocity_threshold(float(settings["MIN_KINETIC_SPEED_THRESHOLD"]))
            except (ValueError, TypeError):
                pass
        if "HAND_MOVEMENT_THRESHOLD" in settings:
            try:
                self.set_hand_movement_threshold(float(settings["HAND_MOVEMENT_THRESHOLD"]))
            except (ValueError, TypeError):
                pass
        if "QUALITY_MIN_AVG_SCORE" in settings:
            try:
                self.set_quality_min_avg_score(float(settings["QUALITY_MIN_AVG_SCORE"]))
            except (ValueError, TypeError):
                pass
        if "QUALITY_MIN_FRAME_SCORE" in settings:
            try:
                self.set_quality_min_frame_score(float(settings["QUALITY_MIN_FRAME_SCORE"]))
            except (ValueError, TypeError):
                pass
        if "QUALITY_MAX_SCORE_DROP" in settings:
            try:
                self.set_quality_max_score_drop(float(settings["QUALITY_MAX_SCORE_DROP"]))
            except (ValueError, TypeError):
                pass
        if "MEDIAPIPE_MIN_DETECTION_CONFIDENCE" in settings:
            try:
                self.set_mediapipe_min_detection_confidence(float(settings["MEDIAPIPE_MIN_DETECTION_CONFIDENCE"]))
            except (ValueError, TypeError):
                pass
        if "MEDIAPIPE_MIN_PRESENCE_CONFIDENCE" in settings:
            try:
                self.set_mediapipe_min_presence_confidence(float(settings["MEDIAPIPE_MIN_PRESENCE_CONFIDENCE"]))
            except (ValueError, TypeError):
                pass
        if "MEDIAPIPE_MIN_TRACKING_CONFIDENCE" in settings:
            try:
                self.set_mediapipe_min_tracking_confidence(float(settings["MEDIAPIPE_MIN_TRACKING_CONFIDENCE"]))
            except (ValueError, TypeError):
                pass
        if "ONE_EURO_ENABLED" in settings:
            val = settings["ONE_EURO_ENABLED"]
            if isinstance(val, bool):
                self.set_one_euro_enabled(val)
            else:
                self.set_one_euro_enabled(str(val).lower() in ("true", "1", "yes"))
        if "ONE_EURO_MIN_CUTOFF" in settings:
            try:
                self.set_one_euro_min_cutoff(float(settings["ONE_EURO_MIN_CUTOFF"]))
            except (ValueError, TypeError):
                pass
        if "ONE_EURO_BETA" in settings:
            try:
                self.set_one_euro_beta(float(settings["ONE_EURO_BETA"]))
            except (ValueError, TypeError):
                pass
        if "ONE_EURO_D_CUTOFF" in settings:
            try:
                self.set_one_euro_d_cutoff(float(settings["ONE_EURO_D_CUTOFF"]))
            except (ValueError, TypeError):
                pass
        if "AI_MODEL_PLUGINS_DIR" in settings:
            self.set_plugins_dir(str(settings["AI_MODEL_PLUGINS_DIR"]))
        elif "PLUGINS_DIR" in settings:
            self.set_plugins_dir(str(settings["PLUGINS_DIR"]))
        if "FINGERTIP_OFFSET_ENABLED" in settings:
            val = settings["FINGERTIP_OFFSET_ENABLED"]
            if isinstance(val, bool):
                self.set_fingertip_offset_enabled(val)
            else:
                self.set_fingertip_offset_enabled(str(val).lower() in ("true", "1", "yes"))
        if "FINGERTIP_FORWARD_OFFSET_MM" in settings:
            try:
                self.set_fingertip_forward_offset_mm(float(settings["FINGERTIP_FORWARD_OFFSET_MM"]))
            except (ValueError, TypeError):
                pass
        elif "FINGERTIP_EXTRA_OFFSET_MM" in settings:
            try:
                self.set_fingertip_forward_offset_mm(float(settings["FINGERTIP_EXTRA_OFFSET_MM"]))
            except (ValueError, TypeError):
                pass
        if "TOUCH_DEBOUNCE_COOLDOWN_S" in settings:
            try:
                self.set_touch_debounce_cooldown_s(float(settings["TOUCH_DEBOUNCE_COOLDOWN_S"]))
            except (ValueError, TypeError):
                pass
        if "PRINTED_MARKER_SIDE_WIDTH_MM" in settings:
            try:
                self.set_printed_marker_side_width_mm(float(settings["PRINTED_MARKER_SIDE_WIDTH_MM"]))
            except (ValueError, TypeError):
                pass
        elif "PRINTED_MARKER_WIDTH_MM" in settings:
            try:
                self.set_printed_marker_side_width_mm(float(settings["PRINTED_MARKER_WIDTH_MM"]))
            except (ValueError, TypeError):
                pass
        elif "PRINTED_MARKER_HEIGHT_MM" in settings:
            try:
                self.set_printed_marker_side_width_mm(float(settings["PRINTED_MARKER_HEIGHT_MM"]))
            except (ValueError, TypeError):
                pass

    @property
    def apriltag_min_markers(self) -> int:
        return self._apriltag_min_markers

    @property
    def apriltag_smoothing(self) -> float:
        return self._apriltag_smoothing

    @property
    def fingertip_offset_enabled(self) -> bool:
        return self._fingertip_offset_enabled

    def set_fingertip_offset_enabled(self, value: bool) -> None:
        self._fingertip_offset_enabled = bool(value)

    @property
    def fingertip_forward_offset_mm(self) -> float:
        return self._fingertip_forward_offset_mm

    def set_fingertip_forward_offset_mm(self, value: float) -> None:
        self._fingertip_forward_offset_mm = float(value)
        self._fingertip_extra_offset_mm = float(value)

    @property
    def touch_debounce_cooldown_s(self) -> float:
        return self._touch_debounce_cooldown_s

    def set_touch_debounce_cooldown_s(self, value: float) -> None:
        self._touch_debounce_cooldown_s = float(value)

    @property
    def fingertip_phalanx_ratio(self) -> float:
        return self._fingertip_phalanx_ratio

    @property
    def fingertip_extra_offset_mm(self) -> float:
        return self._fingertip_forward_offset_mm

    @property
    def fingertip_paper_angle_factor_mm(self) -> float:
        return self._fingertip_paper_angle_factor_mm

    @property
    def fingertip_extra_front_mm(self) -> float:
        return self._fingertip_extra_front_mm

    @property
    def printed_marker_side_width_mm(self) -> float:
        return self._printed_marker_side_width_mm

    def set_printed_marker_side_width_mm(self, value: float) -> None:
        val = max(0.0, float(value))
        self._printed_marker_side_width_mm = val
        self._printed_marker_width_mm = val
        self._printed_marker_height_mm = val

    @property
    def printed_marker_width_mm(self) -> float:
        return self._printed_marker_side_width_mm

    def set_printed_marker_width_mm(self, value: float) -> None:
        self.set_printed_marker_side_width_mm(value)

    @property
    def printed_marker_height_mm(self) -> float:
        return self._printed_marker_side_width_mm

    def set_printed_marker_height_mm(self, value: float) -> None:
        self.set_printed_marker_side_width_mm(value)

