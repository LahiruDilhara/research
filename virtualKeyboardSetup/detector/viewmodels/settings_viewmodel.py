"""
viewmodels/settings_viewmodel.py

ViewModel for the settings view following MVVM and SOLID principles.
Encapsulates settings presentation state, validation, and coordinates with SettingsService.
Supports persistence to both global .env and per-layout XML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from config.app_config import AppConfig
from config.constants import (
    FINGERTIP_VELOCITY_THRESHOLD,
    HAND_MOVEMENT_THRESHOLD,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    QUALITY_MAX_SCORE_DROP,
    QUALITY_MIN_AVG_SCORE,
    QUALITY_MIN_FRAME_SCORE,
    TARGET_FPS,
    TOUCH_PROBABILITY_THRESHOLD,
)
from services.settings_service import SettingsService
from utils.logger import setup_logger

logger = setup_logger("SettingsViewModel")


class SettingsViewModel(QObject):
    """ViewModel managing user-adjustable settings state and actions."""

    settings_saved = Signal(str)
    settings_loaded = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        env_path: str | Path,
        config: AppConfig | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._service = SettingsService(env_path)
        self._layout_xml_path: str | None = None

        # Check if config has last_xml_path and load any XML settings
        if self._config and self._config.last_xml_path:
            self.set_layout_xml_path(self._config.last_xml_path)

    # ── Layout XML Association ─────────────────────────────────────────────────

    @property
    def layout_xml_path(self) -> str | None:
        return self._layout_xml_path

    def set_layout_xml_path(self, xml_path: str | None) -> None:
        """
        Associates an active layout XML file.
        Loads any <DetectorSettings> already saved in the XML into config.
        """
        if not xml_path:
            self._layout_xml_path = None
            return

        self._layout_xml_path = str(xml_path)
        p = Path(xml_path)
        if p.exists():
            xml_settings = self._service.load_xml_settings(p)
            if xml_settings and self._config is not None:
                self._config.apply_dict(xml_settings)
                logger.info(
                    "Applied %d settings from layout XML %s into AppConfig",
                    len(xml_settings),
                    p.name,
                )
                self.settings_loaded.emit()

    # ── State Accessors ────────────────────────────────────────────────────────

    @property
    def target_fps(self) -> float:
        if self._config is not None:
            return self._config.target_fps
        return float(TARGET_FPS)

    @property
    def touch_threshold(self) -> float:
        if self._config is not None:
            return self._config.touch_threshold
        return float(TOUCH_PROBABILITY_THRESHOLD)

    @property
    def fingertip_velocity_threshold(self) -> float:
        if self._config is not None:
            return self._config.fingertip_velocity_threshold
        return float(FINGERTIP_VELOCITY_THRESHOLD)

    @property
    def hand_movement_threshold(self) -> float:
        if self._config is not None:
            return self._config.hand_movement_threshold
        return float(HAND_MOVEMENT_THRESHOLD)

    @property
    def mediapipe_min_detection_confidence(self) -> float:
        if self._config is not None:
            return self._config.mediapipe_min_detection_confidence
        return float(MEDIAPIPE_MIN_DETECTION_CONFIDENCE)

    @property
    def mediapipe_min_presence_confidence(self) -> float:
        if self._config is not None:
            return self._config.mediapipe_min_presence_confidence
        return float(MEDIAPIPE_MIN_PRESENCE_CONFIDENCE)

    @property
    def mediapipe_min_tracking_confidence(self) -> float:
        if self._config is not None:
            return self._config.mediapipe_min_tracking_confidence
        return float(MEDIAPIPE_MIN_TRACKING_CONFIDENCE)

    @property
    def quality_min_avg_score(self) -> float:
        if self._config is not None:
            return self._config.quality_min_avg_score
        return float(QUALITY_MIN_AVG_SCORE)

    @property
    def quality_min_frame_score(self) -> float:
        if self._config is not None:
            return self._config.quality_min_frame_score
        return float(QUALITY_MIN_FRAME_SCORE)

    @property
    def quality_max_score_drop(self) -> float:
        if self._config is not None:
            return self._config.quality_max_score_drop
        return float(QUALITY_MAX_SCORE_DROP)

    @property
    def plugins_dir(self) -> str:
        if self._config is not None:
            return self._config.plugins_dir
        return "ai_model_plugins"

    @property
    def env_file_name(self) -> str:
        return self._service.env_path.name

    # ── Actions ────────────────────────────────────────────────────────────────

    def save_settings(
        self,
        fps: float,
        touch_threshold: float,
        velocity_threshold: float,
        hand_movement_threshold: float,
        detection_confidence: float,
        presence_confidence: float,
        tracking_confidence: float,
        quality_min_avg: float,
        quality_min_frame: float,
        quality_max_drop: float,
        plugins_dir: str,
    ) -> bool:
        """
        Validates and persists updated settings.
        Updates AppConfig in memory and writes changes to storage (.env and XML).
        """
        clean_plugins_dir = plugins_dir.strip() or "ai_model_plugins"

        updates: dict[str, Any] = {
            "TARGET_FPS": f"{fps:.1f}",
            "TOUCH_THRESHOLD": f"{touch_threshold:.2f}",
            "FINGERTIP_VELOCITY_THRESHOLD": f"{velocity_threshold:.4f}",
            "MIN_KINETIC_SPEED_THRESHOLD": f"{velocity_threshold:.4f}",
            "HAND_MOVEMENT_THRESHOLD": f"{hand_movement_threshold:.4f}",
            "MEDIAPIPE_MIN_DETECTION_CONFIDENCE": f"{detection_confidence:.2f}",
            "MEDIAPIPE_MIN_PRESENCE_CONFIDENCE": f"{presence_confidence:.2f}",
            "MEDIAPIPE_MIN_TRACKING_CONFIDENCE": f"{tracking_confidence:.2f}",
            "QUALITY_MIN_AVG_SCORE": f"{quality_min_avg:.2f}",
            "QUALITY_MIN_FRAME_SCORE": f"{quality_min_frame:.2f}",
            "QUALITY_MAX_SCORE_DROP": f"{quality_max_drop:.2f}",
            "AI_MODEL_PLUGINS_DIR": clean_plugins_dir,
            "PLUGINS_DIR": clean_plugins_dir,
        }

        # Determine XML path to save into
        xml_save_path = self._layout_xml_path
        if not xml_save_path and self._config and self._config.last_xml_path:
            xml_save_path = self._config.last_xml_path

        try:
            self._service.save_settings(updates, xml_path=xml_save_path)

            if self._config is not None:
                self._config.set_target_fps(fps)
                self._config.set_touch_threshold(touch_threshold)
                self._config.set_fingertip_velocity_threshold(velocity_threshold)
                self._config.set_hand_movement_threshold(hand_movement_threshold)
                self._config.set_mediapipe_min_detection_confidence(detection_confidence)
                self._config.set_mediapipe_min_presence_confidence(presence_confidence)
                self._config.set_mediapipe_min_tracking_confidence(tracking_confidence)
                self._config.set_quality_min_avg_score(quality_min_avg)
                self._config.set_quality_min_frame_score(quality_min_frame)
                self._config.set_quality_max_score_drop(quality_max_drop)
                self._config.set_plugins_dir(clean_plugins_dir)

            targets = [self.env_file_name]
            if xml_save_path and Path(xml_save_path).exists():
                targets.append(Path(xml_save_path).name)

            msg = f"Settings saved to {', '.join(targets)}. Applied immediately to live detector."
            logger.info("Successfully saved settings: %s", updates)
            self.settings_saved.emit(msg)
            return True
        except Exception as exc:
            err_msg = f"Failed to save settings: {exc}"
            logger.error(err_msg)
            self.error_occurred.emit(err_msg)
            return False

    def restore_defaults(self, persist: bool = True) -> bool:
        """
        Restores all settings to their canonical system defaults.
        Optionally persists the restored defaults to storage (.env and active layout XML).
        """
        default_fps = float(TARGET_FPS)
        default_touch_threshold = float(TOUCH_PROBABILITY_THRESHOLD)
        default_velocity_threshold = float(FINGERTIP_VELOCITY_THRESHOLD)
        default_hand_movement_threshold = float(HAND_MOVEMENT_THRESHOLD)
        default_detection_confidence = float(MEDIAPIPE_MIN_DETECTION_CONFIDENCE)
        default_presence_confidence = float(MEDIAPIPE_MIN_PRESENCE_CONFIDENCE)
        default_tracking_confidence = float(MEDIAPIPE_MIN_TRACKING_CONFIDENCE)
        default_quality_min_avg = float(QUALITY_MIN_AVG_SCORE)
        default_quality_min_frame = float(QUALITY_MIN_FRAME_SCORE)
        default_quality_max_drop = float(QUALITY_MAX_SCORE_DROP)
        default_plugins_dir = "ai_model_plugins"

        if persist:
            success = self.save_settings(
                fps=default_fps,
                touch_threshold=default_touch_threshold,
                velocity_threshold=default_velocity_threshold,
                hand_movement_threshold=default_hand_movement_threshold,
                detection_confidence=default_detection_confidence,
                presence_confidence=default_presence_confidence,
                tracking_confidence=default_tracking_confidence,
                quality_min_avg=default_quality_min_avg,
                quality_min_frame=default_quality_min_frame,
                quality_max_drop=default_quality_max_drop,
                plugins_dir=default_plugins_dir,
            )
            self.settings_loaded.emit()
            return success
        else:
            if self._config is not None:
                self._config.set_target_fps(default_fps)
                self._config.set_touch_threshold(default_touch_threshold)
                self._config.set_fingertip_velocity_threshold(default_velocity_threshold)
                self._config.set_hand_movement_threshold(default_hand_movement_threshold)
                self._config.set_mediapipe_min_detection_confidence(default_detection_confidence)
                self._config.set_mediapipe_min_presence_confidence(default_presence_confidence)
                self._config.set_mediapipe_min_tracking_confidence(default_tracking_confidence)
                self._config.set_quality_min_avg_score(default_quality_min_avg)
                self._config.set_quality_min_frame_score(default_quality_min_frame)
                self._config.set_quality_max_score_drop(default_quality_max_drop)
                self._config.set_plugins_dir(default_plugins_dir)
            self.settings_loaded.emit()
            return True
