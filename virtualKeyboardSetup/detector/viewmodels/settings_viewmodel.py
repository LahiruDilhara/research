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
