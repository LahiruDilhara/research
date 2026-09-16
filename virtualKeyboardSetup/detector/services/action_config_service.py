"""
services/action_config_service.py

Reads and writes per-key action mappings as a JSON sidecar file.

The sidecar file is placed alongside the XML layout file:
  my_layout.xml  →  my_layout.actions.json

Format
──────
{
  "button_1": {"type": "keystroke", "value": "a"},
  "button_2": {"type": "shortcut",  "value": "ctrl+c"},
  "button_3": {"type": "shell",     "value": "xdotool key XF86AudioPlay"},
  "button_4": {"type": "macro",     "value": "ctrl+shift+t:200:ctrl+v"},
  "button_5": {"type": "none",      "value": ""}
}
"""

from __future__ import annotations

import json
from pathlib import Path

from core.action.action_executor import ActionData
from utils.logger import setup_logger

logger = setup_logger("ActionConfigService")


class ActionConfigService:
    """Loads and saves key→action mappings as a JSON sidecar next to the XML."""

    VALID_TYPES = {"keystroke", "shortcut", "shell", "macro", "none"}

    def __init__(self) -> None:
        self._config: dict[str, ActionData] = {}
        self._sidecar_path: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, xml_path: str, button_ids: list[str]) -> dict[str, ActionData]:
        """
        Load the sidecar JSON alongside xml_path.
        Missing keys are filled with a 'none' ActionData.
        """
        sidecar = self._sidecar_for(xml_path)
        self._sidecar_path = str(sidecar)
        raw: dict[str, dict] = {}

        if sidecar.exists():
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                logger.info("Action config loaded: %s", sidecar.name)
            except Exception as exc:
                logger.warning("Could not parse action config (%s): %s", sidecar.name, exc)

        self._config = {}
        for btn_id in button_ids:
            entry = raw.get(btn_id, {})
            action_type = entry.get("type", "none")
            if action_type not in self.VALID_TYPES:
                action_type = "none"
            self._config[btn_id] = ActionData(
                type=action_type,
                value=str(entry.get("value", "")),
            )

        return dict(self._config)

    def save(self, xml_path: str, config: dict[str, ActionData]) -> None:
        """Write the action config dict to the sidecar JSON file."""
        sidecar = self._sidecar_for(xml_path)
        raw = {
            btn_id: {"type": a.type, "value": a.value}
            for btn_id, a in config.items()
        }
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
            logger.info("Action config saved: %s", sidecar.name)
        except Exception as exc:
            logger.error("Failed to save action config: %s", exc)
            raise

    def get_action(self, button_id: str) -> ActionData | None:
        return self._config.get(button_id)

    @property
    def config(self) -> dict[str, ActionData]:
        return dict(self._config)

    @property
    def sidecar_path(self) -> str:
        return self._sidecar_path

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sidecar_for(xml_path: str) -> Path:
        p = Path(xml_path)
        return p.with_suffix(".actions.json")

    @staticmethod
    def sidecar_exists(xml_path: str) -> bool:
        return ActionConfigService._sidecar_for(xml_path).exists()
