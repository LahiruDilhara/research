"""
viewmodels/action_config_viewmodel.py

ViewModel for the key action configuration view.
Manages loading, editing, and saving per-key action mappings.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from core.action.action_executor import ActionData
from core.layout.layout_parser import ButtonData, LayoutData
from services.action_config_service import ActionConfigService
from utils.logger import setup_logger

logger = setup_logger("ActionConfigViewModel")


class ActionConfigViewModel(QObject):
    """Manages action config state for all buttons in the loaded layout."""

    config_saved  = Signal(str)   # sidecar path
    error_occurred = Signal(str)

    def __init__(
        self,
        layout: LayoutData,
        xml_path: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._layout   = layout
        self._xml_path = xml_path
        self._svc      = ActionConfigService()

        # Load existing config (or build defaults)
        button_ids = [b.id for b in layout.buttons]
        self._config: dict[str, ActionData] = self._svc.load(xml_path, button_ids)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def buttons(self) -> list[ButtonData]:
        return self._layout.buttons

    @property
    def config(self) -> dict[str, ActionData]:
        return dict(self._config)

    def get_action(self, button_id: str) -> ActionData:
        return self._config.get(button_id, ActionData(type="none", value=""))

    def set_action(self, button_id: str, action_type: str, value: str) -> None:
        self._config[button_id] = ActionData(type=action_type, value=value)

    def save(self) -> bool:
        try:
            self._svc.save(self._xml_path, self._config)
            self.config_saved.emit(self._svc.sidecar_path)
            return True
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return False

    @property
    def xml_path(self) -> str:
        return self._xml_path
