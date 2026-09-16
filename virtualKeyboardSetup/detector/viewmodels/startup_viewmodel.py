"""
viewmodels/startup_viewmodel.py

ViewModel for the startup splash screen.
Coordinates: XML loading, model discovery, and routing to subsequent views.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.interfaces.touch_model import ModelEntry, ModelRegistry
from core.layout.layout_parser import LayoutData
from services.action_config_service import ActionConfigService
from services.layout_service import LayoutService
from services.model_discovery import ModelDiscoveryService
from utils.logger import setup_logger

logger = setup_logger("StartupViewModel")


class StartupViewModel(QObject):
    """Manages startup data: layout + model selection."""

    layout_loaded   = Signal(object)   # LayoutData
    models_ready    = Signal(list)     # list[ModelEntry]
    error_occurred  = Signal(str)

    def __init__(self, plugins_dir: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._layout_svc  = LayoutService()
        self._action_svc  = ActionConfigService()
        self._discover_svc = ModelDiscoveryService(plugins_dir)
        self._discovered   = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def discover_models(self) -> list[ModelEntry]:
        """Run model discovery (idempotent — skips if already done)."""
        if not self._discovered:
            self._discover_svc.discover()
            self._discovered = True
        entries = ModelRegistry.all_entries()
        self.models_ready.emit(entries)
        return entries

    def load_layout(self, xml_path: str) -> LayoutData | None:
        try:
            layout = self._layout_svc.load(xml_path)
            self.layout_loaded.emit(layout)
            return layout
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return None

    @property
    def layout(self) -> LayoutData | None:
        return self._layout_svc.layout

    @property
    def layout_path(self) -> str:
        return self._layout_svc.loaded_path

    def actions_configured(self, xml_path: str) -> bool:
        return ActionConfigService.sidecar_exists(xml_path)

    @property
    def action_config_service(self) -> ActionConfigService:
        return self._action_svc
