"""
services/layout_service.py

Thin service layer around LayoutParser that caches the last loaded layout
and provides path management helpers.
"""

from __future__ import annotations

from pathlib import Path

from core.layout.layout_parser import LayoutData, LayoutParser
from utils.logger import setup_logger

logger = setup_logger("LayoutService")

_parser = LayoutParser()


class LayoutService:
    """Loads and caches a LayoutData from an XML file."""

    def __init__(self) -> None:
        self._layout: LayoutData | None = None
        self._path: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, xml_path: str) -> LayoutData:
        """Parse the XML and cache the result. Raises on parse errors."""
        layout = _parser.parse(xml_path)
        self._layout = layout
        self._path = xml_path
        logger.info(
            "Layout loaded: %s  (%d buttons, %d markers)",
            Path(xml_path).name, len(layout.buttons), len(layout.markers),
        )
        return layout

    @property
    def layout(self) -> LayoutData | None:
        return self._layout

    @property
    def loaded_path(self) -> str:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return self._layout is not None
