"""
Export Service Interface (OCP & DIP).
Defines standard export interface for XML, PDF, and image outcome renderers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from core.models.paper_layout import PaperLayoutModel
from config.app_config import AppConfig


class IExportService(ABC):
    @abstractmethod
    def export(self, filepath: str | Path, layout: PaperLayoutModel, config: AppConfig) -> None:
        """Export layout model to target output file."""
        pass
