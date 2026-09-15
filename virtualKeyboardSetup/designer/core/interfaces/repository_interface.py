"""
Project Repository Interface (ISP & DIP).
Defines persistence operations for paper layout projects.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from core.models.paper_layout import PaperLayoutModel


class IProjectRepository(ABC):
    @abstractmethod
    def save(self, filepath_or_id: str | Path, layout: PaperLayoutModel) -> None:
        """Save paper layout to persistence store."""
        pass

    @abstractmethod
    def load(self, filepath_or_id: str | Path) -> PaperLayoutModel:
        """Load paper layout from persistence store."""
        pass
