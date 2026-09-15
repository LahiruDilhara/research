"""
Paper Layout Domain Model.
Aggregate root representing paper surface dimensions, markers, and contained buttons.
"""

from dataclasses import dataclass, field
from config.constants import (
    DEFAULT_PAPER_WIDTH_MM,
    DEFAULT_PAPER_HEIGHT_MM,
    PAPER_MARGIN_MM,
    MARKER_SIZE_MM,
    MARKER_SPACING_MM,
    MARKER_FAMILY,
    DEFAULT_MARKER_MIN_GAP_RATIO,
    BUTTON_STROKE_WIDTH_MM,
    BUTTON_CORNER_RADIUS_MM,
    BUTTON_MIN_WIDTH_MM,
    BUTTON_MIN_HEIGHT_MM,
    BUTTON_MIN_GAP_MM,
    DEFAULT_FONT_SIZE_PT,
    INTERIOR_BUFFER_MM,
    DEFAULT_GRID_SIZE_MM,
)
from .button_model import ButtonModel
from .marker_model import MarkerModel


@dataclass
class PaperLayoutModel:
    project_name: str = "My Paper Keyboard"
    paper_width_mm: float = DEFAULT_PAPER_WIDTH_MM
    paper_height_mm: float = DEFAULT_PAPER_HEIGHT_MM
    paper_margin_mm: float = PAPER_MARGIN_MM
    marker_size_mm: float = MARKER_SIZE_MM
    marker_spacing_mm: float = MARKER_SPACING_MM
    marker_family: str = MARKER_FAMILY
    marker_min_gap_ratio: float = DEFAULT_MARKER_MIN_GAP_RATIO
    show_outer_markers: bool = True
    use_custom_markers: bool = False
    button_stroke_width_mm: float = BUTTON_STROKE_WIDTH_MM
    button_corner_radius_mm: float = BUTTON_CORNER_RADIUS_MM
    button_min_width_mm: float = BUTTON_MIN_WIDTH_MM
    button_min_height_mm: float = BUTTON_MIN_HEIGHT_MM
    button_min_gap_mm: float = BUTTON_MIN_GAP_MM
    default_font_size_pt: int = DEFAULT_FONT_SIZE_PT
    interior_buffer_mm: float = INTERIOR_BUFFER_MM
    grid_snap_enabled: bool = True
    grid_size_mm: float = DEFAULT_GRID_SIZE_MM
    buttons: list[ButtonModel] = field(default_factory=list)
    markers: list[MarkerModel] = field(default_factory=list)
    custom_markers: list[MarkerModel] = field(default_factory=list)

    def add_button(self, button: ButtonModel) -> None:
        self.buttons.append(button)

    def remove_button(self, button_id: str) -> None:
        self.buttons = [b for b in self.buttons if b.id != button_id]

    def get_button(self, button_id: str) -> ButtonModel | None:
        for b in self.buttons:
            if b.id == button_id:
                return b
        return None

    def add_custom_marker(self, marker: MarkerModel) -> None:
        self.custom_markers.append(marker)
        self.use_custom_markers = True

    def remove_custom_marker(self, marker_id: int) -> None:
        self.custom_markers = [m for m in self.custom_markers if m.id != marker_id]
        if not self.custom_markers:
            self.use_custom_markers = False

    def clear(self) -> None:
        self.buttons.clear()
        self.custom_markers.clear()
        self.use_custom_markers = False
