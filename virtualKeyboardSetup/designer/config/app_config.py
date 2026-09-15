"""
Application Configuration Manager.
Loads settings from .env file and provides strongly typed properties for user-configurable
paper dimensions and fixed system layout constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from .constants import (
    DEFAULT_PAPER_WIDTH_MM,
    DEFAULT_PAPER_HEIGHT_MM,
    PAPER_MARGIN_MM,
    MARKER_SIZE_MM,
    MARKER_SPACING_MM,
    DEFAULT_MARKER_MIN_GAP_RATIO,
    MARKER_FAMILY,
    BUTTON_STROKE_WIDTH_MM,
    BUTTON_CORNER_RADIUS_MM,
    BUTTON_MIN_WIDTH_MM,
    BUTTON_MIN_HEIGHT_MM,
    BUTTON_MIN_GAP_MM,
    DEFAULT_FONT_SIZE_PT,
    INTERIOR_BUFFER_MM,
    DEFAULT_GRID_SIZE_MM,
)


class AppConfig:
    def __init__(self, env_path: str | Path | None = None):
        if env_path and Path(env_path).exists():
            load_dotenv(env_path)
        else:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
            else:
                load_dotenv()

        # User Configurable Paper Surface Dimensions & AprilTag Marker Size
        self._paper_width_mm = float(os.getenv("PAPER_WIDTH_MM", DEFAULT_PAPER_WIDTH_MM))
        self._paper_height_mm = float(os.getenv("PAPER_HEIGHT_MM", DEFAULT_PAPER_HEIGHT_MM))
        self._paper_margin_mm = float(os.getenv("PAPER_MARGIN_MM", PAPER_MARGIN_MM))
        self._marker_size_mm = float(os.getenv("MARKER_SIZE_MM", MARKER_SIZE_MM))

        # Button & Grid Settings
        self._button_stroke_width_mm = float(os.getenv("BUTTON_STROKE_WIDTH_MM", BUTTON_STROKE_WIDTH_MM))
        self._button_corner_radius_mm = float(os.getenv("BUTTON_CORNER_RADIUS_MM", BUTTON_CORNER_RADIUS_MM))
        self._button_min_width_mm = float(os.getenv("BUTTON_MIN_WIDTH_MM", BUTTON_MIN_WIDTH_MM))
        self._button_min_height_mm = float(os.getenv("BUTTON_MIN_HEIGHT_MM", BUTTON_MIN_HEIGHT_MM))
        self._button_min_gap_mm = float(os.getenv("BUTTON_MIN_GAP_MM", BUTTON_MIN_GAP_MM))
        self._default_font_size_pt = int(os.getenv("DEFAULT_FONT_SIZE_PT", DEFAULT_FONT_SIZE_PT))

        self._db_path = os.getenv("DB_PATH", "designer_data.db")
        self._app_title = os.getenv("APP_TITLE", "Virtual Keyboard Paper Layout Designer")
        self._app_theme = os.getenv("APP_THEME", "Dark")
        self._grid_snap_enabled = os.getenv("GRID_SNAP_ENABLED", "true").lower() in ("true", "1", "yes")
        # AprilTag Marker Settings
        self._marker_min_gap_ratio = float(os.getenv("MARKER_MIN_GAP_RATIO", DEFAULT_MARKER_MIN_GAP_RATIO))
        self._show_outer_markers = os.getenv("SHOW_OUTER_MARKERS", "true").lower() in ("true", "1", "yes")
        self._grid_size_mm = float(os.getenv("GRID_SIZE_MM", DEFAULT_GRID_SIZE_MM))

    # User Configurable Properties
    @property
    def paper_width_mm(self) -> float:
        return self._paper_width_mm

    @paper_width_mm.setter
    def paper_width_mm(self, value: float) -> None:
        self._paper_width_mm = max(50.0, float(value))

    @property
    def paper_height_mm(self) -> float:
        return self._paper_height_mm

    @paper_height_mm.setter
    def paper_height_mm(self, value: float) -> None:
        self._paper_height_mm = max(50.0, float(value))

    @property
    def paper_margin_mm(self) -> float:
        return self._paper_margin_mm

    @paper_margin_mm.setter
    def paper_margin_mm(self, value: float) -> None:
        self._paper_margin_mm = max(0.0, float(value))

    @property
    def marker_size_mm(self) -> float:
        return self._marker_size_mm

    @marker_size_mm.setter
    def marker_size_mm(self, value: float) -> None:
        self._marker_size_mm = max(5.0, float(value))

    @property
    def marker_min_gap_ratio(self) -> float:
        return self._marker_min_gap_ratio

    @property
    def show_outer_markers(self) -> bool:
        return self._show_outer_markers

    @show_outer_markers.setter
    def show_outer_markers(self, value: bool) -> None:
        self._show_outer_markers = bool(value)

    @property
    def marker_spacing_mm(self) -> float:
        return MARKER_SPACING_MM

    @property
    def marker_family(self) -> str:
        return MARKER_FAMILY

    @property
    def button_stroke_width_mm(self) -> float:
        return self._button_stroke_width_mm

    @button_stroke_width_mm.setter
    def button_stroke_width_mm(self, value: float) -> None:
        self._button_stroke_width_mm = max(0.1, float(value))


    @property
    def button_corner_radius_mm(self) -> float:
        return self._button_corner_radius_mm

    @button_corner_radius_mm.setter
    def button_corner_radius_mm(self, value: float) -> None:
        self._button_corner_radius_mm = max(0.0, float(value))

    @property
    def button_min_width_mm(self) -> float:
        return self._button_min_width_mm

    @property
    def button_min_height_mm(self) -> float:
        return self._button_min_height_mm

    @property
    def button_min_gap_mm(self) -> float:
        return self._button_min_gap_mm

    @button_min_gap_mm.setter
    def button_min_gap_mm(self, value: float) -> None:
        self._button_min_gap_mm = max(0.0, float(value))


    @property
    def default_font_size_pt(self) -> int:
        return self._default_font_size_pt

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def app_title(self) -> str:
        return self._app_title

    @property
    def app_theme(self) -> str:
        return self._app_theme

    @property
    def grid_snap_enabled(self) -> bool:
        return self._grid_snap_enabled

    @grid_snap_enabled.setter
    def grid_snap_enabled(self, value: bool) -> None:
        self._grid_snap_enabled = value

    @property
    def grid_size_mm(self) -> float:
        return self._grid_size_mm

    @grid_size_mm.setter
    def grid_size_mm(self, value: float) -> None:
        self._grid_size_mm = max(1.0, float(value))


    # Derived Zone Properties
    @property
    def marker_ring_outer_min(self) -> float:
        return self.paper_margin_mm

    @property
    def marker_ring_outer_max_x(self) -> float:
        return self.paper_width_mm - self.paper_margin_mm

    @property
    def marker_ring_outer_max_y(self) -> float:
        return self.paper_height_mm - self.paper_margin_mm

    @property
    def marker_ring_inner_min(self) -> float:
        return self.paper_margin_mm + self.marker_size_mm

    @property
    def marker_ring_inner_max_x(self) -> float:
        return self.paper_width_mm - self.marker_ring_inner_min

    @property
    def marker_ring_inner_max_y(self) -> float:
        return self.paper_height_mm - self.marker_ring_inner_min

    @property
    def interior_x_min(self) -> float:
        return self.marker_ring_inner_min + INTERIOR_BUFFER_MM

    @property
    def interior_y_min(self) -> float:
        return self.marker_ring_inner_min + INTERIOR_BUFFER_MM

    @property
    def interior_x_max(self) -> float:
        return self.paper_width_mm - self.interior_x_min

    @property
    def interior_y_max(self) -> float:
        return self.paper_height_mm - self.interior_y_min

    @property
    def interior_width_mm(self) -> float:
        return self.interior_x_max - self.interior_x_min

    @property
    def interior_height_mm(self) -> float:
        return self.interior_y_max - self.interior_y_min
