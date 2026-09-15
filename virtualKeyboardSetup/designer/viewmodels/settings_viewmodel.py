"""
Settings View Model (MVVM Architecture).
Manages application configuration, user customizable paper surface dimensions,
grid snap settings, preset paper sizes, and visual theme options.
"""

from PySide6.QtCore import QObject, Signal
from config.app_config import AppConfig


class SettingsViewModel(QObject):
    # Signals for View Layer Binding
    paper_dimensions_changed = Signal(float, float)  # (paper_width_mm, paper_height_mm)
    theme_changed = Signal(str)                      # ("light" or "dark")
    status_message = Signal(str, str)                # (title, content)

    PRESETS = [
        ("A4 Landscape (297 x 210 mm)", 297.0, 210.0),
        ("A4 Portrait (210 x 297 mm)", 210.0, 297.0),
        ("A3 Landscape (420 x 297 mm)", 420.0, 297.0),
        ("Letter Landscape (279 x 216 mm)", 279.0, 216.0),
        ("Custom Dimensions", None, None),
    ]

    def __init__(self, config: AppConfig, parent: QObject | None = None):
        super().__init__(parent)
        self.config = config

    @property
    def paper_width_mm(self) -> float:
        return self.config.paper_width_mm

    @property
    def paper_height_mm(self) -> float:
        return self.config.paper_height_mm

    @property
    def marker_size_mm(self) -> float:
        return self.config.marker_size_mm

    @property
    def show_outer_markers(self) -> bool:
        return self.config.show_outer_markers

    @property
    def grid_snap_enabled(self) -> bool:
        return self.config.grid_snap_enabled

    @property
    def grid_size_mm(self) -> float:
        return self.config.grid_size_mm

    @property
    def paper_margin_mm(self) -> float:
        return self.config.paper_margin_mm

    @property
    def button_min_gap_mm(self) -> float:
        return self.config.button_min_gap_mm

    @property
    def button_corner_radius_mm(self) -> float:
        return self.config.button_corner_radius_mm

    @property
    def button_stroke_width_mm(self) -> float:
        return self.config.button_stroke_width_mm

    @property
    def db_path(self) -> str:
        return self.config.db_path

    def get_preset_dimensions(self, index: int) -> tuple[float, float] | None:
        """Return (width_mm, height_mm) for a preset index, or None if Custom."""
        if 0 <= index < len(self.PRESETS):
            _, w, h = self.PRESETS[index]
            if w is not None and h is not None:
                return (w, h)
        return None

    def apply_settings(
        self,
        width_mm: float,
        height_mm: float,
        marker_size_mm: float,
        show_outer_markers: bool,
        grid_snap: bool,
        grid_size: float,
        paper_margin_mm: float = 10.0,
        button_min_gap_mm: float = 5.0,
        button_corner_radius_mm: float = 0.0,
        button_stroke_width_mm: float = 1.5,
    ) -> None:
        """Apply user configuration updates to AppConfig and notify subscribers."""
        self.config.paper_width_mm = width_mm
        self.config.paper_height_mm = height_mm
        self.config.marker_size_mm = marker_size_mm
        self.config.show_outer_markers = show_outer_markers
        self.config.grid_snap_enabled = grid_snap
        self.config.grid_size_mm = grid_size
        self.config.paper_margin_mm = paper_margin_mm
        self.config.button_min_gap_mm = button_min_gap_mm
        self.config.button_corner_radius_mm = button_corner_radius_mm
        self.config.button_stroke_width_mm = button_stroke_width_mm

        self.paper_dimensions_changed.emit(width_mm, height_mm)
        self.status_message.emit(
            "Settings Updated",
            f"Updated paper layout settings: margin ({paper_margin_mm:.1f} mm), button gap ({button_min_gap_mm:.1f} mm), stroke width ({button_stroke_width_mm:.1f} mm), and surface dimensions.",
        )


