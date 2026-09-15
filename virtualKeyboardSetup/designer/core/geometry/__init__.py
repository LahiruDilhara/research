"""Geometry logic package."""
from .marker_generator import generate_marker_layout
from .layout_geometry import rects_overlap, rect_within_interior, snap_to_grid

__all__ = ["generate_marker_layout", "rects_overlap", "rect_within_interior", "snap_to_grid"]
