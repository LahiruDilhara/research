"""
Layout Geometry Validation Logic.
Provides rect overlap detection, interior boundary enforcement, and grid snapping math.
"""

from config.app_config import AppConfig


def rects_overlap(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
    gap_mm: float = 0.0,
) -> bool:
    """
    Axis-aligned overlap test for two rects (x_mm, y_mm, width_mm, height_mm).
    gap_mm: minimum required gap between button outer edges.
    """
    ax1, ay1 = rect_a[0], rect_a[1]
    ax2, ay2 = rect_a[0] + rect_a[2], rect_a[1] + rect_a[3]

    bx1, by1 = rect_b[0], rect_b[1]
    bx2, by2 = rect_b[0] + rect_b[2], rect_b[1] + rect_b[3]

    return not (
        ax2 + gap_mm <= bx1
        or bx2 + gap_mm <= ax1
        or ay2 + gap_mm <= by1
        or by2 + gap_mm <= ay1
    )


def rect_within_interior(
    rect: tuple[float, float, float, float],
    config: AppConfig | None = None,
) -> bool:
    """Checks if rect (x, y, w, h in mm) fits completely inside paper interior active region."""
    if config is None:
        config = AppConfig()

    x, y, w, h = rect
    return (
        x >= config.interior_x_min
        and y >= config.interior_y_min
        and x + w <= config.interior_x_max
        and y + h <= config.interior_y_max
    )


def snap_to_grid(value_mm: float, grid_size_mm: float = 5.0) -> float:
    """Snaps value in mm to nearest grid increment."""
    if grid_size_mm <= 0:
        return value_mm
    return round(value_mm / grid_size_mm) * grid_size_mm


def validate_layout_geometry(layout, config: AppConfig) -> bool:
    """Validate entire layout geometry for active surface zone bounds and mutual button/marker overlaps."""
    buttons = layout.buttons
    for i, b in enumerate(buttons):
        rect = b.rect_tuple
        if not rect_within_interior(rect, config):
            return False

        for j, other_b in enumerate(buttons):
            if i != j and rects_overlap(rect, other_b.rect_tuple, config.button_min_gap_mm):
                return False

        if layout.use_custom_markers:
            for m in layout.custom_markers:
                if rects_overlap(rect, m.rect_tuple, 2.0):
                    return False

    if layout.use_custom_markers:
        for m in layout.custom_markers:
            m_rect = m.rect_tuple
            if (
                m_rect[0] < config.paper_margin_mm
                or m_rect[1] < config.paper_margin_mm
                or m_rect[0] + m_rect[2] > config.paper_width_mm - config.paper_margin_mm
                or m_rect[1] + m_rect[3] > config.paper_height_mm - config.paper_margin_mm
            ):
                return False

    return True

