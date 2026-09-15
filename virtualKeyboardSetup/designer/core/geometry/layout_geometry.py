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

    eps = 1e-4
    return not (
        ax2 + gap_mm - eps <= bx1
        or bx2 + gap_mm - eps <= ax1
        or ay2 + gap_mm - eps <= by1
        or by2 + gap_mm - eps <= ay1
    )


def rect_within_interior(
    rect: tuple[float, float, float, float],
    config: AppConfig | None = None,
) -> bool:
    """Checks if rect (x, y, w, h in mm) fits completely inside paper interior active region."""
    if config is None:
        config = AppConfig()

    x, y, w, h = rect
    eps = 1e-4
    return (
        x + eps >= config.interior_x_min
        and y + eps >= config.interior_y_min
        and x + w - eps <= config.interior_x_max
        and y + h - eps <= config.interior_y_max
    )


def snap_to_grid(value_mm: float, grid_size_mm: float = 5.0) -> float:
    """Snaps value in mm to nearest grid increment."""
    if grid_size_mm <= 0:
        return value_mm
    return round(value_mm / grid_size_mm) * grid_size_mm


def validate_layout_geometry(layout, config: AppConfig) -> tuple[bool, str]:
    """Validate layout geometry for active surface bounds, button/marker overlaps, and unique AprilTag marker IDs."""
    buttons = layout.buttons

    print(f"\n==================== [LAYOUT VALIDATION LOG] ====================")
    print(f"Project Name: '{layout.project_name}' | Paper Size: {config.paper_width_mm}x{config.paper_height_mm} mm")
    print(f"Active Surface Bounds: X:[{config.interior_x_min:.1f} .. {config.interior_x_max:.1f}] Y:[{config.interior_y_min:.1f} .. {config.interior_y_max:.1f}]")
    print(f"Buttons Count: {len(buttons)} | Custom Markers Count: {len(layout.custom_markers)} (use_custom={layout.use_custom_markers})")

    # 1. Validate Buttons
    for i, b in enumerate(buttons):
        rect = b.rect_tuple
        print(f" -> Button #{i+1} [{b.id}] text='{b.text}' rect=(x={b.x_mm:.2f}, y={b.y_mm:.2f}, w={b.width_mm:.2f}, h={b.height_mm:.2f})")

        if not rect_within_interior(rect, config):
            err = f"Key button '{b.text or b.id}' (x={b.x_mm:.1f}, y={b.y_mm:.1f}) is placed outside active surface zone."
            print(f"    [VALIDATION FAILED] {err}")
            print(f"=================================================================\n")
            return False, err

        for j, other_b in enumerate(buttons):
            if i != j:
                if rects_overlap(rect, other_b.rect_tuple, gap_mm=config.button_min_gap_mm):
                    err = f"Key button '{b.text or b.id}' overlaps or is too close to button '{other_b.text or other_b.id}' (minimum {config.button_min_gap_mm:.1f} mm gap required)."
                    print(f"    [VALIDATION FAILED] {err}")
                    print(f"    Button A: {rect}")
                    print(f"    Button B: {other_b.rect_tuple}")
                    print(f"=================================================================\n")
                    return False, err

        if layout.use_custom_markers:
            btn_marker_gap = max(config.button_min_gap_mm, config.marker_min_gap_mm)
            for m in layout.custom_markers:
                if rects_overlap(rect, m.rect_tuple, gap_mm=btn_marker_gap):
                    err = f"Key button '{b.text or b.id}' overlaps or is too close to custom AprilTag marker #{m.id} (minimum {btn_marker_gap:.1f} mm gap required)."
                    print(f"    [VALIDATION FAILED] {err}")
                    print(f"=================================================================\n")
                    return False, err

    # 2. Validate Custom Markers
    if layout.use_custom_markers:
        custom_ids = [m.id for m in layout.custom_markers]
        if len(custom_ids) != len(set(custom_ids)):
            err = "Duplicate custom AprilTag marker IDs found. Each custom marker must have a unique Tag ID."
            print(f"    [VALIDATION FAILED] {err}")
            print(f"=================================================================\n")
            return False, err

        if config.show_outer_markers:
            from core.geometry.marker_generator import generate_marker_layout
            outer_markers = generate_marker_layout(config)
            outer_ids = {m.id for m in outer_markers}
            for m_id in custom_ids:
                if m_id in outer_ids:
                    err = f"Custom AprilTag marker ID #{m_id} conflicts with outer perimeter marker ID #{m_id}."
                    print(f"    [VALIDATION FAILED] {err}")
                    print(f"=================================================================\n")
                    return False, err

            for m in layout.custom_markers:
                for om in outer_markers:
                    if rects_overlap(m.rect_tuple, om.rect_tuple, gap_mm=config.marker_min_gap_mm):
                        err = f"Custom AprilTag marker #{m.id} overlaps or is too close to outer perimeter marker #{om.id} (minimum {config.marker_min_gap_mm:.1f} mm gap required)."
                        print(f"    [VALIDATION FAILED] {err}")
                        print(f"=================================================================\n")
                        return False, err

        for i, m1 in enumerate(layout.custom_markers):
            m_rect = m1.rect_tuple
            print(f" -> Custom Marker [{m1.id}] center=(x={m1.x_mm:.2f}, y={m1.y_mm:.2f}) size={m1.size_mm:.2f}")
            if not rect_within_interior(m_rect, config):
                err = f"Custom AprilTag marker #{m1.id} (x={m1.x_mm:.1f}, y={m1.y_mm:.1f}) is placed outside active surface region."
                print(f"    [VALIDATION FAILED] {err}")
                print(f"=================================================================\n")
                return False, err

            for j, m2 in enumerate(layout.custom_markers):
                if i != j and rects_overlap(m_rect, m2.rect_tuple, gap_mm=config.marker_min_gap_mm):
                    err = f"Custom AprilTag marker #{m1.id} overlaps or is too close to custom marker #{m2.id} (minimum {config.marker_min_gap_mm:.1f} mm gap required)."
                    print(f"    [VALIDATION FAILED] {err}")
                    print(f"=================================================================\n")
                    return False, err

    print(" [VALIDATION SUCCESS] All buttons and markers are valid and non-overlapping.")
    print(f"=================================================================\n")
    return True, ""




