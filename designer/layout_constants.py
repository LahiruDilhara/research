"""
layout_constants.py

Fixed constants for the A4-landscape button layout designer, and the
logic that auto-generates the AprilTag marker ring around the border.
Markers are NOT user-placed — their positions are always derived from
these constants, so every exported layout is consistent and every
marker's real-world position is known exactly.
"""

import math

# =============================================================
# PAPER CONSTANTS (A4 landscape)
# =============================================================
PAPER_WIDTH_MM = 297.0
PAPER_HEIGHT_MM = 210.0

# =============================================================
# MARKER & MARGIN CONSTANTS
# =============================================================
PAPER_MARGIN_MM = 8.0        # gap from paper edge to the outer boundary of the marker ring (8mm)
MARKER_SIZE_MM = 15.0        # physical size of the printed AprilTag square (15mm)
MARKER_MARGIN_MM = PAPER_MARGIN_MM  # gap from paper edge to marker outer edge
MARKER_SPACING_MM = 40.0     # target center-to-center spacing of markers along each side
MARKER_FAMILY = "DICT_APRILTAG_36h11"

# =============================================================
# BUTTON CONSTANTS
# =============================================================
BUTTON_STROKE_WIDTH_MM = 1.5
BUTTON_CORNER_RADIUS_MM = 0
BUTTON_MIN_WIDTH_MM = 15.0
BUTTON_MIN_HEIGHT_MM = 10.0
BUTTON_DEFAULT_FONT_SIZE_PT = 14
BUTTON_MIN_GAP_MM = 10.0

# =============================================================
# DERIVED LAYOUT ZONES:
# Zone 1: Paper Outer Margin (0 to PAPER_MARGIN_MM)
# Zone 2: Marker Ring Zone (PAPER_MARGIN_MM to PAPER_MARGIN_MM + MARKER_SIZE_MM)
# Zone 3: Interior Safety Buffer (_INTERIOR_BUFFER_MM)
# Zone 4: Interior Printable Layout Area (INTERIOR_X_MIN..MAX, INTERIOR_Y_MIN..MAX)
# =============================================================
_INTERIOR_BUFFER_MM = 5.0

# Marker ring outer boundary
MARKER_RING_OUTER_MIN = PAPER_MARGIN_MM
MARKER_RING_OUTER_MAX_X = PAPER_WIDTH_MM - PAPER_MARGIN_MM
MARKER_RING_OUTER_MAX_Y = PAPER_HEIGHT_MM - PAPER_MARGIN_MM

# Marker ring inner boundary
MARKER_RING_INNER_MIN = PAPER_MARGIN_MM + MARKER_SIZE_MM
MARKER_RING_INNER_MAX_X = PAPER_WIDTH_MM - MARKER_RING_INNER_MIN
MARKER_RING_INNER_MAX_Y = PAPER_HEIGHT_MM - MARKER_RING_INNER_MIN

# Interior button layout zone
INTERIOR_X_MIN = MARKER_RING_INNER_MIN + _INTERIOR_BUFFER_MM
INTERIOR_Y_MIN = MARKER_RING_INNER_MIN + _INTERIOR_BUFFER_MM
INTERIOR_X_MAX = PAPER_WIDTH_MM - INTERIOR_X_MIN
INTERIOR_Y_MAX = PAPER_HEIGHT_MM - INTERIOR_Y_MIN


def generate_marker_layout():
    """
    Auto-generate the marker ring around the paper border, evenly
    spaced along each side, in clockwise order starting with ID 0 at
    the top-left corner.

    Returns a list of dicts:
        {"id": int, "x_mm": float, "y_mm": float}
    where (x_mm, y_mm) is the marker's CENTER position, in the paper
    coordinate system (origin = ID 0's center, x right, y down).
    """
    left_x = MARKER_MARGIN_MM + MARKER_SIZE_MM / 2.0
    right_x = PAPER_WIDTH_MM - MARKER_MARGIN_MM - MARKER_SIZE_MM / 2.0
    top_y = MARKER_MARGIN_MM + MARKER_SIZE_MM / 2.0
    bottom_y = PAPER_HEIGHT_MM - MARKER_MARGIN_MM - MARKER_SIZE_MM / 2.0

    def evenly_spaced_positions(start, end, spacing):
        """Positions from start to end inclusive, spaced as close to
        `spacing` as possible while landing exactly on both ends."""
        length = abs(end - start)
        count = max(1, round(length / spacing))
        return [start + (end - start) * (i / count) for i in range(count + 1)]

    top_xs = evenly_spaced_positions(left_x, right_x, MARKER_SPACING_MM)
    right_ys = evenly_spaced_positions(top_y, bottom_y, MARKER_SPACING_MM)
    bottom_xs = evenly_spaced_positions(right_x, left_x, MARKER_SPACING_MM)
    left_ys = evenly_spaced_positions(bottom_y, top_y, MARKER_SPACING_MM)

    points = []
    # Top edge, left -> right (includes both top corners)
    for x in top_xs:
        points.append((x, top_y))
    # Right edge, top -> bottom (skip top-right corner, already added)
    for y in right_ys[1:]:
        points.append((right_x, y))
    # Bottom edge, right -> left (skip bottom-right corner, already added)
    for x in bottom_xs[1:]:
        points.append((x, bottom_y))
    # Left edge, bottom -> top (skip both remaining corners already added)
    for y in left_ys[1:-1]:
        points.append((left_x, y))

    markers = []
    for i, (x, y) in enumerate(points):
        markers.append({"id": i, "x_mm": x, "y_mm": y})

    return markers


def marker_corners_mm(center_x_mm, center_y_mm, size_mm=MARKER_SIZE_MM):
    """4 corners of a marker in paper mm: TL, TR, BR, BL (clockwise)."""
    h = size_mm / 2.0
    return [
        (center_x_mm - h, center_y_mm - h),  # top-left
        (center_x_mm + h, center_y_mm - h),  # top-right
        (center_x_mm + h, center_y_mm + h),  # bottom-right
        (center_x_mm - h, center_y_mm + h),  # bottom-left
    ]


def rects_overlap(a, b, gap_mm=0.0):
    """
    Axis-aligned overlap test for two button rects.
    Each rect is (x_mm, y_mm, width_mm, height_mm), (x_mm, y_mm) = top-left.
    gap_mm: minimum required clear gap between the outer edges of two buttons.
    """
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    # Two rects violate the minimum gap if their expanded bounding boxes overlap by less than gap_mm
    return not (ax2 + gap_mm <= bx1 or bx2 + gap_mm <= ax1 or ay2 + gap_mm <= by1 or by2 + gap_mm <= ay1)


def rect_within_interior(rect):
    """Check a button rect (x,y,w,h in mm) fits inside the interior area."""
    x, y, w, h = rect
    return (
        x >= INTERIOR_X_MIN and y >= INTERIOR_Y_MIN and
        x + w <= INTERIOR_X_MAX and y + h <= INTERIOR_Y_MAX
    )
