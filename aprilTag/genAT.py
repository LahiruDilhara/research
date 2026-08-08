"""
Generate a printable page (default A4) containing AprilTag markers.

- 4 markers are always placed at the corners.
- You set how many markers go along the wide side and how many along
  the long side (corners count as part of these totals).
- Optionally add a grid of markers in the middle of the page.

Requires: opencv-python (cv2.aruco has built-in AprilTag dictionaries).
The dt-apriltags package is a DETECTOR only (it can't draw tags), so tag
images are generated here with cv2.aruco instead -- the bit patterns are
identical to the AprilTag family, so tags made here will be detected fine
by dt-apriltags, apriltag, or cv2.aruco detectors alike.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# ROOT CONSTANTS -- edit these to configure the output
# ============================================================

# Page size in millimeters. Defaults to A4 portrait.
PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0

# Print resolution. 300 DPI is a good default for laser/inkjet printing.
DPI = 1200

# Margin from the page edge to the outer edge of the corner markers (mm).
MARGIN_MM = 4

# Physical size of each marker's black square (mm). Keep in mind: bigger
# markers are more robust at distance and extreme angles.
MARKER_SIZE_MM = 15

# How many markers along the WIDE side (top and bottom rows), including
# the 2 corner markers on that row.
MARKERS_ALONG_WIDTH = 4

# How many markers along the LONG side (left and right columns),
# including the 2 corner markers on that column.
MARKERS_ALONG_HEIGHT = 6

# Whether to also place a grid of markers in the middle of the page.
ADD_MIDDLE_MARKERS = False

# Size of the middle grid (rows x cols), only used if ADD_MIDDLE_MARKERS.
MIDDLE_MARKERS_ROWS = 1
MIDDLE_MARKERS_COLS = 1

# AprilTag family / dictionary to use.
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# Starting marker ID (IDs increase by 1 for each marker placed).
START_ID = 0

# Draw the numeric ID under each marker (helpful for setup/debugging).
DRAW_ID_LABELS = False

# Output file paths.
OUTPUT_PNG = "./apriltag_sheet.png"
OUTPUT_PDF = "./apriltag_sheet.pdf"

# ============================================================
# END OF CONSTANTS
# ============================================================


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def generate_marker_image(marker_id: int, size_px: int) -> np.ndarray:
    """Return a size_px x size_px grayscale AprilTag image (0=black, 255=white)."""
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)


def compute_positions():
    """
    Compute the (x_mm, y_mm, id) top-left placement for every marker on the page,
    without duplicating the 4 shared corner markers.
    """
    positions = []
    next_id = [START_ID]

    def new_id():
        i = next_id[0]
        next_id[0] += 1
        return i

    usable_w = PAGE_WIDTH_MM - 2 * MARGIN_MM - MARKER_SIZE_MM
    usable_h = PAGE_HEIGHT_MM - 2 * MARGIN_MM - MARKER_SIZE_MM

    xs = np.linspace(MARGIN_MM, MARGIN_MM + usable_w, MARKERS_ALONG_WIDTH)
    ys = np.linspace(MARGIN_MM, MARGIN_MM + usable_h, MARKERS_ALONG_HEIGHT)

    placed = {}  # (round(x), round(y)) -> id, to avoid duplicate corners

    def place(x, y):
        key = (round(x, 3), round(y, 3))
        if key in placed:
            return
        mid = new_id()
        placed[key] = mid
        positions.append((x, y, mid))

    # Top row and bottom row (spans the wide side)
    for x in xs:
        place(x, ys[0])
        place(x, ys[-1])

    # Left column and right column (spans the long side), corners already placed
    for y in ys:
        place(xs[0], y)
        place(xs[-1], y)

    # Middle grid
    if ADD_MIDDLE_MARKERS and MIDDLE_MARKERS_ROWS > 0 and MIDDLE_MARKERS_COLS > 0:
        mid_x_start = MARGIN_MM + MARKER_SIZE_MM * 1.5
        mid_x_end = PAGE_WIDTH_MM - MARGIN_MM - MARKER_SIZE_MM * 1.5
        mid_y_start = MARGIN_MM + MARKER_SIZE_MM * 1.5
        mid_y_end = PAGE_HEIGHT_MM - MARGIN_MM - MARKER_SIZE_MM * 1.5

        if MIDDLE_MARKERS_COLS == 1:
            mxs = [(mid_x_start + mid_x_end) / 2]
        else:
            mxs = np.linspace(mid_x_start, mid_x_end, MIDDLE_MARKERS_COLS)

        if MIDDLE_MARKERS_ROWS == 1:
            mys = [(mid_y_start + mid_y_end) / 2]
        else:
            mys = np.linspace(mid_y_start, mid_y_end, MIDDLE_MARKERS_ROWS)

        for my in mys:
            for mx in mxs:
                place(mx, my)

    return positions


def build_page():
    page_w_px = mm_to_px(PAGE_WIDTH_MM)
    page_h_px = mm_to_px(PAGE_HEIGHT_MM)
    marker_size_px = mm_to_px(MARKER_SIZE_MM)

    canvas = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)

    positions = compute_positions()
    print(f"Placing {len(positions)} markers (IDs {START_ID} to {START_ID + len(positions) - 1})")

    label_data = []  # (x_px, y_px_below_marker, text)

    for x_mm, y_mm, marker_id in positions:
        x_px = mm_to_px(x_mm)
        y_px = mm_to_px(y_mm)

        tag_img = generate_marker_image(marker_id, marker_size_px)

        # Guard against markers that would spill off the canvas due to rounding.
        y_end = min(y_px + marker_size_px, page_h_px)
        x_end = min(x_px + marker_size_px, page_w_px)
        canvas[y_px:y_end, x_px:x_end] = tag_img[: y_end - y_px, : x_end - x_px]

        if DRAW_ID_LABELS:
            label_data.append((x_px, y_end, f"ID {marker_id}"))

    # Convert to PIL to draw text labels and export as PNG/PDF.
    page_img = Image.fromarray(canvas).convert("RGB")

    if DRAW_ID_LABELS:
        draw = ImageDraw.Draw(page_img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, mm_to_px(3)))
        except OSError:
            font = ImageFont.load_default()
        for x_px, y_end, text in label_data:
            draw.text((x_px, y_end + 2), text, fill=(0, 0, 0), font=font)

    page_img.save(OUTPUT_PNG, dpi=(DPI, DPI))
    page_img.save(OUTPUT_PDF, "PDF", resolution=DPI)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_page()