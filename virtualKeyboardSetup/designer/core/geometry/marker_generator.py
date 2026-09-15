import math
from core.models.marker_model import MarkerModel
from config.app_config import AppConfig


def generate_marker_layout(config: AppConfig | None = None) -> list[MarkerModel]:
    if config is None:
        config = AppConfig()

    paper_w = config.paper_width_mm
    paper_h = config.paper_height_mm
    margin = config.paper_margin_mm
    marker_sz = config.marker_size_mm
    min_gap_ratio = config.marker_min_gap_ratio  # Default 1.5x configured in .env

    left_x = margin + marker_sz / 2.0
    right_x = paper_w - margin - marker_sz / 2.0
    top_y = margin + marker_sz / 2.0
    bottom_y = paper_h - margin - marker_sz / 2.0

    # 1. Horizontal Edge Calculation (Top and Bottom Edges)
    dist_h = right_x - left_x
    min_step_h = marker_sz * (1.0 + min_gap_ratio)

    if dist_h >= min_step_h:
        k_h = max(1, math.floor(dist_h / min_step_h))
    else:
        k_h = 1

    step_h = dist_h / k_h
    top_xs = [left_x + i * step_h for i in range(k_h + 1)]

    # 2. Vertical Edge Calculation (Left and Right Edges)
    dist_v = bottom_y - top_y
    min_step_v = marker_sz * (1.0 + min_gap_ratio)

    if dist_v >= min_step_v:
        k_v = max(1, math.floor(dist_v / min_step_v))
    else:
        k_v = 1

    step_v = dist_v / k_v
    right_ys = [top_y + j * step_v for j in range(k_v + 1)]

    # 3. Assemble Clockwise Perimeter Loop
    points = []
    # Top edge (left to right, excluding right corner)
    for x in top_xs[:-1]:
        points.append((x, top_y))

    # Right edge (top to bottom, excluding bottom corner)
    for y in right_ys[:-1]:
        points.append((right_x, y))

    # Bottom edge (right to left, excluding left corner)
    for x in reversed(top_xs[1:]):
        points.append((x, bottom_y))

    # Left edge (bottom to top, excluding top corner)
    for y in reversed(right_ys[1:]):
        points.append((left_x, y))

    # Construct MarkerModel objects with sequential AprilTag Tag IDs
    markers = []
    for idx, (x, y) in enumerate(points):
        markers.append(MarkerModel(id=idx, x_mm=x, y_mm=y, size_mm=marker_sz))

    return markers
