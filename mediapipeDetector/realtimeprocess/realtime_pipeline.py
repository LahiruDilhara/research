"""
realtimeprocess/realtime_pipeline.py

Modular Online Real-Time Processing Pipeline & Feature Extractor.

Delegates pipeline execution stages to clean, modular stage components in realtimeprocess/stages/:
- Stage 1: HandScaleNormalizer (stage1_normalizer.py)
- Stage 2: OneEuroFilter1D & OneEuroFilterBank (stage2_euro_filter.py)
- Stage 3: compute_window_velocities (stage3_velocities.py)
- Stage 4: validate_realtime_window_quality (stage4_quality_filter.py)
- Stage 5: unroll_per_finger_window (stage5_finger_unroll.py)
- Stage 6: extract_variant_tensor (stage6_variant_extractor.py)

Executes exact process.sh per-frame sequence:
1. Converts raw MediaPipe [0,1] 3D coordinates to pixel space (x_px = raw_x*w, y_px = raw_y*h, z_px = raw_z*w).
2. Applies HandScaleNormalizer (8-distance palm RMS scale L_hand and wrist translation centering).
3. Applies continuous 1€ Filter (min=3.0, beta=1.4, d=1.0) on scale-normalized coordinates (process.sh Step 3).
"""

import math

from realtimeprocess.stages.stage1_normalizer import (
    HandScaleNormalizer,
    WRIST_INDEX,
    INDEX_MCP_INDEX,
    MIDDLE_MCP_INDEX,
    RING_MCP_INDEX,
    PINKY_MCP_INDEX,
)
from realtimeprocess.stages.stage2_euro_filter import (
    OneEuroFilter1D,
    OneEuroFilterBank,
    ALL_21_LANDMARK_NAMES,
)
from realtimeprocess.stages.stage3_velocities import compute_window_velocities
from realtimeprocess.stages.stage4_quality_filter import validate_realtime_window_quality
from realtimeprocess.stages.stage5_finger_unroll import (
    unroll_per_finger_window,
    FINGERS,
    COMMON_PALM_JOINTS,
    FINGER_JOINT_MAP,
)
from realtimeprocess.stages.stage6_variant_extractor import extract_variant_tensor


def process_streaming_frame(
    raw_pts: list[tuple[float, float, float]],
    frame_w: float,
    frame_h: float,
    t_sec: float,
    normalizer: HandScaleNormalizer,
    euro_filter_bank: OneEuroFilterBank
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    """
    Executes exact process.sh per-frame sequence:
    1. Converts raw MediaPipe [0,1] 3D coordinates to pixel space: x_px = raw_x*w, y_px = raw_y*h, z_px = raw_z*w.
    2. Applies HandScaleNormalizer (8-distance palm RMS scale L_hand and wrist translation centering).
    3. Applies continuous 1€ Filter (min=3.0, beta=1.4, d=1.0) ON THE NORMALIZED COORDINATES (process.sh Step 3).
    Returns (f_dict, smooth_pts_px) where smooth_pts_px contains filtered 2D pixel coordinates for silk-smooth GUI rendering.
    """
    pts_px = [(x * frame_w, y * frame_h, z * frame_w) for (x, y, z) in raw_pts]
    norm_pts = normalizer.normalize(pts_px, center_wrist=True)
    filtered_norm_pts = euro_filter_bank.filter_frame(t_sec, norm_pts)

    f_dict = {}
    for lm_idx, lm_name in enumerate(ALL_21_LANDMARK_NAMES):
        nx, ny, nz = filtered_norm_pts[lm_idx]
        f_dict[f"{lm_name}_x"] = nx
        f_dict[f"{lm_name}_y"] = ny
        f_dict[f"{lm_name}_z"] = nz

    # Reconstruct 1€ filtered smooth pixel coordinates for zero-jitter screen rendering
    w_x, w_y, _ = pts_px[WRIST_INDEX]
    i_x, i_y, _ = pts_px[INDEX_MCP_INDEX]
    m_x, m_y, _ = pts_px[MIDDLE_MCP_INDEX]
    r_x, r_y, _ = pts_px[RING_MCP_INDEX]
    p_x, p_y, _ = pts_px[PINKY_MCP_INDEX]

    d_sq = [
        (i_x - w_x) ** 2 + (i_y - w_y) ** 2,
        (m_x - w_x) ** 2 + (m_y - w_y) ** 2,
        (r_x - w_x) ** 2 + (r_y - w_y) ** 2,
        (p_x - w_x) ** 2 + (p_y - w_y) ** 2,
        (m_x - i_x) ** 2 + (m_y - i_y) ** 2,
        (r_x - m_x) ** 2 + (r_y - m_y) ** 2,
        (p_x - r_x) ** 2 + (p_y - r_y) ** 2,
        (p_x - i_x) ** 2 + (p_y - i_y) ** 2,
    ]
    l_hand = math.sqrt(sum(d_sq) / 8.0)
    if l_hand <= 0:
        l_hand = 1.0

    smooth_pts_px = [
        (w_x + nx * l_hand, w_y + ny * l_hand)
        for (nx, ny, _) in filtered_norm_pts
    ]

    return f_dict, smooth_pts_px


__all__ = [
    "ALL_21_LANDMARK_NAMES",
    "WRIST_INDEX",
    "INDEX_MCP_INDEX",
    "MIDDLE_MCP_INDEX",
    "RING_MCP_INDEX",
    "PINKY_MCP_INDEX",
    "COMMON_PALM_JOINTS",
    "FINGERS",
    "FINGER_JOINT_MAP",
    "OneEuroFilter1D",
    "OneEuroFilterBank",
    "HandScaleNormalizer",
    "process_streaming_frame",
    "compute_window_velocities",
    "validate_realtime_window_quality",
    "unroll_per_finger_window",
    "extract_variant_tensor",
]
