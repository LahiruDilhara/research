"""
realtimeprocess/stages/stage1_normalizer.py

Stage 1: Scale Normalization & Wrist Translation Centering.

Computes 8-distance palm RMS scale L_hand and normalizes 21 MediaPipe hand joint coordinates:
L_hand = sqrt((d1^2 + d2^2 + d3^2 + d4^2 + d5^2 + d6^2 + d7^2 + d8^2) / 8.0)
Wrist translation centering moves wrist landmark 0 to (0.0, 0.0, 0.0).

Matches datacreator/normalize_landmarks.py 100%.
"""

import math

WRIST_INDEX = 0
INDEX_MCP_INDEX = 5
MIDDLE_MCP_INDEX = 9
RING_MCP_INDEX = 13
PINKY_MCP_INDEX = 17


class HandScaleNormalizer:
    """Computes 8-distance palm RMS scale L_hand and normalizes 21 joint 3D coordinates."""

    def normalize(self, pts_px: list[tuple[float, float, float]], center_wrist: bool = True) -> list[tuple[float, float, float]]:
        """
        Given 21 pixel-space (or raw) 3D landmark coordinates [(x, y, z), ...]:
        Calculates 8 symmetric palm distances and RMS scale L_hand.
        Translates wrist to (0,0,0) and divides coordinates by L_hand.
        """
        w_x, w_y, w_z = pts_px[WRIST_INDEX]
        i_x, i_y, _   = pts_px[INDEX_MCP_INDEX]
        m_x, m_y, _   = pts_px[MIDDLE_MCP_INDEX]
        r_x, r_y, _   = pts_px[RING_MCP_INDEX]
        p_x, p_y, _   = pts_px[PINKY_MCP_INDEX]

        # 8 Symmetric Palm Skeleton Segments
        d1_sq = (i_x - w_x) ** 2 + (i_y - w_y) ** 2  # Wrist -> Index
        d2_sq = (m_x - w_x) ** 2 + (m_y - w_y) ** 2  # Wrist -> Middle
        d3_sq = (r_x - w_x) ** 2 + (r_y - w_y) ** 2  # Wrist -> Ring
        d4_sq = (p_x - w_x) ** 2 + (p_y - w_y) ** 2  # Wrist -> Pinky
        d5_sq = (m_x - i_x) ** 2 + (m_y - i_y) ** 2  # Index -> Middle
        d6_sq = (r_x - m_x) ** 2 + (r_y - m_y) ** 2  # Middle -> Ring
        d7_sq = (p_x - r_x) ** 2 + (p_y - r_y) ** 2  # Ring -> Pinky
        d8_sq = (p_x - i_x) ** 2 + (p_y - i_y) ** 2  # Index -> Pinky

        # Root-Mean-Square (RMS) of 8 symmetric palm distances
        l_hand = math.sqrt((d1_sq + d2_sq + d3_sq + d4_sq + d5_sq + d6_sq + d7_sq + d8_sq) / 8.0)
        if l_hand <= 0:
            l_hand = 1.0

        offset_x = w_x if center_wrist else 0.0
        offset_y = w_y if center_wrist else 0.0
        offset_z = w_z if center_wrist else 0.0

        return [
            ((px - offset_x) / l_hand, (py - offset_y) / l_hand, (pz - offset_z) / l_hand)
            for px, py, pz in pts_px
        ]
