"""
core/pipeline/normalizer.py

Hand landmark scale normaliser.
Replicates mediapipeDetector/realtimeprocess/stages/stage1_normalizer.py
exactly so that plugin models receive the same feature values as during training.

L_hand = sqrt( (d1² + d2² + ... + d8²) / 8 )
where d1..d8 are 8 symmetric palm skeleton segment lengths.
Coordinates are then translated to wrist-centred and divided by L_hand.
"""

from __future__ import annotations

import math

from config.constants import (
    ALL_21_LANDMARK_NAMES,
    WRIST_INDEX,
    INDEX_MCP_INDEX,
    MIDDLE_MCP_INDEX,
    RING_MCP_INDEX,
    PINKY_MCP_INDEX,
)


class HandScaleNormalizer:
    """
    Computes the 8-distance RMS palm scale L_hand and produces
    wrist-centred, scale-normalised 3-D landmark coordinates.
    Matches process.sh Step 2 / normalize_landmarks.py 100%.
    """

    def compute_l_hand(self, pts_px: list[tuple[float, float, float]]) -> float:
        """Computes 8-distance RMS palm scale L_hand in pixel units."""
        w_x, w_y, _ = pts_px[WRIST_INDEX]
        i_x, i_y, _ = pts_px[INDEX_MCP_INDEX]
        m_x, m_y, _ = pts_px[MIDDLE_MCP_INDEX]
        r_x, r_y, _ = pts_px[RING_MCP_INDEX]
        p_x, p_y, _ = pts_px[PINKY_MCP_INDEX]

        d_sq = [
            (i_x - w_x) ** 2 + (i_y - w_y) ** 2,  # wrist → index
            (m_x - w_x) ** 2 + (m_y - w_y) ** 2,  # wrist → middle
            (r_x - w_x) ** 2 + (r_y - w_y) ** 2,  # wrist → ring
            (p_x - w_x) ** 2 + (p_y - w_y) ** 2,  # wrist → pinky
            (m_x - i_x) ** 2 + (m_y - i_y) ** 2,  # index → middle
            (r_x - m_x) ** 2 + (r_y - m_y) ** 2,  # middle → ring
            (p_x - r_x) ** 2 + (p_y - r_y) ** 2,  # ring → pinky
            (p_x - i_x) ** 2 + (p_y - i_y) ** 2,  # index → pinky
        ]
        l_hand = math.sqrt(sum(d_sq) / 8.0)
        return l_hand if l_hand > 0.0 else 1.0

    def normalize(
        self,
        pts_px: list[tuple[float, float, float]],
        center_wrist: bool = True,
    ) -> list[tuple[float, float, float]]:
        """
        Parameters
        ----------
        pts_px      : 21 landmark coordinates in pixel space [(x, y, z), ...].
                      z should be raw_z * frame_width (same scaling as x).
        center_wrist: If True, translate wrist landmark to (0, 0, 0).

        Returns
        -------
        21 scale-normalised coordinates [(nx, ny, nz), ...].
        """
        l_hand = self.compute_l_hand(pts_px)
        w_x, w_y, _ = pts_px[WRIST_INDEX]

        w_x_off = w_x if center_wrist else 0.0
        w_y_off = w_y if center_wrist else 0.0
        w_z_off = pts_px[WRIST_INDEX][2] if center_wrist else 0.0

        return [
            (
                (px - w_x_off) / l_hand,
                (py - w_y_off) / l_hand,
                (pz - w_z_off) / l_hand,
            )
            for px, py, pz in pts_px
        ]

    def build_norm_dict(
        self,
        norm_pts: list[tuple[float, float, float]],
        pts_px: list[tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        """
        Converts 21 normalised (x, y, z) tuples into the flat dict expected by
        plugin predict() calls: {"wrist_x": 0.0, "wrist_y": 0.0, ..., "pinky_tip_z": ...}.
        Also attaches _raw_stationary coordinates for transit displacement filtering (Step 7).
        """
        d: dict[str, Any] = {}
        for idx, name in enumerate(ALL_21_LANDMARK_NAMES):
            nx, ny, nz = norm_pts[idx]
            d[f"{name}_x"] = nx
            d[f"{name}_y"] = ny
            d[f"{name}_z"] = nz

        if pts_px is not None and len(pts_px) == 21:
            w_x, w_y, _ = pts_px[WRIST_INDEX]
            i_x, i_y, _ = pts_px[INDEX_MCP_INDEX]
            m_x, m_y, _ = pts_px[MIDDLE_MCP_INDEX]
            r_x, r_y, _ = pts_px[RING_MCP_INDEX]
            p_x, p_y, _ = pts_px[PINKY_MCP_INDEX]
            l_hand = self.compute_l_hand(pts_px)

            d["_raw_stationary"] = {
                "wrist": (w_x, w_y),
                "index_mcp": (i_x, i_y),
                "middle_mcp": (m_x, m_y),
                "ring_mcp": (r_x, r_y),
                "pinky_mcp": (p_x, p_y),
                "l_hand": l_hand,
            }
        return d
