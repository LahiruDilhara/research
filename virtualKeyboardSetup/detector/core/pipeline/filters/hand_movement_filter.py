"""
core/pipeline/filters/hand_movement_filter.py

Whole-Hand Transit Movement Displacement Filter.
Exact replication of mediapipeDetector/process.sh Step 7 and
datacreator/filter_hand_movement.py.

Algorithm:
- Evaluates 5 stationary landmarks: wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp.
- Computes displacement between Frame 1 and Frame 5:
    displacement = sqrt((x_5 - x_1)^2 + (y_5 - y_1)^2) / L_hand(Frame 1)
- If max(displacement) > threshold (0.155), the hand is translating across
  the keyboard surface and touch classification is suppressed.
"""

from __future__ import annotations

import math
from typing import Any

from config.constants import HAND_MOVEMENT_THRESHOLD, STATIONARY_LANDMARK_NAMES


class HandMovementFilter:
    """Validates whether hand is stationary or moving across 5-frame sequence window."""

    def __init__(self, threshold: float = HAND_MOVEMENT_THRESHOLD) -> None:
        self.threshold = threshold

    def compute_displacement(self, norm_frames_5: Any) -> float:
        """
        Computes maximum stationary joint displacement between Frame 1 and Frame 5
        normalized by Frame 1 palm scale L_hand.
        """
        if norm_frames_5 is None or len(norm_frames_5) < 5:
            return 0.0

        import numpy as np
        if isinstance(norm_frames_5, np.ndarray):
            if norm_frames_5.ndim >= 3 and norm_frames_5.shape[0] >= 5:
                # Stationary joints: wrist=0, index_mcp=5, middle_mcp=9, ring_mcp=13, pinky_mcp=17
                stat_indices = [0, 5, 9, 13, 17]
                p1 = norm_frames_5[0, stat_indices, :2]
                p5 = norm_frames_5[4, stat_indices, :2]
                diffs = p5 - p1
                dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
                return float(np.max(dists))
            return 0.0

        if not isinstance(norm_frames_5[0], dict):
            return 0.0

        f1_raw = norm_frames_5[0].get("_raw_stationary")
        f5_raw = norm_frames_5[4].get("_raw_stationary")
        if not f1_raw or not f5_raw:
            return 0.0

        l_hand = float(f1_raw.get("l_hand", 1.0))
        if l_hand <= 0.0:
            l_hand = 1.0

        max_disp = 0.0
        for name in STATIONARY_LANDMARK_NAMES:
            p1 = f1_raw.get(name)
            p5 = f5_raw.get(name)
            if p1 and p5:
                dx = p5[0] - p1[0]
                dy = p5[1] - p1[1]
                dist = math.sqrt(dx * dx + dy * dy) / l_hand
                if dist > max_disp:
                    max_disp = dist

        return max_disp

    def validate(
        self,
        norm_frames_5: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> tuple[bool, float, str]:
        """
        Validates if hand movement is below transit threshold.

        Returns
        -------
        (is_stationary, max_displacement, reason)
        """
        limit = self.threshold if threshold is None else threshold
        max_disp = self.compute_displacement(norm_frames_5)
        if max_disp > limit:
            return False, max_disp, f"Hand Moving ({max_disp:.3f} > {limit:.3f} L_hand)"
        return True, max_disp, "Stationary"
