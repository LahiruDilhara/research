"""
core/pipeline/filters/kinetic_motion_filter.py

Zero-Velocity Touch Suppression Filter.
Exact replication of mediapipeDetector/process.sh Step 9
(--remove-zero-vel-touch) and datacreator/filter_dataset.py.

Validation rule:
- A physical touch impact requires active fingertip movement across the 5 frames.
- If maximum fingertip 2D speed across all 4 velocity steps is below the kinetic
  noise floor (default: 0.008 L_hand per step), the finger is stationary/resting
  or experiencing sensor jitter, so touch onset is suppressed.
"""

from __future__ import annotations

from config.constants import MIN_KINETIC_SPEED_THRESHOLD

FINGER_TIP_MAP = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "pinky": "pinky_tip",
    "Thumb": "thumb_tip",
    "Index": "index_tip",
    "Middle": "middle_tip",
    "Ring": "ring_tip",
    "Pinky": "pinky_tip",
}


class KineticMotionFilter:
    """Verifies physical kinetic impact dynamics to reject resting/zero-velocity touches."""

    def __init__(self, threshold: float = MIN_KINETIC_SPEED_THRESHOLD) -> None:
        self.threshold = threshold

    def get_max_tip_speed(
        self,
        v_steps_4: list[dict[str, float]] | None,
        finger_name: str,
    ) -> float:
        """Computes maximum 2D fingertip speed across velocity steps."""
        if not v_steps_4:
            return 0.0

        tip_lm = FINGER_TIP_MAP.get(finger_name, f"{finger_name.lower()}_tip")
        max_speed = 0.0
        for step in v_steps_4:
            s2 = step.get(f"{tip_lm}_speed_2d", 0.0)
            if s2 > max_speed:
                max_speed = s2
        return max_speed

    def validate(
        self,
        v_steps_4: list[dict[str, float]] | None,
        finger_name: str,
        threshold: float | None = None,
    ) -> bool:
        """
        Returns True if the fingertip speed in any of the 4 steps exceeds kinetic threshold.
        """
        if not v_steps_4:
            return True

        limit = self.threshold if threshold is None else threshold
        return self.get_max_tip_speed(v_steps_4, finger_name) >= limit
