"""
realtimeprocess/stages/stage4_quality_filter.py

Stage 4: Sequence Window Quality and Cleaning Filters.

Applies online sequence window validation corresponding to process.sh pipeline stages 9 and 10:
1. Hand Score Confidence Quality (--min-avg-score 0.65, --min-frame-score 0.45, --max-score-drop 0.35):
   Rejects window if hand confidence scores do not meet dataset quality requirements.
2. Zero Velocity Touch Filter (--remove-zero-vel-touch):
   Rejects touch classification if fingertip velocities across 4 steps are zero or below kinetic noise floor.

Matches process.sh Steps 9 and 10 and datacreator/filter_window_quality.py 100%.
"""

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

FINGER_TIP_MAP = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "pinky": "pinky_tip",
}

# Kinetic motion threshold (relative to L_hand per step):
# Below this threshold, fingertip motion is sensor jitter, not intentional physical contact.
MIN_KINETIC_SPEED_THRESHOLD = 0.008


def validate_realtime_window_quality(
    v_steps_4: list[dict[str, float]],
    hand_scores_5: list[float] = None,
    min_avg_score: float = 0.65,
    min_frame_score: float = 0.45,
    max_score_drop: float = 0.35,
) -> tuple[bool, str]:
    """
    Applies real-time window validation corresponding to process.sh pipeline stage 10:
    1. Hand confidence average score across 5 frames >= min_avg_score (0.65).
    2. Hand confidence minimum frame score >= min_frame_score (0.45).
    3. Hand confidence maximum score fluctuation <= max_score_drop (0.35).
    Returns (is_valid: bool, rejection_reason: str).
    """
    if hand_scores_5 and len(hand_scores_5) > 0:
        valid_scores = [s for s in hand_scores_5 if s > 0.0]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            score_diff = max_score - min_score

            if min_avg_score is not None and avg_score < min_avg_score:
                return False, f"Low Hand Avg Score ({avg_score:.2f} < {min_avg_score:.2f})"

            if min_frame_score is not None and min_score < min_frame_score:
                return False, f"Low Hand Frame Score ({min_score:.2f} < {min_frame_score:.2f})"

            if max_score_drop is not None and score_diff > max_score_drop:
                return False, f"High Score Fluctuation ({score_diff:.2f} > {max_score_drop:.2f})"

    return True, "OK"


def validate_finger_kinetic_motion(v_steps_4: list[dict[str, float]], finger_name: str) -> bool:
    """
    Implements process.sh Step 9 (--remove-zero-vel-touch):
    A physical touch onset requires active fingertip movement across the 5-frame window.
    If the fingertip speed across all 4 velocity steps is below the sensor noise floor,
    the finger is completely stationary/resting, so touch is suppressed.
    """
    tip_lm = FINGER_TIP_MAP.get(finger_name, f"{finger_name}_tip")
    max_tip_speed = 0.0

    for v_step in v_steps_4:
        s2 = v_step.get(f"{tip_lm}_speed_2d", 0.0)
        if s2 > max_tip_speed:
            max_tip_speed = s2

    return max_tip_speed >= MIN_KINETIC_SPEED_THRESHOLD

