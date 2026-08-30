"""
realtimeprocess/stages/stage4_quality_filter.py

Stage 4: Sequence Window Quality & Cleaning Filters.

Applies online sequence window validation corresponding to process.sh pipeline stages 8 & 9:
1. Zero Velocity Spike Check (--remove-zero-vel-touch):
   Rejects window if all 4 velocity steps across all 21 joints are exactly 0.0.
2. Hand Score Confidence Quality (--min-avg-score 0.65):
   Rejects window if average hand confidence score across 5 frames < 0.65.

Matches process.sh Steps 8 & 9 and datacreator/filter_window_quality.py 100%.
"""

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def validate_realtime_window_quality(
    v_steps_4: list[dict[str, float]],
    hand_scores_5: list[float] = None,
    min_avg_score: float = 0.65,
    check_zero_vel: bool = True
) -> tuple[bool, str]:
    """
    Applies real-time window validation corresponding to process.sh pipeline stages 8 & 9.
    Returns (is_valid: bool, rejection_reason: str).
    """
    # 1. Check Hand Score Quality
    if hand_scores_5 and len(hand_scores_5) > 0:
        valid_scores = [s for s in hand_scores_5 if s > 0.0]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            if avg_score < min_avg_score:
                return False, f"Low Hand Score Quality ({avg_score:.2f} < {min_avg_score:.2f})"

    # 2. Check Zero Velocity (Stationary / Glitch drop)
    if check_zero_vel:
        all_zero = True
        for v_step in v_steps_4:
            for lm_name in ALL_21_LANDMARK_NAMES:
                vx = v_step.get(f"{lm_name}_vx", 0.0)
                vy = v_step.get(f"{lm_name}_vy", 0.0)
                if abs(vx) > 1e-7 or abs(vy) > 1e-7:
                    all_zero = False
                    break
            if not all_zero:
                break

        if all_zero:
            return False, "Zero Velocity Across Window"

    return True, "OK"
