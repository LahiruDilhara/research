"""
realtimeprocess/stages/stage7_hand_movement.py

Stage 7: Whole-Hand Transit Movement Displacement Filter.

Filters whole-hand movement across 5-frame sequence windows using stationary landmark
displacement relative to rigid palm scale L_hand:
- Evaluates Wrist and 4 MCP knuckles (index_mcp, middle_mcp, ring_mcp, pinky_mcp).
- Computes displacement between Frame 1 and Frame 5:
    dist = sqrt((x_5 - x_1)^2 + (y_5 - y_1)^2) / L_hand(Frame 1)
- If max(dist) > threshold (default: 0.175), the whole hand is moving across the surface,
  so window is marked as moving and touch classification is suppressed.

Matches process.sh Step 7 and datacreator/filter_hand_movement.py 100%.
"""

import math

STATIONARY_NAMES = ["wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]


def compute_hand_displacement(norm_frames_5: list[dict]) -> float:
    """
    Computes maximum stationary joint displacement across 5 frames normalized by Frame 1 palm scale.
    """
    if not norm_frames_5 or len(norm_frames_5) < 5:
        return 0.0

    f1_raw = norm_frames_5[0].get("_raw_stationary")
    f5_raw = norm_frames_5[4].get("_raw_stationary")
    if not f1_raw or not f5_raw:
        return 0.0

    l_hand = f1_raw.get("l_hand", 1.0)
    if l_hand <= 0:
        l_hand = 1.0

    max_disp = 0.0
    for name in STATIONARY_NAMES:
        p1 = f1_raw.get(name)
        p5 = f5_raw.get(name)
        if p1 and p5:
            dist = math.sqrt((p5[0] - p1[0]) ** 2 + (p5[1] - p1[1]) ** 2) / l_hand
            if dist > max_disp:
                max_disp = dist

    return max_disp


def validate_hand_movement(norm_frames_5: list[dict], threshold: float = 0.175) -> tuple[bool, float, str]:
    """
    Validates whether the hand is stationary or moving during the 5-frame window.
    Returns (is_stationary: bool, max_disp: float, reason: str).
    """
    max_disp = compute_hand_displacement(norm_frames_5)
    if max_disp > threshold:
        return False, max_disp, f"Hand Moving ({max_disp:.3f} > {threshold:.3f} L_hand)"
    return True, max_disp, "Stationary"
