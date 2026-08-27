"""
datacreator/hand_selection.py

Reusable single-hand selection logic for MediaPipe hand tracking pipelines.
Prioritizes "Right" hand if multiple hands are visible in a frame.
Falls back to "Left" hand if "Right" is not detected.
"""

def select_single_hand(raw_hands_data: list, primary_preference: str = "Right") -> list:
    """
    Selects at most one hand from detected raw hands list.
    - If primary_preference ('Right') is present, selects it.
    - Otherwise, falls back to the first available hand ('Left').
    Returns a list with 1 hand dict, or [] if no hands detected.
    """
    if not raw_hands_data:
        return []

    # Priority 1: Check for preferred hand (e.g. "Right")
    for hand_info in raw_hands_data:
        if hand_info.get("hand") == primary_preference:
            return [hand_info]

    # Priority 2: Fallback to any detected hand (e.g. "Left")
    return [raw_hands_data[0]]
