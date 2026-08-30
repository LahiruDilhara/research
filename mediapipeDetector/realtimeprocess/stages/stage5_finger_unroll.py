"""
realtimeprocess/stages/stage5_finger_unroll.py

Stage 5: Per-Finger Dataset Unrolling.

Unrolls 5-frame sequence window & 4-step velocity window into 5 distinct per-finger sequence rows:
- thumb, index, middle, ring, pinky
Common palm joints: wrist, thumb_cmc, index_mcp, middle_mcp, ring_mcp, pinky_mcp.
Mapped finger-specific joints: pip, dip, tip.

Matches process.sh Step 10 and datacreator/split_fingers.py 100%.
"""

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

COMMON_PALM_JOINTS = [
    "wrist", "thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
]

FINGER_JOINT_MAP = {
    "thumb":  {"pip": "thumb_mcp", "dip": "thumb_ip",  "tip": "thumb_tip"},
    "index":  {"pip": "index_pip", "dip": "index_dip", "tip": "index_tip"},
    "middle": {"pip": "middle_pip", "dip": "middle_dip", "tip": "middle_tip"},
    "ring":   {"pip": "ring_pip",   "dip": "ring_dip",   "tip": "ring_tip"},
    "pinky":  {"pip": "pinky_pip",  "dip": "pinky_dip",  "tip": "pinky_tip"},
}


def unroll_per_finger_window(norm_frames_5: list[dict[str, float]], v_steps_4: list[dict[str, float]]) -> dict[str, dict]:
    """
    Unrolls 5-frame landmark window & 4-step velocity window into 5 distinct per-finger data dicts.
    Returns dict mapping finger_name -> unrolled_row_dict containing wrist, pip, dip, tip coordinates & velocities.
    """
    finger_rows = {}

    for finger in FINGERS:
        j_map = FINGER_JOINT_MAP[finger]
        row = {"finger_name": finger}

        # 1. Coordinate steps (k = 1..5)
        for k in range(1, 6):
            f_data = norm_frames_5[k - 1]
            # Common palm joints
            for j in COMMON_PALM_JOINTS:
                row[f"{j}{k}_x"] = f_data[f"{j}_x"]
                row[f"{j}{k}_y"] = f_data[f"{j}_y"]
                row[f"{j}{k}_z"] = f_data[f"{j}_z"]

            # Mapped finger-specific joints (pip, dip, tip)
            for j_role, orig_name in j_map.items():
                row[f"{j_role}{k}_x"] = f_data[f"{orig_name}_x"]
                row[f"{j_role}{k}_y"] = f_data[f"{orig_name}_y"]
                row[f"{j_role}{k}_z"] = f_data[f"{orig_name}_z"]

        # 2. Velocity steps (v = 1..4)
        for v in range(1, 5):
            v_data = v_steps_4[v - 1]
            # Common palm joints
            for j in COMMON_PALM_JOINTS:
                row[f"{j}{v}_vx"] = v_data[f"{j}_vx"]
                row[f"{j}{v}_vy"] = v_data[f"{j}_vy"]
                row[f"{j}{v}_vz"] = v_data[f"{j}_vz"]

            # Mapped finger-specific joints (pip, dip, tip)
            for j_role, orig_name in j_map.items():
                row[f"{j_role}{v}_vx"] = v_data[f"{orig_name}_vx"]
                row[f"{j_role}{v}_vy"] = v_data[f"{orig_name}_vy"]
                row[f"{j_role}{v}_vz"] = v_data[f"{orig_name}_vz"]
                row[f"{j_role}{v}_speed_2d"] = v_data[f"{orig_name}_speed_2d"]
                row[f"{j_role}{v}_speed_3d"] = v_data[f"{orig_name}_speed_3d"]

        finger_rows[finger] = row

    return finger_rows
