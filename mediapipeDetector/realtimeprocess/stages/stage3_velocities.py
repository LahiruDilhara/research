"""
realtimeprocess/stages/stage3_velocities.py

Stage 3: 4-Step Velocity & Speed Calculation.

Computes 4 frame-to-frame velocity steps (v = 1..4) across 5 sequence frames for all 21 hand joints:
vx = x_{v+1} - x_v
vy = y_{v+1} - y_v
vz = z_{v+1} - z_v
speed_2d = sqrt(vx^2 + vy^2)
speed_3d = sqrt(vx^2 + vy^2 + vz^2)

Matches process.sh Step 7 and datacreator/calculate_velocities.py 100%.
"""

import math

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def compute_window_velocities(norm_frames_5: list[dict[str, float]]) -> list[dict[str, float]]:
    """
    Given a list of 5 normalized landmark frame dicts (k = 1..5):
    Computes 4 frame-to-frame velocity steps (v = 1..4) for all 21 joints.
    Returns list of 4 velocity step dicts containing vx, vy, vz, speed_2d, speed_3d.
    """
    velocity_steps = []
    for v in range(1, 5):
        curr_f = norm_frames_5[v]
        prev_f = norm_frames_5[v - 1]
        v_dict = {}

        for lm_name in ALL_21_LANDMARK_NAMES:
            cx, cy, cz = curr_f[f"{lm_name}_x"], curr_f[f"{lm_name}_y"], curr_f[f"{lm_name}_z"]
            px, py, pz = prev_f[f"{lm_name}_x"], prev_f[f"{lm_name}_y"], prev_f[f"{lm_name}_z"]

            vx = cx - px
            vy = cy - py
            vz = cz - pz
            speed_2d = math.sqrt(vx * vx + vy * vy)
            speed_3d = math.sqrt(vx * vx + vy * vy + vz * vz)

            v_dict[f"{lm_name}_vx"] = vx
            v_dict[f"{lm_name}_vy"] = vy
            v_dict[f"{lm_name}_vz"] = vz
            v_dict[f"{lm_name}_speed_2d"] = speed_2d
            v_dict[f"{lm_name}_speed_3d"] = speed_3d

        velocity_steps.append(v_dict)

    return velocity_steps
