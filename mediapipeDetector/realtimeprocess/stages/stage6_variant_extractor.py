"""
realtimeprocess/stages/stage6_variant_extractor.py

Stage 6: Research Feature Variant Tensor Extractor.

Extracts numerical feature tensors of shape (5, seq_len, feature_dim) for all 14 research feature variants:
1. coords_2d (5x8)
2. coords_3d (5x12)
3. vel_2d (4x8)
4. vel_3d (4x12)
5. vel_speed_2d (4x12)
6. vel_speed_3d (4x16)
7. combined_2d (4x16)
8. combined_3d (4x24)
9. all_joints_vel (4x18)
10. all_joints_coords_vel (4x36)
11. z_kinematics (4x8)
12. super_combined (4x28)
13. wrist_relative_3d (4x21)
14. fingertip_velocity_ratios (4x16)

Matches deepLearningModels/model_arch.py parse_variant_csv() 100%.
"""

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def extract_variant_tensor(finger_rows: dict[str, dict], variant_name: str) -> np.ndarray:
    """
    Given unrolled finger_rows dict for 5 fingers:
    Extracts numerical feature tensor of shape (5, seq_len, feature_dim) matching target variant schema.
    Returns float32 NumPy array of shape (5, seq_len, feature_dim).
    """
    variant_name = variant_name.lower().strip()
    finger_list = FINGERS

    def _g(r, k):
        return r.get(k, 0.0)

    if variant_name == "coords_2d":
        seq_len, feature_dim = 5, 8
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for k in range(1, 6):
                X[i, k - 1, :] = [_g(row, f"wrist{k}_x"), _g(row, f"wrist{k}_y"), _g(row, f"pip{k}_x"), _g(row, f"pip{k}_y"), _g(row, f"dip{k}_x"), _g(row, f"dip{k}_y"), _g(row, f"tip{k}_x"), _g(row, f"tip{k}_y")]
        return X

    elif variant_name == "coords_3d":
        seq_len, feature_dim = 5, 12
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for k in range(1, 6):
                X[i, k - 1, :] = [_g(row, f"wrist{k}_x"), _g(row, f"wrist{k}_y"), _g(row, f"wrist{k}_z"), _g(row, f"pip{k}_x"), _g(row, f"pip{k}_y"), _g(row, f"pip{k}_z"), _g(row, f"dip{k}_x"), _g(row, f"dip{k}_y"), _g(row, f"dip{k}_z"), _g(row, f"tip{k}_x"), _g(row, f"tip{k}_y"), _g(row, f"tip{k}_z")]
        return X

    elif variant_name in ("vel_2d", "vel_velocities"):
        seq_len, feature_dim = 4, 8
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                X[i, v - 1, :] = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")]
        return X

    elif variant_name == "vel_3d":
        seq_len, feature_dim = 4, 12
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                X[i, v - 1, :] = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")]
        return X

    elif variant_name in ("vel_speed_2d", "vel_speed"):
        seq_len, feature_dim = 4, 12
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                vels = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")]
                speeds = [_g(row, f"wrist{v}_speed_2d"), _g(row, f"pip{v}_speed_2d"), _g(row, f"dip{v}_speed_2d"), _g(row, f"tip{v}_speed_2d")]
                X[i, v - 1, :] = vels + speeds
        return X

    elif variant_name == "vel_speed_3d":
        seq_len, feature_dim = 4, 16
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                vels = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")]
                speeds = [_g(row, f"wrist{v}_speed_3d"), _g(row, f"pip{v}_speed_3d"), _g(row, f"dip{v}_speed_3d"), _g(row, f"tip{v}_speed_3d")]
                X[i, v - 1, :] = vels + speeds
        return X

    elif variant_name in ("combined_2d", "combined"):
        seq_len, feature_dim = 4, 16
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                pos = [_g(row, f"wrist{v}_x"), _g(row, f"wrist{v}_y"), _g(row, f"pip{v}_x"), _g(row, f"pip{v}_y"), _g(row, f"dip{v}_x"), _g(row, f"dip{v}_y"), _g(row, f"tip{v}_x"), _g(row, f"tip{v}_y")]
                vel = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")]
                X[i, v - 1, :] = pos + vel
        return X

    elif variant_name == "combined_3d":
        seq_len, feature_dim = 4, 24
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                pos = [_g(row, f"wrist{v}_x"), _g(row, f"wrist{v}_y"), _g(row, f"wrist{v}_z"), _g(row, f"pip{v}_x"), _g(row, f"pip{v}_y"), _g(row, f"pip{v}_z"), _g(row, f"dip{v}_x"), _g(row, f"dip{v}_y"), _g(row, f"dip{v}_z"), _g(row, f"tip{v}_x"), _g(row, f"tip{v}_y"), _g(row, f"tip{v}_z")]
                vel = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")]
                X[i, v - 1, :] = pos + vel
        return X

    elif variant_name == "all_joints_vel":
        seq_len, feature_dim = 4, 18
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                X[i, v - 1, :] = [
                    _g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"),
                    _g(row, f"thumb_cmc{v}_vx"), _g(row, f"thumb_cmc{v}_vy"),
                    _g(row, f"index_mcp{v}_vx"), _g(row, f"index_mcp{v}_vy"),
                    _g(row, f"middle_mcp{v}_vx"), _g(row, f"middle_mcp{v}_vy"),
                    _g(row, f"ring_mcp{v}_vx"), _g(row, f"ring_mcp{v}_vy"),
                    _g(row, f"pinky_mcp{v}_vx"), _g(row, f"pinky_mcp{v}_vy"),
                    _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"),
                    _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"),
                    _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")
                ]
        return X

    elif variant_name in ("all_joints_coords_vel", "all_combined"):
        seq_len, feature_dim = 4, 36
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                pos = [
                    _g(row, f"wrist{v}_x"), _g(row, f"wrist{v}_y"),
                    _g(row, f"thumb_cmc{v}_x"), _g(row, f"thumb_cmc{v}_y"),
                    _g(row, f"index_mcp{v}_x"), _g(row, f"index_mcp{v}_y"),
                    _g(row, f"middle_mcp{v}_x"), _g(row, f"middle_mcp{v}_y"),
                    _g(row, f"ring_mcp{v}_x"), _g(row, f"ring_mcp{v}_y"),
                    _g(row, f"pinky_mcp{v}_x"), _g(row, f"pinky_mcp{v}_y"),
                    _g(row, f"pip{v}_x"), _g(row, f"pip{v}_y"),
                    _g(row, f"dip{v}_x"), _g(row, f"dip{v}_y"),
                    _g(row, f"tip{v}_x"), _g(row, f"tip{v}_y")
                ]
                vel = [
                    _g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"),
                    _g(row, f"thumb_cmc{v}_vx"), _g(row, f"thumb_cmc{v}_vy"),
                    _g(row, f"index_mcp{v}_vx"), _g(row, f"index_mcp{v}_vy"),
                    _g(row, f"middle_mcp{v}_vx"), _g(row, f"middle_mcp{v}_vy"),
                    _g(row, f"ring_mcp{v}_vx"), _g(row, f"ring_mcp{v}_vy"),
                    _g(row, f"pinky_mcp{v}_vx"), _g(row, f"pinky_mcp{v}_vy"),
                    _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"),
                    _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"),
                    _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")
                ]
                X[i, v - 1, :] = pos + vel
        return X

    elif variant_name == "z_kinematics":
        seq_len, feature_dim = 4, 8
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                z_pos = [_g(row, f"wrist{v}_z"), _g(row, f"pip{v}_z"), _g(row, f"dip{v}_z"), _g(row, f"tip{v}_z")]
                z_vel = [_g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vz")]
                X[i, v - 1, :] = z_pos + z_vel
        return X

    elif variant_name in ("super_combined", "super"):
        seq_len, feature_dim = 4, 28
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                pos = [_g(row, f"wrist{v}_x"), _g(row, f"wrist{v}_y"), _g(row, f"wrist{v}_z"), _g(row, f"pip{v}_x"), _g(row, f"pip{v}_y"), _g(row, f"pip{v}_z"), _g(row, f"dip{v}_x"), _g(row, f"dip{v}_y"), _g(row, f"dip{v}_z"), _g(row, f"tip{v}_x"), _g(row, f"tip{v}_y"), _g(row, f"tip{v}_z")]
                vel = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")]
                speed = [_g(row, f"wrist{v}_speed_3d"), _g(row, f"pip{v}_speed_3d"), _g(row, f"dip{v}_speed_3d"), _g(row, f"tip{v}_speed_3d")]
                X[i, v - 1, :] = pos + vel + speed
        return X

    elif variant_name in ("wrist_relative_3d", "wrist_rel_3d"):
        seq_len, feature_dim = 4, 21
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                wx, wy, wz = _g(row, f"wrist{v}_x"), _g(row, f"wrist{v}_y"), _g(row, f"wrist{v}_z")
                px, py, pz = _g(row, f"pip{v}_x") - wx, _g(row, f"pip{v}_y") - wy, _g(row, f"pip{v}_z") - wz
                dx, dy, dz = _g(row, f"dip{v}_x") - wx, _g(row, f"dip{v}_y") - wy, _g(row, f"dip{v}_z") - wz
                tx, ty, tz = _g(row, f"tip{v}_x") - wx, _g(row, f"tip{v}_y") - wy, _g(row, f"tip{v}_z") - wz
                vel = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")]
                X[i, v - 1, :] = [px, py, pz, dx, dy, dz, tx, ty, tz] + vel
        return X

    elif variant_name in ("fingertip_velocity_ratios", "tip_vel_ratios"):
        seq_len, feature_dim = 4, 16
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                w_vx, w_vy, w_vz = _g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz")
                t_vx, t_vy, t_vz = _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")
                rel_vx, rel_vy, rel_vz = t_vx - w_vx, t_vy - w_vy, t_vz - w_vz
                tip_speed = _g(row, f"tip{v}_speed_3d")
                wrist_speed = _g(row, f"wrist{v}_speed_3d")
                speed_ratio = (tip_speed + 1e-5) / (wrist_speed + 1e-5)
                vel = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"wrist{v}_vz"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"pip{v}_vz"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"dip{v}_vz"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy"), _g(row, f"tip{v}_vz")]
                X[i, v - 1, :] = vel + [rel_vx, rel_vy, rel_vz, speed_ratio]
        return X

    else:
        raise ValueError(f"Unsupported variant_name: '{variant_name}'")
