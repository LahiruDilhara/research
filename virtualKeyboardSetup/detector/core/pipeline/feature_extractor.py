"""
core/pipeline/feature_extractor.py

Feature extraction pipeline for real-time inference matching the training pipeline 100%.

Stages:
  1. compute_window_velocities: 4 frame-to-frame velocity steps (vx, vy, vz, speed_2d, speed_3d) across 5 frames
  2. unroll_per_finger_window: 5 per-finger dictionary records mapping palm + finger joints
  3. extract_variant_tensor: numerical tensor (5 fingers, seq_len, feature_dim) for any research feature variant
  4. scale_variant_tensor: StandardScaler normalization matching training dataset distributions
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

ALL_21_LANDMARK_NAMES: list[str] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

FINGERS: list[str] = ["thumb", "index", "middle", "ring", "pinky"]

COMMON_PALM_JOINTS: list[str] = [
    "wrist", "thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
]

FINGER_JOINT_MAP: dict[str, dict[str, str]] = {
    "thumb":  {"pip": "thumb_mcp", "dip": "thumb_ip",  "tip": "thumb_tip"},
    "index":  {"pip": "index_pip", "dip": "index_dip", "tip": "index_tip"},
    "middle": {"pip": "middle_pip", "dip": "middle_dip", "tip": "middle_tip"},
    "ring":   {"pip": "ring_pip",   "dip": "ring_dip",   "tip": "ring_tip"},
    "pinky":  {"pip": "pinky_pip",  "dip": "pinky_dip",  "tip": "pinky_tip"},
}

# ── Scaler Cache ───────────────────────────────────────────────────────────────

_SCALER_DATA: dict[str, np.ndarray] | None = None


def _load_scalers() -> dict[str, np.ndarray]:
    global _SCALER_DATA
    if _SCALER_DATA is None:
        scaler_file = Path(__file__).resolve().parent.parent.parent / "plugins" / "scalers.npz"
        if scaler_file.exists():
            _SCALER_DATA = dict(np.load(scaler_file))
        else:
            _SCALER_DATA = {}
    return _SCALER_DATA


def scale_variant_tensor(X: np.ndarray, variant_name: str) -> np.ndarray:
    """
    Applies the fitted StandardScaler mean and scale to the input feature tensor.
    X shape: (5_fingers, seq_len, feature_dim)
    """
    scalers = _load_scalers()
    mean_key = f"{variant_name}_mean"
    scale_key = f"{variant_name}_scale"

    if mean_key in scalers and scale_key in scalers:
        mean = scalers[mean_key]
        scale = scalers[scale_key]
        N_b, T, C = X.shape
        X_flat = X.reshape(N_b, -1)
        # Apply standard scaler: (X - mean) / scale
        X_scaled = (X_flat - mean) / np.where(scale == 0, 1.0, scale)
        return X_scaled.reshape(N_b, T, C).astype(np.float32)

    return X.astype(np.float32)


# ── Feature computation ────────────────────────────────────────────────────────

def compute_window_velocities(
    norm_frames_5: list[dict[str, float]],
) -> list[dict[str, float]]:
    """
    Computes 4 velocity steps (v = 1..4) between 5 scale-normalized frames.
    """
    velocity_steps: list[dict[str, float]] = []
    for v in range(1, 5):
        curr_f = norm_frames_5[v]
        prev_f = norm_frames_5[v - 1]
        v_dict: dict[str, float] = {}

        for lm_name in ALL_21_LANDMARK_NAMES:
            cx = curr_f.get(f"{lm_name}_x", 0.0)
            cy = curr_f.get(f"{lm_name}_y", 0.0)
            cz = curr_f.get(f"{lm_name}_z", 0.0)
            px = prev_f.get(f"{lm_name}_x", 0.0)
            py = prev_f.get(f"{lm_name}_y", 0.0)
            pz = prev_f.get(f"{lm_name}_z", 0.0)


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


def unroll_per_finger_window(
    norm_frames_5: list[dict[str, float]],
    v_steps_4: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Unrolls 5-frame landmark window & 4-step velocity window into 5 distinct per-finger data dicts.
    """
    finger_rows: dict[str, dict[str, float]] = {}

    for finger in FINGERS:
        j_map = FINGER_JOINT_MAP[finger]
        row: dict[str, Any] = {"finger_name": finger}

        # 1. Coordinate steps (k = 1..5)
        for k in range(1, 6):
            f_data = norm_frames_5[k - 1]
            for j in COMMON_PALM_JOINTS:
                row[f"{j}{k}_x"] = f_data.get(f"{j}_x", 0.0)
                row[f"{j}{k}_y"] = f_data.get(f"{j}_y", 0.0)
                row[f"{j}{k}_z"] = f_data.get(f"{j}_z", 0.0)

            for j_role, orig_name in j_map.items():
                row[f"{j_role}{k}_x"] = f_data.get(f"{orig_name}_x", 0.0)
                row[f"{j_role}{k}_y"] = f_data.get(f"{orig_name}_y", 0.0)
                row[f"{j_role}{k}_z"] = f_data.get(f"{orig_name}_z", 0.0)

        # 2. Velocity steps (v = 1..4)
        for v in range(1, 5):
            v_data = v_steps_4[v - 1]
            for j in COMMON_PALM_JOINTS:
                row[f"{j}{v}_vx"] = v_data.get(f"{j}_vx", 0.0)
                row[f"{j}{v}_vy"] = v_data.get(f"{j}_vy", 0.0)
                row[f"{j}{v}_vz"] = v_data.get(f"{j}_vz", 0.0)

            for j_role, orig_name in j_map.items():
                row[f"{j_role}{v}_vx"] = v_data.get(f"{orig_name}_vx", 0.0)
                row[f"{j_role}{v}_vy"] = v_data.get(f"{orig_name}_vy", 0.0)
                row[f"{j_role}{v}_vz"] = v_data.get(f"{orig_name}_vz", 0.0)
                row[f"{j_role}{v}_speed_2d"] = v_data.get(f"{orig_name}_speed_2d", 0.0)
                row[f"{j_role}{v}_speed_3d"] = v_data.get(f"{orig_name}_speed_3d", 0.0)

        finger_rows[finger] = row

    return finger_rows


def extract_variant_tensor(finger_rows: dict[str, dict], variant_name: str) -> np.ndarray:
    """
    Extracts numerical feature tensor of shape (5, seq_len, feature_dim) matching the research variant.
    """
    variant_name = variant_name.lower().strip()
    finger_list = FINGERS

    def _g(r, k):
        return r.get(k, 0.0)

    if variant_name in ("combined_2d", "combined"):
        seq_len, feature_dim = 4, 16
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                pos = [_g(row, f"wrist{v}_x"), _g(row, f"wrist{v}_y"), _g(row, f"pip{v}_x"), _g(row, f"pip{v}_y"), _g(row, f"dip{v}_x"), _g(row, f"dip{v}_y"), _g(row, f"tip{v}_x"), _g(row, f"tip{v}_y")]
                vel = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")]
                X[i, v - 1, :] = pos + vel
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

    elif variant_name in ("vel_2d", "vel_velocities"):
        seq_len, feature_dim = 4, 8
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                X[i, v - 1, :] = [_g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"), _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"), _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"), _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")]
        return X

    elif variant_name == "coords_2d":
        seq_len, feature_dim = 5, 8
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for k in range(1, 6):
                X[i, k - 1, :] = [_g(row, f"wrist{k}_x"), _g(row, f"wrist{k}_y"), _g(row, f"pip{k}_x"), _g(row, f"pip{k}_y"), _g(row, f"dip{k}_x"), _g(row, f"dip{k}_y"), _g(row, f"tip{k}_x"), _g(row, f"tip{k}_y")]
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

    elif variant_name in ("fingertip_velocity_ratios", "tip_vel_ratios"):
        seq_len, feature_dim = 4, 16
        X = np.zeros((5, seq_len, feature_dim), dtype=np.float32)
        for i, f in enumerate(finger_list):
            row = finger_rows[f]
            for v in range(1, 5):
                w_vx, w_vy = _g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy")
                t_vx, t_vy = _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")
                rel_vx, rel_vy = t_vx - w_vx, t_vy - w_vy
                tip_speed = _g(row, f"tip{v}_speed_2d")
                wrist_speed = _g(row, f"wrist{v}_speed_2d")
                pip_speed = _g(row, f"pip{v}_speed_2d")
                dip_speed = _g(row, f"dip{v}_speed_2d")
                speed_ratio = (tip_speed + 1e-5) / (wrist_speed + 1e-5)
                vel_vals = [
                    _g(row, f"wrist{v}_vx"), _g(row, f"wrist{v}_vy"),
                    _g(row, f"pip{v}_vx"), _g(row, f"pip{v}_vy"),
                    _g(row, f"dip{v}_vx"), _g(row, f"dip{v}_vy"),
                    _g(row, f"tip{v}_vx"), _g(row, f"tip{v}_vy")
                ]
                rel_kinematics = [rel_vx, rel_vy, wrist_speed, tip_speed, pip_speed, dip_speed, speed_ratio, 0.0]
                X[i, v - 1, :] = vel_vals + rel_kinematics
        return X

    else:
        raise ValueError(f"Unsupported variant: {variant_name}")
