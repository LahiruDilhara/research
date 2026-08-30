"""
realtimeprocess/realtime_pipeline.py

Pure online streaming pre-processor and feature extractor for live MediaPipe 12 FPS hand landmark streams.

Performs:
1. Online 1€ filtering across 21 hand joints (x, y, z).
2. Per-frame 8-distance palm RMS scale normalization (L_hand) and wrist translation centering (0,0,0).
3. 4-step frame-to-frame velocity calculation (vx, vy, vz, speed_2d, speed_3d) across 5-frame sequence windows.
4. Per-finger unrolling (thumb, index, middle, ring, pinky) mapping common palm joints and finger-specific joints (pip, dip, tip).
5. Feature tensor extraction matching any model variant (coords_2d, vel_2d, combined_2d, coords_3d, vel_3d, combined_3d, all_joints_vel, all_joints_coords_vel, etc.).
"""

import math
import numpy as np

# ── Landmark & Joint Definitions ──────────────────────────────────────────────
ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

WRIST_INDEX = 0
INDEX_MCP_INDEX = 5
MIDDLE_MCP_INDEX = 9
RING_MCP_INDEX = 13
PINKY_MCP_INDEX = 17

COMMON_PALM_JOINTS = [
    "wrist", "thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
]

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

FINGER_JOINT_MAP = {
    "thumb":  {"pip": "thumb_mcp", "dip": "thumb_ip",  "tip": "thumb_tip"},
    "index":  {"pip": "index_pip", "dip": "index_dip", "tip": "index_tip"},
    "middle": {"pip": "middle_pip", "dip": "middle_dip", "tip": "middle_tip"},
    "ring":   {"pip": "ring_pip",   "dip": "ring_dip",   "tip": "ring_tip"},
    "pinky":  {"pip": "pinky_pip",  "dip": "pinky_dip",  "tip": "pinky_tip"},
}


# ── 1€ Filter Implementation ──────────────────────────────────────────────────
class OneEuroFilter1D:
    def __init__(self, t0: float, x0: float, min_cutoff: float = 3.0, beta: float = 1.4, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_elapsed: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_elapsed
        return r / (r + 1.0)

    def filter(self, t: float, x: float) -> float:
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0:
            return self.x_prev

        a_d = self._smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_elapsed, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilterBank:
    """Maintains continuous online 1€ filtering across all 21 hand joint 3D coordinates (process.sh defaults: min=3.0, beta=1.4, d=1.0)."""

    def __init__(self, min_cutoff: float = 3.0, beta: float = 1.4, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters = {}

    def filter_frame(self, t_sec: float, norm_pts: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        """
        Given timestamp t_sec and 21 scale-normalized wrist-centered 3D coordinates (nx, ny, nz):
        Applies 1€ filter per joint normalized coordinate (exact process.sh Step 3 order).
        Returns 1€ filtered [(nx, ny, nz), ...].
        """
        filtered_pts = []
        for idx, (nx, ny, nz) in enumerate(norm_pts):
            lm_name = ALL_21_LANDMARK_NAMES[idx]
            fnx, fny, fnz = nx, ny, nz
            for axis, val in zip(["x", "y", "z"], [nx, ny, nz]):
                key = f"{lm_name}_{axis}"
                if key not in self.filters:
                    self.filters[key] = OneEuroFilter1D(t_sec, val, self.min_cutoff, self.beta, self.d_cutoff)
                    f_val = val
                else:
                    f_val = self.filters[key].filter(t_sec, val)

                if axis == "x":
                    fnx = f_val
                elif axis == "y":
                    fny = f_val
                else:
                    fnz = f_val

            filtered_pts.append((fnx, fny, fnz))

        return filtered_pts


# ── Scale Normalization & Centering ───────────────────────────────────────────
class HandScaleNormalizer:
    """Computes 8-distance palm RMS scale L_hand and normalizes 21 joint 3D coordinates."""

    def normalize(self, pts_px: list[tuple[float, float, float]], center_wrist: bool = True) -> list[tuple[float, float, float]]:
        w_x, w_y, w_z = pts_px[WRIST_INDEX]
        i_x, i_y, _   = pts_px[INDEX_MCP_INDEX]
        m_x, m_y, _   = pts_px[MIDDLE_MCP_INDEX]
        r_x, r_y, _   = pts_px[RING_MCP_INDEX]
        p_x, p_y, _   = pts_px[PINKY_MCP_INDEX]

        d1_sq = (i_x - w_x) ** 2 + (i_y - w_y) ** 2
        d2_sq = (m_x - w_x) ** 2 + (m_y - w_y) ** 2
        d3_sq = (r_x - w_x) ** 2 + (r_y - w_y) ** 2
        d4_sq = (p_x - w_x) ** 2 + (p_y - w_y) ** 2
        d5_sq = (m_x - i_x) ** 2 + (m_y - i_y) ** 2
        d6_sq = (r_x - m_x) ** 2 + (r_y - m_y) ** 2
        d7_sq = (p_x - r_x) ** 2 + (p_y - r_y) ** 2
        d8_sq = (p_x - i_x) ** 2 + (p_y - i_y) ** 2

        l_hand = math.sqrt((d1_sq + d2_sq + d3_sq + d4_sq + d5_sq + d6_sq + d7_sq + d8_sq) / 8.0)
        if l_hand <= 0:
            l_hand = 1.0

        offset_x = w_x if center_wrist else 0.0
        offset_y = w_y if center_wrist else 0.0
        offset_z = w_z if center_wrist else 0.0

        return [
            ((px - offset_x) / l_hand, (py - offset_y) / l_hand, (pz - offset_z) / l_hand)
            for px, py, pz in pts_px
        ]


def process_streaming_frame(
    raw_pts: list[tuple[float, float, float]],
    frame_w: float,
    frame_h: float,
    t_sec: float,
    normalizer: HandScaleNormalizer,
    euro_filter_bank: OneEuroFilterBank
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    """
    Executes exact process.sh per-frame sequence:
    1. Converts raw MediaPipe [0,1] 3D coordinates to pixel space: x_px = raw_x*w, y_px = raw_y*h, z_px = raw_z*w.
    2. Applies HandScaleNormalizer (8-distance palm RMS scale L_hand and wrist translation centering).
    3. Applies continuous 1€ Filter (min=3.0, beta=1.4, d=1.0) ON THE NORMALIZED COORDINATES (process.sh Step 3).
    Returns (f_dict, smooth_pts_px) where smooth_pts_px contains filtered 2D pixel coordinates for silk-smooth GUI rendering.
    """
    pts_px = [(x * frame_w, y * frame_h, z * frame_w) for (x, y, z) in raw_pts]
    norm_pts = normalizer.normalize(pts_px, center_wrist=True)
    filtered_norm_pts = euro_filter_bank.filter_frame(t_sec, norm_pts)

    f_dict = {}
    for lm_idx, lm_name in enumerate(ALL_21_LANDMARK_NAMES):
        nx, ny, nz = filtered_norm_pts[lm_idx]
        f_dict[f"{lm_name}_x"] = nx
        f_dict[f"{lm_name}_y"] = ny
        f_dict[f"{lm_name}_z"] = nz

    # Reconstruct 1€ filtered smooth pixel coordinates for zero-jitter screen rendering
    w_x, w_y, _ = pts_px[WRIST_INDEX]
    i_x, i_y, _ = pts_px[INDEX_MCP_INDEX]
    m_x, m_y, _ = pts_px[MIDDLE_MCP_INDEX]
    r_x, r_y, _ = pts_px[RING_MCP_INDEX]
    p_x, p_y, _ = pts_px[PINKY_MCP_INDEX]

    d_sq = [
        (i_x - w_x) ** 2 + (i_y - w_y) ** 2,
        (m_x - w_x) ** 2 + (m_y - w_y) ** 2,
        (r_x - w_x) ** 2 + (r_y - w_y) ** 2,
        (p_x - w_x) ** 2 + (p_y - w_y) ** 2,
        (m_x - i_x) ** 2 + (m_y - i_y) ** 2,
        (r_x - m_x) ** 2 + (r_y - m_y) ** 2,
        (p_x - r_x) ** 2 + (p_y - r_y) ** 2,
        (p_x - i_x) ** 2 + (p_y - i_y) ** 2,
    ]
    l_hand = math.sqrt(sum(d_sq) / 8.0)
    if l_hand <= 0:
        l_hand = 1.0

    smooth_pts_px = [
        (w_x + nx * l_hand, w_y + ny * l_hand)
        for (nx, ny, _) in filtered_norm_pts
    ]

    return f_dict, smooth_pts_px


# ── Velocity Calculation for 5-Frame Window ──────────────────────────────────
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


# ── Quality & Dataset Cleaning Filters (process.sh Steps 8 & 9) ─────────────
def validate_realtime_window_quality(
    v_steps_4: list[dict[str, float]],
    hand_scores_5: list[float] = None,
    min_avg_score: float = 0.65,
    check_zero_vel: bool = True
) -> tuple[bool, str]:
    """
    Applies real-time window validation corresponding to process.sh pipeline stages 8 & 9:
    1. Check Zero Velocity Spikes (--remove-zero-vel-touch):
       If all 4 velocity steps across all 21 joints are exactly 0.0, reject window.
    2. Check Hand Score Quality (--min-avg-score 0.65):
       If average hand confidence score across 5 frames < 0.65, reject window.

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


# ── Per-Finger Unrolling ──────────────────────────────────────────────────────
def unroll_per_finger_window(norm_frames_5: list[dict[str, float]], v_steps_4: list[dict[str, float]]) -> dict[str, dict]:
    """
    Unrolls 5-frame landmark window & 4-step velocity window into 5 distinct per-finger data dicts:
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
                row[f"{j}{v}_speed_2d"] = v_data[f"{j}_speed_2d"]
                row[f"{j}{v}_speed_3d"] = v_data[f"{j}_speed_3d"]

            # Mapped finger-specific joints (pip, dip, tip)
            for j_role, orig_name in j_map.items():
                row[f"{j_role}{v}_vx"] = v_data[f"{orig_name}_vx"]
                row[f"{j_role}{v}_vy"] = v_data[f"{orig_name}_vy"]
                row[f"{j_role}{v}_vz"] = v_data[f"{orig_name}_vz"]
                row[f"{j_role}{v}_speed_2d"] = v_data[f"{orig_name}_speed_2d"]
                row[f"{j_role}{v}_speed_3d"] = v_data[f"{orig_name}_speed_3d"]

        finger_rows[finger] = row

    return finger_rows


# ── Variant Feature Tensor Extractor ──────────────────────────────────────────
def extract_variant_tensor(finger_rows: dict[str, dict], variant_name: str) -> np.ndarray:
    """
    Given unrolled finger_rows dict for 5 fingers:
    Extracts numerical feature tensor of shape (5, seq_len, feature_dim) matching target variant schema.
    Returns float32 NumPy array of shape (5, seq_len, feature_dim).
    """
    variant_name = variant_name.lower().strip()
    finger_list = FINGERS  # [thumb, index, middle, ring, pinky]

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

    else:
        raise ValueError(f"Unsupported variant_name: '{variant_name}'")
