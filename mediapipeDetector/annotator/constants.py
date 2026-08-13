"""
annotator/constants.py

All shared constants for the Touch Detection Data Annotator.
Filter parameters are kept identical to live_camera.py.
"""
import os

# ── Landmark configuration (mirrors live_camera.py) ─────────────────────────
FINGER_THREE_LANDMARKS = {
    "Thumb":  [2, 3, 4],
    "Index":  [6, 7, 8],
    "Middle": [10, 11, 12],
    "Ring":   [14, 15, 16],
    "Pinky":  [18, 19, 20],
}
WRIST_INDEX = 0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_COLORS_BGR = {
    "Thumb":  (0, 140, 255),
    "Index":  (255, 200, 0),
    "Middle": (0, 255, 0),
    "Ring":   (255, 0, 255),
    "Pinky":  (255, 0, 0),
}

# Hex color mapping matching OpenCV BGR colors for UI controls
FINGER_COLORS_HEX = {
    "Thumb":  "#ff8c00",   # Vibrant Orange
    "Index":  "#00c8ff",   # Electric Cyan
    "Middle": "#00e676",   # Bright Emerald Green
    "Ring":   "#ff00ff",   # Magenta / Pink
    "Pinky":  "#3d7eff",   # Bright Blue
}

# ── 1€ Filter & Deadband (must match live_camera.py exactly) ─────────────────
FILTER_MIN_CUTOFF = 1.5
FILTER_BETA = 5.0
DEADBAND_VELOCITY_THRESHOLD = 0.4
MISSING_FRAMES_TOLERANCE = 2

# ── Sliding-window parameters ────────────────────────────────────────────────
WINDOW_SIZE = 5      # frames per window
WINDOW_OVERLAP = 2   # shared frames between consecutive windows
WINDOW_STEP = 3      # = WINDOW_SIZE - WINDOW_OVERLAP  (new frames per step)

# ── Annotation option lists ──────────────────────────────────────────────────
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
JOINT_LABELS = ["MCP", "PIP", "DIP"]   # labels for the 3 tracked joints per finger

POV_OPTIONS = ["front", "left", "right"]

ANY_DIFF_PRESETS = [
    "",
    "slightly right-front",
    "slightly left-front",
    "extreme angle",
    "partial occlusion",
    "lighting variation",
    "blurry / fast motion",
    "hand partially cut off",
]

# ── Model path (root of project) ─────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "hand_landmarker.task")


# ── CSV header builder ───────────────────────────────────────────────────────
def _build_csv_headers() -> list[str]:
    h: list[str] = [
        "video_file", "video_hash", "duration_ms",
        "start_ms", "end_ms", "start_frame", "end_frame",
    ]

    # ── 160 Landmark coordinates (5 frames per window: f_step 1..5) ─────────
    for f_step in range(1, 6):
        h += [f"wrist{f_step}_x", f"wrist{f_step}_y"]
        for finger in FINGERS:
            for joint in JOINT_LABELS:
                h += [
                    f"{finger.lower()}{f_step}_{joint.lower()}_x",
                    f"{finger.lower()}{f_step}_{joint.lower()}_y",
                ]

    # ── 128 Joint velocities (4 transitions per 5-frame window: v_step 1..4) ──
    for v_step in range(1, 5):
        h += [f"wrist{v_step}_vx", f"wrist{v_step}_vy"]
        for finger in FINGERS:
            for joint in JOINT_LABELS:
                h += [
                    f"{finger.lower()}{v_step}_{joint.lower()}_vx",
                    f"{finger.lower()}{v_step}_{joint.lower()}_vy",
                ]

    # ── Annotation columns ───────────────────────────────────────────────────
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        h.append(f"{finger}_touch")
    h += [
        "hand_move", "hand_point_of_view", "hand_closer",
        "hovering", "daylight", "hand_visible", "out_of_sync", "rightHand", "any_difference",
    ]
    return h


CSV_HEADERS: list[str] = _build_csv_headers()
