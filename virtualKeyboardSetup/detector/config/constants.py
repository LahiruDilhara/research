"""
config/constants.py

App-wide constants for the detector pipeline.
All timing, buffer, and detection parameters are defined here
so they can be referenced from every layer without circular imports.
"""

from __future__ import annotations

# ── Pipeline timing ────────────────────────────────────────────────────────────
TARGET_FPS: float = 12.0
FRAME_INTERVAL: float = 1.0 / TARGET_FPS       # ~83.3 ms

# ── Sliding window ─────────────────────────────────────────────────────────────
WINDOW_SIZE: int = 5          # frames kept in ring buffer
SHIFT_SIZE: int = 3           # frames captured before triggering inference (5 frames, 2-frame overlap -> stride 3)

# ── MediaPipe ──────────────────────────────────────────────────────────────────
MEDIAPIPE_NUM_HANDS: int = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
MEDIAPIPE_MIN_PRESENCE_CONFIDENCE: float = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
MEDIAPIPE_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MEDIAPIPE_MODEL_FILENAME: str = "hand_landmarker.task"

# ── Hand landmark names (21 joints, matching process.sh order) ─────────────────
ALL_21_LANDMARK_NAMES: list[str] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip",  "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp",  "ring_pip",  "ring_dip",  "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Landmark index of each fingertip in the 21-landmark array
FINGERTIP_INDICES: dict[str, int] = {
    "Thumb":  4,
    "Index":  8,
    "Middle": 12,
    "Ring":   16,
    "Pinky":  20,
}

FINGERS: list[str] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ── Hand skeleton drawing ──────────────────────────────────────────────────────
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Thumb":  (0, 140, 255),
    "Index":  (255, 200, 0),
    "Middle": (0, 255, 0),
    "Ring":   (255, 0, 255),
    "Pinky":  (255, 0, 0),
}

# ── AprilTag homography ────────────────────────────────────────────────────────
APRILTAG_FAMILY: str = "tag36h11"
APRILTAG_MIN_MARKERS: int = 1
APRILTAG_SMOOTHING_ALPHA: float = 0.75   # blending weight for previous H
APRILTAG_NTHREADS: int = 1

# ── Touch detection & debouncing ───────────────────────────────────────────────
TOUCH_PROBABILITY_THRESHOLD: float = 0.55
TOUCH_ONSET_THRESHOLD: float = 0.55
TOUCH_RELEASE_THRESHOLD: float = 0.40

# ── Filtration thresholds (matching process.sh & datacreator) ──────────────────
# Whole-hand transit movement threshold (Step 7: max stationary joint displacement / L_hand)
HAND_MOVEMENT_THRESHOLD: float = 0.155
STATIONARY_LANDMARK_NAMES: list[str] = [
    "wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
]

# Window quality & confidence thresholds (Step 10: filter_window_quality.py)
QUALITY_MIN_AVG_SCORE: float = 0.65
QUALITY_MIN_FRAME_SCORE: float = 0.45
QUALITY_MAX_SCORE_DROP: float = 0.35

# Kinetic motion threshold (Step 9: filter_dataset.py --remove-zero-vel-touch)
MIN_KINETIC_SPEED_THRESHOLD: float = 0.008
FINGERTIP_VELOCITY_THRESHOLD: float = 0.008

# ── One Euro (1€) Landmark Coordinate Filter ──────────────────────────────────
ONE_EURO_ENABLED: bool = False
ONE_EURO_MIN_CUTOFF: float = 0.02
ONE_EURO_BETA: float = 0.08
ONE_EURO_D_CUTOFF: float = 1.0

# ── Scale normalisation (HandScaleNormalizer, stage1) ─────────────────────────
WRIST_INDEX: int = 0
INDEX_MCP_INDEX: int = 5
MIDDLE_MCP_INDEX: int = 9
RING_MCP_INDEX: int = 13
PINKY_MCP_INDEX: int = 17

# ── Fingertip Physical Contact Extrapolation ──────────────────────────────────
# Compensates for MediaPipe nail-bed landmark placement and paper perspective tilt
FINGERTIP_OFFSET_ENABLED: bool = True
FINGERTIP_FORWARD_OFFSET_MM: float = 5.0     # Millimeter forward extension along finger direction
FINGERTIP_PHALANX_RATIO: float = 0.45       # fraction of distal phalanx length
FINGERTIP_EXTRA_OFFSET_MM: float = 5.0      # extra base millimeter offset
FINGERTIP_PAPER_ANGLE_FACTOR_MM: float = 6.0 # paper respective cos(theta) factor
FINGERTIP_EXTRA_FRONT_MM: float = 3.0       # frontward paper 90-degree compensation
TOUCH_DEBOUNCE_COOLDOWN_S: float = 0.35     # Minimum cooldown between discrete key taps (prevents sliding window double triggers)

# ── Print scale calibration ──────────────────────────────────────────────────
# Measured marker size from physical printed paper (ruler measurement in mm).
# AprilTag markers are square, so side width defines the marker box size.
# 0.0 means uncalibrated / use layout default (scale factor = 1.0).
PRINTED_MARKER_SIDE_WIDTH_MM: float = 0.0
PRINTED_MARKER_WIDTH_MM: float = 0.0
PRINTED_MARKER_HEIGHT_MM: float = 0.0

DIP_INDICES: dict[str, int] = {
    "Thumb": 3,
    "Index": 7,
    "Middle": 11,
    "Ring": 15,
    "Pinky": 19,
}
PIP_INDICES: dict[str, int] = {
    "Thumb": 2,
    "Index": 6,
    "Middle": 10,
    "Ring": 14,
    "Pinky": 18,
}
MCP_INDICES: dict[str, int] = {
    "Thumb": 1,
    "Index": 5,
    "Middle": 9,
    "Ring": 13,
    "Pinky": 17,
}

# ── UI theme colours (matches designer) ───────────────────────────────────────
UI_BG_DARK     = "#202020"
UI_BG_CARD     = "#1A1A24"
UI_BG_CARD_HVR = "#1E1E2B"
UI_ACCENT      = "#009FEF"
UI_TEXT_PRI    = "#F8FAFC"
UI_TEXT_SEC    = "#94A3B8"
UI_SUCCESS     = "#00DC64"
UI_DANGER      = "#FF4D4D"
