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
SHIFT_SIZE: int = 2           # frames captured before triggering inference

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

# ── Touch detection ────────────────────────────────────────────────────────────
TOUCH_PROBABILITY_THRESHOLD: float = 0.5

# ── Scale normalisation (HandScaleNormalizer, stage1) ─────────────────────────
WRIST_INDEX: int = 0
INDEX_MCP_INDEX: int = 5
MIDDLE_MCP_INDEX: int = 9
RING_MCP_INDEX: int = 13
PINKY_MCP_INDEX: int = 17

# ── UI theme colours (matches designer) ───────────────────────────────────────
UI_BG_DARK     = "#202020"
UI_BG_CARD     = "#1A1A24"
UI_BG_CARD_HVR = "#1E1E2B"
UI_ACCENT      = "#009FEF"
UI_TEXT_PRI    = "#F8FAFC"
UI_TEXT_SEC    = "#94A3B8"
UI_SUCCESS     = "#00DC64"
UI_DANGER      = "#FF4D4D"
