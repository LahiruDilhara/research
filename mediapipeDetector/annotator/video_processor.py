"""
annotator/video_processor.py

Sliding-window management and per-window CSV data extraction.
"""
import io
import math
import os

from PIL import Image

from annotator.constants import (
    FINGERS, JOINT_LABELS, WINDOW_SIZE, WINDOW_STEP,
)


# ── Window utilities ─────────────────────────────────────────────────────────

def window_count(total_frames: int) -> int:
    """Number of valid 5-frame windows that can be formed."""
    count = 0
    while count * WINDOW_STEP + WINDOW_SIZE <= total_frames:
        count += 1
    return count


def get_window_frames(frame_data: list, window_idx: int) -> list | None:
    """
    Return the 5 frame_data dicts for window_idx, or None if there are not
    enough frames (i.e. fewer than 3 new frames available past the overlap).
    """
    start = window_idx * WINDOW_STEP
    end = start + WINDOW_SIZE
    if end > len(frame_data):
        return None
    return frame_data[start:end]


def window_idx_from_start_frame(start_frame: int) -> int:
    return start_frame // WINDOW_STEP


# ── PIL image helpers ────────────────────────────────────────────────────────

def frames_to_pil(window_frames: list) -> list[Image.Image]:
    """Convert JPEG-cached annotated bytes in window_frames to PIL Images."""
    return [
        Image.open(io.BytesIO(fd["annotated_jpg_bytes"]))
        for fd in window_frames
    ]


# ── CSV data extraction ──────────────────────────────────────────────────────

def _best_vel(vel_list: list) -> tuple[float, float]:
    """Return the velocity with highest magnitude from a list of (vx, vy)."""
    valid = [v for v in vel_list if v is not None]
    if not valid:
        return 0.0, 0.0
    return max(valid, key=lambda v: math.hypot(v[0], v[1]))


def extract_window_record_data(
    window_frames: list,
    video_file: str,
    video_hash: str,
    duration_ms: int,
) -> dict:
    """
    Build the non-annotation portion of a CSV record from 5 processed frames.

    Coordinates : taken from the middle frame (index 2) for positional stability.
    Velocities  : max-magnitude velocity across all 5 frames per joint — this
                  captures the peak touch impulse which is what matters for LSTM.
    """
    sf = window_frames[0]
    ef = window_frames[-1]
    mid = window_frames[2]

    rec: dict = {
        "video_file": os.path.basename(video_file),
        "video_hash": video_hash,
        "duration_ms": duration_ms,
        "start_ms": sf["timestamp_ms"],
        "end_ms": ef["timestamp_ms"],
        "start_frame": sf["frame_idx"],
        "end_frame": ef["frame_idx"],
    }

    # ── Coordinates from middle frame ────────────────────────────────────────
    hd = mid.get("hand_data")
    if hd:
        wx, wy = hd["wrist"]
        rec["wrist_x"] = round(wx, 4)
        rec["wrist_y"] = round(wy, 4)
        for finger in FINGERS:
            joints = hd["fingers"].get(finger, [(0.0, 0.0)] * 3)
            for j_idx, jlabel in enumerate(JOINT_LABELS):
                x, y = joints[j_idx] if j_idx < len(joints) else (0.0, 0.0)
                rec[f"{finger.lower()}_{jlabel.lower()}_x"] = round(x, 4)
                rec[f"{finger.lower()}_{jlabel.lower()}_y"] = round(y, 4)
    else:
        rec["wrist_x"] = 0.0
        rec["wrist_y"] = 0.0
        for finger in FINGERS:
            for jlabel in JOINT_LABELS:
                rec[f"{finger.lower()}_{jlabel.lower()}_x"] = 0.0
                rec[f"{finger.lower()}_{jlabel.lower()}_y"] = 0.0

    # ── Velocities: max-magnitude across 5 frames ────────────────────────────
    wrist_vels: list = []
    finger_vels: dict = {fn: [[] for _ in JOINT_LABELS] for fn in FINGERS}

    for fd in window_frames:
        vd = fd.get("velocity_data")
        if vd is None:
            continue
        wv = vd.get("wrist_velocity")
        if wv is not None:
            wrist_vels.append(wv)
        for fn in FINGERS:
            jvs = vd.get("finger_velocities", {}).get(fn, [])
            for j_idx in range(len(JOINT_LABELS)):
                jv = jvs[j_idx] if j_idx < len(jvs) else None
                if jv is not None:
                    finger_vels[fn][j_idx].append(jv)

    bwv = _best_vel(wrist_vels)
    rec["wrist_vx"] = bwv[0]
    rec["wrist_vy"] = bwv[1]

    for fn in FINGERS:
        for j_idx, jlabel in enumerate(JOINT_LABELS):
            bv = _best_vel(finger_vels[fn][j_idx])
            rec[f"{fn.lower()}_{jlabel.lower()}_vx"] = bv[0]
            rec[f"{fn.lower()}_{jlabel.lower()}_vy"] = bv[1]

    return rec
