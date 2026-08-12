"""
annotator/video_processor.py

Sliding window utilities and per-window record feature extraction.
Extracts 160 landmark coordinates (5 frames) and 128 velocities (4 transitions)
per 5-frame window record for LSTM sequence input.
"""
import io
import math
import os
from PIL import Image

from annotator.constants import (
    CSV_HEADERS, FINGERS, JOINT_LABELS, WINDOW_OVERLAP, WINDOW_SIZE, WINDOW_STEP,
)


def window_count(total_frames: int) -> int:
    """Calculate the total number of 5-frame sliding windows available."""
    if total_frames < WINDOW_SIZE:
        return 0
    return 1 + (total_frames - WINDOW_SIZE) // WINDOW_STEP


def get_window_frames(frame_data: list, window_idx: int) -> list | None:
    """Return the list of 5 frame dicts for a given window index."""
    start_idx = window_idx * WINDOW_STEP
    end_idx = start_idx + WINDOW_SIZE
    if end_idx > len(frame_data):
        return None
    return frame_data[start_idx:end_idx]


def window_idx_from_start_frame(start_frame: int) -> int:
    """Calculate window index from start_frame number."""
    return max(0, start_frame // WINDOW_STEP)


def frames_to_pil(window_frames: list) -> list[Image.Image]:
    """Convert annotated frame JPEG bytes or RGB arrays to PIL Image instances."""
    res = []
    for fd in window_frames:
        if "annotated_jpg_bytes" in fd:
            res.append(Image.open(io.BytesIO(fd["annotated_jpg_bytes"])))
        elif "annotated_frame_rgb" in fd:
            res.append(Image.fromarray(fd["annotated_frame_rgb"]))
    return res


def extract_window_record_data(
    window_frames: list,
    video_file: str,
    video_hash: str,
    duration_ms: int,
) -> dict:
    """
    Extract per-window feature vector:
    - Metadata: video_file, video_hash, duration_ms, start/end timestamps & frame indices
    - 160 Coordinate columns (5 frames x 16 joints x 2 (x,y)):
      wrist1_x..wrist5_y, thumb1_mcp_x..pinky5_dip_y
    - 128 Velocity columns (4 transitions x 16 joints x 2 (vx,vy)):
      wrist1_vx..wrist4_vy, thumb1_mcp_vx..pinky4_dip_vy
    """
    sf = window_frames[0]
    ef = window_frames[-1]

    rec: dict = {
        "video_file": os.path.basename(video_file),
        "video_hash": video_hash,
        "duration_ms": duration_ms,
        "start_ms": sf["timestamp_ms"],
        "end_ms": ef["timestamp_ms"],
        "start_frame": sf["frame_idx"],
        "end_frame": ef["frame_idx"],
    }

    # ── 160 Landmark coordinates for all 5 frames (f_step 1..5) ─────────────
    for f_idx, fd in enumerate(window_frames):
        f_step = f_idx + 1
        hd = fd.get("hand_data")
        if hd:
            wx, wy = hd["wrist"]
            rec[f"wrist{f_step}_x"] = round(wx, 4)
            rec[f"wrist{f_step}_y"] = round(wy, 4)
            for finger in FINGERS:
                joints = hd["fingers"].get(finger, [(0.0, 0.0)] * 3)
                for j_idx, jlabel in enumerate(JOINT_LABELS):
                    x, y = joints[j_idx] if j_idx < len(joints) else (0.0, 0.0)
                    rec[f"{finger.lower()}{f_step}_{jlabel.lower()}_x"] = round(x, 4)
                    rec[f"{finger.lower()}{f_step}_{jlabel.lower()}_y"] = round(y, 4)
        else:
            rec[f"wrist{f_step}_x"] = 0.0
            rec[f"wrist{f_step}_y"] = 0.0
            for finger in FINGERS:
                for jlabel in JOINT_LABELS:
                    rec[f"{finger.lower()}{f_step}_{jlabel.lower()}_x"] = 0.0
                    rec[f"{finger.lower()}{f_step}_{jlabel.lower()}_y"] = 0.0

    # ── 128 Joint velocities for 4 transitions (v_step 1..4) ────────────────
    # Transition v_step (1..4) corresponds to window_frames[1..4]
    for v_idx in range(1, 5):
        v_step = v_idx
        if v_idx < len(window_frames):
            fd = window_frames[v_idx]
            vd = fd.get("velocity_data")
        else:
            vd = None

        if vd:
            wv = vd.get("wrist_velocity") or (0.0, 0.0)
            rec[f"wrist{v_step}_vx"] = round(wv[0], 4)
            rec[f"wrist{v_step}_vy"] = round(wv[1], 4)

            fvels = vd.get("finger_velocities", {})
            for finger in FINGERS:
                jvs = fvels.get(finger, [(0.0, 0.0)] * 3)
                for j_idx, jlabel in enumerate(JOINT_LABELS):
                    jv = jvs[j_idx] if (j_idx < len(jvs) and jvs[j_idx] is not None) else (0.0, 0.0)
                    rec[f"{finger.lower()}{v_step}_{jlabel.lower()}_vx"] = round(jv[0], 4)
                    rec[f"{finger.lower()}{v_step}_{jlabel.lower()}_vy"] = round(jv[1], 4)
        else:
            rec[f"wrist{v_step}_vx"] = 0.0
            rec[f"wrist{v_step}_vy"] = 0.0
            for finger in FINGERS:
                for jlabel in JOINT_LABELS:
                    rec[f"{finger.lower()}{v_step}_{jlabel.lower()}_vx"] = 0.0
                    rec[f"{finger.lower()}{v_step}_{jlabel.lower()}_vy"] = 0.0

    return rec
