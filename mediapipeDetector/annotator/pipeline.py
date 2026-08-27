"""
annotator/pipeline.py

Full MediaPipe hand-landmark pipeline — mirrors live_camera.py exactly:
  scale normalisation → 1€ filter → velocity with deadband.

Includes regular flat fingertip text annotations (Thumb, Index, Middle, Ring, Pinky)
overlayed on the frame image for clear visual identification.
"""
import logging
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

from one_euro_filter import FingertipFilter
from annotator.constants import (
    FINGER_THREE_LANDMARKS, WRIST_INDEX, HAND_CONNECTIONS, FINGER_COLORS_BGR,
    FILTER_MIN_CUTOFF, FILTER_BETA, DEADBAND_VELOCITY_THRESHOLD,
    MISSING_FRAMES_TOLERANCE, MODEL_PATH, TARGET_FPS,
)

logger = logging.getLogger("Annotator.Pipeline")


# ── Stateful pipeline state (one instance per video) ────────────────────────

class _PipelineState:
    """Holds all mutable filter / tracker state across sequential video frames."""

    def __init__(self):
        # Scale normalisation
        self.prev_scales: dict = {}
        self.scale_stale: dict = {}
        # 1€ filter
        self.filters: dict = {}
        self.filter_stale: dict = {}
        # Velocity
        self.prev_hands: dict = {}
        self.vel_stale: dict = {}


# ── Hand-scale calculation (identical to live_camera.py) ────────────────────

def _hand_scale(pts: list, prev: float | None = None, alpha: float = 1.0) -> float:
    w_x, w_y = pts[WRIST_INDEX]
    m_x, m_y = pts[9]
    i_x, i_y = pts[5]
    p_x, p_y = pts[17]
    palm_len = math.hypot(m_x - w_x, m_y - w_y)
    palm_wid = math.hypot(p_x - i_x, p_y - i_y)
    raw = math.hypot(palm_len, palm_wid)
    if raw <= 0:
        raw = 1.0
    return alpha * raw + (1.0 - alpha) * prev if (prev is not None and alpha < 1.0) else raw


# ── Step 1 → MediaPipe detection + scale normalisation ──────────────────────

def _analyze(frame_bgr: any, landmarker, timestamp_ms: int,
             state: _PipelineState) -> list:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms)

    if not result.hand_landmarks or not result.handedness:
        for lbl in list(state.prev_scales):
            state.scale_stale[lbl] = state.scale_stale.get(lbl, 0) + 1
            if state.scale_stale[lbl] >= MISSING_FRAMES_TOLERANCE:
                state.prev_scales.pop(lbl, None)
                state.scale_stale.pop(lbl, None)
        return []

    raw, seen = [], set()
    for idx, lms in enumerate(result.hand_landmarks):
        lbl = result.handedness[idx][0].category_name
        seen.add(lbl)
        state.scale_stale[lbl] = 0
        pts_px = [(lm.x * w, lm.y * h) for lm in lms]
        l_hand = _hand_scale(pts_px, state.prev_scales.get(lbl), alpha=1.0)
        state.prev_scales[lbl] = l_hand
        pts_n = [(px / l_hand, py / l_hand) for px, py in pts_px]
        raw.append({
            "hand": lbl,
            "wrist": pts_n[WRIST_INDEX],
            "wrist_pixel": pts_px[WRIST_INDEX],
            "all_pts": pts_n,
            "all_pts_pixel": pts_px,
            "fingers": {fn: [pts_n[i] for i in idxs]
                        for fn, idxs in FINGER_THREE_LANDMARKS.items()},
            "fingers_pixel": {fn: [pts_px[i] for i in idxs]
                              for fn, idxs in FINGER_THREE_LANDMARKS.items()},
            "l_hand": l_hand,
        })
    for lbl in set(state.prev_scales) - seen:
        state.scale_stale[lbl] = state.scale_stale.get(lbl, 0) + 1
        if state.scale_stale[lbl] >= MISSING_FRAMES_TOLERANCE:
            state.prev_scales.pop(lbl, None)
            state.scale_stale.pop(lbl, None)
    return raw


# ── Step 2 → single-hand selection (prefer Left) ────────────────────────────

def _select(raw: list) -> list:
    if not raw:
        return []
    for h in raw:
        if h["hand"] == "Left":
            return [h]
    return [raw[0]]


# ── Step 3 → 1€ filter on normalised landmarks ──────────────────────────────

def _filter(raw: list, state: _PipelineState, t: float) -> list:
    if not raw:
        for lbl in list(state.filters):
            state.filter_stale[lbl] = state.filter_stale.get(lbl, 0) + 1
            if state.filter_stale[lbl] >= MISSING_FRAMES_TOLERANCE:
                state.filters.pop(lbl, None)
                state.filter_stale.pop(lbl, None)
        return []

    out, seen = [], set()
    for hi in raw:
        lbl = hi["hand"]
        seen.add(lbl)
        state.filter_stale[lbl] = 0
        l_hand = hi["l_hand"]
        if lbl not in state.filters:
            state.filters[lbl] = [
                FingertipFilter(FILTER_MIN_CUTOFF, FILTER_BETA) for _ in range(21)
            ]
        fls = state.filters[lbl]
        fpts = [fls[i].update(t, x, y) for i, (x, y) in enumerate(hi["all_pts"])]
        fpts_px = [(round(x * l_hand, 2), round(y * l_hand, 2)) for x, y in fpts]
        out.append({
            "hand": lbl,
            "wrist": fpts[WRIST_INDEX],
            "wrist_pixel": fpts_px[WRIST_INDEX],
            "all_pts": fpts,
            "all_pts_pixel": fpts_px,
            "fingers": {fn: [fpts[i] for i in idxs]
                        for fn, idxs in FINGER_THREE_LANDMARKS.items()},
            "fingers_pixel": {fn: [fpts_px[i] for i in idxs]
                              for fn, idxs in FINGER_THREE_LANDMARKS.items()},
            "l_hand": l_hand,
        })
    for lbl in set(state.filters) - seen:
        state.filter_stale[lbl] = state.filter_stale.get(lbl, 0) + 1
        if state.filter_stale[lbl] >= MISSING_FRAMES_TOLERANCE:
            state.filters.pop(lbl, None)
            state.filter_stale.pop(lbl, None)
    return out


# ── Step 4 → frame-rate-independent velocity + deadband ─────────────────────

def _velocities(hands: list, state: _PipelineState, t: float) -> list:
    if not hands:
        for lbl in list(state.prev_hands):
            state.vel_stale[lbl] = state.vel_stale.get(lbl, 0) + 1
            if state.vel_stale[lbl] >= MISSING_FRAMES_TOLERANCE:
                state.prev_hands.pop(lbl, None)
                state.vel_stale.pop(lbl, None)
        return []

    out, seen = [], set()
    for hi in hands:
        lbl = hi["hand"]
        seen.add(lbl)
        state.vel_stale[lbl] = 0
        cw, cf = hi["wrist"], hi["fingers"]
        wv, fv = None, {}

        if lbl in state.prev_hands:
            ph = state.prev_hands[lbl]
            dt = t - ph["timestamp"]
            if dt <= 0:
                dt = 1.0 / 30.0
            pw = ph.get("wrist")
            if pw is not None:
                vx = round((cw[0] - pw[0]) / dt, 4)
                vy = round((cw[1] - pw[1]) / dt, 4)
                wv = (0.0, 0.0) if math.hypot(vx, vy) < DEADBAND_VELOCITY_THRESHOLD else (vx, vy)
            pf = ph.get("fingers", {})
            for fn, cc in cf.items():
                if fn in pf and pf[fn] is not None:
                    jvels = []
                    for c, p in zip(cc, pf[fn]):
                        fx = round((c[0] - p[0]) / dt, 4)
                        fy = round((c[1] - p[1]) / dt, 4)
                        jvels.append(
                            (0.0, 0.0) if math.hypot(fx, fy) < DEADBAND_VELOCITY_THRESHOLD
                            else (fx, fy)
                        )
                    fv[fn] = jvels
                else:
                    fv[fn] = [None, None, None]
        else:
            wv = None
            fv = {fn: [None, None, None] for fn in cf}

        state.prev_hands[lbl] = {"wrist": cw, "fingers": cf, "timestamp": t}
        out.append({"hand": lbl, "wrist_velocity": wv, "finger_velocities": fv})

    for lbl in set(state.prev_hands) - seen:
        state.vel_stale[lbl] = state.vel_stale.get(lbl, 0) + 1
        if state.vel_stale[lbl] >= MISSING_FRAMES_TOLERANCE:
            state.prev_hands.pop(lbl, None)
            state.vel_stale.pop(lbl, None)
    return out


# ── Skeleton drawing helper ──────────────────────────────────────────────────

def draw_skeleton(frame_bgr: any, hand_data: dict) -> any:
    """
    Overlay MediaPipe skeleton and regular flat fingertip text labels on a BGR frame.
    Returns a new annotated frame.
    """
    frame = frame_bgr.copy()
    pts = hand_data.get("all_pts_pixel", [])
    
    # Draw connections
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame,
                     (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])),
                     (200, 200, 200), 1)

    # Draw wrist
    wp = hand_data.get("wrist_pixel", (0, 0))
    cv2.circle(frame, (int(wp[0]), int(wp[1])), 6, (0, 255, 255), -1)

    # Draw finger joint circles and regular fingertip labels
    for fn, coords in hand_data.get("fingers_pixel", {}).items():
        color = FINGER_COLORS_BGR.get(fn, (255, 255, 255))
        for j, pt in enumerate(coords):
            is_tip = (j == 2)
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(frame, (px, py), 6 if is_tip else 4, color, -1)

            # Annotate finger name on the fingertip as a regular flat word
            if is_tip:
                text_pos = (px + 8, py - 4)
                cv2.putText(frame, fn, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)

    return frame


# ── Full video processing entry point ────────────────────────────────────────

def process_video(
    video_path: str,
    model_path: str = MODEL_PATH,
    progress_cb=None,
) -> tuple[list, float, int, int]:
    """
    Runs the full pipeline on every frame of the video.
    """
    logger.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise RuntimeError(f"Cannot open video: {video_path}")

    header_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    logger.info(
        f"Video opened: {w}x{h} resolution, {native_fps:.2f} FPS, header frame count {header_total}"
    )

    # ── FPS validation & downsampling to 12 FPS target ─────────────────────────
    if native_fps < TARGET_FPS:
        cap.release()
        msg = f"Video FPS ({native_fps:.2f}) is less than {TARGET_FPS:.0f} FPS. Processing aborted."
        logger.error(msg)
        raise ValueError(msg)

    frame_step = max(1, round(native_fps / TARGET_FPS))
    logger.info(
        f"FPS normalisation: native {native_fps:.2f} FPS → "
        f"processing every {frame_step} frame(s) → effective {TARGET_FPS:.0f} FPS"
    )

    # ── MediaPipe init ────────────────────────────────────────────────────────
    logger.info(f"Initializing MediaPipe HandLandmarker from model: {model_path}")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)
    logger.info("MediaPipe HandLandmarker successfully loaded.")

    state = _PipelineState()
    frame_data = []
    native_fi = 0
    processed_fi = 0

    # Phase 1: Read all raw frames and timestamps into memory
    all_raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        all_raw_frames.append((frame, msec))

    cap.release()

    if not all_raw_frames:
        logger.info("Empty video file.")
        landmarker.close()
        return [], TARGET_FPS, 0, 0

    total_native_frames = len(all_raw_frames)
    last_msec = all_raw_frames[-1][1]
    duration_sec = last_msec / 1000.0
    calculated_native_fps = total_native_frames / duration_sec if duration_sec > 0 else 0
    expected_target_frames = int(duration_sec * TARGET_FPS)

    logger.info("=== Video Scan Statistics ===")
    logger.info(f"Total Native Frames Available : {total_native_frames} frames")
    logger.info(f"Full Length of Video        : {duration_sec:.3f} seconds ({last_msec:.1f} ms)")
    logger.info(f"Estimated Native FPS        : {calculated_native_fps:.2f} FPS")
    logger.info(f"Targeting Processing FPS    : {TARGET_FPS} FPS")
    logger.info(f"Expected Processed Frames   : ~{expected_target_frames} frames")
    logger.info("=============================")

    # Phase 2: Downsample to exactly 12 FPS by nearest timestamp
    buffered_frames = []
    processed_fi = 0
    native_idx = 0

    while native_idx < total_native_frames:
        curr_frame, curr_time = all_raw_frames[native_idx]
        target_time = processed_fi * (1000.0 / TARGET_FPS)

        # Stop if we are on the very last frame and target_time has surpassed it
        is_last_frame = (native_idx == total_native_frames - 1)
        if is_last_frame and target_time > curr_time:
            break

        if not is_last_frame:
            _, next_time = all_raw_frames[native_idx + 1]
            if abs(next_time - target_time) < abs(curr_time - target_time):
                native_idx += 1
                continue

        buffered_frames.append((curr_frame, int(target_time)))
        processed_fi += 1

    # Discard any trailing frames that cannot form a complete sliding window before MediaPipe processing
    from annotator.constants import WINDOW_SIZE, WINDOW_STEP
    total_extracted = len(buffered_frames)
    if total_extracted >= WINDOW_SIZE:
        valid_len = ((total_extracted - WINDOW_SIZE) // WINDOW_STEP) * WINDOW_STEP + WINDOW_SIZE
        if valid_len < total_extracted:
            logger.info(f"Throwing away {total_extracted - valid_len} trailing frame(s) to align with exact window boundaries before processing.")
            buffered_frames = buffered_frames[:valid_len]
    else:
        logger.info("Not enough frames to form a single window. Discarding all.")
        buffered_frames = []

    # Run MediaPipe and 1Euro filter on the exact valid frames
    for i, (frame, ts_ms) in enumerate(buffered_frames):
        t_s = ts_ms / 1000.0

        raw = _analyze(frame, landmarker, ts_ms, state)
        single = _select(raw)
        filtered = _filter(single, state, t_s)
        vel = _velocities(filtered, state, t_s)

        annotated = frame.copy()
        hd = filtered[0] if filtered else None
        if hd:
            annotated = draw_skeleton(annotated, hd)
        cv2.putText(annotated, f"F:{i}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(annotated, f"{ts_ms}ms", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpg_bytes = buf.tobytes()

        frame_data.append({
            "frame_idx": i,
            "timestamp_ms": ts_ms,
            "annotated_jpg_bytes": jpg_bytes,
            "hand_data": hd,
            "velocity_data": vel[0] if vel else None,
        })

        denom = len(buffered_frames)
        if progress_cb:
            progress_cb(i + 1, denom)

        # Yield GIL briefly
        time.sleep(0.001)

    landmarker.close()

    actual_total = len(frame_data)
    duration_ms = int((actual_total / TARGET_FPS) * 1000)
    logger.info(
        f"MediaPipe video processing finished. Extracted {actual_total} frames "
        f"at effective {TARGET_FPS:.0f} FPS (from {native_fps:.2f} FPS source, step={frame_step}), "
        f"duration {duration_ms} ms."
    )
    return frame_data, TARGET_FPS, actual_total, duration_ms
