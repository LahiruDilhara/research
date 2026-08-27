"""
datacreator/resample_12fps.py

Resamples input video files to exactly 12.0 FPS using nearest-timestamp matching,
identical to the frame downsampling calculation in annotator/pipeline.py.
"""

import argparse
import os
import cv2


TARGET_FPS = 12.0


def resample_video_to_12fps(input_path: str, output_path: str, target_fps: float = TARGET_FPS):
    """
    Reads all video frames, calculates actual time range and frame count dynamically (does not rely on header FPS),
    downsamples to target_fps (12.0 FPS) by nearest timestamp, and saves out an MP4 video file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    header_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Phase 1: Read all raw frames and capture actual timestamps from video stream
    all_raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        all_raw_frames.append((frame, msec))

    cap.release()

    if not all_raw_frames:
        print("[Warning] Video is empty. Skipping.")
        return

    total_native_frames = len(all_raw_frames)
    first_msec = all_raw_frames[0][1]
    last_msec = all_raw_frames[-1][1]

    # If stream timestamps are missing or invalid, generate timestamps based on header/default FPS
    if last_msec <= first_msec or any(m < 0 for _, m in all_raw_frames):
        frame_interval_ms = 1000.0 / (header_fps if header_fps > 0 else 30.0)
        all_raw_frames = [(f, idx * frame_interval_ms) for idx, (f, _) in enumerate(all_raw_frames)]
        first_msec = all_raw_frames[0][1]
        last_msec = all_raw_frames[-1][1]

    duration_ms = last_msec - first_msec
    duration_sec = duration_ms / 1000.0
    actual_native_fps = (total_native_frames / duration_sec) if duration_sec > 0 else header_fps

    print("=== Video Frame & Time Range Statistics ===")
    print(f" Input Video Path       : {input_path}")
    print(f" Resolution             : {w}x{h}")
    print(f" Total Native Frames    : {total_native_frames} frames")
    print(f" Actual Video Duration  : {duration_sec:.3f} seconds ({duration_ms:.1f} ms)")
    print(f" Calculated Native FPS  : {actual_native_fps:.2f} FPS (Header reported: {header_fps:.2f})")
    print(f" Target Output FPS      : {target_fps:.1f} FPS")
    print("===========================================")

    if actual_native_fps < target_fps:
        msg = f"Video FPS ({actual_native_fps:.2f}) is less than required {target_fps:.1f} FPS threshold. Processing aborted."
        print(f"[Error] {msg}")
        raise ValueError(msg)

    # Phase 2: Downsample to target_fps by nearest timestamp matching
    resampled_frames = []
    processed_fi = 0
    native_idx = 0

    while native_idx < total_native_frames:
        curr_frame, curr_time = all_raw_frames[native_idx]
        target_time = first_msec + (processed_fi * (1000.0 / target_fps))

        is_last_frame = (native_idx == total_native_frames - 1)
        if is_last_frame and target_time > curr_time:
            break

        if not is_last_frame:
            _, next_time = all_raw_frames[native_idx + 1]
            if abs(next_time - target_time) < abs(curr_time - target_time):
                native_idx += 1
                continue

        resampled_frames.append(curr_frame)
        processed_fi += 1
        native_idx += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    for frame in resampled_frames:
        out.write(frame)

    out.release()

    print(f"[Success] 12 FPS MP4 Video saved to: {output_path}")
    print(f"          Output Frame Count: {len(resampled_frames)} frames ({len(resampled_frames) / target_fps:.2f} seconds)")


def main():
    parser = argparse.ArgumentParser(description="Resample video to 12.0 FPS matching annotator.py logic")
    parser.add_argument("-i", "--input", required=True, help="Input video file path")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")
    parser.add_argument("--fps", type=float, default=12.0, help="Target FPS (default: 12.0)")

    args = parser.parse_args()
    resample_video_to_12fps(args.input, args.output, target_fps=args.fps)


if __name__ == "__main__":
    main()
