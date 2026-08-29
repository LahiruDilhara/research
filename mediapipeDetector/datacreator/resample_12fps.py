# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.2",
#     "opencv-python>=5.0.0.93",
# ]
# ///

"""
datacreator/resample_12fps.py

Resamples input video files to exactly 12.0 FPS using nearest-timestamp matching,
identical to the frame downsampling calculation in annotator/pipeline.py.

Supports processing a single video file or a list/glob pattern of video files (e.g. '*', '*.mp4', 'videos/*').
All output files are saved into a specified output directory with the same base name and .mp4 extension.

Default behavior: Overwrites existing destination video files.
If '--skip' is passed: Skips conversion if the destination video file already exists.
"""

import argparse
import glob
import os
import sys
import cv2

TARGET_FPS = 12.0


def resample_video_to_12fps(
    input_path: str,
    output_path: str,
    target_fps: float = TARGET_FPS,
    skip_existing: bool = False
) -> str | None:
    """
    Reads all video frames, calculates actual time range and frame count dynamically (does not rely on header FPS),
    downsamples to target_fps (12.0 FPS) by nearest timestamp, and saves out an MP4 video file.

    If output_path is an existing directory (or ends with a path separator), the resampled video will be saved inside
    output_path using the base name of input_path with a .mp4 extension.

    If skip_existing is True and destination file exists, skips conversion.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    # If output_path is specified as a directory, append base_name.mp4
    if os.path.isdir(output_path) or output_path.endswith(os.sep) or output_path.endswith("/") or output_path.endswith("\\"):
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{base_name}.mp4")

    # Check --skip condition
    if skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"[Skip] Destination video '{output_path}' already exists. Skipping (--skip enabled).")
        return output_path

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
        print(f"[Warning] Video '{input_path}' is empty. Skipping.")
        return None

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
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    for frame in resampled_frames:
        out.write(frame)

    out.release()

    print(f"[Success] {target_fps:.1f} FPS MP4 Video saved to: {output_path}")
    print(f"          Output Frame Count: {len(resampled_frames)} frames ({len(resampled_frames) / target_fps:.2f} seconds)")
    return output_path


def collect_input_files(input_patterns: list[str]) -> list[str]:
    """
    Expands glob patterns or directory paths into a list of file paths.
    """
    matched_files = []
    for pattern in input_patterns:
        search_pattern = os.path.join(pattern, "*") if os.path.isdir(pattern) else pattern

        glob_matches = glob.glob(search_pattern, recursive=True)
        if glob_matches:
            for filepath in sorted(glob_matches):
                if os.path.isfile(filepath) and filepath not in matched_files:
                    matched_files.append(filepath)
        elif os.path.isfile(pattern) and pattern not in matched_files:
            matched_files.append(pattern)
        else:
            print(f"[Warning] No files found for input pattern/path: '{pattern}'")

    return matched_files


def main():
    parser = argparse.ArgumentParser(
        description="Resample video(s) to 12.0 FPS MP4 format matching annotator.py logic"
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input video file path(s) or glob pattern(s) (e.g. video.mp4, '*.mp4', 'videos/*', 'v1.mov v2.avi')"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory path where processed MP4 files will be saved"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Target FPS (default: 12.0)"
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip conversion if destination video file already exists (by default, existing destination videos are overwritten)"
    )

    args = parser.parse_args()

    input_files = collect_input_files(args.input)
    if not input_files:
        print("[Error] No valid input video files found. Exiting.")
        sys.exit(1)

    output_dir = args.output
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        print(f"[Error] Output path '{output_dir}' exists but is not a directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(input_files)} video file(s) to process:")
    for f in input_files:
        print(f"  - {f}")
    print(f"Output directory: {output_dir}")
    if args.skip:
        print("Flag --skip enabled: Will skip conversion if output video already exists.\n")
    else:
        print("Default mode: Existing output video files will be overwritten.\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, input_file in enumerate(input_files, start=1):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        out_file_path = os.path.join(output_dir, f"{base_name}.mp4")

        print(f"[{idx}/{len(input_files)}] Processing: {input_file} -> {out_file_path}")
        try:
            res = resample_video_to_12fps(input_file, out_file_path, target_fps=args.fps, skip_existing=args.skip)
            if res:
                if args.skip and os.path.exists(out_file_path) and os.path.getsize(out_file_path) > 0 and "[Skip]" in str(res):
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"[Failed] Could not process '{input_file}': {e}")
            fail_count += 1
        print()

    print("==========================================")
    print("Batch Processing Finished:")
    print(f"  Success: {success_count}/{len(input_files)}")
    if skip_count > 0:
        print(f"  Skipped: {skip_count}/{len(input_files)}")
    print(f"  Failed : {fail_count}/{len(input_files)}")
    print("==========================================")

    if success_count == 0 and skip_count == 0 and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
