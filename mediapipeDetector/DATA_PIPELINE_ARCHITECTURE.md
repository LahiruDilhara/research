# MediaPipe Touch Detection Data Extractor Architecture

## Overview & Data Flow Pipeline

The Touch Detection Data Extractor converts pre-recorded video streams and live camera feeds into unitless, scale-normalized, frame-rate-independent feature vectors formatted for sequence classification models (such as LSTM or Transformer architectures).

Data Flow:
Raw Video Input -> Layer 1 (Sub-sampling & Pacing) -> Layer 2 (Scale Normalization) -> Layer 3 (1€ Denoising) -> Layer 4 (Velocity Calculation) -> Layer 5 (Sliding Window Assembly) -> Layer 6 (CSV Output)

---

## Layer 1: Frame Sub-Sampling & 12 FPS Target Pacing (`annotator/pipeline.py`)

- Purpose: Normalises varying source video frame rates (such as 13 FPS, 15 FPS, 24 FPS, 30 FPS, or 60 FPS) to a uniform 12.0 FPS target.
- Input: Raw video stream (OpenCV `cv2.VideoCapture`).
- Logic:
  1. Inspect the native frame rate of the video: `native_fps = cap.get(cv2.CAP_PROP_FPS)`.
  2. Reject any video where `native_fps < 12.0`.
  3. Calculate the frame skip step:
     `frame_step = max(1, round(native_fps / 12.0))`
     Example: For a 30 FPS video, `frame_step = round(30 / 12) = 2`. The pipeline processes frames 0, 2, 4, 6, etc.
  4. Process only frames matching `native_frame_index % frame_step == 0`.
  5. Synthesize clean 12.0 FPS timestamps so every frame interval dt is exactly 1/12 second (approx. 0.0833 seconds / 83.3 ms):
     `timestamp_ms = int((processed_frame_index / 12.0) * 1000)`
     `t_seconds = timestamp_ms / 1000.0`
- Output: Array of 12 FPS image frames with synchronized 12 FPS timestamps.

---

## Layer 2: Landmark Extraction & Scale Normalization (`_analyze` in `pipeline.py`)

- Purpose: Makes hand landmark coordinates completely unitless and independent of camera resolution (480p, 720p, 1080p, 4K) or the hand's distance from the camera.
- Input: Image frame, MediaPipe HandLandmarker object, timestamp.
- Logic:
  1. MediaPipe detects 21 hand joint pixel coordinates (X_pixel, Y_pixel) for joints 0 through 20.
  2. Calculate the physical hand scale in pixels (`L_hand`) using palm length (Wrist to Middle Finger MCP) and palm width (Index Finger MCP to Pinky Finger MCP):
     `palm_length = sqrt((X_middle_mcp - X_wrist)^2 + (Y_middle_mcp - Y_wrist)^2)`
     `palm_width = sqrt((X_pinky_mcp - X_index_mcp)^2 + (Y_pinky_mcp - Y_index_mcp)^2)`
     `L_raw = sqrt(palm_length^2 + palm_width^2)`
  3. Smooth the hand scale across frames using Exponential Moving Average (alpha = 0.2) to eliminate frame-to-frame skeleton flickering:
     `L_hand = 0.2 * L_raw + 0.8 * previous_L_hand`
     Why EMA is necessary: MediaPipe's bounding box fluctuates by a few pixels on every frame due to sensor noise or micro-shadows. Without EMA smoothing, dividing coordinates by a fluctuating scale creates artificial jitter and fake velocity noise even when the hand is standing completely still.
  4. Divide all 21 pixel coordinates by `L_hand`:
     `X_normalized = X_pixel / L_hand`
     `Y_normalized = Y_pixel / L_hand`
- Output: Normalized landmark coordinates (X_normalized, Y_normalized) measured in unitless hand-lengths.

---

## Layer 3: 1€ Filter Denoising (`_filter` in `pipeline.py`)

- Purpose: Eliminates high-frequency landmark jitter while preserving rapid hand movement trajectories.
- Input: Normalized coordinates (X_normalized, Y_normalized) and frame timestamp in seconds.
- Logic:
  1. Pass each of the 21 normalized landmarks through dedicated 1€ Filters (`min_cutoff = 1.5`, `beta = 5.0`):
     `(X_filtered, Y_filtered) = FingertipFilter.update(t_seconds, X_normalized, Y_normalized)`
  2. Cutoff Adaptation:
     - When movement is slow or still: The filter applies a low cutoff frequency (1.5 Hz) to smooth static hand tremors.
     - When movement is fast: The filter automatically increases the cutoff frequency (scaled by beta = 5.0) to eliminate tracking lag.
- Output: Smooth, noise-filtered normalized coordinates (X_filtered, Y_filtered).

---

## Layer 4: Velocity Calculation & Deadband Hysteresis (`_velocities` in `pipeline.py`)

- Purpose: Computes normalized joint velocities in hand-lengths per second and suppresses stationary micro-jitter.
- Input: Filtered landmark coordinates from the current frame and previous frame, and the time step `dt = current_t - previous_t`.
- Logic:
  1. Calculate raw velocity components for the wrist and 15 finger joints:
     `Vx = (current_X - previous_X) / dt`
     `Vy = (current_Y - previous_Y) / dt`
     Unit: Hand-lengths per second.
  2. Apply velocity deadband threshold (`DEADBAND_VELOCITY_THRESHOLD = 0.4`):
     If `sqrt(Vx^2 + Vy^2) < 0.4`, set `Vx = 0.0` and `Vy = 0.0`.
- Output: Velocity pairs (Vx, Vy) for Wrist + 15 finger joints.

---

## Layer 5: Dynamic Sliding Window Feature Assembly (`video_processor.py`)

- Purpose: Groups processed sequential frames into sliding window sequence samples configured by Window Size (N) and Overlap (O).
- Parameters:
  - `WINDOW_SIZE` (N): Number of frames per window (default: N = 5).
  - `WINDOW_OVERLAP` (O): Shared frames between consecutive windows (default: O = 2).
  - `WINDOW_STEP` (S = N - O): New frames advanced per step (default: S = 3).
- Feature Vector Layout for Each Window Record:
  1. Metadata Columns:
     `video_file`, `video_hash`, `duration_ms`, `start_ms`, `end_ms`, `start_frame`, `end_frame`
  2. Landmark Coordinate Columns (N frames * 16 joints * 2 = 32*N columns):
     `wrist1_x`, `wrist1_y`, `thumb1_mcp_x` ... `pinkyN_dip_y`
  3. Joint Velocity Columns ((N - 1) transitions * 16 joints * 2 = 32*(N - 1) columns):
     `wrist1_vx`, `wrist1_vy`, `thumb1_mcp_vx` ... `pinky(N-1)_dip_vy`
- Feature Count Summary:
  - For default 5-frame window (N = 5):
    7 Metadata + 160 Coordinate Columns + 128 Velocity Columns + 14 Annotation Columns = 309 Total CSV Columns.

---

## Layer 6: Crash-Safe CSV Storage (`csv_manager.py`)

- Purpose: Saves feature vector records to disk using a crash-safe write-and-close pattern.
- Logic:
  1. Open CSV file in append mode (`"a"`).
  2. Format row dictionary to match dynamic headers built by `build_csv_headers(WINDOW_SIZE)`.
  3. Write record row and immediately close the file handle so no data is lost if execution interrupts.

---

## Live Camera Inference Architecture (`live_camera.py`)

Data Flow for Real-Time Inference:
Live Camera Feed -> Threaded Capture -> Stage 1 (12 FPS Pacing) -> Stage 2 (Landmark Analysis & Scale Normalization) -> Stage 3 (1€ Denoising & Hand Selection) -> Stage 4 (Real-time Velocities & Deadband) -> Stage 5 (Dynamic Sliding Window Queue) -> Real-time Model Inference

### Stage 1: Threaded Frame Capture & 12 FPS Pacing (`WebcamStreamThread` & `run_live_camera`)
1. Threaded Capture: `WebcamStreamThread` continuously fetches background camera frames without blocking the main event loop.
2. 12 FPS Pacing: `target_frame_duration = 1.0 / 12.0` (83.3 ms per frame). Loop sleeps for any remaining frame budget:
   `wait_time = target_frame_duration - loop_elapsed`
   `if wait_time > 0: time.sleep(wait_time)`

### Stage 2: MediaPipe Landmark Analysis & Scale Normalization (`analyze_landmarks`)
1. MediaPipe inference extracts 21 2D joint coordinates.
2. Calculates hand scale `L_raw` in pixels from palm length and palm width.
3. EMA scale smoothing (alpha = 0.2): `L_hand = 0.2 * L_raw + 0.8 * previous_L_hand`.
4. Coordinates normalized: `X_norm = X_pixel / L_hand`, `Y_norm = Y_pixel / L_hand`.

### Stage 3: Single-Hand Selection & 1€ Denoising (`filter_landmarks`)
1. Single-hand priority: Always selects 'Left' hand when multiple hands are in view.
2. 1€ Coordinate Filter: Filters 21 normalized landmarks with `min_cutoff = 1.5` and `beta = 5.0`.

### Stage 4: Real-time Velocity & Deadband (`calculate_velocities`)
1. Real-time time delta: `dt = current_t - previous_t` (fallback to 1/12 s if dt <= 0).
2. Velocity components: `Vx = (current_X - previous_X) / dt`, `Vy = (current_Y - previous_Y) / dt`.
3. Velocity deadband: If `sqrt(Vx^2 + Vy^2) < 0.4`, sets `Vx = 0.0` and `Vy = 0.0`.

### Stage 5: Live Sliding Window Velocity Queue (`update_velocity_queue`)
1. Maintains per-finger velocity queues: `collections.deque(maxlen=WINDOW_SIZE)`.
2. Formats 8-element velocity vectors per finger: `[wrist_vx, wrist_vy, mcp_vx, mcp_vy, pip_vx, pip_vy, dip_vx, dip_vy]`.
3. Appends vector to finger queue on every frame. When the queue fills to `WINDOW_SIZE` (default: 5 frames), it forms a complete sliding window sequence ready for model prediction.

---

## Summary of Parity Between Extractor (`pipeline.py`) & Inference (`live_camera.py`)

- Target Rate: Extractor sub-samples video to 12 FPS target; Live Camera paces stream at 12 FPS.
- Scale Normalization: Both use palm scale L_hand with EMA alpha = 0.2.
- Denoising: Both use 1€ Filter with min_cutoff = 1.5 and beta = 5.0.
- Velocity & Deadband: Both calculate velocity in hand-lengths per second with deadband threshold 0.4.
- Sliding Window: Both assemble sequence windows of size WINDOW_SIZE (default: 5 frames).
