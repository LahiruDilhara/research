"""
realtimeprocess/camera_thread.py

Synchronized 12 FPS Processing Pipeline with 5-Queue Per-Finger Routing.

Design:
1. Frame-Lock Video and Skeleton Synchronization:
   - MediaPipe hand joint detection and skeleton overlays are rendered directly onto the exact 12 FPS captured frame.
   - Eliminates temporal mismatch so the skeleton never lags behind the hand in the video preview.
2. Continuous Camera Hardware Draining:
   - The capture thread continuously reads from cv2.VideoCapture to keep the V4L2/OS driver buffer empty.
   - Sub-samples fresh, instantaneous frames at the strict 12 FPS rate (~83.3 ms).
3. Exact process.sh Filtration & 5-Queue Routing:
   - Maintains a 5-frame sliding window with 2-frame shift (stride 3).
   - Enforces whole-hand transit displacement check (threshold: 0.155 L_hand).
   - Computes 4-step frame-to-frame velocities and 2D speeds.
   - Enforces quality checks (min_avg_score: 0.65, min_frame_score: 0.45, max_score_drop: 0.35, zero velocity check).
   - Unrolls valid windows into 5 distinct per-finger records (thumb, index, middle, ring, pinky).
   - Pushes into 5 dedicated queues and signals the inference worker via threading.Event.
4. Event-Driven Inference Worker:
   - Awakens immediately when all 5 queues receive finger records.
   - Runs PyTorch inference using the defined model (e.g. LSTM_All_Combined).
   - Updates HUD predictions and latency.
"""

import time
import queue
import threading
from collections import deque
from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp

from realtimeprocess.stages.stage1_normalizer import HandScaleNormalizer
from realtimeprocess.stages.stage3_velocities import compute_window_velocities
from realtimeprocess.stages.stage4_quality_filter import validate_realtime_window_quality
from realtimeprocess.stages.stage5_finger_unroll import (
    unroll_per_finger_window,
    FINGERS,
)
from realtimeprocess.stages.stage7_hand_movement import (
    validate_hand_movement,
    DEFAULT_DISPLACEMENT_THRESHOLD,
)
from realtimeprocess.realtime_pipeline import (
    process_streaming_frame,
    ALL_21_LANDMARK_NAMES,
)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]


class CameraCaptureThread(threading.Thread):
    """
    Dedicated capture thread continuously draining the webcam buffer.
    Pushes instantaneous frames at the strict 12 FPS interval into the pipeline queue.
    """

    def __init__(self, src=0, target_fps: float = 12.0, frame_queue: queue.Queue = None):
        super().__init__(name="CameraCaptureThread", daemon=True)
        self.src = src
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # 1.0 / 12.0 = 0.08333s (~83.3 ms)
        self.jitter_tolerance = 0.5 * (1.0 / 30.0)  # ~16.7 ms tolerance for discrete webcam frame steps
        self.frame_queue = frame_queue
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.src)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        except Exception:
            pass

        if not cap.isOpened():
            print(f"[CameraCaptureThread] Error: Could not open video source: {self.src}")
            self.running = False
            return

        last_sample_time = 0.0

        try:
            while self.running:
                # Continuously read from camera to keep the hardware driver buffer empty
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                now = time.perf_counter()
                elapsed = now - last_sample_time

                # Sub-sample at the strict 12 FPS interval with jitter tolerance
                if elapsed >= (self.frame_interval - self.jitter_tolerance):
                    last_sample_time = now
                    h, w, _ = frame.shape
                    frame_mirror = cv2.flip(frame, 1)

                    if self.frame_queue is not None:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self.frame_queue.put_nowait((frame_mirror, now, w, h))
                        except queue.Full:
                            pass

        except Exception as e:
            if self.running:
                print(f"[CameraCaptureThread] Error in capture loop: {e}")
        finally:
            cap.release()
            print("[CameraCaptureThread] Camera capture loop terminated.")

    def stop(self):
        self.running = False


class PipelineProcessingThread(threading.Thread):
    """
    Dedicated worker thread processing 12 FPS frames, extracting landmarks,
    rendering synchronized skeleton overlays onto the same frame, and pushing
    unrolled records into 5 dedicated finger queues.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        finger_queues: dict[str, queue.Queue],
        queue_event: threading.Event,
    ):
        super().__init__(name="PipelineProcessingThread", daemon=True)
        self.frame_queue = frame_queue
        self.finger_queues = finger_queues
        self.queue_event = queue_event
        self.running = False

        self.normalizer = HandScaleNormalizer()
        self.landmark_buffer = deque(maxlen=5)
        self.score_buffer = deque(maxlen=5)
        self.shift_counter = 0
        self.last_timestamp_ms = 0

        self.latest_annotated = None
        self.hand_detected = False
        self.frame_width = 640
        self.frame_height = 480
        self.actual_fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.perf_counter()
        self.lock = threading.Lock()

    def run(self):
        self.running = True

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )

        project_root = Path(__file__).resolve().parent.parent
        model_path = str(project_root / "hand_landmarker.task")

        if not Path(model_path).exists():
            print("[PipelineProcessingThread] Downloading hand_landmarker.task model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = HandLandmarker.create_from_options(options)

        try:
            while self.running:
                try:
                    item = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_mirror, now, w, h = item

                # Monotonically increasing timestamp for MediaPipe VIDEO running mode
                timestamp_ms = int(now * 1000)
                if timestamp_ms <= self.last_timestamp_ms:
                    timestamp_ms = self.last_timestamp_ms + 1
                self.last_timestamp_ms = timestamp_ms

                frame_rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Annotate directly onto the exact same 12 FPS frame to eliminate spatial lag
                annotated = frame_mirror.copy()
                detected = False
                raw_pts = []
                hand_score = 0.0
                norm_f_dict = None

                if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                    detected = True
                    landmarks = result.hand_landmarks[0]
                    for lm in landmarks:
                        raw_pts.append((lm.x, lm.y, lm.z))

                    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                        hand_score = float(result.handedness[0][0].score)
                    else:
                        hand_score = 0.85

                    # Exact process.sh Step 2 & 3: Scale normalization on raw coordinates
                    norm_f_dict, smooth_pts_px = process_streaming_frame(
                        raw_pts, w, h, now, self.normalizer
                    )

                    # Draw skeleton lines onto the matching frame
                    for s_idx, e_idx in HAND_CONNECTIONS:
                        x1, y1 = int(smooth_pts_px[s_idx][0]), int(smooth_pts_px[s_idx][1])
                        x2, y2 = int(smooth_pts_px[e_idx][0]), int(smooth_pts_px[e_idx][1])
                        cv2.line(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)

                    # Draw joint circles onto the matching frame
                    for idx, (sx, sy) in enumerate(smooth_pts_px):
                        px, py = int(sx), int(sy)
                        if idx in [4, 8, 12, 16, 20]:
                            cv2.circle(annotated, (px, py), 6, (0, 255, 255), -1)
                        else:
                            cv2.circle(annotated, (px, py), 4, (0, 165, 255), -1)

                with self.lock:
                    self.latest_annotated = annotated
                    self.hand_detected = detected
                    self.frame_width = w
                    self.frame_height = h
                    self.frame_count += 1
                    wall_now = time.perf_counter()
                    fps_dur = wall_now - self.fps_start_time
                    if fps_dur >= 1.0:
                        self.actual_fps = self.frame_count / fps_dur
                        self.frame_count = 0
                        self.fps_start_time = wall_now

                if detected and norm_f_dict is not None:
                    self.landmark_buffer.append(norm_f_dict)
                    self.score_buffer.append(hand_score)
                    self.shift_counter += 1

                    # Form 5-frame window with 2-frame overlap (stride 3) matching process.sh
                    if len(self.landmark_buffer) == 5 and self.shift_counter >= 3:
                        self.shift_counter = 0
                        norm_window_5 = list(self.landmark_buffer)
                        scores_5 = list(self.score_buffer)

                        # Step 8: Compute 4-step velocities and 2D speeds
                        v_steps_4 = compute_window_velocities(norm_window_5)

                        # Step 11: Unroll sequence window into 5 distinct per-finger records
                        finger_rows = unroll_per_finger_window(norm_window_5, v_steps_4)

                        # Partition and push into the 5 dedicated finger queues
                        for finger in FINGERS:
                            q = self.finger_queues[finger]
                            if q.full():
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    pass
                            try:
                                q.put_nowait({
                                    "type": "WINDOW",
                                    "finger": finger,
                                    "row": finger_rows[finger],
                                    "norm_window_5": norm_window_5,
                                    "scores_5": scores_5,
                                    "v_steps_4": v_steps_4,
                                })
                            except queue.Full:
                                pass

                        # Signal inference worker
                        self.queue_event.set()

                else:
                    # Hand lost: clear sequence buffers and drain queues
                    self.landmark_buffer.clear()
                    self.score_buffer.clear()
                    self.shift_counter = 0

                    for finger in FINGERS:
                        q = self.finger_queues[finger]
                        while not q.empty():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                break
                        try:
                            q.put_nowait({
                                "type": "NO_HAND",
                                "finger": finger,
                            })
                        except queue.Full:
                            pass

                    self.queue_event.set()

        except Exception as e:
            if self.running:
                print(f"[PipelineProcessingThread] Error in processing loop: {e}")
        finally:
            try:
                landmarker.close()
            except Exception:
                pass
            print("[PipelineProcessingThread] Pipeline processing loop terminated.")

    def stop(self):
        self.running = False

    def reset_buffers(self):
        """Clears sequence ring buffers to reset state."""
        self.landmark_buffer.clear()
        self.score_buffer.clear()
        self.shift_counter = 0


class InferenceWorkerThread(threading.Thread):
    """
    Dedicated inference worker consuming from the 5 finger queues,
    combining the unrolled finger streams, applying process.sh filters,
    and running forward-pass evaluations using the defined deep learning model.
    """

    def __init__(
        self,
        finger_queues: dict[str, queue.Queue],
        queue_event: threading.Event,
        model_manager,
        hand_movement_threshold: float = DEFAULT_DISPLACEMENT_THRESHOLD,
        min_avg_score: float = 0.65,
        min_frame_score: float = 0.45,
        max_score_drop: float = 0.35,
        callback=None
    ):
        super().__init__(name="InferenceWorkerThread", daemon=True)
        self.finger_queues = finger_queues
        self.queue_event = queue_event
        self.model_manager = model_manager
        self.hand_movement_threshold = hand_movement_threshold
        self.min_avg_score = min_avg_score
        self.min_frame_score = min_frame_score
        self.max_score_drop = max_score_drop
        self.callback = callback
        self.running = False

    def run(self):
        self.running = True

        try:
            while self.running:
                signaled = self.queue_event.wait(timeout=0.1)
                if not self.running:
                    break
                if not signaled:
                    continue
                self.queue_event.clear()

                # Check if all 5 finger queues have data available
                has_data = True
                for f in FINGERS:
                    if self.finger_queues[f].empty():
                        has_data = False
                        break

                if not has_data:
                    continue

                finger_rows = {}
                norm_window_5 = None
                scores_5 = None
                v_steps_4 = None
                is_no_hand = False

                for f in FINGERS:
                    try:
                        item = self.finger_queues[f].get_nowait()
                        if item.get("type") == "NO_HAND":
                            is_no_hand = True
                        else:
                            finger_rows[f] = item["row"]
                            if norm_window_5 is None:
                                norm_window_5 = item.get("norm_window_5")
                                scores_5 = item.get("scores_5")
                                v_steps_4 = item.get("v_steps_4")
                    except queue.Empty:
                        has_data = False
                        break

                if not has_data:
                    continue

                if is_no_hand:
                    if self.model_manager is not None:
                        self.model_manager.reset_touch_states()
                    no_hand_preds = {
                        f: {
                            "touch": False,
                            "prob": 0.0,
                            "reason": "No Hand Detected",
                            "hand_moving": False,
                            "disp": 0.0,
                        }
                        for f in FINGERS
                    }
                    if self.callback:
                        self.callback(no_hand_preds, 0.0)
                    continue

                if len(finger_rows) < 5 or norm_window_5 is None:
                    continue

                t0 = time.perf_counter()

                # Step 7 Filtration: Hand transit movement displacement check (threshold: 0.155)
                is_stationary, max_disp, hm_reason = validate_hand_movement(
                    norm_window_5, threshold=self.hand_movement_threshold
                )

                if not is_stationary:
                    if self.model_manager is not None:
                        self.model_manager.reset_touch_states()
                    rejection_preds = {
                        f: {
                            "touch": False,
                            "prob": 0.0,
                            "reason": hm_reason,
                            "hand_moving": True,
                            "disp": max_disp,
                        }
                        for f in FINGERS
                    }
                    if self.callback:
                        self.callback(rejection_preds, 0.0)
                    continue

                # Step 10 Filtration: Hand score confidence quality checks
                is_valid, q_reason = validate_realtime_window_quality(
                    v_steps_4,
                    scores_5,
                    min_avg_score=self.min_avg_score,
                    min_frame_score=self.min_frame_score,
                    max_score_drop=self.max_score_drop,
                )

                if not is_valid:
                    if self.model_manager is not None:
                        self.model_manager.reset_touch_states()
                    rejection_preds = {
                        f: {
                            "touch": False,
                            "prob": 0.0,
                            "reason": q_reason,
                            "hand_moving": False,
                            "disp": max_disp,
                        }
                        for f in FINGERS
                    }
                    if self.callback:
                        self.callback(rejection_preds, 0.0)
                    continue

                # Combine 5-finger streams and execute PyTorch model inference
                if self.model_manager is not None:
                    preds = self.model_manager.predict_unrolled_fingers(
                        finger_rows,
                        max_disp=max_disp,
                        v_steps_4=v_steps_4
                    )
                else:
                    preds = {
                        f: {
                            "touch": False,
                            "prob": 0.0,
                            "reason": "No Model Manager",
                            "hand_moving": False,
                            "disp": max_disp,
                        }
                        for f in FINGERS
                    }

                latency_ms = (time.perf_counter() - t0) * 1000.0

                if self.callback:
                    try:
                        self.callback(preds, latency_ms)
                    except Exception as e:
                        if self.running:
                            print(f"[InferenceWorkerThread] Callback error: {e}")

        except Exception as e:
            if self.running:
                print(f"[InferenceWorkerThread] Error: {e}")
        finally:
            print("[InferenceWorkerThread] Inference worker loop terminated.")

    def stop(self):
        self.running = False


class CameraThread:
    """
    Orchestrates the 12 FPS Camera Capture Thread, Pipeline Processing Thread,
    5-Queue Per-Finger Routing, and Model Inference Worker Thread.
    """

    def __init__(
        self,
        src=0,
        target_fps: float = 12.0,
        callback=None,
        model_manager=None,
        hand_movement_threshold: float = DEFAULT_DISPLACEMENT_THRESHOLD,
        min_avg_score: float = 0.65,
        min_frame_score: float = 0.45,
        max_score_drop: float = 0.35,
    ):
        self.src = src
        self.target_fps = target_fps
        self.callback = callback
        self.model_manager = model_manager
        self.hand_movement_threshold = hand_movement_threshold
        self.min_avg_score = min_avg_score
        self.min_frame_score = min_frame_score
        self.max_score_drop = max_score_drop

        # 1. Raw frame queue between capture thread and processing pipeline
        self.frame_queue = queue.Queue(maxsize=2)

        # 2. 5 Dedicated queues for per-finger sequence evaluation
        self.finger_queues = {f: queue.Queue(maxsize=5) for f in FINGERS}

        # 3. Thread synchronization event between pipeline and inference worker
        self.queue_event = threading.Event()

        # 4. Initialize worker threads
        self.capture_thread = CameraCaptureThread(
            src=self.src,
            target_fps=self.target_fps,
            frame_queue=self.frame_queue,
        )

        self.pipeline_thread = PipelineProcessingThread(
            frame_queue=self.frame_queue,
            finger_queues=self.finger_queues,
            queue_event=self.queue_event,
        )

        self.inference_thread = InferenceWorkerThread(
            finger_queues=self.finger_queues,
            queue_event=self.queue_event,
            model_manager=self.model_manager,
            hand_movement_threshold=self.hand_movement_threshold,
            min_avg_score=self.min_avg_score,
            min_frame_score=self.min_frame_score,
            max_score_drop=self.max_score_drop,
            callback=self.callback,
        )

    @property
    def running(self) -> bool:
        return (
            self.capture_thread.running
            and self.pipeline_thread.running
            and self.inference_thread.running
        )

    def start(self):
        """Starts capture thread, pipeline processing thread, and inference worker thread."""
        self.capture_thread.start()
        self.pipeline_thread.start()
        self.inference_thread.start()
        print("[CameraThread] Started synchronized 12 FPS capture, processing, and 5-queue inference pipeline.")

    def stop(self):
        """Clean shutdown of all worker threads."""
        self.capture_thread.stop()
        self.pipeline_thread.stop()
        self.inference_thread.stop()
        self.queue_event.set()

        try:
            self.capture_thread.join(timeout=0.3)
            self.pipeline_thread.join(timeout=0.3)
            self.inference_thread.join(timeout=0.3)
        except Exception:
            pass

    def reset_buffers(self):
        """Clears sequence ring buffers and 5 finger queues when switching models or settings."""
        self.pipeline_thread.reset_buffers()
        if self.model_manager is not None:
            self.model_manager.reset_touch_states()
        for f in FINGERS:
            q = self.finger_queues[f]
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def get_latest_frame_data(self):
        """
        Returns the synchronized 12 FPS frame with its matching hand skeleton overlay.
        The skeleton is rendered on the exact image it was detected from, eliminating spatial lag.
        """
        with self.pipeline_thread.lock:
            if self.pipeline_thread.latest_annotated is None:
                return None, False, 0.0, 640, 480
            annotated = self.pipeline_thread.latest_annotated.copy()
            detected = self.pipeline_thread.hand_detected
            fps = self.pipeline_thread.actual_fps
            w = self.pipeline_thread.frame_width
            h = self.pipeline_thread.frame_height

        return annotated, detected, fps, w, h

