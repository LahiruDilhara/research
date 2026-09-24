"""
viewmodels/detector_viewmodel.py

Orchestrates the full live detection pipeline.

Responsibilities
────────────────
- Creates and manages the CameraWorker QThread.
- Receives window_ready signal → calls active model's predict() → resolves touch.
- Receives frame_ready signal → forwards to UI.
- Calls ActionExecutor when a key press is confirmed.
- Exposes signals consumed by DetectorView (frame updates, finger probs, touch events).
- Supports hot-swapping the active model without restarting the camera thread.
"""

from enum import Enum
import time

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from config.app_config import AppConfig
from config.constants import FINGERS, TOUCH_PROBABILITY_THRESHOLD
from core.action.action_executor import ActionData, ActionExecutor
from core.interfaces.touch_model import ITouchModel, ModelEntry
from core.layout.layout_parser import LayoutData
from core.pipeline.camera_worker import CameraWorker
from core.pipeline.touch_resolver import TouchResolver
from services.touch_pipeline_service import TouchPipelineService
from utils.logger import setup_logger

logger = setup_logger("DetectorViewModel")


class ExecutionMode(str, Enum):
    """Operational mode of the virtual keyboard detector."""
    PLAY = "play"  # Visual testing mode: video rendering, simulated touches, no OS keystrokes
    RUN  = "run"   # Headless production mode: zero video rendering, low RAM, live telemetry, executes OS keystrokes


class DetectorViewModel(QObject):
    """ViewModel for the live detector view."""

    # ── Signals ────────────────────────────────────────────────────────────────
    frame_updated        = Signal(object, float, bool, bool)
    # (frame: np.ndarray | None, fps: float, hand_detected: bool, layout_found: bool)

    finger_probs_updated = Signal(dict)
    # {"Thumb": {"touch": bool, "prob": float}, ...}

    touch_event          = Signal(str, str, float)
    # (key_id: str, finger: str, probability: float)

    action_executed      = Signal(str, str, str, str)
    # (action_type: str, payload: str, key_id: str, finger: str)

    latency_updated      = Signal(float)
    # inference latency in milliseconds

    fps_updated          = Signal(float)
    # current pipeline execution FPS

    mode_changed         = Signal(str)
    # "play" or "run"

    model_changed        = Signal(str)
    # active model name

    camera_changed       = Signal(int)
    # active camera index

    touch_threshold_changed = Signal(float)
    # active touch threshold (0.01 to 1.00)

    pipeline_error       = Signal(str)

    def __init__(
        self,
        layout: LayoutData,
        action_config: dict[str, ActionData],
        config: AppConfig,
        camera_index: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._layout       = layout
        self._action_config = action_config
        self._config       = config
        self._camera_index = camera_index
        self._mode = ExecutionMode.PLAY
        self._active_model_name: str = ""

        self._active_model: ITouchModel | None = None
        self._current_H: np.ndarray | None = None
        self._layout_found = False

        self._pipeline_service = TouchPipelineService(
            window_size=config.window_size,
            shift_size=config.shift_size,
            hand_movement_threshold=config.hand_movement_threshold,
            quality_min_avg_score=config.quality_min_avg_score,
            quality_min_frame_score=config.quality_min_frame_score,
            quality_max_score_drop=config.quality_max_score_drop,
            min_kinetic_speed=config.min_kinetic_speed,
            touch_onset_threshold=config.touch_onset_threshold,
            touch_release_threshold=config.touch_release_threshold,
            velocity_threshold=config.fingertip_velocity_threshold,
            one_euro_enabled=config.one_euro_enabled,
            one_euro_min_cutoff=config.one_euro_min_cutoff,
            one_euro_beta=config.one_euro_beta,
            one_euro_d_cutoff=config.one_euro_d_cutoff,
        )
        self._resolver = TouchResolver(
            layout,
            offset_enabled=config.fingertip_offset_enabled,
            forward_offset_mm=config.fingertip_forward_offset_mm,
        )
        self._executor = ActionExecutor()
        self._worker: CameraWorker | None = None
        self._overlay_enabled: bool = True
        self._last_pressed_key: str | None = None
        self._last_press_time: float = 0.0
        self._debounce_cooldown: float = config.touch_debounce_cooldown_s

    # ── Operational Mode & Properties ──────────────────────────────────────────

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._mode

    def set_execution_mode(self, mode: ExecutionMode | str) -> None:
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower())
        self._mode = mode
        if self._worker is not None:
            self._worker.render_video = (mode == ExecutionMode.PLAY)
        self.mode_changed.emit(self._mode.value)
        logger.info("DetectorViewModel execution mode set to: %s", self._mode.value)

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    @property
    def layout(self) -> LayoutData:
        return self._layout

    @property
    def action_config(self) -> dict[str, ActionData]:
        return self._action_config

    def set_action_config(self, config: dict[str, ActionData]) -> None:
        self._action_config = config
        logger.info("Updated DetectorViewModel action configuration (%d keys).", len(config))

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the camera worker thread."""
        if self._worker and self._worker.isRunning():
            return
        self._worker = CameraWorker(
            camera_index=self._camera_index,
            layout=self._layout,
            config=self._config,
            pipeline_service=self._pipeline_service,
            parent=None,
        )
        self._worker.render_video = (self._mode == ExecutionMode.PLAY)
        self._worker.set_show_overlay(self._overlay_enabled)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.window_ready.connect(self._on_window_ready)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        logger.info("Detection pipeline started (camera=%d, mode=%s).", self._camera_index, self._mode.value)

    def stop(self) -> None:
        """Stop the camera worker thread cleanly."""
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        logger.info("Detection pipeline stopped.")

    def update_settings(self, config: AppConfig) -> None:
        """
        Dynamically update live detector thresholds and configuration.
        Takes effect immediately on the next inference window without stopping camera.
        """
        self._config = config
        self._pipeline_service.set_velocity_threshold(config.fingertip_velocity_threshold)
        self._pipeline_service.set_touch_threshold(config.touch_threshold)
        self._pipeline_service.set_hand_movement_threshold(config.hand_movement_threshold)
        self._pipeline_service.set_quality_filter_thresholds(
            min_avg=config.quality_min_avg_score,
            min_frame=config.quality_min_frame_score,
            max_drop=config.quality_max_score_drop,
        )
        self._pipeline_service.update_one_euro_settings(
            enabled=config.one_euro_enabled,
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta,
            d_cutoff=config.one_euro_d_cutoff,
        )
        self._resolver.set_fingertip_offset(
            config.fingertip_offset_enabled,
            config.fingertip_forward_offset_mm,
        )
        self._debounce_cooldown = config.touch_debounce_cooldown_s
        logger.info(
            "DetectorViewModel live thresholds updated: velocity_threshold=%.4f, touch_threshold=%.2f, hand_movement=%.4f, one_euro=%s",
            config.fingertip_velocity_threshold,
            config.touch_threshold,
            config.hand_movement_threshold,
            config.one_euro_enabled,
        )
        self.touch_threshold_changed.emit(config.touch_threshold)

    @property
    def touch_threshold(self) -> float:
        """Current touch probability threshold."""
        return self._pipeline_service.touch_threshold

    def set_touch_threshold(self, threshold: float) -> None:
        """Dynamically update touch probability threshold in service and config."""
        val = max(0.01, min(1.00, float(threshold)))
        self._pipeline_service.set_touch_threshold(val)
        if self._config:
            self._config.set_touch_threshold(val)
        self.touch_threshold_changed.emit(val)
        logger.info("DetectorViewModel touch threshold updated: %.2f (%.0f%%)", val, val * 100.0)

    # ── Model hot-swap ─────────────────────────────────────────────────────────

    def set_model(self, entry: ModelEntry) -> None:
        """
        Instantiate and load the given model entry as the active model.
        Can be called while the camera is running, since the swap is atomic
        at the Python object level (GIL).
        """
        if not entry.weights_path:
            logger.warning("Model '%s' has no resolved weights path.", entry.name)
            return
        try:
            instance = entry.cls()
            instance.load(entry.weights_path)
            self._active_model = instance
            self._active_model_name = entry.name
            entry.instance = instance
            self.model_changed.emit(entry.name)
            logger.info("Active model set: %s", entry.name)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", entry.name, exc)
            self.pipeline_error.emit(f"Failed to load model '{entry.name}':\n{exc}")

    def set_model_by_name(self, name: str, registry_entries: list[ModelEntry]) -> None:
        entry = next((e for e in registry_entries if e.name == name), None)
        if entry:
            self.set_model(entry)

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def set_camera_index(self, camera_index: int) -> None:
        """Switch camera device dynamically without stopping the workspace."""
        if self._camera_index == camera_index and self._worker and self._worker.isRunning():
            return
        logger.info("Switching camera device to index: %d", camera_index)
        self._camera_index = camera_index
        self._config._camera_index = camera_index
        self.camera_changed.emit(camera_index)
        was_running = self._worker is not None and self._worker.isRunning()
        if was_running:
            self.stop()
            self.start()

    @property
    def current_H(self) -> np.ndarray | None:
        return self._current_H if self._layout_found else None

    @property
    def layout_found(self) -> bool:
        return self._layout_found

    def simulate_touch_at_pixel(self, px: float, py: float) -> str | None:
        """Simulate touch at raw camera pixel coordinates replicating analyzer/main.py."""
        if self._current_H is None or not self._layout_found:
            return None
        import cv2
        pt_src = np.array([[[float(px), float(py)]]], dtype=np.float32)
        pt_dst = cv2.perspectiveTransform(pt_src, self._current_H.astype(np.float32))
        mm_x, mm_y = float(pt_dst[0][0][0]), float(pt_dst[0][0][1])

        hit = self._layout.find_button_at(mm_x, mm_y) if hasattr(self._layout, "find_button_at") else None
        if hit is None:
            for b in self._layout.buttons:
                if b.contains_mm(mm_x, mm_y):
                    hit = b
                    break

        if hit is not None:
            if self._worker:
                self._worker.set_active_button(hit.id)
            logger.info("Simulated canvas touch at (%d, %d)px -> (%.1f, %.1f)mm -> Key '%s'", px, py, mm_x, mm_y, hit.id)
            self.touch_event.emit(hit.id, "Index", 1.0)
            return hit.id
        return None

    def set_keyboard_overlay_enabled(self, enabled: bool) -> None:
        """Enable or disable projected keyboard overlay on video feed."""
        self._overlay_enabled = enabled
        if self._worker:
            self._worker.set_show_overlay(enabled)

    # ── Slots ──────────────────────────────────────────────────────────────────

    @Slot(object, float, bool, object, bool)
    def _on_frame_ready(
        self,
        frame: np.ndarray | None,
        fps: float,
        hand_detected: bool,
        H,
        layout_found: bool,
    ) -> None:
        self._current_H = H if layout_found else None
        self._layout_found = bool(layout_found and self._current_H is not None)
        if not hand_detected:
            self._pipeline_service.reset_touch_states()
            self._last_pressed_key = None
        self.fps_updated.emit(fps)
        self.frame_updated.emit(frame, fps, hand_detected, self._layout_found)

    @Slot(list, list, int, int)
    def _on_window_ready(
        self,
        norm_window: list[dict],
        pixel_window: list[list],
        frame_w: int,
        frame_h: int,
    ) -> None:
        if self._active_model is None:
            logger.warning("DetectorViewModel: Window received, but no AI model is active. Please select a model.")
            return

        # ── Parallel 5-Finger Model Inference (Service Layer) ───────────────
        t0 = time.perf_counter()
        results = self._pipeline_service.run_parallel_inference(self._active_model, norm_window)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.latency_updated.emit(latency_ms)
        self.finger_probs_updated.emit(results)

        # ── Touch candidate monitoring ───────────────────────────────────────
        touch_fingers = [f for f, data in results.items() if data.get("touch", False)]
        if touch_fingers:
            candidates_str = ", ".join(f"{f}: {results[f].get('prob', 0.0):.2f}" for f in touch_fingers)
            logger.info("Touch candidate onset: [%s] (latency=%.1f ms)", candidates_str, latency_ms)
        else:
            max_f = max(FINGERS, key=lambda f: results[f].get("prob", 0.0))
            max_p = results[max_f].get("prob", 0.0)
            reason = results[max_f].get("reason", "Below Threshold")
            logger.info(
                "Window evaluated (inference=%.1f ms): No touch detected (%s | highest=%s at %.2f)",
                latency_ms, reason, max_f, max_p,
            )
            return

        if not self._layout_found or self._current_H is None:
            logger.warning(
                "Touch candidate(s) [%s] detected, but blocked: AprilTag paper layout is not tracked (homography missing).",
                candidates_str,
            )
            return

        # Check if all fingers have released
        all_released = not any(data.get("touch", False) for data in results.values())
        if all_released:
            self._last_pressed_key = None

        probs = {f: float(data.get("prob", 0.0)) for f, data in results.items()}

        # Single Primary Touch: highest confidence touching finger resolves first
        primary_hit = self._resolver.resolve(
            touch_fingers, probs, pixel_window, self._current_H
        )
        if primary_hit is None:
            logger.info(
                "Touch candidate(s) [%s] rejected: Fingertip contact point did not hit any button in layout.",
                candidates_str,
            )
            return

        key_id, finger, prob = primary_hit
        now = time.perf_counter()

        # Check debounce cooldown:
        # Prevents continuous re-triggers across sliding windows during the same physical tap,
        # and prevents drifting to an adjacent key during landing/rebound.
        is_cooldown_active = (now - self._last_press_time) < self._debounce_cooldown
        if is_cooldown_active and self._last_pressed_key is not None:
            cooldown_rem = self._debounce_cooldown - (now - self._last_press_time)
            logger.info(
                "Debounce cooldown active (%.2fs remaining): repeat/drift press suppressed for key='%s' (current active='%s')",
                cooldown_rem, key_id, self._last_pressed_key,
            )
            if self._worker:
                self._worker.set_active_buttons([self._last_pressed_key])
            return

        self._last_pressed_key = key_id
        self._last_press_time = now

        if self._worker:
            self._worker.set_active_buttons([key_id])
            self._worker.set_contact_points(self._resolver.last_contact_points)

        # ── Action execution (Run Mode only) ─────────────────────────────────
        if self._mode == ExecutionMode.RUN:
            action = self._action_config.get(key_id)
            if action and action.is_active:
                self._executor.execute(action)
                self.action_executed.emit(action.type, action.value, key_id, finger)
                logger.info(
                    "[RUN MODE ACTION EXECUTED] Key='%s' Finger=%s Action=[%s: %s] (prob=%.2f, latency=%.1f ms)",
                    key_id, finger, action.type.upper(), action.value, prob, latency_ms,
                )
            else:
                logger.info(
                    "[RUN MODE TOUCH] Key='%s' Finger=%s (No digital action bound) (prob=%.2f)",
                    key_id, finger, prob,
                )
        else:
            action = self._action_config.get(key_id)
            action_desc = f"[{action.type.upper()}: {action.value}]" if (action and action.is_active) else "[No Action]"
            logger.info(
                "[PLAY MODE SIMULATED TOUCH] Key='%s' Finger=%s Simulated=%s (prob=%.2f, latency=%.1f ms)",
                key_id, finger, action_desc, prob, latency_ms,
            )

        self.touch_event.emit(key_id, finger, prob)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        logger.error("Pipeline error: %s", message)
        self.pipeline_error.emit(message)
