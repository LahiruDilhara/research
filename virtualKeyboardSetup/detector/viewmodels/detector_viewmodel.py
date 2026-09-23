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

from __future__ import annotations

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


class DetectorViewModel(QObject):
    """ViewModel for the live detector view."""

    # ── Signals ────────────────────────────────────────────────────────────────
    frame_updated       = Signal(object, float, bool, bool)
    # (frame: np.ndarray, fps: float, hand_detected: bool, layout_found: bool)

    finger_probs_updated = Signal(dict)
    # {"Thumb": 0.0..1.0, "Index": ..., ...}

    touch_event         = Signal(str, str, float)
    # (key_id: str, finger: str, probability: float)

    pipeline_error      = Signal(str)

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
        )
        self._resolver = TouchResolver(layout)
        self._executor = ActionExecutor()
        self._worker: CameraWorker | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the camera worker thread."""
        if self._worker and self._worker.isRunning():
            return
        self._worker = CameraWorker(
            camera_index=self._camera_index,
            layout=self._layout,
            config=self._config,
            parent=None,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.window_ready.connect(self._on_window_ready)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        logger.info("Detection pipeline started (camera=%d).", self._camera_index)

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
        logger.info(
            "DetectorViewModel live thresholds updated: velocity_threshold=%.4f, touch_threshold=%.2f",
            config.fingertip_velocity_threshold,
            config.touch_threshold,
        )

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
            entry.instance = instance
            logger.info("Active model set: %s", entry.name)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", entry.name, exc)
            self.pipeline_error.emit(f"Failed to load model '{entry.name}':\n{exc}")

    def set_model_by_name(self, name: str, registry_entries: list[ModelEntry]) -> None:
        entry = next((e for e in registry_entries if e.name == name), None)
        if entry:
            self.set_model(entry)

    # ── Slots ──────────────────────────────────────────────────────────────────

    @Slot(object, float, bool, object, bool)
    def _on_frame_ready(
        self,
        frame: np.ndarray,
        fps: float,
        hand_detected: bool,
        H,
        layout_found: bool,
    ) -> None:
        self._current_H = H
        self._layout_found = layout_found
        if not hand_detected:
            self._pipeline_service.reset_touch_states()
        self.frame_updated.emit(frame, fps, hand_detected, layout_found)

    @Slot(list, list, int, int)
    def _on_window_ready(
        self,
        norm_window: list[dict],
        pixel_window: list[list],
        frame_w: int,
        frame_h: int,
    ) -> None:
        if self._active_model is None:
            return

        # ── Parallel 5-Finger Model Inference (Service Layer) ───────────────
        results = self._pipeline_service.run_parallel_inference(self._active_model, norm_window)
        self.finger_probs_updated.emit(results)

        # ── Touch resolution ─────────────────────────────────────────────────
        if not self._layout_found or self._current_H is None:
            return

        # Enforce exact process.sh logic: only evaluate fingers with active touch
        touch_fingers = [f for f, data in results.items() if data.get("touch", False)]
        if not touch_fingers:
            return

        probs = {f: float(data.get("prob", 0.0)) for f, data in results.items()}
        result = self._resolver.resolve(
            touch_fingers, probs, pixel_window, self._current_H
        )
        if result is None:
            return

        key_id, finger, prob = result

        # ── Action execution ─────────────────────────────────────────────────
        action = self._action_config.get(key_id)
        if action and action.is_active:
            self._executor.execute(action)

        self.touch_event.emit(key_id, finger, prob)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        logger.error("Pipeline error: %s", message)
        self.pipeline_error.emit(message)
