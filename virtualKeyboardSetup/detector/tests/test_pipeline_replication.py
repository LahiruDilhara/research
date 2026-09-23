"""
tests/test_pipeline_replication.py

Automated test suite verifying the replication of mediapipeDetector filtration,
data processing, and deep learning model invocation in virtualKeyboardSetup/detector.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.constants import (
    ALL_21_LANDMARK_NAMES,
    FINGERS,
    HAND_MOVEMENT_THRESHOLD,
    MIN_KINETIC_SPEED_THRESHOLD,
    QUALITY_MAX_SCORE_DROP,
    QUALITY_MIN_AVG_SCORE,
    QUALITY_MIN_FRAME_SCORE,
)
from core.pipeline.feature_extractor import (
    compute_window_velocities,
    extract_variant_tensor,
    scale_variant_tensor,
    unroll_per_finger_window,
)
from core.pipeline.filters import (
    HandMovementFilter,
    KineticMotionFilter,
    WindowQualityFilter,
)
from core.pipeline.normalizer import HandScaleNormalizer
from ai_model_plugins.lstm_all_combined.lstm_all_combined_plugin import LSTMAllCombinedPlugin
from services.touch_pipeline_service import TouchPipelineService


def _create_synthetic_landmarks(offset_x: float = 0.0, offset_y: float = 0.0) -> list[tuple[float, float, float]]:
    """Generates synthetic 21 hand landmarks in pixel space with realistic palm scale."""
    base_pts = [
        (300.0, 400.0, 0.0),  # 0: wrist
        (260.0, 370.0, 0.0),  # 1: thumb_cmc
        (230.0, 340.0, 0.0),  # 2: thumb_mcp
        (210.0, 310.0, 0.0),  # 3: thumb_ip
        (190.0, 280.0, 0.0),  # 4: thumb_tip
        (280.0, 290.0, 0.0),  # 5: index_mcp
        (275.0, 240.0, 0.0),  # 6: index_pip
        (270.0, 200.0, 0.0),  # 7: index_dip
        (265.0, 160.0, 0.0),  # 8: index_tip
        (310.0, 285.0, 0.0),  # 9: middle_mcp
        (310.0, 230.0, 0.0),  # 10: middle_pip
        (310.0, 185.0, 0.0),  # 11: middle_dip
        (310.0, 140.0, 0.0),  # 12: middle_tip
        (340.0, 290.0, 0.0),  # 13: ring_mcp
        (345.0, 240.0, 0.0),  # 14: ring_pip
        (350.0, 200.0, 0.0),  # 15: ring_dip
        (355.0, 165.0, 0.0),  # 16: ring_tip
        (370.0, 310.0, 0.0),  # 17: pinky_mcp
        (380.0, 270.0, 0.0),  # 18: pinky_pip
        (390.0, 240.0, 0.0),  # 19: pinky_dip
        (400.0, 210.0, 0.0),  # 20: pinky_tip
    ]
    return [(x + offset_x, y + offset_y, z) for x, y, z in base_pts]


def test_hand_scale_normalizer():
    """Verify palm scale L_hand and wrist-centered normalization."""
    normalizer = HandScaleNormalizer()
    pts = _create_synthetic_landmarks()
    l_hand = normalizer.compute_l_hand(pts)
    assert l_hand > 50.0, f"Expected realistic palm scale, got {l_hand}"

    norm_pts = normalizer.normalize(pts, center_wrist=True)
    assert len(norm_pts) == 21
    # Wrist centered -> wrist should be (0, 0, 0)
    assert abs(norm_pts[0][0]) < 1e-5
    assert abs(norm_pts[0][1]) < 1e-5

    norm_dict = normalizer.build_norm_dict(norm_pts, pts_px=pts)
    assert "_raw_stationary" in norm_dict
    assert norm_dict["_raw_stationary"]["l_hand"] == l_hand
    assert "wrist_x" in norm_dict and "pinky_tip_y" in norm_dict


def test_hand_movement_filter():
    """Verify whole-hand transit movement detection (Step 7: threshold = 0.155)."""
    normalizer = HandScaleNormalizer()
    filter_mod = HandMovementFilter(threshold=HAND_MOVEMENT_THRESHOLD)

    # Stationary hand (offset=0 throughout 5 frames)
    stationary_window = []
    for _ in range(5):
        pts = _create_synthetic_landmarks(0.0, 0.0)
        norm_pts = normalizer.normalize(pts)
        d = normalizer.build_norm_dict(norm_pts, pts_px=pts)
        stationary_window.append(d)

    is_stat, disp, reason = filter_mod.validate(stationary_window)
    assert is_stat is True
    assert disp < 0.01
    assert reason == "Stationary"

    # Moving hand (frame 5 displaced significantly: e.g. 30 pixels when L_hand ~ 100)
    moving_window = []
    for i in range(5):
        offset = float(i * 10.0)  # Total shift = 40 pixels
        pts = _create_synthetic_landmarks(offset, offset)
        norm_pts = normalizer.normalize(pts)
        d = normalizer.build_norm_dict(norm_pts, pts_px=pts)
        moving_window.append(d)

    is_stat, disp, reason = filter_mod.validate(moving_window)
    assert is_stat is False, "Expected hand transit displacement to be detected"
    assert disp > HAND_MOVEMENT_THRESHOLD
    assert "Hand Moving" in reason


def test_window_quality_filter():
    """Verify window tracking confidence quality filter (Step 10)."""
    q_filter = WindowQualityFilter(
        min_avg_score=QUALITY_MIN_AVG_SCORE,
        min_frame_score=QUALITY_MIN_FRAME_SCORE,
        max_score_drop=QUALITY_MAX_SCORE_DROP,
    )

    # Valid scores: avg ~ 0.85, all > 0.70, drop ~ 0.10
    valid_scores = [0.85, 0.88, 0.80, 0.82, 0.89]
    ok, reason = q_filter.validate(valid_scores)
    assert ok is True
    assert reason == "OK"

    # Low average score (< 0.65)
    low_avg = [0.55, 0.60, 0.50, 0.58, 0.62]
    ok, reason = q_filter.validate(low_avg)
    assert ok is False
    assert "Low Hand Avg Score" in reason

    # Low single frame score (< 0.45)
    low_frame = [0.90, 0.90, 0.40, 0.90, 0.90]
    ok, reason = q_filter.validate(low_frame)
    assert ok is False
    assert "Low Hand Frame Score" in reason

    # High score drop (> 0.35)
    high_drop = [0.90, 0.85, 0.50, 0.75, 0.80]
    ok, reason = q_filter.validate(high_drop)
    assert ok is False
    assert "High Score Fluctuation" in reason


def test_kinetic_motion_filter():
    """Verify zero-velocity touch suppression (Step 9)."""
    k_filter = KineticMotionFilter(threshold=MIN_KINETIC_SPEED_THRESHOLD)

    # Active movement step
    active_steps = [
        {"index_tip_speed_2d": 0.002},
        {"index_tip_speed_2d": 0.015},  # > 0.008
        {"index_tip_speed_2d": 0.005},
        {"index_tip_speed_2d": 0.001},
    ]
    assert k_filter.validate(active_steps, "Index") is True

    # Zero velocity / noise floor steps
    quiescent_steps = [
        {"index_tip_speed_2d": 0.001},
        {"index_tip_speed_2d": 0.002},
        {"index_tip_speed_2d": 0.003},
        {"index_tip_speed_2d": 0.001},
    ]
    assert k_filter.validate(quiescent_steps, "Index") is False


def test_feature_extraction_and_scaling():
    """Verify 4-step velocity calculation, per-finger unrolling, and tensor scaling."""
    normalizer = HandScaleNormalizer()
    window_5 = []
    for i in range(5):
        pts = _create_synthetic_landmarks(i * 1.5, i * 1.0)
        norm_pts = normalizer.normalize(pts)
        d = normalizer.build_norm_dict(norm_pts, pts_px=pts)
        window_5.append(d)

    # 1. 4 velocity steps
    v_steps = compute_window_velocities(window_5)
    assert len(v_steps) == 4
    assert "index_tip_speed_2d" in v_steps[0]

    # 2. Unroll 5 per-finger rows
    finger_rows = unroll_per_finger_window(window_5, v_steps)
    assert len(finger_rows) == 5
    for f in ["thumb", "index", "middle", "ring", "pinky"]:
        assert f in finger_rows

    # 3. Extract all_joints_coords_vel tensor
    X_raw = extract_variant_tensor(finger_rows, "all_joints_coords_vel")
    assert X_raw.shape == (5, 4, 36)

    # 4. Standard scale via scalers.npz
    X_scaled = scale_variant_tensor(X_raw, "all_joints_coords_vel")
    assert X_scaled.shape == (5, 4, 36)
    assert not np.isnan(X_scaled).any()


def test_lstm_all_combined_plugin_inference():
    """Verify that the 94.3% LSTM_All_Combined plugin loads weights and runs inference."""
    weights_path = PROJECT_ROOT / "ai_model_plugins" / "lstm_all_combined" / "LSTM_All_Combined_cfg01.pth"
    assert weights_path.exists(), f"Weights missing at {weights_path}"

    plugin = LSTMAllCombinedPlugin()
    plugin.load(str(weights_path))

    normalizer = HandScaleNormalizer()
    window_5 = []
    for i in range(5):
        pts = _create_synthetic_landmarks(i * 0.5, i * 0.5)
        norm_pts = normalizer.normalize(pts)
        d = normalizer.build_norm_dict(norm_pts, pts_px=pts)
        window_5.append(d)

    probs = plugin.predict(window_5)
    assert len(probs) == 5
    for f in FINGERS:
        assert f in probs
        assert 0.0 <= probs[f] <= 1.0


def test_touch_pipeline_service_full_flow():
    """Verify TouchPipelineService queueing, filtration, and inference integration."""
    service = TouchPipelineService()
    weights_path = PROJECT_ROOT / "ai_model_plugins" / "lstm_all_combined" / "LSTM_All_Combined_cfg01.pth"
    plugin = LSTMAllCombinedPlugin()
    plugin.load(str(weights_path))

    # Ingest 5 frames with active finger motion so velocity threshold passes
    for i in range(5):
        pts = _create_synthetic_landmarks()
        # Move index tip downwards to simulate tapping motion
        pts[8] = (pts[8][0], pts[8][1] + i * 10.0, pts[8][2])
        # Mock MediaPipe landmark objects
        class MockLM:
            def __init__(self, x, y, z):
                self.x = x / 640.0
                self.y = y / 480.0
                self.z = z / 640.0

        mock_raw = [MockLM(x, y, z) for x, y, z in pts]
        win_ready, norm_win, pix_win = service.process_frame(
            raw_landmarks=mock_raw,
            hand_label="Right",
            frame_w=640,
            frame_h=480,
            hand_score=0.92,
        )

    assert win_ready is True
    assert norm_win is not None
    assert len(norm_win) == 5

    # Run inference through service with full filtration
    results = service.run_parallel_inference(plugin, norm_win)
    assert len(results) == 5
    assert service.last_status == "OK"
    for f in FINGERS:
        assert "touch" in results[f]
        assert "prob" in results[f]
        assert "reason" in results[f]


def test_velocity_threshold_ignores_stationary_window():
    """Verify that when fingertip speed is below the threshold, the window is ignored."""
    service = TouchPipelineService(velocity_threshold=0.008)
    weights_path = PROJECT_ROOT / "ai_model_plugins" / "lstm_all_combined" / "LSTM_All_Combined_cfg01.pth"
    plugin = LSTMAllCombinedPlugin()
    plugin.load(str(weights_path))

    # Ingest 5 completely stationary frames (speed = 0.0)
    for _ in range(5):
        pts = _create_synthetic_landmarks()

        class MockLM:
            def __init__(self, x, y, z):
                self.x = x / 640.0
                self.y = y / 480.0
                self.z = z / 640.0

        mock_raw = [MockLM(x, y, z) for x, y, z in pts]
        win_ready, norm_win, pix_win = service.process_frame(
            raw_landmarks=mock_raw,
            hand_label="Right",
            frame_w=640,
            frame_h=480,
            hand_score=0.95,
        )

    assert win_ready is True
    results = service.run_parallel_inference(plugin, norm_win)

    # Window should be ignored before running deep learning model forward pass
    assert "Window Ignored" in service.last_status
    assert "below velocity threshold" in service.last_status
    for f in FINGERS:
        assert results[f]["touch"] is False
        assert results[f]["prob"] == 0.0
        assert "below velocity threshold" in results[f]["reason"]


def test_velocity_threshold_adjustment():
    """Verify that velocity threshold can be dynamically adjusted."""
    service = TouchPipelineService(velocity_threshold=0.008)
    assert service.velocity_threshold == 0.008

    service.set_velocity_threshold(0.025)
    assert service.velocity_threshold == 0.025

    service.set_velocity_threshold(0.001)
    assert service.velocity_threshold == 0.001


def test_pinky_touch_does_not_trigger_index():
    """Verify that when only the pinky moves to touch, resting index finger does not trigger."""
    service = TouchPipelineService()
    weights_path = PROJECT_ROOT / "ai_model_plugins" / "lstm_all_combined" / "LSTM_All_Combined_cfg01.pth"
    plugin = LSTMAllCombinedPlugin()
    plugin.load(str(weights_path))

    # Ingest 5 frames where only the pinky moves downward, other fingers remain resting
    for i in range(5):
        pts = _create_synthetic_landmarks()
        # Move pinky downward by 12px each frame
        pts[20] = (pts[20][0], pts[20][1] + i * 12.0, pts[20][2])
        pts[19] = (pts[19][0], pts[19][1] + i * 8.0, pts[19][2])
        pts[18] = (pts[18][0], pts[18][1] + i * 4.0, pts[18][2])

        class MockLM:
            def __init__(self, x, y, z):
                self.x = x / 640.0
                self.y = y / 480.0
                self.z = z / 640.0

        mock_raw = [MockLM(x, y, z) for x, y, z in pts]
        win_ready, norm_win, pix_win = service.process_frame(
            raw_landmarks=mock_raw,
            hand_label="Right",
            frame_w=640,
            frame_h=480,
            hand_score=0.95,
        )

    assert win_ready is True
    results = service.run_parallel_inference(plugin, norm_win)

    # Resting fingers must have kinetic motion filter active (has_kinetic_motion = False)
    # Index must never be touched
    assert results["Index"]["touch"] is False
    assert results["Thumb"]["touch"] is False
    assert results["Middle"]["touch"] is False
    assert results["Ring"]["touch"] is False

    touch_fingers = [f for f, data in results.items() if data.get("touch", False)]
    assert "Index" not in touch_fingers


if __name__ == "__main__":
    test_hand_scale_normalizer()
    test_hand_movement_filter()
    test_window_quality_filter()
    test_kinetic_motion_filter()
    test_feature_extraction_and_scaling()
    test_lstm_all_combined_plugin_inference()
    test_touch_pipeline_service_full_flow()
    test_velocity_threshold_ignores_stationary_window()
    test_velocity_threshold_adjustment()
    test_pinky_touch_does_not_trigger_index()
    print("\nAll 10 pipeline replication, velocity threshold, and model verification tests passed successfully!")

