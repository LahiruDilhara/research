"""
tests/test_touch_resolver.py

Unit tests for TouchResolver:
- Verifies kinematic deceleration and trajectory turnaround impact frame selection.
- Verifies rebound taps (bounce back up), resting taps (staying still on key).
- Verifies camera tilt invariance (angled diagonal strokes).
- Verifies key boundary hit testing via Homography matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.constants import FINGERTIP_INDICES
from core.layout.layout_parser import ButtonData, LayoutData
from core.pipeline.touch_resolver import TouchResolver


def _create_mock_layout() -> LayoutData:
    """Creates a mock layout with two buttons."""
    # Button 'A': x in [20, 60] mm, y in [20, 60] mm
    # Button 'B': x in [80, 120] mm, y in [20, 60] mm
    btn_a = ButtonData(
        id="KEY_A",
        label="A",
        x_mm=20.0,
        y_mm=20.0,
        x_max_mm=60.0,
        y_max_mm=60.0,
        width_mm=40.0,
        height_mm=40.0,
        center_x_mm=40.0,
        center_y_mm=40.0,
    )
    btn_b = ButtonData(
        id="KEY_B",
        label="B",
        x_mm=80.0,
        y_mm=20.0,
        x_max_mm=120.0,
        y_max_mm=60.0,
        width_mm=40.0,
        height_mm=40.0,
        center_x_mm=100.0,
        center_y_mm=40.0,
    )
    return LayoutData(
        paper_width_mm=210.0,
        paper_height_mm=297.0,
        marker_size_mm=15.0,
        marker_family="tag36h11",
        buttons=[btn_a, btn_b],
        markers=[],
    )


def test_rebound_tap_at_frame_3():
    """Verify that a downward strike at Frame 3 followed by an upward rebound selects Frame 3."""
    layout = _create_mock_layout()
    resolver = TouchResolver(layout)

    # Identity homography: 1 pixel = 1 mm
    H = np.eye(3, dtype=np.float64)

    # 5 frames of 21 landmarks
    # Index finger tip index is 8
    # Frame 0: still at (40, 10)
    # Frame 1: still at (40, 10)
    # Frame 2: moving down at (40, 25)
    # Frame 3: struck key 'A' at (40, 45) (inside KEY_A: [20..60, 20..60])
    # Frame 4: rebounded back up to (40, 20)
    pixel_window = []
    y_coords = [10.0, 10.0, 25.0, 45.0, 20.0]
    for y in y_coords:
        landmarks = [(40.0, y) for _ in range(21)]
        pixel_window.append(landmarks)

    probs = {"Index": 0.92, "Thumb": 0.10}
    touch_fingers = ["Index"]

    result = resolver.resolve(touch_fingers, probs, pixel_window, H)
    assert result is not None
    key_id, finger, prob = result
    assert key_id == "KEY_A"
    assert finger == "Index"
    assert abs(prob - 0.92) < 1e-4

    # Verify impact frame selection directly
    impact_frame = resolver._find_impact_frame(FINGERTIP_INDICES["Index"], pixel_window)
    assert impact_frame == 3


def test_resting_tap_at_frame_3():
    """Verify that a downward strike at Frame 3 followed by staying still selects Frame 3."""
    layout = _create_mock_layout()
    resolver = TouchResolver(layout)
    H = np.eye(3, dtype=np.float64)

    # Frame 0: (40, 10)
    # Frame 1: (40, 15)
    # Frame 2: (40, 25)
    # Frame 3: lands at (40, 45) (inside KEY_A)
    # Frame 4: resting still at (40, 45)
    pixel_window = []
    y_coords = [10.0, 15.0, 25.0, 45.0, 45.0]
    for y in y_coords:
        landmarks = [(40.0, y) for _ in range(21)]
        pixel_window.append(landmarks)

    impact_frame = resolver._find_impact_frame(FINGERTIP_INDICES["Index"], pixel_window)
    assert impact_frame == 3

    result = resolver.resolve(["Index"], {"Index": 0.88}, pixel_window, H)
    assert result is not None
    assert result[0] == "KEY_A"


def test_early_rebound_at_frame_2():
    """Verify that a fast strike landing at Frame 2 and rebounding selects Frame 2."""
    layout = _create_mock_layout()
    resolver = TouchResolver(layout)
    H = np.eye(3, dtype=np.float64)

    # Frame 0: (100, 10)
    # Frame 1: (100, 25)
    # Frame 2: landed at (100, 50) (inside KEY_B: [80..120, 20..60])
    # Frame 3: rebounded to (100, 30)
    # Frame 4: continuing up to (100, 15)
    pixel_window = []
    y_coords = [10.0, 25.0, 50.0, 30.0, 15.0]
    for y in y_coords:
        landmarks = [(100.0, y) for _ in range(21)]
        pixel_window.append(landmarks)

    impact_frame = resolver._find_impact_frame(FINGERTIP_INDICES["Index"], pixel_window)
    assert impact_frame == 2

    result = resolver.resolve(["Index"], {"Index": 0.85}, pixel_window, H)
    assert result is not None
    assert result[0] == "KEY_B"


def test_angled_camera_motion_invariance():
    """Verify that tilted diagonal strokes (e.g. 45 degrees) resolve the exact turnaround frame."""
    layout = _create_mock_layout()
    resolver = TouchResolver(layout)

    # Motion along diagonal line at 45 degrees: (x, y)
    # Frame 0: (20, 20)
    # Frame 1: (25, 25)
    # Frame 2: (35, 35)
    # Frame 3: lands at (45, 45) (inside KEY_A)
    # Frame 4: rebounds backward along diagonal to (30, 30)
    pixel_window = []
    coords = [(20.0, 20.0), (25.0, 25.0), (35.0, 35.0), (45.0, 45.0), (30.0, 30.0)]
    for pt in coords:
        landmarks = [pt for _ in range(21)]
        pixel_window.append(landmarks)

    # Camera tilt invariance check: direction reverses along diagonal
    impact_frame = resolver._find_impact_frame(FINGERTIP_INDICES["Index"], pixel_window)
    assert impact_frame == 3

    H = np.eye(3, dtype=np.float64)
    result = resolver.resolve(["Index"], {"Index": 0.90}, pixel_window, H)
    assert result is not None
    assert result[0] == "KEY_A"


def test_simultaneous_multi_touch_resolution():
    """Verify that multiple fingers touching different keys simultaneously are all resolved."""
    layout = _create_mock_layout()
    resolver = TouchResolver(layout)
    H = np.eye(3, dtype=np.float64)

    # Frame 0 to 4:
    # Index finger (idx 8) strikes KEY_A at (40, 45)
    # Middle finger (idx 12) strikes KEY_B at (100, 45)
    pixel_window = []
    for _ in range(5):
        landmarks = [(0.0, 0.0) for _ in range(21)]
        # Index tip
        landmarks[8] = (40.0, 45.0)
        # Middle tip
        landmarks[12] = (100.0, 45.0)
        pixel_window.append(landmarks)

    probs = {"Index": 0.95, "Middle": 0.91}
    touch_fingers = ["Index", "Middle"]

    hits = resolver.resolve_all(touch_fingers, probs, pixel_window, H)
    assert len(hits) == 2
    keys_resolved = {h[0] for h in hits}
    assert keys_resolved == {"KEY_A", "KEY_B"}


def test_forward_offset_extrapolation():
    """Verify that when landmark tip is slightly behind key boundary, forward offset shifts contact into key."""
    layout = _create_mock_layout()
    # KEY_A is in y in [20, 60] mm
    # DIP at (40, 10), TIP at (40, 17) -> direction is +y (forward)
    # Without offset: TIP at y=17 is outside KEY_A
    # With offset=6.0mm: contact is at (40, 23), inside KEY_A!
    resolver = TouchResolver(layout, offset_enabled=True, forward_offset_mm=6.0)
    H = np.eye(3, dtype=np.float64)

    pixel_window = []
    for _ in range(5):
        landmarks = [(40.0, 10.0) for _ in range(21)]
        # Index DIP (idx 7) at (40, 10)
        landmarks[7] = (40.0, 10.0)
        # Index TIP (idx 8) at (40, 17)
        landmarks[8] = (40.0, 17.0)
        pixel_window.append(landmarks)

    result = resolver.resolve(["Index"], {"Index": 0.95}, pixel_window, H)
    assert result is not None
    assert result[0] == "KEY_A"


def test_in_flight_frame_does_not_hijack_landing_target():
    """Verify that in-flight motion over KEY_B does not falsely resolve KEY_B when landing on KEY_A."""
    layout = _create_mock_layout()
    resolver = TouchResolver(layout, offset_enabled=False)
    H = np.eye(3, dtype=np.float64)

    # Frame 0: in air at (100, 10)
    # Frame 1: in air at (100, 20)
    # Frame 2: passing over KEY_B at (100, 40)
    # Frame 3: moving toward KEY_A at (60, 40)
    # Frame 4: landed firmly on KEY_A at (40, 40)
    pixel_window = []
    positions = [(100.0, 10.0), (100.0, 20.0), (100.0, 40.0), (60.0, 40.0), (40.0, 40.0)]
    for pt in positions:
        landmarks = [pt for _ in range(21)]
        pixel_window.append(landmarks)

    result = resolver.resolve(["Index"], {"Index": 0.95}, pixel_window, H)
    assert result is not None
    # Must resolve landed target KEY_A, NOT in-flight KEY_B!
    assert result[0] == "KEY_A"


if __name__ == "__main__":
    test_rebound_tap_at_frame_3()
    test_resting_tap_at_frame_3()
    test_early_rebound_at_frame_2()
    test_angled_camera_motion_invariance()
    test_simultaneous_multi_touch_resolution()
    test_forward_offset_extrapolation()
    test_in_flight_frame_does_not_hijack_landing_target()
    print("\nAll TouchResolver kinematic deceleration, multi-touch, offset, and angle invariance tests passed successfully!")

