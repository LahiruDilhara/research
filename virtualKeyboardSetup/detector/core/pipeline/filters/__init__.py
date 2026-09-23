"""
core/pipeline/filters/__init__.py

Modular filtration components replicating mediapipeDetector/process.sh:
- HandMovementFilter  : Step 7 whole-hand transit displacement validation
- WindowQualityFilter : Step 10 MediaPipe tracking confidence stability
- KineticMotionFilter : Step 9 zero-velocity touch suppression
"""

from __future__ import annotations

from .hand_movement_filter import HandMovementFilter
from .window_quality_filter import WindowQualityFilter
from .kinetic_motion_filter import KineticMotionFilter

__all__ = [
    "HandMovementFilter",
    "WindowQualityFilter",
    "KineticMotionFilter",
]
