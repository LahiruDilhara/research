"""
realtimeprocess/stages/

Modular real-time processing pipeline stages matching process.sh and datacreator/ architecture:

- stage1_normalizer: HandScaleNormalizer (8-distance palm RMS scale L_hand & wrist translation centering).
- stage2_euro_filter: OneEuroFilter1D & OneEuroFilterBank (process.sh defaults: min=3.0, beta=1.4, d=1.0).
- stage3_velocities: 4-step velocity components (vx, vy, vz) & 2D/3D speeds computation across 5 frames.
- stage4_quality_filter: Real-time window quality validation (min-avg-score 0.65 & zero velocity drop check).
- stage5_finger_unroll: Per-finger unrolling (thumb, index, middle, ring, pinky).
- stage6_variant_extractor: Feature tensor extraction for all 14 research variants.
"""

from .stage1_normalizer import HandScaleNormalizer
from .stage2_euro_filter import OneEuroFilter1D, OneEuroFilterBank
from .stage3_velocities import compute_window_velocities
from .stage4_quality_filter import validate_realtime_window_quality
from .stage5_finger_unroll import unroll_per_finger_window
from .stage6_variant_extractor import extract_variant_tensor

__all__ = [
    "HandScaleNormalizer",
    "OneEuroFilter1D",
    "OneEuroFilterBank",
    "compute_window_velocities",
    "validate_realtime_window_quality",
    "unroll_per_finger_window",
    "extract_variant_tensor",
]
