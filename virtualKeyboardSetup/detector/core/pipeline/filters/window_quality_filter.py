"""
core/pipeline/filters/window_quality_filter.py

Sequence Window Hand Confidence and Quality Filter.
Exact replication of mediapipeDetector/process.sh Step 10 and
datacreator/filter_window_quality.py.

Validation rules:
1. Average hand confidence score across 5 frames >= min_avg_score (0.65).
2. Minimum frame score across 5 frames >= min_frame_score (0.45).
3. Score drop / fluctuation max(score) - min(score) <= max_score_drop (0.35).
"""

from __future__ import annotations

from config.constants import (
    QUALITY_MIN_AVG_SCORE,
    QUALITY_MIN_FRAME_SCORE,
    QUALITY_MAX_SCORE_DROP,
)


class WindowQualityFilter:
    """Validates MediaPipe tracking confidence stability across 5 frames."""

    def __init__(
        self,
        min_avg_score: float = QUALITY_MIN_AVG_SCORE,
        min_frame_score: float = QUALITY_MIN_FRAME_SCORE,
        max_score_drop: float = QUALITY_MAX_SCORE_DROP,
    ) -> None:
        self.min_avg_score = min_avg_score
        self.min_frame_score = min_frame_score
        self.max_score_drop = max_score_drop

    def validate(
        self,
        scores_5: list[float] | None,
        min_avg_score: float | None = None,
        min_frame_score: float | None = None,
        max_score_drop: float | None = None,
    ) -> tuple[bool, str]:
        """
        Validates hand confidence quality across 5-frame window.

        Returns
        -------
        (is_valid, reason)
        """
        if not scores_5 or len(scores_5) == 0:
            return True, "OK"

        valid_scores = [s for s in scores_5 if s > 0.0]
        if not valid_scores:
            return True, "OK"

        min_avg = self.min_avg_score if min_avg_score is None else min_avg_score
        min_frame = self.min_frame_score if min_frame_score is None else min_frame_score
        max_drop = self.max_score_drop if max_score_drop is None else max_score_drop

        avg_score = sum(valid_scores) / len(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        score_diff = max_score - min_score

        if min_avg is not None and avg_score < min_avg:
            return False, f"Low Hand Avg Score ({avg_score:.2f} < {min_avg:.2f})"

        if min_frame is not None and min_score < min_frame:
            return False, f"Low Hand Frame Score ({min_score:.2f} < {min_frame:.2f})"

        if max_drop is not None and score_diff > max_drop:
            return False, f"High Score Fluctuation ({score_diff:.2f} > {max_drop:.2f})"

        return True, "OK"
