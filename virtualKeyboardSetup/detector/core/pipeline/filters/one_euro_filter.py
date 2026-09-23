"""
core/pipeline/filters/one_euro_filter.py

One Euro (1€) Adaptive Low-Pass Filter (Casiez et al., 2012).
Applies adaptive speed-based frequency smoothing to coordinates or velocity signals:
- At low speeds: Low cutoff frequency filters high-frequency landmark jitter.
- At high speeds: High cutoff frequency removes lag, ensuring immediate responsiveness.

Cutoff frequency adaptation:
    cutoff = min_cutoff + beta * |dx_hat|
"""

from __future__ import annotations

import math
from typing import Any

from config.constants import (
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_ENABLED,
    ONE_EURO_MIN_CUTOFF,
)


class OneEuroFilter1D:
    """Filters a single scalar time-series value using the 1€ filter algorithm."""

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_elapsed: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_elapsed
        return r / (r + 1.0)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, t: float, x: float) -> float:
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0.0:
            return self.x_prev

        a_d = self._smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_elapsed, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilter:
    """
    Multi-channel One Euro filter managing independent 1D filters for landmark coordinates.
    Filters coordinates per channel/joint index across timestamps.
    """

    def __init__(
        self,
        enabled: bool = ONE_EURO_ENABLED,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._channels: dict[str, OneEuroFilter1D] = {}

    def reset(self) -> None:
        """Clear all active channel states when hand tracking is lost or reset."""
        self._channels.clear()

    def filter_points(
        self,
        pts: list[tuple[float, float, float]],
        timestamp: float,
    ) -> list[tuple[float, float, float]]:
        """
        Filters a list of 3D landmark points (x, y, z) at the given timestamp.
        If filter is disabled, returns pts unmodified.
        """
        if not self.enabled or not pts:
            return pts

        filtered_pts: list[tuple[float, float, float]] = []
        for idx, (px, py, pz) in enumerate(pts):
            kx = f"{idx}_x"
            ky = f"{idx}_y"
            kz = f"{idx}_z"

            if kx not in self._channels:
                self._channels[kx] = OneEuroFilter1D(
                    timestamp, px, self.min_cutoff, self.beta, self.d_cutoff
                )
                fx = px
            else:
                self._channels[kx].min_cutoff = self.min_cutoff
                self._channels[kx].beta = self.beta
                self._channels[kx].d_cutoff = self.d_cutoff
                fx = self._channels[kx].filter(timestamp, px)

            if ky not in self._channels:
                self._channels[ky] = OneEuroFilter1D(
                    timestamp, py, self.min_cutoff, self.beta, self.d_cutoff
                )
                fy = py
            else:
                self._channels[ky].min_cutoff = self.min_cutoff
                self._channels[ky].beta = self.beta
                self._channels[ky].d_cutoff = self.d_cutoff
                fy = self._channels[ky].filter(timestamp, py)

            if kz not in self._channels:
                self._channels[kz] = OneEuroFilter1D(
                    timestamp, pz, self.min_cutoff, self.beta, self.d_cutoff
                )
                fz = pz
            else:
                self._channels[kz].min_cutoff = self.min_cutoff
                self._channels[kz].beta = self.beta
                self._channels[kz].d_cutoff = self.d_cutoff
                fz = self._channels[kz].filter(timestamp, pz)

            filtered_pts.append((fx, fy, fz))

        return filtered_pts
