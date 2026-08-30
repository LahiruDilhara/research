"""
realtimeprocess/stages/stage2_euro_filter.py

Stage 2: 1€ (One Euro) Temporal Landmark Filtering.

Applies adaptive low-pass 1€ filtering across all 21 hand joint coordinates (x, y, z) over time:
- Removes high-frequency jitter when stationary.
- Dynamically increases cutoff frequency during fast motion for zero latency.

Matches process.sh Step 3 defaults: min_cutoff=3.0, beta=1.4, d_cutoff=1.0.
Matches datacreator/filter_landmarks.py 100%.
"""

import math

ALL_21_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class OneEuroFilter1D:
    """Filters a single 1D scalar signal over time using Casiez et al. 1€ filter algorithm."""

    def __init__(self, t0: float, x0: float, min_cutoff: float = 3.0, beta: float = 1.4, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_elapsed: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_elapsed
        return r / (r + 1.0)

    def filter(self, t: float, x: float) -> float:
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0:
            return self.x_prev

        a_d = self._smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_elapsed, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilterBank:
    """Maintains online 1€ filtering state across all 21 hand joint 3D coordinates."""

    def __init__(self, min_cutoff: float = 3.0, beta: float = 1.4, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters = {}

    def reset(self):
        """Resets all joint filter states when hand tracking is interrupted or model switches."""
        self.filters.clear()

    def update_params(self, min_cutoff: float, beta: float, d_cutoff: float):
        """Updates min_cutoff, beta, and d_cutoff parameters on all live filter instances."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        for f in self.filters.values():
            f.min_cutoff = self.min_cutoff
            f.beta = self.beta
            f.d_cutoff = self.d_cutoff

    def filter_frame(self, t_sec: float, norm_pts: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        """
        Given timestamp t_sec and 21 scale-normalized wrist-centered 3D coordinates (nx, ny, nz):
        Applies 1€ filter per joint coordinate in process.sh Step 3 order.
        Returns 1€ filtered [(nx, ny, nz), ...].
        """
        filtered_pts = []
        for idx, (nx, ny, nz) in enumerate(norm_pts):
            lm_name = ALL_21_LANDMARK_NAMES[idx]
            fnx, fny, fnz = nx, ny, nz
            for axis, val in zip(["x", "y", "z"], [nx, ny, nz]):
                key = f"{lm_name}_{axis}"
                if key not in self.filters:
                    self.filters[key] = OneEuroFilter1D(t_sec, val, self.min_cutoff, self.beta, self.d_cutoff)
                    f_val = val
                else:
                    f_val = self.filters[key].filter(t_sec, val)

                if axis == "x":
                    fnx = f_val
                elif axis == "y":
                    fny = f_val
                else:
                    fnz = f_val

            filtered_pts.append((fnx, fny, fnz))

        return filtered_pts
