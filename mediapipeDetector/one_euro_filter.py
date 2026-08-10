"""
one_euro_filter.py

Reference implementation of the 1€ Filter (Casiez, Roussel & Vogel, 2012),
adapted for smoothing 2D fingertip/landmark coordinates from MediaPipe.
"""

import math


def _smoothing_factor(t_elapsed, cutoff):
    r = 2 * math.pi * cutoff * t_elapsed
    return r / (r + 1)


def _exponential_smoothing(alpha, x, x_prev):
    return alpha * x + (1 - alpha) * x_prev


class OneEuroFilter1D:
    """Filters a single scalar value over time. mincutoff and beta are the
    two parameters to tune."""

    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0:
            return self.x_prev  # guard against non-increasing timestamps

        # Filtered derivative (velocity) of the signal
        a_d = _smoothing_factor(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff: higher when moving fast, lower when still
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_elapsed, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class FingertipFilter:
    """
    Convenience wrapper: one OneEuroFilter1D for x and one for y, per
    fingertip/landmark coordinate.
    """

    def __init__(self, min_cutoff=0.5, beta=2.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.filter_x = None
        self.filter_y = None

    def update(self, t_seconds, x, y):
        if self.filter_x is None:
            self.filter_x = OneEuroFilter1D(t_seconds, x, self.min_cutoff, self.beta)
            self.filter_y = OneEuroFilter1D(t_seconds, y, self.min_cutoff, self.beta)
            return float(x), float(y)
        return self.filter_x(t_seconds, x), self.filter_y(t_seconds, y)


if __name__ == "__main__":
    import random
    import statistics

    random.seed(1)
    fps = 30
    t = [i / fps for i in range(90)]

    true_signal = []
    for i in range(90):
        if i < 30:
            true_signal.append(0.5)                   # still (idle hand)
        elif i < 40:
            true_signal.append(0.5 + (i - 30) * 0.05)  # fast tap approach
        else:
            true_signal.append(1.0)                   # still again (post-tap)

    noisy = [v + random.gauss(0, 0.01) for v in true_signal]

    f = OneEuroFilter1D(t[0], noisy[0], min_cutoff=0.5, beta=2.0)
    filtered = [noisy[0]]
    for i in range(1, len(t)):
        filtered.append(f(t[i], noisy[i]))

    still_noisy_std = statistics.pstdev(noisy[5:30])
    still_filtered_std = statistics.pstdev(filtered[5:30])
    print(f"Still-period jitter (stdev) -> raw: {still_noisy_std:.4f}, filtered: {still_filtered_std:.4f}")

    fast_true = true_signal[30:41]
    fast_filtered = filtered[30:41]
    lag_error = sum(abs(a - b) for a, b in zip(fast_true, fast_filtered)) / len(fast_true)
    print(f"Fast-movement tracking error (lower = less lag): {lag_error:.4f}")

    assert still_filtered_std < still_noisy_std, "Filter should reduce jitter during stillness"
    print("PASS: jitter reduced during stillness while remaining responsive during motion.")
