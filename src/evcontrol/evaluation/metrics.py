"""Metrics for a finished run. All functions take plain numpy arrays.

They map onto the questions in the working notes:
    tracking_rmse / max_abs_error -> how well did we follow the reference?
    saturation_fraction           -> how often was the actuator the limiting factor?
    constraint_violations         -> did a state leave its safe set? (BLF goal)
    ultimate_bound                -> what is the empirical UUB radius after the transient?
"""

from __future__ import annotations

import numpy as np


def tracking_rmse(reference, state) -> float:
    """Root-mean-square tracking error."""
    err = np.asarray(reference, dtype=float) - np.asarray(state, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def max_abs_error(reference, state) -> float:
    """Worst-case tracking error."""
    err = np.asarray(reference, dtype=float) - np.asarray(state, dtype=float)
    return float(np.max(np.abs(err)))


def saturation_fraction(delta_u, tol: float = 1e-12) -> float:
    """Fraction of time steps where the actuator clipped the command (|delta_u| > tol)."""
    d = np.asarray(delta_u, dtype=float)
    return float(np.mean(np.abs(d) > tol))


def constraint_violations(x, lower: float, upper: float) -> int:
    """Number of samples outside the safe interval [lower, upper]."""
    x = np.asarray(x, dtype=float)
    return int(np.sum((x < lower) | (x > upper)))


def ultimate_bound(x, settle_fraction: float = 0.5) -> float:
    """Empirical UUB radius: max |x| over the last ``1 - settle_fraction`` of the run.

    The first part of the run is treated as the transient and ignored, mirroring
    "after the transient: ||x(t)|| <= some calculable bound" in the notes.
    """
    if not 0.0 <= settle_fraction < 1.0:
        raise ValueError("settle_fraction must be in [0, 1)")
    x = np.asarray(x, dtype=float)
    start = int(len(x) * settle_fraction)
    return float(np.max(np.abs(x[start:])))
