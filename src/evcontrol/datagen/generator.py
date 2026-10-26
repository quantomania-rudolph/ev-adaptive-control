"""Signal generators for simulation inputs.

CONTRACT with other segments (please don't break without telling the team):
    * every generator returns a 1-D ``numpy.ndarray`` of length ``n_steps``
    * time step ``dt`` is in seconds
    * randomness is controlled ONLY by the ``seed`` argument, so runs are reproducible

Why seeds matter: if a teammate's result can't be reproduced, nobody can debug it.
We use ``numpy.random.default_rng(seed)`` (a local generator) instead of the global
``np.random.seed`` so one segment's randomness can never leak into another's.
"""

from __future__ import annotations

import numpy as np


def time_vector(n_steps: int, dt: float) -> np.ndarray:
    """Return t = [0, dt, 2*dt, ...] with ``n_steps`` samples."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    return np.arange(n_steps) * dt


def generate_reference(
    n_steps: int,
    dt: float,
    kind: str = "step",
    amplitude: float = 1.0,
    frequency_hz: float = 0.5,
) -> np.ndarray:
    """Driver-request-like reference signal r(t).

    kind="step"  : jumps from 0 to ``amplitude`` at t = 0 (worst case for saturation)
    kind="sine"  : amplitude * sin(2*pi*f*t)
    kind="ramp"  : rises linearly from 0 to ``amplitude`` over the whole run
    """
    t = time_vector(n_steps, dt)
    if kind == "step":
        return np.full(n_steps, amplitude, dtype=float)
    if kind == "sine":
        return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    if kind == "ramp":
        return np.linspace(0.0, amplitude, n_steps)
    raise ValueError(f"unknown reference kind: {kind!r}")


def generate_disturbance(n_steps: int, bound: float, seed: int = 0) -> np.ndarray:
    """Bounded random disturbance d(t) with |d(t)| <= ``bound``.

    The hard bound is on purpose: our theorem (BIBS/UUB) assumes a *bounded*
    disturbance, so the simulation inputs must respect that assumption.
    """
    if bound < 0:
        raise ValueError("bound must be non-negative")
    rng = np.random.default_rng(seed)
    return rng.uniform(-bound, bound, size=n_steps)
