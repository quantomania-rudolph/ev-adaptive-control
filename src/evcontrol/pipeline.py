"""End-to-end glue: datagen -> controller + actuator + plant -> evaluation.

THIS FILE IS A PLACEHOLDER on purpose. It uses the simplest possible toy plant so the
integration test has something real to run:

    x_dot = -x + u + d          (stable scalar plant, unit input gain)
    u_c   = k * (r - x)         (plain proportional controller, NOT adaptive yet)
    u     = sat(u_c)            (actuator from evcontrol.actuator)

Why this plant is a good first integration target: since |u| <= u_max and |d| <= d_max,
x_dot <= -x + (u_max + d_max), so |x| can never settle above u_max + d_max. That is a
tiny, checkable BIBS-style bound, which the integration test asserts.

When the controller / plant segments exist, swap ``simulate`` for the real thing but keep
the returned dictionary keys, so evaluation code and tests don't need to change.
"""

from __future__ import annotations

import numpy as np

from evcontrol.actuator import Actuator
from evcontrol.datagen import generate_disturbance, generate_reference, time_vector


def simulate(
    n_steps: int = 2000,
    dt: float = 0.005,
    gain: float = 5.0,
    u_limit: float = 1.0,
    d_bound: float = 0.1,
    ref_kind: str = "step",
    ref_amplitude: float = 3.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run one closed-loop simulation with forward-Euler integration."""
    t = time_vector(n_steps, dt)
    r = generate_reference(n_steps, dt, kind=ref_kind, amplitude=ref_amplitude)
    d = generate_disturbance(n_steps, d_bound, seed=seed)
    actuator = Actuator(-u_limit, u_limit)

    x = np.zeros(n_steps)
    u_c = np.zeros(n_steps)
    u = np.zeros(n_steps)
    delta_u = np.zeros(n_steps)

    for k in range(n_steps - 1):
        u_c[k] = gain * (r[k] - x[k])
        u[k], delta_u[k] = actuator.step(u_c[k])
        x[k + 1] = x[k] + dt * (-x[k] + u[k] + d[k])  # forward Euler step

    # Fill the final sample so every array has length n_steps.
    u_c[-1] = gain * (r[-1] - x[-1])
    u[-1], delta_u[-1] = actuator.step(u_c[-1])

    return {"t": t, "r": r, "d": d, "x": x, "u_c": u_c, "u": u, "delta_u": delta_u}
