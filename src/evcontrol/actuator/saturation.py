"""Actuator saturation, straight from the working notes:

    u = sat(u_c)          what the hardware actually applies
    delta_u = u - u_c     the mismatch the controller/proof must account for

Keeping ``delta_u`` as an explicit output (instead of hiding it) is the whole point:
the anti-windup state ``z' = -A_z z + B_z * delta_u`` and the Lyapunov cross terms
all need it.
"""

from __future__ import annotations

import numpy as np


def saturate(u_c, u_min: float, u_max: float):
    """Clip a command to [u_min, u_max]. Works on scalars and numpy arrays."""
    if u_min > u_max:
        raise ValueError("u_min must be <= u_max")
    return np.clip(u_c, u_min, u_max)


def saturation_mismatch(u_c, u_min: float, u_max: float):
    """delta_u = sat(u_c) - u_c. Zero whenever the command is feasible."""
    return saturate(u_c, u_min, u_max) - np.asarray(u_c, dtype=float)


class Actuator:
    """Static saturating actuator (the simplest possible model).

    Replace or extend this with torque limits that depend on speed / SOC /
    temperature, rate limits, or inverter dynamics. Keep the ``step`` signature
    (``u_c -> (u, delta_u)``) so the pipeline and tests keep working.
    """

    def __init__(self, u_min: float, u_max: float) -> None:
        if u_min > u_max:
            raise ValueError("u_min must be <= u_max")
        self.u_min = float(u_min)
        self.u_max = float(u_max)

    def step(self, u_c: float) -> tuple[float, float]:
        """Apply one command. Returns (u, delta_u)."""
        u = float(saturate(u_c, self.u_min, self.u_max))
        return u, u - float(u_c)
