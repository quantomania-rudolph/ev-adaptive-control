"""Inverter layer of the information firewall: computation delay, angle-error rotation, dead-time and device-drop error.

Order of operations (contract section 6): controller clip -> pre_clip (delay FIFO, then rotation by the angle error) ->
the PLANT's true clip -> post_clip_error added AFTER the clip.  Nothing is clipped in this file.

pre_clip(u_sent, delta): FIFO of cfg.delay_steps entries, initial content zeros; with delay_steps == 0 the command passes through
the same step.  u_delayed = FIFO output, u_true = R(delta) u_delayed, R(phi) = [[cos,-sin],[sin,cos]], delta = theta_m - theta_true.
last_delay_dv = u_delayed - u_sent, last_rot_dv = u_true - u_delayed (exact decomposition: u_true = u_sent + both).  The delay
is applied whatever cfg.enabled says (identity_mode() has delay_steps = 0).

post_clip_error: per phase x, with K = dead_time_s/tau * vdc_true + device_drop_v (Tpwm = tau) and s(i) = tanh(i / smooth_band_a):
    v_x = -K s(i_x_true) + compensation * K s(i_x_meas)
converted with interfaces.park(va, vb, vc, epsilon_true) (full Clarke: the zero-sequence part drops out).  Zero when enabled is False
or K == 0.  DEVIATION from the contract text: the compensation (70 % of the distortion, computed from the MEASURED current sign)
lives here, in the inverter firmware, not in the controller, so the controller never needs to know Td or Vdc.  This is only
bookkeeping of where the term is added; the net effect on the plant is the same.

Bounds (exact, derived in tests/test_inverter.py U16): with |s| <= 1 each net phase value is in [-(1+c)K, (1+c)K]; the Clarke
image of three values in [-V, V] has magnitude at most (4/3) V (vertex pattern (1,1,-1)).  So |error_dq| <= (4/3)(1+c) K in
general, and <= (4/3)|1-c| K when the measured current equals the true one.  The contract's (4/pi) K is only the DC part
(fundamental) of the error for a rotating sinusoidal current; the instantaneous peak includes the 6th-harmonic ripple and is
(4/3) K for the uncompensated sharp-sign limit.  Both are exposed: dc_fundamental(vdc), max_error_dq(vdc).
"""
from __future__ import annotations

import math

import numpy as np

from interfaces import InverterConfig, park


class InverterModel:
    def __init__(self, cfg: InverterConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng            # unused: the model is deterministic; kept for interface symmetry
        self.reset()

    def reset(self) -> None:
        self._fifo = [np.zeros(2) for _ in range(int(self.cfg.delay_steps))]
        self.last_delay_dv = np.zeros(2)
        self.last_rot_dv = np.zeros(2)

    # ------------------------------------------------------------------ before the plant's clip
    def pre_clip(self, u_sent: np.ndarray, delta: float) -> np.ndarray:
        u = np.array(u_sent, dtype=float)
        if self.cfg.delay_steps > 0:
            self._fifo.append(u)
            u_del = self._fifo.pop(0)
        else:
            u_del = u
        c, s = math.cos(delta), math.sin(delta)
        out = np.array([c * u_del[0] - s * u_del[1], s * u_del[0] + c * u_del[1]])
        self.last_delay_dv = u_del - u
        self.last_rot_dv = out - u_del
        return out

    # ------------------------------------------------------------------ after the plant's clip
    def distortion_amplitude(self, vdc_true: float) -> float:
        """K = Td/Tpwm * Vdc + Vdrop (per-phase square-wave amplitude of the uncompensated distortion)."""
        return self.cfg.dead_time_s / self.cfg.tau * max(float(vdc_true), 0.0) + self.cfg.device_drop_v

    def dc_fundamental(self, vdc_true: float) -> float:
        """(4/pi) K: DC dq magnitude of the UNCOMPENSATED error for a rotating sinusoidal current (large vs smooth_band_a)."""
        return 4.0 / math.pi * self.distortion_amplitude(vdc_true)

    def max_error_dq(self, vdc_true: float) -> float:
        """Hard bound on |post_clip_error| for ANY currents (true and measured arbitrary): (4/3)(1+c) K."""
        return 4.0 / 3.0 * (1.0 + self.cfg.compensation) * self.distortion_amplitude(vdc_true)

    def _s(self, i: float) -> float:
        band = self.cfg.smooth_band_a
        return math.tanh(i / band) if band > 0.0 else float(np.sign(i))

    def post_clip_error(self, i_abc_true, i_abc_meas, vdc_true: float, epsilon_true: float) -> np.ndarray:
        cfg = self.cfg
        if not cfg.enabled:
            return np.zeros(2)
        K = self.distortion_amplitude(vdc_true)
        if K == 0.0:
            return np.zeros(2)
        v = [-K * self._s(it) + cfg.compensation * K * self._s(im) for it, im in zip(i_abc_true, i_abc_meas)]
        d, q = park(v[0], v[1], v[2], epsilon_true)
        return np.array([d, q])
