"""Sensor layer of the information firewall: truth dict -> interfaces.Measurements.

Conventions (STATED ONCE, shared with interfaces.py and the contract):
  * delta = theta_m - theta_true.  i_dq_m = R(-delta) i_dq, so i_d_m = +i_q sin(delta) for small delta.
  * Measurement frame for the dq currents: the plant's dq current states are the Park transform of (i_a, i_b, i_c) at the
    PREVIOUS step's electrical angle (truth['epsilon_prev']; verified by the lead to 7e-13).  So
        i_d_m, i_q_m = interfaces.park(i_a_m, i_b_m, i_c_m, epsilon_prev + delta_prev, two_sensor=cfg.two_sensor)
    where delta_prev is the angle error that applied to the previous step's angle.  interfaces.park is the only Park used.
    On the very first call there is no previous error; delta_prev is then taken equal to the current delta_k.

Model, per call to measure() (non-identity mode):
  Per-run draws, made in reset() (which __init__ calls) from the generator, ONE call rng.uniform(-1, 1, 8) scaled by the half-widths:
      u[0:3] -> gain_i (x i_gain_err), u[3:6] -> offset_i (x i_offset_a, amperes), u[6] -> theta_offset (x theta_offset_rad),
      u[7] -> vdc_gain (x vdc_gain_err).   Calling reset() again redraws (the generator keeps advancing; same seed => same sequence).
  Per-step noise: ONE call rng.standard_normal(5), always made (so changing a sigma never shifts the stream), order
      [noise_ia, noise_ib, noise_ic, noise_theta, noise_vdc].
  * Current:  i_m = clip(quantize((1 + g_x) i + b_x + sigma_i n)),  sigma_i = i_noise_lsb * LSB,  LSB = 2 i_range_a / 2**adc_bits.
      quantize(v) = round(v / LSB) * LSB with round-half-to-even (Python round / np.round); a mid-tread quantizer (zero is a code).
      clip to [-i_range_a, +i_range_a] AFTER rounding (i_range_a is itself a multiple of LSB, 2**(adc_bits-1) codes each side).
      With the defaults this gives 4097 levels from -2048 LSB to +2048 LSB.
  * Angle:  delta_k = theta_offset + theta_noise_rad * n + theta_ripple_rad*(sin(th_mech) + sin(2 th_mech)).
      th_mech = (UNWRAPPED electrical angle) / p, where the unwrapped angle accumulates wrap(epsilon_k - epsilon_{k-1}) starting
      from the first epsilon.  DEVIATION from the contract text (th_mech = epsilon_el / p): epsilon_el is wrapped to +/-pi by
      GEM, so epsilon/p would make the ripple jump at every wrap and would not be periodic in the mechanical revolution.
      Identical to the literal formula until the first wrap.  theta_m = epsilon_el + delta_k, NOT wrapped (it can exceed +/-pi
      by |delta_k|); last_gaps['theta_m'] is exactly delta_k.  (No angle quantization: SensorConfig has no field for it.)
  * Speed: second-order PLL on theta_m (never on truth omega).  e = wrap(theta_m - theta_hat) to (-pi, pi];
        omega_hat += dt wn^2 e ;  theta_hat += dt (omega_hat + 2 zeta wn e)   (sequential, omega_hat updated first);
        omega_m_m = omega_hat / p.  dt = cfg.tau.  Start-up: theta_hat = theta_m of the first sample, omega_hat = 0, so a run that
        starts at speed sees a transient (frequency step) of overshoot 4.3 % and settling time ~ 4/(zeta wn) = 19 ms; the lead
        starts runs at rest.  Closed loops: theta_hat/theta_m = (2 zeta wn s + wn^2)/D, omega_hat/theta_m = wn^2 s/D,
        D = s^2 + 2 zeta wn s + wn^2.
      Noise on omega_m_m for white angle noise of sigma_theta per sample: var(omega_hat_el) = sigma_theta^2 dt wn^3 / (4 zeta)
      (from (1/2pi) int |wn^2 jw/D|^2 dw = wn^4/(2*2 zeta wn)); divide the std by p for omega_m_m.
  * DC link: vdc_m = LPF(quantize(clip((1 + gv) V_dc + vdc_noise_v n))), clip to [0, vdc_range_v], 12-bit round-half-even over
      [0, vdc_range_v] (LSB_v = range/4096), first-order low-pass y += alpha (q - y), alpha = 1 - exp(-2 pi fc dt) (exact
      discretization of the step response at the sample instants).  The filter is initialised at the first quantized sample.
      DC gain of the chain = 1 + gv (up to the quantizer).
  * Request: omega_ref passes through unchanged when ref_hold_s == 0 and ref_quant == 0.  Otherwise sample-and-hold: the
      request is re-sampled at the first call of each hold period k = floor(t/ref_hold_s + 1e-6) and quantized to multiples of
      ref_quant (round-half-even); with ref_hold_s == 0 and ref_quant > 0 it is only quantized.  Applies in identity mode too
      (a property of the data source, not a sensor error).
  * identity mode (cfg.identity): i_a/b/c_m, i_d_m, i_q_m (the plant's own dq states, DEVIATION: returned directly rather than via
      park, so legacy equivalence is exact), theta_m = epsilon_el, omega_m_m = truth omega_m, vdc_m = truth V_dc.  No PLL, filter,
      quantization or noise; the generator is not touched per step.

Public state for logging/tests: gain_i, offset_i (np arrays), theta_offset, vdc_gain, last_delta, last_gaps, LSB_i, LSB_v.
bms and ceiling_telemetry are passed through unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from interfaces import TRUTH_KEYS, BMSMsg, Measurements, SensorConfig, park

_TWO_PI = 2.0 * math.pi


def wrap_pi(x: float) -> float:
    """Wrap to (-pi, pi]."""
    return x - _TWO_PI * math.ceil((x - math.pi) / _TWO_PI)


class SensorSuite:
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.LSB_i = 2.0 * cfg.i_range_a / 2 ** cfg.adc_bits
        self.LSB_v = cfg.vdc_range_v / 2 ** 12
        self.dt = float(cfg.tau)
        self._alpha = 1.0 - math.exp(-_TWO_PI * cfg.vdc_corner_hz * self.dt)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        cfg = self.cfg
        u = self.rng.uniform(-1.0, 1.0, 8)
        self.gain_i = u[0:3] * cfg.i_gain_err
        self.offset_i = u[3:6] * cfg.i_offset_a
        self.theta_offset = float(u[6] * cfg.theta_offset_rad)
        self.vdc_gain = float(u[7] * cfg.vdc_gain_err)
        self._theta_hat = 0.0
        self._omega_hat = 0.0
        self._pll_init = False
        self._delta_prev = None
        self._eps_last = None
        self._eps_unw = 0.0
        self._vdc_y = None
        self._ref_k = None
        self._ref_held = 0.0
        self.last_delta = 0.0
        self.last_gaps: dict = {}

    @property
    def pll_theta_hat(self) -> float:
        """PLL phase estimate (read-only, for tests/logging)."""
        return self._theta_hat

    @property
    def pll_omega_hat_el(self) -> float:
        return self._omega_hat

    # ------------------------------------------------------------------ helpers
    def _quant_i(self, v: float) -> float:
        lsb = self.LSB_i
        q = round(v / lsb) * lsb
        r = self.cfg.i_range_a
        return -r if q < -r else (r if q > r else q)

    def _request(self, t: float, omega_ref: float) -> float:
        cfg = self.cfg
        hold, quant = cfg.ref_hold_s, cfg.ref_quant
        if hold == 0.0 and quant == 0.0:
            return omega_ref
        if hold > 0.0:
            k = math.floor(t / hold + 1e-6)
            if k != self._ref_k:
                self._ref_k = k
                self._ref_held = round(omega_ref / quant) * quant if quant > 0.0 else omega_ref
            return self._ref_held
        return round(omega_ref / quant) * quant

    # ------------------------------------------------------------------ main entry
    def measure(self, truth: dict, omega_ref: float, bms: BMSMsg, ceiling_telemetry=None) -> Measurements:
        cfg = self.cfg
        t = truth["t"]
        ref = self._request(t, omega_ref)
        ia, ib, ic = truth["i_a"], truth["i_b"], truth["i_c"]
        eps, eps_prev = truth["epsilon_el"], truth["epsilon_prev"]

        if cfg.identity:
            self.last_delta = 0.0
            m = Measurements(t=t, i_a_m=ia, i_b_m=ib, i_c_m=ic, i_d_m=truth["i_d"], i_q_m=truth["i_q"], theta_m=eps,
                             omega_m_m=truth["omega_m"], vdc_m=truth["V_dc"], bms=bms, omega_ref=ref,
                             ceiling_telemetry=ceiling_telemetry)
            self._set_gaps(m, truth, ref, omega_ref)
            return m

        n = self.rng.standard_normal(5)            # fixed order: ia, ib, ic, theta, vdc
        sig_i = cfg.i_noise_lsb * self.LSB_i
        g, b = self.gain_i, self.offset_i
        ia_m = self._quant_i((1.0 + g[0]) * ia + b[0] + sig_i * n[0])
        ib_m = self._quant_i((1.0 + g[1]) * ib + b[1] + sig_i * n[1])
        ic_m = self._quant_i((1.0 + g[2]) * ic + b[2] + sig_i * n[2])

        # angle error
        if self._eps_last is None:
            self._eps_unw = eps
        else:
            self._eps_unw += wrap_pi(eps - self._eps_last)
        self._eps_last = eps
        th_mech = self._eps_unw / cfg.p
        delta = (self.theta_offset + cfg.theta_noise_rad * n[3]
                 + cfg.theta_ripple_rad * (math.sin(th_mech) + math.sin(2.0 * th_mech)))
        self.last_delta = delta
        theta_m = eps + delta

        # dq currents in the measurement frame (angle error of the previous step's angle)
        dprev = delta if self._delta_prev is None else self._delta_prev
        i_d_m, i_q_m = park(ia_m, ib_m, ic_m, eps_prev + dprev, two_sensor=cfg.two_sensor)
        self._delta_prev = delta

        # PLL
        if not self._pll_init:
            self._theta_hat, self._omega_hat, self._pll_init = theta_m, 0.0, True
        else:
            wn, zeta, dt = cfg.pll_wn, cfg.pll_zeta, self.dt
            e = wrap_pi(theta_m - self._theta_hat)
            self._omega_hat += dt * wn * wn * e
            self._theta_hat = wrap_pi(self._theta_hat + dt * (self._omega_hat + 2.0 * zeta * wn * e))
        omega_m_m = self._omega_hat / cfg.p

        # DC link
        v = (1.0 + self.vdc_gain) * truth["V_dc"] + cfg.vdc_noise_v * n[4]
        v = min(max(v, 0.0), cfg.vdc_range_v)
        q = round(v / self.LSB_v) * self.LSB_v
        self._vdc_y = q if self._vdc_y is None else self._vdc_y + self._alpha * (q - self._vdc_y)

        m = Measurements(t=t, i_a_m=ia_m, i_b_m=ib_m, i_c_m=ic_m, i_d_m=i_d_m, i_q_m=i_q_m, theta_m=theta_m,
                         omega_m_m=omega_m_m, vdc_m=self._vdc_y, bms=bms, omega_ref=ref, ceiling_telemetry=ceiling_telemetry)
        self._set_gaps(m, truth, ref, omega_ref)
        return m

    def _set_gaps(self, m: Measurements, truth: dict, ref: float, omega_ref: float) -> None:
        """Measured minus true per channel (omega_ref: delivered minus requested)."""
        self.last_gaps = {
            "i_a_m": m.i_a_m - truth["i_a"], "i_b_m": m.i_b_m - truth["i_b"], "i_c_m": m.i_c_m - truth["i_c"],
            "i_d_m": m.i_d_m - truth["i_d"], "i_q_m": m.i_q_m - truth["i_q"],
            "theta_m": self.last_delta,
            "omega_m_m": m.omega_m_m - truth["omega_m"], "vdc_m": m.vdc_m - truth["V_dc"],
            "omega_ref": ref - omega_ref,
        }
