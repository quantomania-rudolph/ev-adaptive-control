"""Firewalled cascade controller: the same control laws as control.CascadeController, but it can ONLY read a Measurements.

Differences from the legacy controller (each is a contract item, docs/INFORMATION_CONTRACT.md):
  * inputs: interfaces.Measurements only. No plant object, no ledger, no true state. The constructor takes config dataclasses and
    a `Beliefs` record of ABSOLUTE numbers (built outside this module by draw_beliefs()).
  * speed is the PLL speed everywhere (speed loop, decoupling, adaptation).
  * saturation: the controller clips its own command against ceiling_hat = min(vdc_m / sqrt(3), U_CAP) (or the labelled oracle
    telemetry) and feeds the adaptive law / anti-windup the shortfall IT removed (du_known), never the plant's true shortfall.
  * angle advance: it rotates its sent command forward by theta_adv = d * omega_el_hat * tau to compensate its OWN known delay d.
    (GEM already contains the half-step hold rotation, so there is nothing to compensate for d = 0; legacy equivalence holds.)
  * psi_tilde is NOT computed here (it needs truth); the logger computes it.
Nothing in this file may import the plant, battery, bms, sensors or inverter modules (tests/test_realism_firewall.py checks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces import ControllerConfig, Measurements, U_CAP, rot


@dataclass(frozen=True)
class Beliefs:
    """What the controller believes about the motor, absolute numbers (SI). psi_hat0 is the adaptive law's initial estimate."""
    R: float
    Ld: float
    Lq: float
    psi_name: float        # nameplate flux used for Kt and the governor ellipse
    J: float
    psi_hat0: float
    psi_range: tuple = (0.03, 0.10)


class FirewalledController:
    def __init__(self, cfg: ControllerConfig, beliefs: Beliefs):
        self.cfg, self.b = cfg, beliefs
        self.p = cfg.p
        self.psi_hat, self.psi_nom = beliefs.psi_hat0, beliefs.psi_hat0
        Kt = 1.5 * self.p * beliefs.psi_name
        self.Kt = Kt
        self.Kp_w = beliefs.J * cfg.speed_bw / Kt
        self.Ki_w = self.Kp_w * cfg.speed_bw / 4.0
        self.int_w = 0.0
        self.iq_ref = 0.0
        self.i_ref_clip = 0.0
        self.i_lim_eff = cfg.i_lim
        self.z = 0.0
        self.n = 0
        self.dt_w = 10 * cfg.tau
        self.du_q_known = 0.0       # shortfall the controller itself removed on the previous step (controller frame)
        self.last: dict = {}

    # ------------------------------------------------------------------------------------------------ helpers
    def _ceiling_hat(self, m: Measurements) -> float:
        c = min(m.vdc_m / np.sqrt(3.0), U_CAP)
        if m.ceiling_telemetry is not None:       # labelled oracle-ceiling event only (non-physical; reproduces experiments A-D)
            c = min(c, float(m.ceiling_telemetry))
        return float(c)

    def _i_feasible(self, w_m: float, m: Measurements) -> float:
        cfg, b = self.cfg, self.b
        w_el = max(abs(self.p * w_m), 1.0)
        u_nom = (m.vdc_m / np.sqrt(3.0)) if cfg.vdc_aware else U_CAP
        room = (u_nom / w_el) ** 2 - b.psi_name ** 2
        i_v = np.sqrt(max(room, 0.0)) / b.Lq
        lim = min(cfg.i_lim, cfg.gov_factor * i_v)
        if cfg.bms_aware:                          # BMS current limit -> iq limit through the power balance (belief eta)
            p_lim = m.vdc_m * m.bms.i_dis_lim * cfg.eta
            lim = min(lim, p_lim / (self.Kt * max(abs(w_m), 1.0)))
        return lim

    def _speed_loop(self, omega_ref: float, w_m: float, m: Measurements):
        e_w = omega_ref - w_m
        unsat = self.Kp_w * e_w + self.int_w
        lim = self._i_feasible(w_m, m) if self.cfg.governor else self.cfg.i_lim
        self.i_lim_eff = lim
        self.iq_ref = float(np.clip(unsat, -lim, lim))
        self.i_ref_clip = self.iq_ref - unsat
        self.int_w += self.dt_w * (self.Ki_w * e_w + 5.0 * self.i_ref_clip)

    # ------------------------------------------------------------------------------------------------ the control step
    def command(self, m: Measurements):
        """Return (u_sent, internals). u_sent is the dq voltage the controller hands to the inverter (already advanced)."""
        cfg, b = self.cfg, self.b
        if self.n % 10 == 0:
            self._speed_loop(m.omega_ref, m.omega_m_m, m)
        self.n += 1
        w_el, i_d, i_q = self.p * m.omega_m_m, m.i_d_m, m.i_q_m
        dt = cfg.tau

        e = i_q - self.iq_ref
        e_a = e
        if cfg.mode == "aw":
            self.z += dt * (-cfg.k_i * self.z + self.du_q_known) / b.Lq
            e_a = e - self.z
        dpsi = -cfg.gamma * w_el * e_a
        if cfg.mode == "robust":
            dpsi -= cfg.sigma * (self.psi_hat - self.psi_nom)
        self.psi_hat += dt * dpsi
        if cfg.mode == "robust" or cfg.clamp_psi:
            self.psi_hat = float(np.clip(self.psi_hat, *b.psi_range))

        u_q = b.R * i_q + w_el * (b.Ld * i_d + self.psi_hat) - cfg.k_i * e
        k_d = cfg.k_i * (b.Ld / b.Lq) if cfg.d_gain_by_L else cfg.k_i
        u_d = b.R * i_d - w_el * b.Lq * i_q - k_d * i_d
        u_cmd = np.array([u_d, u_q])

        # the controller's own clip against what it BELIEVES the bus allows (radial projection, same as the plant's)
        ceil_hat = self._ceiling_hat(m)
        mag = float(np.hypot(*u_cmd))
        u_clip = u_cmd if mag <= ceil_hat else u_cmd * (ceil_hat / mag)
        self.du_q_known = float(u_clip[1] - u_cmd[1])          # d_u_q convention: delivered - requested, controller frame

        theta_adv = cfg.delay_steps * w_el * dt if cfg.angle_advance else 0.0
        u_sent = u_clip if theta_adv == 0.0 else rot(theta_adv) @ u_clip

        self.last = {
            "iq_ref": self.iq_ref, "e_q": e, "e_qa": e_a, "z": self.z, "psi_hat": self.psi_hat,
            "omega_ref": m.omega_ref, "i_ref_clip": self.i_ref_clip, "i_lim_eff": self.i_lim_eff,
            "u_d_ctrl": float(u_cmd[0]), "u_q_ctrl": float(u_cmd[1]),            # pre-clip command, controller frame
            "u_d_clip": float(u_clip[0]), "u_q_clip": float(u_clip[1]),          # after the controller's own clip
            "ceiling_hat": ceil_hat, "du_q_known": self.du_q_known, "theta_adv": theta_adv,
        }
        return u_sent, self.last
