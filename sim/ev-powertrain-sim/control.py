"""Cascade controller with an adaptive q-axis current loop. Exposes its internals for V(t) analysis.

Structure (field-oriented control, i_d regulated to 0):
    speed PI (1 kHz) -> i_q*  ->  q-axis ADAPTIVE current loop (10 kHz) -> u_q,c
                                  d-axis certainty-equivalence loop      -> u_d,c

Why adaptive on the q axis, with psi (flux linkage) as the unknown parameter:
    L_q di_q/dt = u_q - R i_q - w_el (L_d i_d + psi_p)          <- plant (verified against GEM to 1e-10 A)
    control   u_q,c = R i_q + w_el (L_d i_d + psi_hat) - k e      with e = i_q - i_q*, psi_tilde = psi_hat - psi_p
    error dyn L_q e_dot = -k e + w_el psi_tilde  (+ du_q when the actuator saturates)
    Lyapunov  V = 0.5 L_q e^2 + psi_tilde^2 / (2 Gamma)
              V_dot = -k e^2 + psi_tilde (w_el e + psi_hat_dot/Gamma)   -> choose psi_hat_dot = -Gamma w_el e
This is exactly the notes' "theta_hat_dot = -Gamma phi e" with phi = w_el. The unknown term multiplies
speed, so the estimator is strongest exactly where the voltage ceiling bites (high speed).
(An earlier draft had the sign of the w_el*psi_tilde term wrong and the baseline diverged even without
saturation. That is the single most likely bug when the team re-derives this: check the sign by simulation.)

Modes (the notes' Experiments B, C, D):
    naive : psi_hat_dot = -Gamma w_el e
    robust: sigma-modification (leak toward the NOMINAL value; see note) + projection onto a physical range
    aw    : naive law driven by the compensated error e_a = e - C z, with  Lq z_dot = -k z + du_q
            (z is the part of the error caused by the actuator; for this plant it is exact, so
             e_a obeys the UNSATURATED error dynamics and the estimator is not fooled)
"""
from __future__ import annotations

import numpy as np


class CascadeController:
    def __init__(self, plant, mode="naive", psi_hat0=None, gamma=2.0e-5, k_i=3.0, sigma=2.0,
                 i_lim=250.0, speed_bw=20.0, psi_range=(0.03, 0.10), governor=False, gov_factor=0.75, vdc_aware=False):
        self.pl, self.mode = plant, mode
        self.vdc_aware, self._vdc = vdc_aware, plant.V_dc   # vdc_aware: governor uses the MEASURED bus voltage, not the nominal
        self.governor, self.gov_factor, self.i_lim_eff = governor, gov_factor, i_lim
        self.Rs, self.Ld, self.Lq, self.p = plant.Rs, plant.Ld, plant.Lq, plant.p
        self.psi_nom = plant.psi_p * 0.7 if psi_hat0 is None else psi_hat0   # deliberately 30 % wrong
        self.psi_hat = self.psi_nom
        self.gamma, self.k_i, self.sigma, self.psi_range = gamma, k_i, sigma, psi_range
        self.i_lim = i_lim
        # speed PI: Kp = J*bw/Kt, Ki = Kp*bw/4 ; Kt = 1.5 p psi
        Kt = 1.5 * self.p * plant.psi_p
        self.Kp_w = plant.J * speed_bw / Kt
        self.Ki_w = self.Kp_w * speed_bw / 4.0
        self.int_w = 0.0
        self.iq_ref = 0.0
        self.z = 0.0                                    # anti-windup auxiliary state
        self.n = 0
        self.dt_w = 10 * plant.tau                      # speed loop runs every 10 current-loop steps
        self.last = {}

    # speed loop (1 kHz) with back-calculation anti-windup, so the SPEED integrator is not the culprit
    def _i_feasible(self, omega):
        """Reference-governor-in-miniature: the largest i_q the NOMINAL voltage ceiling can sustain at this speed.

        Steady-state voltage ellipse with i_d = 0:  (L_q i_q)^2 + psi^2 <= (u_max / w_el)^2.
        A fixed 250 A clamp is infeasible at 200 rad/s (found in experiments C and D: after the event the drive
        sat in a trap with 230 A and only 18 N.m because the d-axis cross-coupling voltage consumed the budget).
        A safety factor leaves room for R*i and transients. Uses the NOMINAL ceiling only: an unmodeled
        sag (the experiment's event) is exactly what the governor cannot know in advance.
        """
        w_el = max(abs(self.p * omega), 1.0)
        u_nom = self._vdc / np.sqrt(3.0) if self.vdc_aware else self.pl.u_max_hw
        room = (u_nom / w_el) ** 2 - self.pl.psi_p ** 2
        i_v = np.sqrt(max(room, 0.0)) / self.Lq
        return min(self.i_lim, self.gov_factor * i_v)

    def _speed_loop(self, omega_ref, omega):
        e_w = omega_ref - omega
        unsat = self.Kp_w * e_w + self.int_w
        lim = self._i_feasible(omega) if self.governor else self.i_lim
        self.i_lim_eff = lim
        self.iq_ref = float(np.clip(unsat, -lim, lim))
        self.i_ref_clip = self.iq_ref - unsat           # M4 request-vs-delivered at the current reference
        self.int_w += self.dt_w * (self.Ki_w * e_w + 5.0 * self.i_ref_clip)

    def command(self, omega_ref, st, prev_ledger):
        """Return u_c = (u_d,c, u_q,c) and a dict of controller internals for logging."""
        self._vdc = st["V_dc"]
        if self.n % 10 == 0:
            self._speed_loop(omega_ref, st["omega_m"])
        self.n += 1
        w_el, i_d, i_q = st["omega_el"], st["i_d"], st["i_q"]
        dt = self.pl.tau
        self._vdc = st["V_dc"]

        # ---- adaptation (uses the saturation mismatch of the PREVIOUS step, which is all a real controller has)
        e = i_q - self.iq_ref
        du_q = prev_ledger.get("d_u_q", 0.0)
        e_a = e
        if self.mode == "aw":
            # z_dot = (-k z + du_q) / Lq   (Euler)
            self.z += dt * (-self.k_i * self.z + du_q) / self.Lq
            e_a = e - self.z
        dpsi = -self.gamma * w_el * e_a
        if self.mode == "robust":
            # sigma-modification (leakage), sigma in 1/s. NOTE: the notes write -sigma*theta_hat (leak to 0).
            # psi is far from 0, so leaking to zero would bias it badly; we leak toward the nominal value
            # instead. This is the standard "switching/nominal sigma" variant. Theory team: confirm.
            dpsi -= self.sigma * (self.psi_hat - self.psi_nom)
        self.psi_hat += dt * dpsi
        if self.mode == "robust":
            self.psi_hat = float(np.clip(self.psi_hat, *self.psi_range))   # projection

        # ---- control laws
        u_q = self.Rs * i_q + w_el * (self.Ld * i_d + self.psi_hat) - self.k_i * e
        u_d = self.Rs * i_d - w_el * self.Lq * i_q - self.k_i * i_d            # i_d* = 0
        self.last = {
            "iq_ref": self.iq_ref, "e_q": e, "e_qa": e_a, "z": self.z, "psi_hat": self.psi_hat,
            "psi_tilde": self.psi_hat - self.pl.psi_p, "omega_ref": omega_ref,
            "i_ref_clip": getattr(self, "i_ref_clip", 0.0), "i_lim_eff": self.i_lim_eff,
        }
        return np.array([u_d, u_q]), self.last
