"""Battery pack + DC-link model for the Tier 1 plant (replaces the constant 300 V supply).

    battery (OCV(SOC), R0(T), one RC branch, BMS current limits)
        --- busbar Rc ---> DC-link capacitor C_dc (voltage state V_dc) ---> inverter ---> motor

Why this structure (learning note):
    * The inverter ceiling is V_dc/sqrt(3). Making V_dc a STATE, driven by real current, is what turns the notes'
      "sudden unmodeled actuator saturation" from an imposed number into something the powertrain produces itself.
    * The DC link is where energy requested by the motor and energy the battery can give meet. Their difference
      is stored in (or drained from) the capacitor, so it is physically observable as dV_dc/dt. That is the
      battery-side mismatch row (M10 in the plan).
    * The BMS current limit is the realistic way for a pack to be unable to deliver: cold, low SOC, or a derate.
      When it bites, the capacitor drains in milliseconds and V_dc collapses.

Numerics: the capacitor/resistor loop has time constant C_dc*Rs ~ 0.3 ms, close to our 0.1 ms step, so explicit
Euler would be marginal. We solve the V_dc update implicitly (one linear equation, no iteration) and the RC branch
with its exact exponential. Inverter power is treated as constant-power over a step using the previous V_dc.

ASSUMPTIONS (generic Li-ion NMC numbers, NOT fitted to a specific pack; see plan register): OCV table, R0, R1, tau1,
BMS limits, inverter loss coefficients, thermal mass. Tier 2 must replace them with a real pack's data.
"""
from __future__ import annotations

import numpy as np

# cell open-circuit voltage vs state of charge (generic NMC shape)
_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00])
_OCV = np.array([3.00, 3.30, 3.45, 3.58, 3.65, 3.78, 3.93, 4.10, 4.20])


class Battery:
    def __init__(self, soc0=0.5, temp0_k=298.15, n_series=80, q_ah=60.0, r0_cell=1.0e-3, r1_cell=0.6e-3,
                 tau1=20.0, r_bus=5.0e-3, c_dc=2.0e-3, i_dis_max=250.0, i_chg_max=150.0,
                 r_inv=8.0e-3, p_inv_idle=30.0, c_th=4.0e5, r_th=0.05, t_amb=298.15, v_min=20.0):
        self.ns, self.q_as = n_series, q_ah * 3600.0
        self.R0_25, self.R1, self.tau1 = n_series * r0_cell, n_series * r1_cell, tau1
        self.Rc, self.C, self.i_dis, self.i_chg = r_bus, c_dc, i_dis_max, i_chg_max
        self.r_inv, self.p_idle, self.c_th, self.r_th, self.t_amb, self.v_min = r_inv, p_inv_idle, c_th, r_th, t_amb, v_min
        self.soc, self.T, self.v1, self.i_b = soc0, temp0_k, 0.0, 0.0
        self.v_dc = self.ocv(soc0)                      # start at rest: bus at open-circuit voltage
        self.n_nominal = float(self.v_dc)

    def ocv(self, soc):
        return self.ns * float(np.interp(soc, _SOC, _OCV))

    def r0(self, temp_k):
        # Arrhenius-style rise when cold: about x2.15 at 0 C, x0.59 at 45 C (the formula's value; an earlier comment said x1.9 and x0.7) (generic, flagged as assumption)
        return self.R0_25 * float(np.exp(2500.0 * (1.0 / temp_k - 1.0 / 298.15)))

    def step(self, dt, p_el, r0_mult=1.0, idis_scale=1.0):
        """Advance one step. p_el = motor electrical power (W, + = motoring). Returns a dict of channels."""
        # inverter loss: fixed gate-drive/idle loss plus conduction loss ~ R_inv * I_dc^2 (simple, an ASSUMPTION)
        p_loss_inv = self.p_idle + self.r_inv * (p_el / max(self.v_dc, self.v_min)) ** 2
        p_dc = p_el + p_loss_inv                         # power the inverter draws from the bus
        v_old = self.v_dc
        i_dc = p_dc / max(v_old, self.v_min)             # inverter DC current (+ = drawing from the bus)

        e_oc = self.ocv(self.soc)
        r0 = self.r0(self.T) * r0_mult
        rs = r0 + self.Rc
        self.v1 += (self.i_b * self.R1 - self.v1) * (1.0 - np.exp(-dt / self.tau1))   # exact RC update
        e_eff = e_oc - self.v1

        # implicit bus update: C dV/dt = i_b - i_dc, i_b = (e_eff - V)/Rs
        v_new = (v_old * self.C / dt + e_eff / rs - i_dc) / (self.C / dt + 1.0 / rs)
        i_b = (e_eff - v_new) / rs
        i_hi, i_lo = self.i_dis * idis_scale, -self.i_chg
        limited = 0.0
        if i_b > i_hi or i_b < i_lo:
            limited = 1.0
            i_b = float(np.clip(i_b, i_lo, i_hi))
            v_new = v_old + dt * (i_b - i_dc) / self.C   # BMS holds the current; the capacitor takes the rest
        v_new = max(v_new, self.v_min)

        # whatever voltage the limiter drops between pack and bus is dissipated there (keeps the energy books closed)
        p_bms = (e_eff - rs * i_b - v_new) * i_b
        self.soc = float(np.clip(self.soc - i_b * dt / self.q_as, 0.0, 1.0))
        p_heat = i_b ** 2 * (r0 + self.R1) + 0.0
        self.T += dt * (p_heat - (self.T - self.t_amb) / self.r_th) / self.c_th
        self.v_dc, self.i_b = v_new, i_b
        return {
            "V_dc": v_new, "i_dc": i_dc, "i_batt": i_b, "soc": self.soc, "V_oc": e_oc, "V_rc": self.v1,
            "V_term": v_new + self.Rc * i_b, "T_batt": self.T, "R0_eff": r0, "bms_limited": limited,
            "i_batt_max": i_hi, "P_inv_loss": p_loss_inv, "P_dc": p_dc, "P_bms_clip": p_bms,
        }
