"""Telemetry layer: turns a raw run log into the data product the analysts consume.

Why this file exists (the learning note):
    The plant logs what it MEASURES (states, delivered voltage). The notes' plant is
        x_dot = f(x) + g(x) u + d
    and the data team needs the pieces of that equation as separate columns, plus the energy books.
    Everything here is a pure function of logged columns + motor parameters, so it can be re-derived
    from any run (or from a Tier-2 car by swapping the parameter dict). Nothing is hidden in code.

Sign convention everywhere:  delta = DELIVERED - REQUESTED   (negative = the actuator gave less than asked).

State vector contract  x = [i_d, i_q, omega_m]   input u = [u_d, u_q] (delivered)   disturbance d = [d_id, d_iq, d_w]
    di_d/dt = ( -R i_d + w_el Lq i_q )/Ld                 + u_d/Ld            + (du_d)/Ld         <- f, g u, d (actuator)
    di_q/dt = ( -R i_q - w_el (Ld i_d + psi) )/Lq         + u_q/Lq            + (du_q)/Lq
    dw/dt   = ( 1.5 p (psi + (Ld-Lq) i_d) i_q - T_fric )/J                     - T_dist/J      <- d (external road load)
  where u in the g(x)u term is what the controller REQUESTED (u_cmd) and du = u_app - u_cmd is the part the
  saturation took away. So "d" is literally the notes' d(u,x): a disturbance that depends on the input.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

# (name, unit, group, role, description)
CATALOG = [
    # ---- time / reference
    ("t", "s", "time", "index", "simulation time"),
    ("omega_ref", "rad/s", "reference", "input", "driver/cycle speed request"),
    ("iq_ref", "A", "reference", "input", "speed loop -> q-current request (after current clip)"),
    # ---- natural process: measured plant state
    ("omega_m", "rad/s", "state", "state", "mechanical speed"),
    ("omega_el", "rad/s", "state", "state", "electrical speed = p * omega_m"),
    ("epsilon_el", "rad", "state", "state", "electrical rotor angle"),
    ("i_d", "A", "state", "state", "d-axis current"),
    ("i_q", "A", "state", "state", "q-axis current"),
    ("i_a", "A", "phase", "state", "phase a current"),
    ("i_b", "A", "phase", "state", "phase b current"),
    ("i_c", "A", "phase", "state", "phase c current"),
    ("u_a", "V", "phase", "state", "phase a voltage (plant, measured)"),
    ("u_b", "V", "phase", "state", "phase b voltage"),
    ("u_c", "V", "phase", "state", "phase c voltage"),
    ("T_elec", "N.m", "state", "state", "electromagnetic torque (plant-reported)"),
    ("V_dc", "V", "supply", "state", "DC-link (bus) voltage, true value; sets the inverter ceiling V_dc/sqrt(3)"),
    ("V_oc", "V", "battery", "state", "pack open-circuit voltage from SOC"),
    ("V_rc", "V", "battery", "state", "pack RC-branch (diffusion) voltage"),
    ("V_term", "V", "battery", "state", "pack terminal voltage"),
    ("soc", "-", "battery", "state", "state of charge"),
    ("T_batt", "K", "battery", "state", "pack temperature"),
    ("R0_eff", "ohm", "battery", "state", "pack ohmic resistance now (temperature and fault dependent)"),
    ("i_batt", "A", "battery", "state", "pack current (+ = discharging); this is what the battery DELIVERS"),
    ("i_batt_max", "A", "battery", "input", "BMS discharge-current limit in force (drops in a derate)"),
    ("bms_limited", "0/1", "battery", "mismatch", "flag: BMS current limit is holding the pack back"),
    ("i_dc", "A", "supply", "input", "inverter DC current; this is what the motor side REQUESTS from the bus"),
    ("P_el", "W", "energy", "state", "electrical power into the motor (from delivered voltage and current)"),
    ("P_inv_loss", "W", "energy", "derived", "inverter loss (idle + conduction, assumed)"),
    ("P_dc", "W", "energy", "derived", "power drawn from the DC link = P_el + inverter loss"),
    ("P_bms_clip", "W", "energy", "derived", "power burned in the BMS current limiter"),
    ("bus_sag", "V", "mismatch", "mismatch", "M9: V_dc minus open-circuit voltage: total voltage the pack path loses"),
    ("ceiling_loss", "V", "mismatch", "mismatch", "M9b: inverter ceiling now minus nominal ceiling (173.2 V); how much headroom the bus took away"),
    ("d_i_dc", "A", "mismatch", "mismatch", "M10: pack current delivered minus inverter current requested (negative = bus starved, capacitor drains)"),
    ("E_bus_unserved", "J", "energy", "mismatch", "M10b: cumulative energy the inverter wanted that the pack did not supply"),
    ("P_chem", "W", "energy", "derived", "chemical power V_oc * i_batt"),
    ("P_batt_loss", "W", "energy", "derived", "pack + busbar loss: RC branch, R0 and Rc heating"),
    ("P_bus_store", "W", "energy", "derived", "power into (+) or out of (-) the DC-link capacitor C V dV/dt"),
    ("resid_chain", "W", "validation", "validation", "chem - losses - bus - BMS clip: pack energy books, should be ~0"),
    ("resid_bus", "W", "validation", "validation", "V_dc*(i_batt - i_dc) - C V dV/dt: DC-link current balance, should be ~0"),
    ("lam_d", "Wb", "natural", "derived", "d flux linkage Ld*i_d + psi"),
    ("lam_q", "Wb", "natural", "derived", "q flux linkage Lq*i_q"),
    ("bemf", "V", "natural", "derived", "back-EMF magnitude w_el*psi: the voltage the motor 'spends' just spinning"),
    ("v_demand_mag", "V", "natural", "derived", "|u_cmd|: voltage the controller wants"),
    ("v_headroom", "V", "natural", "derived", "u_limit - |u_cmd|: negative = demand exceeds supply"),
    # ---- control inputs
    ("u_d_cmd", "V", "input", "input", "REQUESTED d voltage"),
    ("u_q_cmd", "V", "input", "input", "REQUESTED q voltage"),
    ("u_d_app", "V", "input", "input", "DELIVERED d voltage (after our saturation)"),
    ("u_q_app", "V", "input", "input", "DELIVERED q voltage"),
    ("duty_peak", "-", "input", "input", "largest inverter leg duty (|1| = leg railed)"),
    ("T_cmd", "N.m", "input", "input", "torque implied by the current request"),
    # ---- f, g, d split of the model
    ("f_id", "A/s", "model", "derived", "f(x) for i_d"),
    ("f_iq", "A/s", "model", "derived", "f(x) for i_q"),
    ("f_w", "rad/s^2", "model", "derived", "f(x) for omega (torque minus friction)"),
    ("ts_f1", "rad/s^2", "model", "derived", "two-state drift f1(x) for omega: -T_fric/J (friction + drag, no i_d, no road load)"),
    ("ts_g1", "rad/s^2/A", "model", "derived", "two-state input gain g1 = 1.5 p psi / J (torque constant over inertia; constant)"),
    ("ts_d1", "rad/s^2", "disturbance", "disturbance", "two-state lumped disturbance on omega: (-T_dist + 1.5 p (Ld-Lq) i_d i_q)/J = road load + reluctance term"),
    ("ts_f2", "A/s", "model", "derived", "two-state drift f2(x) for i_q with i_d = 0: (-R i_q - p omega_m psi)/Lq"),
    ("ts_g2", "A/s/V", "model", "derived", "two-state input gain g2 = 1/Lq (constant)"),
    ("ts_d2", "A/s", "disturbance", "disturbance", "two-state lumped disturbance on i_q: (du_q - omega_el Ld i_d)/Lq = actuator mismatch + d-axis cross-coupling"),
    ("ts_d1_red", "rad/s^2", "disturbance", "disturbance", "part of ts_d1 that exists only because i_d is not 0 (reluctance torque / J): the size of the two-state reduction"),
    ("ts_d2_red", "A/s", "disturbance", "disturbance", "part of ts_d2 that exists only because i_d is not 0 (-omega_el Ld i_d / Lq)"),
    ("ts_z1", "rad/s", "controller", "derived", "first tracking error z1 = omega_m - omega_ref (velocity)"),
    ("ts_z2", "A", "controller", "derived", "second tracking error z2 = i_q - iq_ref, current minus its virtual control, using the TRUE current (the controller's e_q uses the MEASURED current: ts_z2 = e_q - gap_i_q in firewalled runs)"),
    ("ts_res_w", "rad/s^2", "validation", "validation", "measured d omega/dt minus (f1 + g1 i_q + d1); must equal resid_w (valid at log step <= 0.4 ms)"),
    ("ts_res_iq", "A/s", "validation", "validation", "forward-difference d i_q/dt minus (f2 + g2 u_q_delivered + d2 - du_q/Lq) with u_q_delivered the same-row voltage; equals resid_iq"),
    ("g_id", "A/s/V", "model", "derived", "g for i_d = 1/Ld"),
    ("g_iq", "A/s/V", "model", "derived", "g for i_q = 1/Lq"),
    ("d_id", "A/s", "disturbance", "disturbance", "actuator-mismatch disturbance on i_d = du_d/Ld"),
    ("d_iq", "A/s", "disturbance", "disturbance", "actuator-mismatch disturbance on i_q = du_q/Lq"),
    ("d_w", "rad/s^2", "disturbance", "disturbance", "external road-load disturbance = -T_dist/J"),
    ("d_w_fric", "rad/s^2", "disturbance", "disturbance", "friction/drag acceleration = -T_fric/J (inside f, shown for analysis)"),
    ("T_fric", "N.m", "disturbance", "disturbance", "friction + drag torque"),
    ("T_dist", "N.m", "disturbance", "disturbance", "injected road/grade torque (positive opposes motion)"),
    ("resid_id", "A/s", "validation", "validation", "measured di_d/dt minus model(f+g u_app): should be small"),
    ("resid_iq", "A/s", "validation", "validation", "measured di_q/dt minus model"),
    ("resid_w", "rad/s^2", "validation", "validation", "measured dw/dt minus model"),
    # ---- mismatch ledger (requested vs delivered)
    ("d_u_d", "V", "mismatch", "mismatch", "M2: delivered - requested, d axis"),
    ("d_u_q", "V", "mismatch", "mismatch", "M2: delivered - requested, q axis"),
    ("d_u_norm", "-", "mismatch", "mismatch", "M2 normalised: |du| / limit"),
    ("u_limit", "V", "mismatch", "mismatch", "available voltage ceiling at this instant"),
    ("saturated", "0/1", "mismatch", "mismatch", "flag: actuator saturated this step"),
    ("d_u_plant_d", "V", "mismatch", "mismatch", "M3: plant-reported minus ours (hidden clipping), expect 0"),
    ("d_u_plant_q", "V", "mismatch", "mismatch", "M3 q axis"),
    ("i_ref_clip", "A", "mismatch", "mismatch", "M4: delivered - requested current reference"),
    ("d_T", "N.m", "mismatch", "mismatch", "M5: actual torque - torque the current request implied"),
    ("e_speed", "rad/s", "mismatch", "mismatch", "M1: speed request - actual speed (what the driver did not get)"),
    ("P_req_elec", "W", "energy", "mismatch", "electrical power requested: 1.5 (u_d_cmd i_d + u_q_cmd i_q)"),
    ("P_del_elec", "W", "energy", "mismatch", "electrical power delivered: 1.5 (u_d_app i_d + u_q_app i_q)"),
    ("dP_elec", "W", "energy", "mismatch", "M6: delivered - requested electrical power"),
    ("P_mech_req", "W", "energy", "mismatch", "shaft power the current request implied: T_cmd * omega_m"),
    ("dP_mech", "W", "energy", "mismatch", "M6b: delivered - requested shaft power (the physically clean mismatch)"),
    ("E_mech_unserved", "J", "energy", "mismatch", "M8b: cumulative shaft energy requested but not delivered (preferred, always interpretable)"),
    ("P_cu", "W", "energy", "derived", "copper loss 1.5 R (i_d^2 + i_q^2)"),
    ("P_mag", "W", "energy", "derived", "rate of change of magnetic energy"),
    ("P_mech", "W", "energy", "derived", "shaft power T_elec * omega_m"),
    ("P_load", "W", "energy", "derived", "power into friction + road disturbance"),
    ("P_kin", "W", "energy", "derived", "rate of change of kinetic energy J w dw/dt"),
    ("P_balance_resid", "W", "validation", "validation", "P_del_elec - P_cu - P_mag - P_mech: should be ~0"),
    ("E_req", "J", "energy", "mismatch", "cumulative requested electrical energy"),
    ("E_del", "J", "energy", "mismatch", "cumulative delivered electrical energy"),
    ("E_unserved", "J", "energy", "mismatch", "M8: cumulative requested minus delivered (the missing energy)"),
    ("E_cu", "J", "energy", "derived", "cumulative copper loss"),
    ("dist_i_limit", "-", "margin", "margin", "distance to current limit (1 = far, 0 = at limit)"),
    # ---- electrical request, rebuilt so the Theory team gets electrical (not shaft) quantities. See docs/ELECTRICAL_MISMATCH.md
    ("P_req_ref", "W", "energy", "mismatch", "reference-implied ELECTRICAL power: 1.5 R iq*^2 + T_cmd w_m. Real power; depends only on the request and speed, never on delivered currents"),
    ("P_cu_ref", "W", "energy", "derived", "copper loss at the requested current: 1.5 R (id*^2 + iq*^2)"),
    ("dP_cu", "W", "energy", "mismatch", "copper-loss mismatch: P_cu minus P_cu_ref"),
    ("dP_elec_ref", "W", "energy", "mismatch", "M6c: delivered electrical power minus reference-implied electrical power = dP_mech + dP_cu + P_mag (exact)"),
    ("E_elec_ref_unserved", "J", "energy", "mismatch", "M8c: cumulative electrical energy the reference implied and the motor did not receive"),
    ("E_mag_store", "J", "energy", "derived", "magnetic energy stored in the windings since t=0 (integral of P_mag; bounded, so it does not drift)"),
    ("resid_elec_decomp", "W", "validation", "validation", "dP_elec_ref - (dP_mech + dP_cu + P_mag): decomposition check, should be ~0"),
    ("P_that", "W", "energy", "derived", "shaft power the voltage command assumes: estimated torque (uses psi_hat) times w_m"),
    ("P_psi_err", "W", "energy", "derived", "power mismatch caused by the flux-estimate error: -1.5 w_el psi_tilde i_q"),
    ("P_fb", "W", "energy", "derived", "feedback-correction power inside the voltage command: P_req_elec - P_cu - P_that"),
    ("resid_req_decomp", "W", "validation", "validation", "dP_elec - (P_psi_err + P_mag - P_fb): decomposition of the voltage-command mismatch, should be ~0"),
    ("resid_du_power", "W", "validation", "validation", "dP_elec - 1.5 (du_d i_d + du_q i_q): actuator-mismatch identity, should be ~0"),
    ("req_sign_conflict", "0/1", "energy", "mismatch", "flag: voltage-command power and reference-implied power disagree in sign (counterfactual warning)"),
    # ---- the disturbance vector, split by physical source (sums to T_fric + T_dist)
    ("T_roll", "N.m", "disturbance", "disturbance", "rolling-resistance torque on the shaft (constant term of the load, follows the sign of speed)"),
    ("T_visc", "N.m", "disturbance", "disturbance", "viscous / drivetrain friction torque b * w_m"),
    ("T_aero", "N.m", "disturbance", "disturbance", "aerodynamic drag torque c * w_m * |w_m|"),
    # ---- what the DATA asks, before any controller (cycle runs only)
    ("T_req_data", "N.m", "reference", "input", "shaft torque the speed request needs, from the cycle and vehicle equations (no controller involved)"),
    ("T_avail", "N.m", "reference", "input", "most torque the motor can hold at this speed with the nominal ceiling and the current limit"),
    ("data_ratio", "-", "reference", "mismatch", "|T_req_data| / T_avail. Above 1 the request is infeasible for ANY controller"),
    ("data_infeasible", "0/1", "reference", "mismatch", "flag: data_ratio above 1. Mismatch at these instants is attributable to the data, not the controller"),
    # ---- controller internals (for the Theory team)
    ("psi_hat", "Wb", "controller", "controller", "adaptive flux estimate"),
    ("psi_tilde", "Wb", "controller", "controller", "estimation error"),
    ("e_q", "A", "controller", "controller", "q current tracking error"),
    ("e_qa", "A", "controller", "controller", "anti-windup compensated error"),
    ("z", "A", "controller", "controller", "anti-windup auxiliary state"),
    ("i_lim_eff", "A", "controller", "controller", "current limit in force (governor)"),
    ("V", "-", "proof", "controller", "candidate Lyapunov function V_e + V_theta + V_aw"),
]

# ---- realism layer (information firewall) columns. role tells you WHO knows the value:
#   truth = plant truth, never given to the controller; measured = what the controller received; error = measured - truth;
#   belief = the controller's (wrong) model constants; internal = controller-side signal; ledger = voltage mismatch piece (delivered - requested)
CATALOG += [
    ("meas_i_a", "A", "measured", "measured", "phase-a current as measured: gain/offset/noise/12-bit quantization/clip"),
    ("meas_i_b", "A", "measured", "measured", "phase-b current as measured"),
    ("meas_i_c", "A", "measured", "measured", "phase-c current as measured"),
    ("meas_i_d", "A", "measured", "measured", "d current the controller sees (Park of measured phases at the measured angle)"),
    ("meas_i_q", "A", "measured", "measured", "q current the controller sees"),
    ("meas_theta", "rad", "measured", "measured", "resolver electrical angle as measured (offset, noise, ripple)"),
    ("meas_omega", "rad/s", "measured", "measured", "mechanical speed as estimated by the PLL from the measured angle"),
    ("meas_vdc", "V", "measured", "measured", "DC-bus voltage as measured (divider gain error, noise, first-order low-pass)"),
    ("gap_i_a", "A", "measured", "error", "measured minus true phase-a current"),
    ("gap_i_b", "A", "measured", "error", "measured minus true phase-b current"),
    ("gap_i_c", "A", "measured", "error", "measured minus true phase-c current"),
    ("gap_i_d", "A", "measured", "error", "measured minus true d current"),
    ("gap_i_q", "A", "measured", "error", "measured minus true q current"),
    ("gap_theta", "rad", "measured", "error", "measured minus true electrical angle"),
    ("gap_omega", "rad/s", "measured", "error", "estimated minus true mechanical speed"),
    ("gap_vdc", "V", "measured", "error", "measured minus true DC-bus voltage"),
    ("bms_i_dis_lim", "A", "measured", "measured", "BMS discharge-current limit as sent (latency, 1 A quantization, multiplicative error)"),
    ("bms_i_chg_lim", "A", "measured", "measured", "BMS charge-current limit as sent"),
    ("bms_v_pack", "V", "measured", "measured", "pack voltage reported by the BMS (delayed)"),
    ("bms_age_s", "s", "measured", "measured", "age of the BMS message when used"),
    ("belief_R", "ohm", "model", "belief", "stator resistance the controller believes (constant per run, drawn from the belief-error range)"),
    ("belief_Ld", "H", "model", "belief", "d inductance the controller believes"),
    ("belief_Lq", "H", "model", "belief", "q inductance the controller believes"),
    ("belief_J", "kg m2", "model", "belief", "inertia the controller believes"),
    ("belief_psi_name", "Wb", "model", "belief", "nameplate flux the controller uses for Kt and the governor (not the adaptive estimate)"),
    ("ceiling_hat", "V", "controller", "internal", "voltage ceiling the controller believes, min(vdc_m/sqrt(3), 173.2) (or the labelled oracle)"),
    ("du_q_known", "V", "controller", "internal", "q-voltage change from the controller's own clip, the saturation it knows about"),
    ("theta_adv", "rad", "controller", "internal", "angle advance applied to the command to offset the computation delay"),
    ("u_d_ctrl", "V", "controller", "internal", "d command before the controller's own clip (controller frame)"),
    ("u_q_ctrl", "V", "controller", "internal", "q command before the controller's own clip"),
    ("u_d_clip", "V", "controller", "internal", "d command after the controller's own clip"),
    ("u_q_clip", "V", "controller", "internal", "q command after the controller's own clip"),
    ("u_d_plant", "V", "supply", "state", "d voltage reported by the plant model"),
    ("u_q_plant", "V", "supply", "state", "q voltage reported by the plant model"),
    ("u_d_extra", "V", "mismatch", "ledger", "inverter dead-time/device-drop error added after the plant clip, d axis"),
    ("u_q_extra", "V", "mismatch", "ledger", "inverter dead-time/device-drop error, q axis"),
    ("d_u_ctrl_clip_d", "V", "mismatch", "ledger", "ledger piece 1: controller's own clip (known saturation), d"),
    ("d_u_ctrl_clip_q", "V", "mismatch", "ledger", "ledger piece 1: controller's own clip, q"),
    ("d_u_adv_d", "V", "mismatch", "ledger", "ledger piece 2: angle advance rotation, d"),
    ("d_u_adv_q", "V", "mismatch", "ledger", "ledger piece 2: angle advance rotation, q"),
    ("d_u_delay_d", "V", "mismatch", "ledger", "ledger piece 3: computation delay (delayed command minus current command), d"),
    ("d_u_delay_q", "V", "mismatch", "ledger", "ledger piece 3: computation delay, q"),
    ("d_u_rot_d", "V", "mismatch", "ledger", "ledger piece 4: rotation by the angle error, d"),
    ("d_u_rot_q", "V", "mismatch", "ledger", "ledger piece 4: rotation by the angle error, q"),
    ("d_u_plant_clip_d", "V", "mismatch", "ledger", "ledger piece 5: plant clip against the TRUE ceiling (hidden saturation), d"),
    ("d_u_plant_clip_q", "V", "mismatch", "ledger", "ledger piece 5: plant clip against the true ceiling, q"),
    ("d_u_inv_d", "V", "mismatch", "ledger", "ledger piece 6: inverter dead-time/device-drop error, d"),
    ("d_u_inv_q", "V", "mismatch", "ledger", "ledger piece 6: inverter error, q"),
    ("sat_ctrl", "0/1", "mismatch", "mismatch", "flag: the controller's own command exceeded the ceiling it believes"),
    ("sat_plant", "0/1", "mismatch", "mismatch", "flag: the plant clipped the command against the true ceiling"),
    ("V_e", "J", "proof", "proof", "Lyapunov term 0.5 Lq e_a^2 (current-error energy)"),
    ("V_theta", "J", "proof", "proof", "Lyapunov term psi_tilde^2 / (2 Gamma) (parameter-error energy; uses TRUE flux, truth column)"),
    ("V_aw", "J", "proof", "proof", "Lyapunov term 0.5 Lq z^2 (anti-windup auxiliary state)"),
    ("v_ref_mps", "m/s", "reference", "input", "cycle speed request in vehicle units (cycle runs only)"),
]


def params_from_plant(pl) -> dict:
    P = dict(p=pl.p, Ld=pl.Ld, Lq=pl.Lq, R=pl.Rs, psi=pl.psi_p, J=pl.J, tau=pl.tau, id_ref=0.0,
             a_load=float(pl.load._a), b_load=float(pl.load._b), c_load=float(pl.load._c))
    if pl.battery is not None:
        P.update(C_dc=pl.battery.C, R_bus=pl.battery.Rc, V_nom_ceiling=pl.u_max_hw)
    return P


def derive(df: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Add all derived channels. P = motor/load parameters (swap this dict for a Tier-2 vehicle)."""
    d = df.copy()
    # The time step in P must be the spacing of the LOGGED rows, because every derivative below is a difference of rows.
    # Cycle runs integrate at 0.2 ms but log every 5 ms; using the wrong step silently scales P_mag and the residuals
    # (found by the pipeline red-team). Trust the time axis, and say so when P disagreed.
    if len(d) > 2:
        tau_t = float(np.median(np.diff(d["t"].to_numpy())))
        if abs(tau_t - P["tau"]) > 0.01 * tau_t:
            warnings.warn(f"derive(): P['tau']={P['tau']:.3g} s disagrees with the logged row spacing {tau_t:.3g} s; using the row spacing", stacklevel=2)
            P = dict(P, tau=tau_t)
    Ld, Lq, R, psi, J, p, tau = P["Ld"], P["Lq"], P["R"], P["psi"], P["J"], P["p"], P["tau"]
    d.attrs["tau_used"] = tau
    # Rows spaced more than ~4x the 0.1 ms integration step cannot resolve the current dynamics, so the DERIVATIVE channels
    # (P_mag, resid_*, P_balance_resid, resid_elec_decomp, resid_req_decomp, E_mag_store) are approximate there. The pointwise
    # mismatch channels (M1 to M6d, M9, M10) stay exact at any logging rate.
    d.attrs["derivative_channels_valid"] = bool(tau <= 4e-4)
    we, i_d, i_q = d["omega_el"], d["i_d"], d["i_q"]

    d["lam_d"], d["lam_q"], d["bemf"] = Ld * i_d + psi, Lq * i_q, we * psi
    d["v_demand_mag"] = np.hypot(d["u_d_cmd"], d["u_q_cmd"])
    d["v_headroom"] = d["u_limit"] - d["v_demand_mag"]

    d["f_id"] = (-R * i_d + we * Lq * i_q) / Ld
    d["f_iq"] = (-R * i_q - we * (Ld * i_d + psi)) / Lq
    d["f_w"] = (d["T_elec"] - d["T_fric"]) / J
    d["g_id"], d["g_iq"] = 1.0 / Ld, 1.0 / Lq
    d["d_id"], d["d_iq"] = d["d_u_d"] / Ld, d["d_u_q"] / Lq
    d["d_w"] = -d["T_dist"] / J
    d["d_w_fric"] = -d["T_fric"] / J

    # Validation: does f + g u + d reproduce the measured derivative?
    # Timing (found 2026-11-02): row k holds the state at the START of a step and the voltage delivered DURING that step, so the
    # exact discrete check is a FORWARD difference (x[k+1]-x[k])/tau against f(x[k]) + g u_app[k]. The earlier version paired a
    # central difference with the previous row's voltage and inflated the i_q residual about 18x (1195 vs 67 A/s on a full-rate run).
    def _fwd(col):
        x = d[col].to_numpy(dtype=float)
        return np.r_[(x[1:] - x[:-1]) / tau, 0.0]          # last row has no next sample: set to 0 (excluded from stats by the audit)
    d["resid_id"] = _fwd("i_d") - (d["f_id"] + d["u_d_app"] / Ld)
    d["resid_iq"] = _fwd("i_q") - (d["f_iq"] + d["u_q_app"] / Lq)
    d["resid_w"] = _fwd("omega_m") - (d["f_w"] + d["d_w"])

    d["e_speed"] = d["omega_ref"] - d["omega_m"]

    # ---- Two-state reduction x2 = [omega_m, i_q]  (exact regrouping of the three-state model; i_d terms become disturbance)
    #   d omega/dt = f1 + g1 i_q + d1        f1 = -T_fric/J   g1 = 1.5 p psi/J   d1 = (-T_dist + 1.5 p (Ld-Lq) i_d i_q)/J
    #   d i_q/dt   = f2 + g2 u_q + d2        f2 = (-R i_q - p omega_m psi)/Lq   g2 = 1/Lq   d2 = (du_q - omega_el Ld i_d)/Lq
    d["ts_f1"] = -d["T_fric"] / J
    d["ts_g1"] = 1.5 * p * psi / J
    d["ts_d1_red"] = 1.5 * p * (Ld - Lq) * i_d * i_q / J
    d["ts_d1"] = d["d_w"] + d["ts_d1_red"]
    d["ts_f2"] = (-R * i_q - we * psi) / Lq
    d["ts_g2"] = 1.0 / Lq
    d["ts_d2_red"] = -we * Ld * i_d / Lq
    d["ts_d2"] = d["d_u_q"] / Lq + d["ts_d2_red"]
    d["ts_z1"] = d["omega_m"] - d["omega_ref"]
    d["ts_z2"] = i_q - d["iq_ref"] if "iq_ref" in d else np.nan
    d["ts_res_w"] = _fwd("omega_m") - (d["ts_f1"] + d["ts_g1"] * i_q + d["ts_d1"])
    d["ts_res_iq"] = _fwd("i_q") - (d["ts_f2"] + d["ts_g2"] * d["u_q_app"] + d["ts_d2_red"])
    d["P_req_elec"] = 1.5 * (d["u_d_cmd"] * i_d + d["u_q_cmd"] * i_q)
    d["P_del_elec"] = 1.5 * (d["u_d_app"] * i_d + d["u_q_app"] * i_q)
    d["dP_elec"] = d["P_del_elec"] - d["P_req_elec"]
    d["P_cu"] = 1.5 * R * (i_d ** 2 + i_q ** 2)
    di_d, di_q = np.gradient(i_d.to_numpy(), tau), np.gradient(i_q.to_numpy(), tau)
    # magnetic power = 1.5 (lam_d di_d/dt + lam_q di_q/dt) minus the part that is motional EMF (goes to shaft)
    d["P_mag"] = 1.5 * (Ld * i_d * di_d + Lq * i_q * di_q)
    d["P_mech"] = d["T_elec"] * d["omega_m"]
    d["P_load"] = (d["T_fric"] + d["T_dist"]) * d["omega_m"]
    dw = np.gradient(d["omega_m"].to_numpy(), tau)
    d["P_kin"] = J * d["omega_m"] * dw
    d["P_balance_resid"] = d["P_del_elec"] - d["P_cu"] - d["P_mag"] - d["P_mech"]
    for src, dst in (("P_req_elec", "E_req"), ("P_del_elec", "E_del"), ("P_cu", "E_cu")):
        d[dst] = np.cumsum(d[src].to_numpy()) * tau
    d["E_unserved"] = d["E_req"] - d["E_del"]
    # CAVEAT (found by checking, see docs): u_cmd . i mixes real power with the reactive cross-coupling voltages
    # (-w Lq i_q in u_d etc.), and the current responds to what was DELIVERED, so 'requested electrical power' is a
    # counterfactual and can even be negative. The shaft-power version below is unambiguous: T_cmd*w vs T_elec*w.
    d["P_mech_req"] = d["T_cmd"] * d["omega_m"]
    d["dP_mech"] = d["P_mech"] - d["P_mech_req"]
    d["E_mech_unserved"] = -np.cumsum(d["dP_mech"].to_numpy()) * tau

    # ---- ELECTRICAL request rebuilt from the controller's own request (see docs/ELECTRICAL_MISMATCH.md).
    # Exact chain:  P_del = P_cu + P_mag + P_mech   (physics, checked by P_balance_resid)
    #               P_req_ref = P_cu(i*) + T_cmd w_m (the electrical power a steady motor needs to deliver the requested torque)
    # so            dP_elec_ref = P_del - P_req_ref = dP_mech + dP_cu + P_mag     <- shaft mismatch + copper-loss mismatch + magnetic rate
    iq_ref = d["iq_ref"] if "iq_ref" in d else np.nan          # legacy logs without the request: reference-based columns become NaN
    d["P_cu_ref"] = 1.5 * R * (P.get("id_ref", 0.0) ** 2 + iq_ref ** 2)
    d["P_req_ref"] = d["P_cu_ref"] + d["P_mech_req"]
    d["dP_cu"] = d["P_cu"] - d["P_cu_ref"]
    d["dP_elec_ref"] = d["P_del_elec"] - d["P_req_ref"]
    d["resid_elec_decomp"] = d["dP_elec_ref"] - (d["dP_mech"] + d["dP_cu"] + d["P_mag"])
    d["E_elec_ref_unserved"] = -np.cumsum(d["dP_elec_ref"].to_numpy()) * tau
    d["E_mag_store"] = np.cumsum(d["P_mag"].to_numpy()) * tau
    # The voltage-command power splits into loss + assumed shaft power + feedback correction, which exposes how the
    # parameter error enters:  dP_elec = -1.5 w_el psi_tilde i_q  +  P_mag  -  P_fb
    d["P_that"] = 1.5 * we * ((Ld - Lq) * i_d + d["psi_hat"]) * i_q
    d["P_psi_err"] = -1.5 * we * d["psi_tilde"] * i_q
    d["P_fb"] = d["P_req_elec"] - d["P_cu"] - d["P_that"]
    d["resid_req_decomp"] = d["dP_elec"] - (d["P_psi_err"] + d["P_mag"] - d["P_fb"])
    d["resid_du_power"] = d["dP_elec"] - 1.5 * (d["d_u_d"] * i_d + d["d_u_q"] * i_q)
    d["req_sign_conflict"] = ((d["P_req_elec"] * d["P_req_ref"] < 0) & (d["P_req_elec"].abs() > 10) & (d["P_req_ref"].abs() > 10)).astype(float)
    # disturbance vector by physical source
    if "a_load" in P:
        d["T_aero"] = P["c_load"] * d["omega_m"] * d["omega_m"].abs()
        d["T_visc"] = P["b_load"] * d["omega_m"]
        d["T_roll"] = d["T_fric"] - d["T_visc"] - d["T_aero"]
    if "data_ratio" in d:
        d["data_infeasible"] = (d["data_ratio"] > 1.0).astype(float)
    if "i_batt" in d:                                   # battery runs only
        ib, vdc = d["i_batt"], d["V_dc"]
        d["bus_sag"] = vdc - d["V_oc"]
        d["ceiling_loss"] = np.minimum(vdc / np.sqrt(3), P["V_nom_ceiling"]) - P["V_nom_ceiling"]     # bus-caused loss of inverter headroom
        d["d_i_dc"] = ib - d["i_dc"]
        d["E_bus_unserved"] = np.cumsum((vdc * (d["i_dc"] - ib)).to_numpy()) * tau
        d["P_chem"] = d["V_oc"] * ib
        d["P_batt_loss"] = d["V_rc"] * ib + (d["R0_eff"] + P["R_bus"]) * ib ** 2
        d["P_bus_store"] = P["C_dc"] * vdc * vdc.diff().fillna(0.0) / tau        # backward difference = how the solver steps
        d["resid_chain"] = d["P_chem"] - d["P_batt_loss"] - vdc * ib - d["P_bms_clip"]
        d["resid_bus"] = vdc * (ib - d["i_dc"]) - d["P_bus_store"]
    return d


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(CATALOG, columns=["name", "unit", "group", "role", "description"])
