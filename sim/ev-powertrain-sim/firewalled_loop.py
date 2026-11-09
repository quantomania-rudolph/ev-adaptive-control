"""The firewalled experiment loop: truth -> sensors -> controller -> inverter -> plant, with truth and measurements logged apart.

    python firewalled_loop.py --name F1 --mode aw --governor --battery --event --seed 3

Same scenario as run_exp.py (speed ramp, 10 N.m grade, unmodeled event) so the legacy and firewalled runs are comparable.
Column groups in the output parquet:
  truth        the legacy columns (omega_m, i_d, i_q, u_*_plant, V_dc, ...). Unchanged names and meaning.
  meas_*       exactly what the controller was handed (Measurements). bms_* the BMS message fields.
  gap_*        measured minus truth, per channel.
  est_/ctl     controller internals (psi_hat, z, iq_ref, ...). psi_tilde is computed HERE from truth, never in the controller.
  d_u_*        ledger. d_u_d/q = delivered(true frame) - controller's pre-clip command, split EXACTLY into
               d_u_ctrl_clip (controller's own clip = du_known), d_u_adv, d_u_delay, d_u_rot, d_u_plant_clip (hidden saturation), d_u_inv.
  belief_*     what the controller believed (constant per run).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import run_exp as R                       # scenario functions are looked up as R.omega_ref(...) so tests can monkeypatch them
from battery import Battery
from control_fw import Beliefs, FirewalledController
from interfaces import (d_loop_spectral_radius, BATTERY_TRUTH_KEYS, BMSConfig, BMSMsg, BeliefConfig, ControllerConfig, InverterConfig, SensorConfig,
                        TAU_PLAIN, TRUTH_KEYS, V_SCALE_GEM, rng_streams, rot, validate_fit_dict)
from plant import GEMPlant


def draw_beliefs(plant: GEMPlant, cfg: BeliefConfig, rng: np.random.Generator):
    """Absolute controller beliefs from relative draws against the truth. Fixed draw order: R, L, J, psi_name (always drawn)."""
    dR, dL, dJ, dpsi = (rng.uniform(*cfg.dR_range), rng.uniform(*cfg.dL_range), rng.uniform(*cfg.dJ_range), rng.uniform(*cfg.dpsi_range))
    if cfg.exact:
        dR = dL = dJ = dpsi = 0.0
    b = Beliefs(R=plant.Rs * (1 + dR), Ld=plant.Ld * (1 + dL), Lq=plant.Lq * (1 + dL), psi_name=plant.psi_p * (1 + dpsi),
                J=plant.J * (1 + dJ), psi_hat0=cfg.psi_start_frac * plant.psi_p, psi_range=cfg.psi_range)
    return b, dict(dR=dR, dL=dL, dJ=dJ, dpsi=dpsi)


def _batt_truth(plant: GEMPlant, led: dict | None) -> dict:
    if led and "V_dc" in led and plant.battery is not None:
        return {k: led[k] for k in BATTERY_TRUTH_KEYS}
    b = plant.battery
    return dict(V_dc=b.v_dc, i_batt=0.0, soc=b.soc, T_batt=b.T, V_term=b.v_dc, i_batt_max=b.i_dis, bms_limited=0.0, i_dc=0.0)


def run_fw(name, mode="naive", event=False, governor=False, battery=False, vdc_aware=False, bms_aware=False, oracle_ceiling=False,
           seed=0, identity=False, sensor_cfg=None, inv_cfg=None, bms_cfg=None, belief_cfg=None, ctl_overrides=None,
           t_end=None, fit_path=None, outdir="runs_fw", write=True, tau=TAU_PLAIN,
           plant=None, omega_ref_fn=None, log_every=1, extra_cols=None, check_d_loop=True):
    from sensors import SensorSuite            # imported here: agents deliver these modules; the controller never imports them
    from inverter import InverterModel

    t_end = R.T_END if t_end is None else t_end
    streams = rng_streams(seed)
    if identity:
        sensor_cfg, inv_cfg = SensorConfig.identity_mode(tau), InverterConfig.identity_mode(tau)
        bms_cfg, belief_cfg = BMSConfig(enabled=False), BeliefConfig(exact=True)
        ctl_overrides = {**(ctl_overrides or {}), "angle_advance": False, "delay_steps": 0}
    sensor_cfg = sensor_cfg or SensorConfig(tau=tau)
    inv_cfg = inv_cfg or InverterConfig(tau=tau)
    bms_cfg, belief_cfg = bms_cfg or BMSConfig(), belief_cfg or BeliefConfig()
    ctl_kw = dict(mode=mode, governor=governor, vdc_aware=vdc_aware, bms_aware=bms_aware, tau=tau, delay_steps=inv_cfg.delay_steps)
    ctl_kw.update(ctl_overrides or {})            # overrides win (the controller's belief about its own delay must match the inverter's)
    ctl_cfg = ControllerConfig(**ctl_kw)
    d_steps = inv_cfg.delay_steps

    fit = None
    batt = None
    if battery and plant is not None and fit_path:
        fit = json.loads(pathlib.Path(fit_path).read_text())
        assert not validate_fit_dict(fit)
    if battery and plant is None:
        if fit_path:
            fit = json.loads(pathlib.Path(fit_path).read_text())
            bad = validate_fit_dict(fit)
            assert not bad, bad
            batt = Battery(**fit["toy_battery_kwargs"])
        else:
            batt = Battery()
    wref = omega_ref_fn or (lambda t: R.omega_ref(t))
    if plant is None:
        plant = GEMPlant(disturbance_fn=R.disturbance, battery=batt, fault_fn=R.bms_derate if (battery and event) else None, tau=tau)
    beliefs, draws = draw_beliefs(plant, belief_cfg, streams["beliefs"])
    for nm, t_cfg in (("sensor", sensor_cfg.tau), ("inverter", inv_cfg.tau), ("controller", ctl_cfg.tau), ("plant", plant.tau)):
        if abs(t_cfg - tau) > 1e-12 * max(tau, 1.0):
            raise ValueError(f"time step mismatch: run tau={tau:g} s but the {nm} uses {t_cfg:g} s")
    k_d = ctl_cfg.k_i * (beliefs.Ld / beliefs.Lq if ctl_cfg.d_gain_by_L else 1.0)
    rho = d_loop_spectral_radius(k_d, tau, plant.Ld, d_steps)
    if rho >= 1.0 and check_d_loop:
        raise ValueError(f"the d-axis current loop is unstable (root magnitude {rho:.3f} >= 1) at tau={tau:g} s with delay {d_steps}: "
                         f"use d_gain_by_L=True or a smaller k_i (audit 2026-11-09); pass check_d_loop=False to run it anyway")
    ctl = FirewalledController(ctl_cfg, beliefs)
    sensors = SensorSuite(sensor_cfg, streams["sensors"])
    inv = InverterModel(inv_cfg, streams["inverter"])
    bms = None
    if battery and fit is not None:
        from bms import BMSModel               # only needed (and only delivered by Agent B) for battery runs with a fit
        bms = BMSModel(bms_cfg, fit, streams["bms"])

    n = int(round(t_end / plant.tau))
    rows, led, eps_prev = [], None, plant.state()["epsilon_el"]
    t0 = time.time()
    no_bms = BMSMsg(i_dis_lim=1.0e9, i_chg_lim=1.0e9, v_pack=V_SCALE_GEM, age_s=0.0)
    for k in range(n):
        st = plant.state()
        truth = {key: st[key] for key in TRUTH_KEYS if key != "epsilon_prev"}
        truth["epsilon_prev"] = eps_prev
        t = st["t"]
        msg = bms.step(t, _batt_truth(plant, led)) if bms is not None else no_bms
        u_max = R.U_EVENT if (event and not battery and R.T_EVENT[0] <= t < R.T_EVENT[1]) else None
        m = sensors.measure(truth, wref(t), msg, plant.true_ceiling(u_max) if oracle_ceiling else None)
        u_sent, ci = ctl.command(m)
        delta = sensors.last_delta - d_steps * st["omega_el"] * plant.tau          # angle error at application (contract section 2)
        u_pre = inv.pre_clip(u_sent, delta)
        u_extra = inv.post_clip_error(np.array([st["i_a"], st["i_b"], st["i_c"]]), np.array([m.i_a_m, m.i_b_m, m.i_c_m]), st["V_dc"], st["epsilon_el"])
        led = plant.apply_voltage(u_pre, u_max, u_extra=u_extra)
        eps_prev = st["epsilon_el"]

        if k % log_every == 0:
            # ---- exact ledger decomposition, sequential: a (ctrl cmd) -> b (ctrl clip) -> c (advance) -> d (delay) -> e (rotation) -> f (plant clip) -> g (+inverter error)
            a = np.array([ci["u_d_ctrl"], ci["u_q_ctrl"]])
            b = np.array([ci["u_d_clip"], ci["u_q_clip"]])
            c = u_sent
            e = u_pre
            d_u_delay, d_u_rot = np.asarray(inv.last_delay_dv), np.asarray(inv.last_rot_dv)
            f = np.array([led["u_d_app"], led["u_q_app"]])
            g = f + np.array([led["u_d_extra"], led["u_q_extra"]])
            gap = {}
            for nm, mv, tv in (("i_a", m.i_a_m, st["i_a"]), ("i_b", m.i_b_m, st["i_b"]), ("i_c", m.i_c_m, st["i_c"]), ("i_d", m.i_d_m, st["i_d"]),
                               ("i_q", m.i_q_m, st["i_q"]), ("theta", m.theta_m, st["epsilon_el"]), ("omega", m.omega_m_m, st["omega_m"]),
                               ("vdc", m.vdc_m, st["V_dc"])):
                gap["gap_" + nm] = mv - tv
            row = {**st, **ci, **{kk: vv for kk, vv in led.items() if kk not in ("u_d_cmd", "u_q_cmd", "u_d_app", "u_q_app", "d_u_d", "d_u_q", "d_u_norm", "saturated")},
                   "u_d_cmd": a[0], "u_q_cmd": a[1], "u_d_app": g[0], "u_q_app": g[1],
                   "d_u_d": g[0] - a[0], "d_u_q": g[1] - a[1],
                   "d_u_ctrl_clip_d": b[0] - a[0], "d_u_ctrl_clip_q": b[1] - a[1],
                   "d_u_adv_d": c[0] - b[0], "d_u_adv_q": c[1] - b[1],
                   "d_u_delay_d": d_u_delay[0], "d_u_delay_q": d_u_delay[1],
                   "d_u_rot_d": d_u_rot[0], "d_u_rot_q": d_u_rot[1],
                   "d_u_plant_clip_d": f[0] - e[0], "d_u_plant_clip_q": f[1] - e[1],
                   "d_u_inv_d": g[0] - f[0], "d_u_inv_q": g[1] - f[1],
                   "d_u_norm": (np.hypot(*((b - a) + (f - e))) / led["u_limit"]) if np.hypot(*a) else 0.0,
                   "saturated": float(np.hypot(*a) > ci["ceiling_hat"] or led["saturated"]),
                   "sat_ctrl": float(np.hypot(*a) > ci["ceiling_hat"]), "sat_plant": float(led["saturated"]),
                   "psi_tilde": ci["psi_hat"] - plant.psi_p,
                   "T_cmd": 1.5 * plant.p * plant.psi_p * ci["iq_ref"],
                   "meas_i_a": m.i_a_m, "meas_i_b": m.i_b_m, "meas_i_c": m.i_c_m, "meas_i_d": m.i_d_m, "meas_i_q": m.i_q_m,
                   "meas_theta": m.theta_m, "meas_omega": m.omega_m_m, "meas_vdc": m.vdc_m,
                   "bms_i_dis_lim": m.bms.i_dis_lim, "bms_i_chg_lim": m.bms.i_chg_lim, "bms_v_pack": m.bms.v_pack, "bms_age_s": m.bms.age_s,
                   **gap}
            row["d_T"] = st["T_elec"] - row["T_cmd"]
            row["dist_i_limit"] = 1.0 - np.hypot(st["i_d"], st["i_q"]) / plant.limits["i"]
            if extra_cols is not None:
                row.update(extra_cols(t))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["V_e"] = 0.5 * plant.Lq * df["e_qa"] ** 2
    df["V_theta"] = df["psi_tilde"] ** 2 / (2 * ctl_cfg.gamma)
    df["V_aw"] = 0.5 * plant.Lq * df["z"] ** 2
    df["V"] = df["V_e"] + df["V_theta"] + df["V_aw"]
    for nm, v in (("R", beliefs.R), ("Ld", beliefs.Ld), ("Lq", beliefs.Lq), ("psi_name", beliefs.psi_name), ("J", beliefs.J)):
        df["belief_" + nm] = v
    manifest = dict(name=name, mode=mode, event=event, governor=governor, battery=battery, vdc_aware=vdc_aware, bms_aware=bms_aware,
                    oracle_ceiling=oracle_ceiling, seed=seed, identity=identity, T_END=t_end, tau=plant.tau, steps=n,
                    wall_s=round(time.time() - t0, 1), fit_path=str(fit_path) if fit_path else None,
                    sensor_cfg=asdict(sensor_cfg), inverter_cfg=asdict(inv_cfg), bms_cfg=asdict(bms_cfg), belief_cfg=asdict(belief_cfg),
                    controller_cfg=asdict(ctl_cfg), belief_draws=draws, beliefs=asdict(beliefs),
                    psi_true=plant.psi_p, scenario=dict(t_event=R.T_EVENT, u_event=R.U_EVENT), gem="3.0.3")
    if write:
        out = pathlib.Path(outdir)
        out.mkdir(exist_ok=True)
        df.to_parquet(out / f"{name}.parquet")
        (out / f"{name}.json").write_text(json.dumps(manifest, indent=2, default=float))
    return df, manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="naive", choices=["naive", "robust", "aw"])
    ap.add_argument("--event", action="store_true")
    ap.add_argument("--governor", action="store_true")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--vdc-aware", action="store_true")
    ap.add_argument("--bms-aware", action="store_true")
    ap.add_argument("--oracle-ceiling", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", default=None)
    ap.add_argument("--outdir", default="runs_fw")
    a = ap.parse_args()
    _, mf = run_fw(a.name, a.mode, a.event, a.governor, a.battery, a.vdc_aware, a.bms_aware, a.oracle_ceiling, a.seed, a.identity,
                   fit_path=a.fit, outdir=a.outdir)
    print("done", a.name, mf["wall_s"], "s")
