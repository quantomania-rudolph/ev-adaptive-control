"""Experiment runner for the notes' A-D matrix on the GEM PMSM plant.

    python run_exp.py --name B --mode naive  --event
    python run_exp.py --name A --mode naive              (no saturation event)
    python run_exp.py --name C --mode robust --event
    python run_exp.py --name D --mode aw     --event

Scenario (identical across experiments so the controller is the only difference):
    speed reference ramps 0 -> 200 rad/s in 4 s, grade-like disturbance +10 N.m at t=1.5 s,
    then (event runs only) an UNMODELED voltage ceiling drops from 173 V to 25 V for t in [4.5, 5.5] s.
    At 200 rad/s the back-EMF is ~40 V, so a 25 V ceiling makes the commanded current unreachable.
"""
import argparse, json, subprocess, sys, time, pathlib
import numpy as np
import pandas as pd
from plant import GEMPlant
from control import CascadeController
from battery import Battery

T_END = 8.0
T_EVENT = (4.5, 5.5)
U_EVENT = 25.0


def omega_ref(t):
    return 200.0 * min(t / 4.0, 1.0)


def disturbance(t):
    return 10.0 if t >= 1.5 else 0.0


def bms_derate(t):
    """Physical event for battery runs: the BMS cuts the allowed pack current to 2 % (5 A, below the ~12.6 A cruise demand) for 1 s (cold, low SOC, fault)."""
    return {"idis_scale": 0.02 if T_EVENT[0] <= t < T_EVENT[1] else 1.0}


def run(name, mode, event, governor=False, outdir="runs", battery=False, vdc_aware=False):
    batt = Battery() if battery else None
    plant = GEMPlant(disturbance_fn=disturbance, battery=batt, fault_fn=bms_derate if (battery and event) else None)
    ctl = CascadeController(plant, mode=mode, governor=governor, vdc_aware=vdc_aware)
    rows, led = [], {}
    n = int(round(T_END / plant.tau))
    t0 = time.time()
    for k in range(n):
        st = plant.state()
        u_c, internals = ctl.command(omega_ref(st["t"]), st, led)
        u_max = U_EVENT if (event and not battery and T_EVENT[0] <= st["t"] < T_EVENT[1]) else None
        led = plant.apply_voltage(u_c, u_max)
        T_cmd = 1.5 * plant.p * plant.psi_p * internals["iq_ref"]
        row = {**st, **internals, **led,
               "T_cmd": T_cmd, "d_T": st["T_elec"] - T_cmd,                       # M5
               "dist_i_limit": 1.0 - np.hypot(st["i_d"], st["i_q"]) / plant.limits["i"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    # proof quantities (offline, from logged internals): V = V_e + V_theta (+ V_aw), with Gamma of the run
    df["V_e"] = 0.5 * plant.Lq * df["e_qa"] ** 2
    df["V_theta"] = df["psi_tilde"] ** 2 / (2 * ctl.gamma)
    df["V_aw"] = 0.5 * plant.Lq * df["z"] ** 2
    df["V"] = df["V_e"] + df["V_theta"] + df["V_aw"]
    out = pathlib.Path(outdir); out.mkdir(exist_ok=True)
    df.to_parquet(out / f"{name}.parquet")
    manifest = dict(name=name, mode=mode, event=event, governor=governor, battery=battery, vdc_aware=vdc_aware, T_END=T_END, tau=plant.tau, gamma=ctl.gamma,
                    sigma=ctl.sigma, k_i=ctl.k_i, psi_true=plant.psi_p, psi_hat0=ctl.psi_nom,
                    wall_s=round(time.time() - t0, 1), steps=n,
                    gem="3.0.3", scenario=dict(t_event=T_EVENT, u_event=U_EVENT))
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    return df, manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="naive", choices=["naive", "robust", "aw"])
    ap.add_argument("--event", action="store_true")
    ap.add_argument("--governor", action="store_true")
    ap.add_argument("--battery", action="store_true", help="real battery + DC link; event becomes a BMS current derate")
    ap.add_argument("--vdc-aware", action="store_true", help="governor uses the measured bus voltage")
    a = ap.parse_args()
    df, m = run(a.name, a.mode, a.event, a.governor, battery=a.battery, vdc_aware=a.vdc_aware)
    print(a.name, "done in", m["wall_s"], "s,", len(df), "rows")
