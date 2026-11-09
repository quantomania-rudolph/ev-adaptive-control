"""Drive-cycle runs through the information firewall (sensors, inverter, BMS, beliefs). Reuses run_fw; only the plant,
the speed request and the logging rate differ.   python firewalled_cycle.py --source udds:PATH --name U_aw --mode aw --t-max 120
"""
import argparse, pathlib, sys, json
import numpy as np
from battery import Battery
from cycles import fit_vehicle_to_motor, data_demand
from plant import GEMPlant
from interfaces import TAU_CYCLE
import run_cycle as RC
from firewalled_loop import run_fw


def run_fw_cycle(cyc, name, mode="aw", governor=False, vdc_aware=False, bms_aware=False, derate=True, seed=0, identity=False,
                 t_max=None, tau=TAU_CYCLE, log_every=25, fit_path="data/tum_battery_fit.json", battery=True, outdir="runs_fw", ctl_extra=None):
    cyc = RC.ensure_standing_start(cyc)
    if t_max is not None:
        keep = cyc.t <= t_max
        cyc.t, cyc.v = cyc.t[keep], cyc.v[keep]
        cyc.grade = None if cyc.grade is None else cyc.grade[keep]
        cyc.measured = {k: a[keep] for k, a in cyc.measured.items()}
    veq, info = fit_vehicle_to_motor(cyc, T_cap=RC.T_CAP)
    grade_T = None
    if cyc.grade is not None:
        gt = veq.grade_torque(cyc.grade)
        grade_T = lambda t: float(np.interp(t, cyc.t, gt))
    fit = json.loads(pathlib.Path(fit_path).read_text()) if (battery and fit_path) else None
    batt = (Battery(**fit["toy_battery_kwargs"]) if fit else Battery()) if battery else None
    fault = None
    if derate and battery:
        t0 = RC.peak_power_time(cyc) - 0.5
        fault = lambda t: {"idis_scale": 0.02 if t0 <= t < t0 + 3.0 else 1.0}
    plant = GEMPlant(disturbance_fn=grade_T, battery=batt, fault_fn=fault, tau=tau, **veq.plant_kwargs())
    wref = lambda t: veq.k * float(np.interp(t, cyc.t, cyc.v))
    dd = data_demand(cyc, veq, J_rotor=plant.J - veq.mass / veq.k ** 2)        # what the DATA asks, before any controller (no effect on the dynamics)
    def dd_at(key, t):
        k = int(min(max(np.searchsorted(cyc.t, t, side="right") - 1, 0), len(cyc.t) - 2))
        return float(dd[key + "_seg"][k])
    return run_fw(name, mode=mode, governor=governor, vdc_aware=vdc_aware, bms_aware=bms_aware and battery, battery=battery,
                  seed=seed, identity=identity, t_end=cyc.duration, tau=tau, fit_path=fit_path if battery else None, outdir=outdir,
                  plant=plant, omega_ref_fn=wref, log_every=log_every,
                  extra_cols=lambda t: {"v_ref_mps": wref(t) / veq.k, "T_req_data": dd_at("T_req", t), "T_avail": dd_at("T_avail", t), "data_ratio": dd_at("ratio", t)}, ctl_overrides={"i_lim": RC.I_LIM, "d_gain_by_L": not identity, **(ctl_extra or {})})   # stable d gain at tau = 2e-4 (audit 2026-11-09); identity mode must equal the legacy controller


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="aw", choices=["naive", "robust", "aw"])
    ap.add_argument("--governor", action="store_true"); ap.add_argument("--vdc-aware", action="store_true"); ap.add_argument("--bms-aware", action="store_true")
    ap.add_argument("--no-derate", action="store_true"); ap.add_argument("--identity", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--t-max", type=float, default=None)
    a = ap.parse_args()
    df, mf = run_fw_cycle(RC._load(a.source), a.name, a.mode, a.governor, a.vdc_aware, a.bms_aware, not a.no_derate, a.seed, a.identity, a.t_max)
    print(a.name, "done", len(df), "rows", mf["wall_s"], "s")
