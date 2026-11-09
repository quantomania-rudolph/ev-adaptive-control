"""Run a real drive cycle through the powertrain (motor + controller + battery + DC link) and log everything.

    python run_cycle.py --source tum:/path/track_0.csv --name T0n --mode naive  --battery --derate
    python run_cycle.py --source tum:/path/track_0.csv --name T0p --mode aw --governor --vdc-aware --battery --derate
    python run_cycle.py --source udds:/path/udds.txt --name U1 --mode aw --battery --t-max 120

What happens (learning note):
  1. The cycle is loaded and sanitised (cycles.py) and its report is saved.
  2. fit_vehicle_to_motor() picks the gear ratio k (so peak vehicle speed maps to ~190 rad/s, below the field-weakening
     speed) and an EQUIVALENT mass the Tier 1 motor can accelerate. Road load and grade are reflected to the shaft.
  3. The speed request is omega_ref(t) = k * v(t). The controller never sees the vehicle, only the speed error.
  4. Optional physical event: the BMS cuts the pack current for a few seconds at the moment of peak acceleration power.
  5. Every `log_every` steps one row is stored (a 3 M-step run at 10 kHz would not fit in memory otherwise).
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time
import numpy as np
import pandas as pd

from battery import Battery
from control import CascadeController
from cycles import Cycle, CycleError, VehicleEq, data_demand, fit_vehicle_to_motor, load_cycle, load_tum_tesla, synthetic_cycle
from plant import GEMPlant

I_LIM = 250.0
T_CAP = 1.5 * 3 * 0.066 * I_LIM        # N.m available with i_d = 0 and the current limit (matches control.i_lim)


def ensure_standing_start(cyc: Cycle, a_ramp=1.5):
    """The simulated motor starts at rest. If the data starts moving, prepend a gentle ramp so the first sample is fair."""
    v0 = float(cyc.v[0])
    if v0 < 0.3:
        return cyc
    n = int(np.ceil(max(v0 / a_ramp, 3.0) / cyc.dt))
    ramp = np.linspace(0.0, v0, n, endpoint=False)
    cyc.v = np.r_[ramp, cyc.v]
    cyc.t = np.arange(cyc.v.size) * cyc.dt
    if cyc.grade is not None:
        cyc.grade = np.r_[np.full(n, cyc.grade[0]), cyc.grade]
    cyc.measured = {k: np.r_[np.full(n, np.nan), a] for k, a in cyc.measured.items()}
    cyc.note("standing_start_added", "fix", f"cycle began at {v0:.1f} m/s; prepended a {n*cyc.dt:.1f} s ramp from rest", n)
    return cyc


def peak_power_time(cyc: Cycle):
    p = cyc.v * cyc.accel()
    return float(cyc.t[int(np.argmax(p))])


def run_cycle(cyc: Cycle, name, *, mode="naive", governor=False, vdc_aware=False, battery=True, derate=None,
              tau=2.0e-4, log_every=25, t_max=None, outdir="runs", allow_road_scale=False, soc0=0.5, fit_smooth_s=None):
    """derate: None, or (t_start, duration, idis_scale). Returns (DataFrame, manifest)."""
    cyc = ensure_standing_start(cyc)
    if t_max is not None:
        keep = cyc.t <= t_max
        cyc.t, cyc.v = cyc.t[keep], cyc.v[keep]
        cyc.grade = None if cyc.grade is None else cyc.grade[keep]
        cyc.measured = {k: a[keep] for k, a in cyc.measured.items()}
        cyc.note("truncated", "info", f"run limited to first {t_max}s", int((~keep).sum()))
    if fit_smooth_s:
        # size the vehicle on a SMOOTHED copy of the request, then run the raw request on it. Used to feed deliberately
        # infeasible data (instant speed steps) to a motor that fits the smooth shape, so data-caused mismatch can be studied.
        from scipy.ndimage import uniform_filter1d
        sm = Cycle(t=cyc.t.copy(), v=uniform_filter1d(cyc.v, size=max(3, int(round(fit_smooth_s / cyc.dt))), mode="nearest"), dt=cyc.dt,
                   grade=None if cyc.grade is None else cyc.grade.copy())
        veq, info = fit_vehicle_to_motor(sm, T_cap=T_CAP, allow_road_scale=allow_road_scale)
        cyc.note("fit_on_smoothed_request", "warn", f"vehicle sized on a {fit_smooth_s:.1f} s moving average of the request; the raw request is run on it", 1)
    else:
        veq, info = fit_vehicle_to_motor(cyc, T_cap=T_CAP, allow_road_scale=allow_road_scale)
    if info["frac_over_cap"] > 0:
        cyc.note("torque_demand_over_capability", "warn", f"{100*info['frac_over_cap']:.2f}% of samples need more than {T_CAP:.0f} N.m; "
                 "the current limit will clip them (logged as M4/M5)", 1)
    if info["road_scale"] != 1.0:
        cyc.note("road_load_scaled", "warn", f"road load scaled by {info['road_scale']:.2f} to fit the Tier 1 motor", 1)
    grade_T = None
    if cyc.grade is not None:
        gt = veq.grade_torque(cyc.grade)
        grade_T = lambda t: float(np.interp(t, cyc.t, gt))
    batt = Battery(soc0=soc0) if battery else None
    fault = None
    if derate and battery:
        t0, dur, sc = derate
        fault = lambda t: {"idis_scale": sc if t0 <= t < t0 + dur else 1.0}
    plant = GEMPlant(disturbance_fn=grade_T, battery=batt, fault_fn=fault, tau=tau, **veq.plant_kwargs())
    ctl = CascadeController(plant, mode=mode, governor=governor, vdc_aware=vdc_aware, i_lim=I_LIM)
    wref = lambda t: veq.k * float(np.interp(t, cyc.t, cyc.v))
    dd = data_demand(cyc, veq, J_rotor=plant.J - veq.mass / veq.k ** 2)        # what the DATA asks, before any controller
    def dd_at(key, t):                                      # value for the INTERVAL the request is in (the request is piecewise linear)
        k = int(min(max(np.searchsorted(cyc.t, t, side="right") - 1, 0), len(cyc.t) - 2))
        return float(dd[key + "_seg"][k])

    n = int(round(cyc.duration / tau))
    rows, led, t_wall = [], {}, time.time()
    for k in range(n):
        st = plant.state()
        u_c, it = ctl.command(wref(st["t"]), st, led)
        led = plant.apply_voltage(u_c)
        if k % log_every == 0:
            T_cmd = 1.5 * plant.p * plant.psi_p * it["iq_ref"]
            rows.append({**st, **it, **led, "T_cmd": T_cmd, "d_T": st["T_elec"] - T_cmd,
                         "dist_i_limit": 1.0 - np.hypot(st["i_d"], st["i_q"]) / plant.limits["i"],
                         "v_ref_mps": wref(st["t"]) / veq.k,
                         "T_req_data": dd_at("T_req", st["t"]), "T_avail": dd_at("T_avail", st["t"]), "data_ratio": dd_at("ratio", st["t"])})
        if not np.isfinite(st["omega_m"]) or abs(st["omega_m"]) > 5 * plant.limits["omega"]:
            cyc.note("simulation_diverged", "error", f"speed non-finite or beyond 5x the limit at t={st['t']:.2f}s; run stopped", 1)
            break
    df = pd.DataFrame(rows)
    df["V_e"] = 0.5 * plant.Lq * df["e_qa"] ** 2
    df["V_theta"] = df["psi_tilde"] ** 2 / (2 * ctl.gamma)
    df["V_aw"] = 0.5 * plant.Lq * df["z"] ** 2
    df["V"] = df["V_e"] + df["V_theta"] + df["V_aw"]
    out = pathlib.Path(outdir); out.mkdir(exist_ok=True)
    df.to_parquet(out / f"{name}.parquet")
    cyc.report_frame().to_csv(out / f"{name}_cycle_report.csv", index=False)
    meas = pd.DataFrame({"t": cyc.t, "v": cyc.v, **cyc.measured})
    meas.to_parquet(out / f"{name}_measured.parquet")
    man = dict(name=name, source=cyc.source, mode=mode, governor=governor, vdc_aware=vdc_aware, battery=battery, derate=derate,
               tau=tau, log_every=log_every, tau_log=tau * log_every, rows=len(df), steps=int(k + 1), wall_s=round(time.time() - t_wall, 1),
               vehicle=dict(veq.__dict__, **{k: v for k, v in info.items()}), T_cap=T_CAP, duration_s=cyc.duration,
               data_demand=dict(frac_ratio_gt_1=float((dd["ratio_seg"] > 1.0).mean()), frac_ratio_gt_085=float((dd["ratio_seg"] > 0.85).mean()),
                                ratio_max=float(dd["ratio_seg"].max()), ratio_q99=float(np.quantile(dd["ratio_seg"], 0.99))),
               report=cyc.report)
    (out / f"{name}.json").write_text(json.dumps(man, indent=2, default=float))
    return df, man


def _load(source, **kw):
    kind, _, path = source.partition(":")
    if kind == "tum":
        return load_tum_tesla(path, **kw)
    if kind == "synthetic":
        name, _, dur = (path or "ramp").partition(":")
        return synthetic_cycle(name, duration=float(dur)) if dur else synthetic_cycle(name)
    if kind in ("udds", "csv"):
        return load_cycle(path, **kw)
    raise CycleError("bad_source", "source must look like tum:/path/track_N.csv, udds:/path/udds.txt or csv:/path/file.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="aw", choices=["naive", "robust", "aw"])
    ap.add_argument("--governor", action="store_true"); ap.add_argument("--vdc-aware", action="store_true")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--derate", action="store_true", help="BMS derate for 3 s at the moment of peak acceleration power")
    ap.add_argument("--derate-scale", type=float, default=0.02); ap.add_argument("--derate-len", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=2e-4); ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--t-max", type=float, default=None); ap.add_argument("--allow-road-scale", action="store_true")
    ap.add_argument("--outdir", default="runs"); ap.add_argument("--fit-smooth", type=float, default=None)
    a = ap.parse_args()
    cyc = _load(a.source)
    dr = (peak_power_time(ensure_standing_start(cyc)) - 0.5, a.derate_len, a.derate_scale) if a.derate else None
    df, man = run_cycle(cyc, a.name, mode=a.mode, governor=a.governor, vdc_aware=a.vdc_aware, battery=a.battery, derate=dr,
                        tau=a.tau, log_every=a.log_every, t_max=a.t_max, outdir=a.outdir, allow_road_scale=a.allow_road_scale, fit_smooth_s=a.fit_smooth)
    print(a.name, "done:", man["rows"], "rows,", man["wall_s"], "s wall; derate", dr)
