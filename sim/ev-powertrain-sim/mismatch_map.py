"""Mismatch map: one record of every requested-vs-delivered gap, for every run, grouped by CLASS of speed request.

    python mismatch_map.py            # reads runs/*.parquet + .json, writes data/mismatch_map.{csv,json}

The `index` is the plain average of the RMS percentages of M1, M2, M4, M5 and M6c: one number to rank runs by ("how big is the
requested-versus-delivered gap overall"). The equal weights are a placeholder; the team should choose real ones.

Convention everywhere: delta = delivered - requested. Each gap is also expressed as a percentage of a natural scale so
rows with different units (rad/s, V, A, N.m, W) can sit on one chart. The scales are fixed numbers, listed in SCALES, so
a percentage means the same thing in every run.

The point of the map is ATTRIBUTION. For the speed gap (M1) the squared error is split into three shares:
  data       within 2 s after an instant where the request was infeasible for ANY controller (data_ratio > 1)
  event      within the injected fault window plus 3 s of recovery (and not already counted as data)
  remainder  everything else: this is the part the controller and plant are responsible for
Runs made before the data-feasibility ledger existed (the 8 s plain-motor experiments) have no data share.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from battery import Battery
from plant import GEMPlant
from telemetry import derive, params_from_plant

T_CAP = 1.5 * 3 * 0.066 * 250.0
P_RATED = T_CAP * 190.0
SCALES = {"M1": ("speed shortfall", "rad/s", 190.0), "M2": ("voltage shortfall |du|", "V", 173.2051), "M3": ("hidden clipping (must be 0)", "V", 173.2051),
          "M4": ("current request clipped", "A", 250.0), "M5": ("torque gap", "N.m", T_CAP), "M6b": ("shaft power gap", "W", P_RATED),
          "M6c": ("electrical power gap, reference-implied", "W", P_RATED), "M6d": ("electrical power gap, voltage command", "W", P_RATED),
          "M9": ("DC-link sag", "V", 300.0), "M10": ("pack minus inverter current", "A", 250.0)}
LEGACY_CLASS = {"A": "8 s ramp, plain motor", "B": "8 s ramp, plain motor", "C": "8 s ramp, plain motor", "D": "8 s ramp, plain motor", "Dg": "8 s ramp, plain motor",
                "Eb": "8 s ramp, battery fault", "Eg": "8 s ramp, battery fault", "T0n": "real mixed (Tesla track 0), full 283 s", "T0p": "real mixed (Tesla track 0), full 283 s"}
LEGACY_CTRL = {"A": "naive, no event", "B": "naive", "C": "robust", "D": "anti-windup", "Dg": "anti-windup + governor", "Eb": "naive", "Eg": "anti-windup + governor",
               "T0n": "naive", "T0p": "anti-windup + governor"}


def controller_label(man):
    if man.get("mode") == "robust":
        return "robust"
    if man.get("mode") == "aw":
        return "anti-windup + governor" if man.get("governor") else "anti-windup"
    return "naive"


def series_for(d: pd.DataFrame) -> dict:
    s = {"M1": d["omega_ref"] - d["omega_m"], "M2": np.hypot(d["d_u_d"], d["d_u_q"]), "M3": np.hypot(d["d_u_plant_d"], d["d_u_plant_q"]),
         "M4": d["i_ref_clip"], "M5": d["d_T"], "M6b": d["dP_mech"],
         # quasi-steady electrical gap = shaft gap + copper-loss gap. The magnetic-storage term is left out on purpose: it
         # integrates to a bounded stored energy, and on decimated logs its finite difference is not meaningful.
         "M6c": d["dP_mech"] + d["dP_cu"], "M6d": d["dP_elec"]}
    if "bus_sag" in d:
        s["M9"], s["M10"] = d["bus_sag"], d["d_i_dc"]
    return s


def summarize_run(name: str, df: pd.DataFrame, man: dict, P0: dict) -> tuple[list, dict]:
    tau = man.get("tau_log", man.get("tau", P0["tau"]))
    d = derive(df, dict(P0, tau=tau))
    t = d["t"].to_numpy()
    if man.get("derate"):
        ev = [man["derate"][0], man["derate"][0] + man["derate"][1]]
    elif man.get("event"):
        ev = [4.5, 5.5]
    else:
        ev = None
    cls = man.get("dataset_class") or LEGACY_CLASS.get(name, "other")
    ctrl = LEGACY_CTRL.get(name) or controller_label(man)
    rows = []
    for mid, ser in series_for(d).items():
        label, unit, scale = SCALES[mid]
        x = ser.to_numpy(float)
        rows.append(dict(run=name, dataset_class=cls, controller=ctrl, event=bool(ev), mismatch=mid, label=label, unit=unit, scale=scale,
                         rms=float(np.sqrt(np.mean(x ** 2))), peak=float(np.max(np.abs(x))), mean=float(x.mean()),
                         rms_pct=float(100 * np.sqrt(np.mean(x ** 2)) / scale), peak_pct=float(100 * np.max(np.abs(x)) / scale),
                         active_pct=float(100 * np.mean(np.abs(x) > 0.01 * scale))))
    # ---- attribution of the speed-error energy
    e2 = (d["omega_ref"] - d["omega_m"]).to_numpy() ** 2
    tot = float(e2.sum()) or 1.0
    in_data = np.zeros(len(t), bool)
    if "data_infeasible" in d:
        bad = np.nonzero(d["data_infeasible"].to_numpy() > 0.5)[0]
        for k in bad:
            in_data |= (t >= t[k]) & (t <= t[k] + 2.0)
    in_event = np.zeros(len(t), bool)
    if ev:
        in_event = (t >= ev[0]) & (t <= ev[1] + 3.0) & ~in_data
    share = dict(data=float(e2[in_data].sum() / tot), event=float(e2[in_event].sum() / tot))
    share["remainder"] = 1.0 - share["data"] - share["event"]
    # shaft and electrical energy owed over the whole run, as a fraction of the shaft energy requested
    e_req = float(np.abs(d["P_mech_req"]).sum() * tau) or 1.0
    core = ["M1", "M2", "M4", "M5", "M6c"]
    index = float(np.mean([r["rms_pct"] for r in rows if r["mismatch"] in core]))
    summ = dict(run=name, dataset_class=cls, controller=ctrl, event=ev, duration=float(t[-1]), m1_share=share, index=index,
                e_shaft_owed_frac=float(-d["dP_mech"].sum() * tau / e_req), e_elec_qs_owed_frac=float(-(d["dP_mech"] + d["dP_cu"]).sum() * tau / e_req),
                data_ratio_gt1_pct=float(100 * (d["data_ratio"] > 1.0).mean()) if "data_ratio" in d else None,
                data_ratio_q99=float(d["data_ratio"].quantile(0.99)) if "data_ratio" in d else None,
                req_sign_conflict_pct=float(100 * d["req_sign_conflict"].mean()), decimated=bool(tau > 1.5e-4 * 3))
    return rows, summ


def build(runs_dir="runs", out_dir="data"):
    P0 = params_from_plant(GEMPlant(battery=Battery()))
    rows, summaries = [], []
    for pq in sorted(pathlib.Path(runs_dir).glob("*.parquet")):
        name = pq.stem
        if name.endswith("_measured"):
            continue
        mj = pq.with_suffix(".json")
        man = json.loads(mj.read_text()) if mj.exists() else {}
        r, s = summarize_run(name, pd.read_parquet(pq), man, P0)
        rows += r; summaries.append(s)
        print("mapped", name, s["dataset_class"], "|", s["controller"])
    out = pathlib.Path(out_dir); out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "mismatch_map.csv", index=False)
    (out / "mismatch_map.json").write_text(json.dumps(dict(scales={k: list(v) for k, v in SCALES.items()}, runs=summaries,
                                                           rows=rows), indent=1, default=float))
    return rows, summaries


if __name__ == "__main__":
    build()
