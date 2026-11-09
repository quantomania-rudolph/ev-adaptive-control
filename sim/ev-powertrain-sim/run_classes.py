"""Run the same three controller set-ups on several CLASSES of speed request, to build the mismatch map.

    python run_classes.py --tum-dir /path/to/ds_tum --udds /path/udds.txt [--workers 2] [--only Yr,U]

Per class we run: n = naive controller + BMS cut at peak power, p = protected controller + the same cut, b = protected
controller with NO fault (the baseline: any mismatch left there comes from the controller and the data, not from an event).
Every run goes through the same cycles.py -> run_cycle.py pipeline, so the classes are directly comparable.
"""
import argparse, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

CLASSES = {   # code: (label, source template, t_max, options)
    "Yr": ("synthetic ramp",                       "synthetic:ramp", 20, {}),
    "Ys": ("synthetic sine",                       "synthetic:sine", 20, {}),
    "Yt": ("synthetic step (infeasible)",          "synthetic:step:30", 30, {"fit_smooth_s": 4.0, "derate_at": (15.0, 2.0)}),   # fault placed on a flat 6 s plateau, away from the jumps
    "U":  ("real urban (EPA UDDS)",                "udds",           120, {}),
    "M":  ("real mixed (Tesla track 0)",           "tum:track_0.csv", 150, {}),
    "X":  ("real suburban (Tesla track 10)",       "tum:track_10.csv", 120, {}),
    "H":  ("real highway (Tesla track 100, road load scaled)", "tum:track_100.csv", 120, {"allow_road_scale": True}),
}
SETUPS = {"n": dict(mode="naive", governor=False, vdc_aware=False, derate=True),
          "p": dict(mode="aw", governor=True, vdc_aware=True, derate=True),
          "b": dict(mode="aw", governor=True, vdc_aware=True, derate=False)}


def job(code, setup, tum_dir, udds, outdir):
    from run_cycle import run_cycle, ensure_standing_start, peak_power_time, _load
    label, src, t_max, opt = CLASSES[code]
    opt = dict(opt)                                                      # job() pops from it; never mutate the shared table
    if src == "udds":
        src = f"udds:{udds}"
    elif src.startswith("tum:"):
        src = f"tum:{tum_dir}/{src[4:]}"
    cyc = _load(src)
    s = SETUPS[setup]
    cyc0 = ensure_standing_start(cyc)
    # derate time is chosen inside the run window, at the moment of peak acceleration power
    keep = cyc0.t <= t_max
    import numpy as np
    p = cyc0.v[keep] * np.gradient(cyc0.v, cyc0.t)[keep]
    t_ev = float(cyc0.t[keep][int(np.argmax(p))]) - 0.5
    fixed = opt.pop("derate_at", None)                                   # a class may pin the fault time (so it cannot overlap data-caused mismatch)
    dr = ((fixed[0], fixed[1], 0.02) if fixed else (max(t_ev, 1.0), 3.0, 0.02)) if s["derate"] else None
    name = f"{code}{setup}"
    df, man = run_cycle(cyc, name, mode=s["mode"], governor=s["governor"], vdc_aware=s["vdc_aware"], battery=True, derate=dr,
                        t_max=t_max, outdir=outdir, **opt)
    man["dataset_class"] = label
    pathlib.Path(outdir, f"{name}.json").write_text(json.dumps(man, indent=2, default=float))
    return name, man["wall_s"], man["rows"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum-dir", required=True); ap.add_argument("--udds", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", default="")
    ap.add_argument("--outdir", default="runs")
    a = ap.parse_args()
    codes = [c for c in CLASSES if not a.only or c in a.only.split(",")]
    jobs = [(c, s) for c in codes for s in SETUPS]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, c, s, a.tum_dir, a.udds, a.outdir): (c, s) for c, s in jobs}
        for f in as_completed(futs):
            try:
                print("done", *f.result(), f"[{time.time()-t0:.0f}s elapsed]", flush=True)
            except Exception as e:
                print("FAILED", futs[f], repr(e), flush=True)
