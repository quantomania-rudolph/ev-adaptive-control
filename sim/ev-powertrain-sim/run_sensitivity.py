"""A7 sensitivity: 8 parameters x 2 levels x {naive, aw} x 3 seeds on the observable-sag battery scenario. Resumable.
Levels are 0.5x and 2x the default; the inverter delay (an integer) uses 0 and 2 steps instead.   python run_sensitivity.py [workers]
"""
import sys, os, json, time, pathlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import replace
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from interfaces import SensorConfig, InverterConfig, BMSConfig, BeliefConfig

OUT = pathlib.Path("data/sensitivity"); OUT.mkdir(parents=True, exist_ok=True)
S, I, B, BL = SensorConfig(), InverterConfig(), BMSConfig(), BeliefConfig()
def scale_rng(r, f): return tuple(f * x for x in r)
# name -> (lo, hi) as functions returning (sensor_cfg, inv_cfg, bms_cfg, belief_cfg)
P = {
 "current_noise":  lambda f: (replace(S, i_noise_lsb=S.i_noise_lsb * f), I, B, BL),
 "current_offset": lambda f: (replace(S, i_offset_a=S.i_offset_a * f), I, B, BL),
 "angle_offset":   lambda f: (replace(S, theta_offset_rad=S.theta_offset_rad * f), I, B, BL),
 "vdc_noise":      lambda f: (replace(S, vdc_noise_v=S.vdc_noise_v * f), I, B, BL),
 "vdc_lag_corner": lambda f: (replace(S, vdc_corner_hz=S.vdc_corner_hz * f), I, B, BL),     # 0.5x corner = MORE lag
 "bms_latency":    lambda f: (S, I, replace(B, latency_s=B.latency_s * f), BL),
 "belief_RL_err":  lambda f: (S, I, B, replace(BL, dR_range=scale_rng(BL.dR_range, f), dL_range=scale_rng(BL.dL_range, f))),
 "inverter_delay": lambda f: (S, replace(I, delay_steps=0 if f < 1 else 2), B, BL),
    # added 2026-11-09: the rows above use d_gain_by_L=False, which is unstable at delay 2 (root 1.08). These two use the stable gain.
    "inverter_delay_dgl": lambda f: (S, replace(I, delay_steps=0 if f < 1 else 2), B, BL),
    "nominal_dgl":        lambda f: (S, I, B, BL),
}
DGL = {"inverter_delay_dgl", "nominal_dgl"}
CTL = {"naive": dict(mode="naive"), "aw": dict(mode="aw")}
FIT = "data/tum_battery_fit.json"


def one(a):
    par, lvl, ctl, seed = a
    from firewalled_loop import run_fw
    from run_matrix_fw import metrics
    sc, ic, bc, blc = P[par](0.5 if lvl == "lo" else 2.0)
    t0 = time.time()
    df, mf = run_fw(f"s_{par}_{lvl}_{ctl}_{seed}", seed=seed, write=False, battery=True, event=True, fit_path=FIT,
                    sensor_cfg=sc, inv_cfg=ic, bms_cfg=bc, belief_cfg=blc,
                    ctl_overrides={"d_gain_by_L": True} if par in DGL else None, check_d_loop=par not in ("inverter_delay",), **CTL[ctl])
    return dict(param=par, level=lvl, controller=ctl, seed=seed, wall_s=round(time.time() - t0, 1), **metrics(df, mf))


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    path = OUT / "metrics.csv"
    done = set()
    if path.exists():
        e = pd.read_csv(path); done = set(zip(e.param, e.level, e.controller, e.seed))
    todo = [(p, l, c, s) for s in range(3) for p in P for l in ("lo", "hi") for c in CTL if (p, l, c, s) not in done]
    print("todo", len(todo), flush=True)
    with ProcessPoolExecutor(w) as ex:
        for fu in as_completed([ex.submit(one, a) for a in todo]):
            r = fu.result(); pd.DataFrame([r]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(r["param"], r["level"], r["controller"], r["seed"], r["wall_s"], flush=True)
