"""Measured bounds on the two-state disturbances d1 (rad/s^2) and d2 (A/s), split by source, per stored seed-0 run.

Learning note: a 'bounded disturbance' assumption needs a number. We cannot prove a bound from simulation, but we can measure the
largest value each piece reaches. Only SEED 0 is stored per scenario, so these are single-run maxima, not envelopes over seeds.
Only pointwise columns are used (no derivatives), so they are valid at any logging rate.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import dataset
warnings.filterwarnings("ignore")
pieces = {"d1_road": ("d_w", 1.0), "d1_reluct": ("ts_d1_red", 1.0), "d1_total": ("ts_d1", 1.0),
          "d2_actuator": ("d_iq", 1.0), "d2_cross": ("ts_d2_red", 1.0), "d2_total": ("ts_d2", 1.0), "d2_unmodeled": ("d2_unmodeled", 1.0), "du_q_unmod_V": ("du_q_unmod_V", 1.0)}
rows = []
for f in sorted(list((ROOT / "data/matrix/runs").glob("*_s0.parquet")) + list((ROOT / "data/cycles/runs").glob("*_s0.parquet"))):
    d = dataset.load_run(f)        # right parameters per file (cycle runs use the fitted vehicle inertia)
    fam = "cycle" if "cycles" in str(f) else "matrix"
    scen, ctl = f.stem.replace("_s0", "").rsplit("_", 1) if fam == "matrix" else f.stem.replace("_s0", "").split("_")
    d["d2_unmodeled"] = (d["d_u_q"] - d["d_u_ctrl_clip_q"]) / dataset.params_for(f)["Lq"]      # mismatch the controller did NOT know about (excludes its own clip)
    d["du_q_unmod_V"] = d["d_u_q"] - d["d_u_ctrl_clip_q"]
    r = dict(family=fam, scenario=scen, controller=ctl, rows=len(d), id_absmax_A=float(d.i_d.abs().max()))
    for k, (c, _) in pieces.items():
        a = d[c].abs().to_numpy()
        r[k + "_max"], r[k + "_p999"], r[k + "_rms"] = float(a.max()), float(np.quantile(a, .999)), float(np.sqrt((a ** 2).mean()))
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(ROOT / "data/disturbance_bounds.csv", index=False)

m = out[out.controller.isin(["awg", "naive"])]
lines = ["| Run | controller | max i_d (A) | road load |d_w| max (rad/s²) | reluctance max (rad/s²) | unmodeled δu_q max (V) | cross-coupling max (A/s) |", "|---|---|---|---|---|---|---|---|"]
for _, r in m.iterrows():
    lines.append(f"| {r.scenario} | {r.controller} | {r.id_absmax_A:.0f} | {r.d1_road_max:.1f} | {r.d1_reluct_max:.1f} | {r.du_q_unmod_V_max:.1f} | {r.d2_cross_max:.0f} |")
print("\n".join(lines))
