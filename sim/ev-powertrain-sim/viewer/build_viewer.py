"""Builds viewer/data_viewer.html: a single self-contained page that shows every exposed channel.

Learning note: the viewer is deliberately dumb. It only plots columns that telemetry.derive() produced, so what
you see here is exactly what a data employee gets from the parquet files. Re-run after any new experiment:
    python viewer/build_viewer.py runs/Eb.parquet runs/Eg.parquet
"""
import json, sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plant import GEMPlant
from battery import Battery
from telemetry import derive, params_from_plant, CATALOG

here = pathlib.Path(__file__).parent
files = sys.argv[1:] or ["runs/T0n.parquet", "runs/T0p.parquet"]
import json as _json
P = params_from_plant(GEMPlant(battery=Battery()))
runs, stats, checks = {}, {}, {}
event = [4.5, 5.5]
TARGET_ROWS = 2000                                  # points per channel on the page; full rate stays in the parquet files
sig = lambda v: float(f"{v:.5g}")
for f in files:
    key = pathlib.Path(f).stem
    man_path = pathlib.Path(f).with_suffix(".json")
    man = _json.loads(man_path.read_text()) if man_path.exists() else {}
    P_run = dict(P, tau=man.get("tau_log", P["tau"]))        # decimated cycle runs integrate with their logging step
    d = derive(pd.read_parquet(f), P_run)
    STEP = max(1, len(d) // TARGET_ROWS)
    if man.get("derate"):
        event = [man["derate"][0], man["derate"][0] + man["derate"][1]]
    names = [c[0] for c in CATALOG if c[0] in d.columns]
    has_batt = "i_batt" in d.columns
    runs[key] = {c: [sig(v) for v in d[c].to_numpy()[::STEP]] for c in names}
    stats[key] = {c: [sig(d[c].min()), sig(d[c].max()), sig(float(np.sqrt((d[c] ** 2).mean())))] for c in names}
    rel = lambda r, f_: float(np.sqrt((d[r] ** 2).mean()) / np.sqrt((d[f_] ** 2).mean()))
    checks[key] = dict(
        rate_hz=int(round(1 / P_run["tau"])), rows=len(d), decimated=bool(P_run["tau"] > 1.5e-4 * 3),
        resid_iq_pct=sig(100 * rel("resid_iq", "f_iq")), resid_id_pct=sig(100 * rel("resid_id", "f_id")),
        resid_w_pct=sig(100 * rel("resid_w", "f_w")),
        power_pct=sig(100 * float(np.sqrt((d.P_balance_resid ** 2).mean()) / np.sqrt((d.P_mech ** 2).mean()))),
        m3_max=float(f"{max(d.d_u_plant_d.abs().max(), d.d_u_plant_q.abs().max()):.2g}"),
        sat_steps=int(d.saturated.sum()), sat_pct=sig(100 * d.saturated.mean()),
        e_mech_kj=sig(d.E_mech_unserved.iloc[-1] / 1000),
        e_mech_event_kj=sig(-d.loc[(d.t >= event[0]) & (d.t < event[1]), "dP_mech"].sum() * P_run["tau"] / 1000),
        peak_du=sig(d.d_u_q.abs().max()),
    )
    if has_batt:
        checks[key].update(chain_w=float(f"{d.resid_chain.abs().max():.2g}"), bus_w=float(f"{np.sqrt((d.resid_bus ** 2).mean()):.2g}"))
cat = [dict(name=n, unit=u, group=g, role=r, desc=ds) for n, u, g, r, ds in CATALOG]
payload = dict(t=runs[next(iter(runs))]["t"], event=event, runs=runs, stats=stats, catalog=cat, checks=checks,
               labels={"B": "B  naive, imposed 25 V ceiling", "Dg": "Dg  protected, imposed 25 V ceiling", "Eb": "Eb  naive controller, BMS derate", "Eg": "Eg  protected, sees bus voltage", "T0n": "T0n  real Tesla drive, naive controller", "T0p": "T0p  real Tesla drive, protected controller"})
tpl = (here / "viewer_template.html").read_text()
out = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
(here / "data_viewer.html").write_text(out)
print("wrote", here / "data_viewer.html", f"{len(out)/1e6:.1f} MB", list(checks.values())[0])
