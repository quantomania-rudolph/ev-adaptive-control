"""Builds dashboard/dashboard2.html: a deliberately small page with three questions.

    python dashboard/build_dash2.py

Learning note: every number comes from files a data teammate can open (data/matrix, data/cycles, data/audit).
Traces are bucket-averaged to ~700 points for the page; the headline mismatch numbers are computed on the FULL-rate
audit files first, because averaging hides spikes. The page therefore ships both.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plant import GEMPlant
from battery import Battery
from telemetry import derive, params_from_plant, CATALOG

P0 = params_from_plant(GEMPlant(battery=Battery()))
sig = lambda v: None if v is None or not np.isfinite(v) else float(f"{v:.4g}")

# ---- 1. headline table -------------------------------------------------------------------------------------------------
def headline():
    rows = []
    for fam in ("plain", "battery"):
        d = pd.read_csv(ROOT / "data/matrix" / f"{fam}_metrics.csv")
        for (sc, ct), g in d.groupby(["scenario", "controller"]):
            rows.append(dict(group=fam, scen=sc, ctl=ct, n=len(g), env=sig(g.envelope_ok.mean()),
                             psi=sig(g.ultimate_psi_err.median()), spd=sig(g.speed_rms_post.median()),
                             eun=sig(g.E_mech_unserved_kj.median()), hid=sig(g.hidden_sat_rms.median()),
                             inv=sig(g.inv_err_rms.median()), tq=sig(g.torque_gap_rms.median())))
    for cyc in ("udds", "tesla"):
        d = pd.read_csv(ROOT / "data/cycles" / f"{cyc}_metrics.csv")
        for ct, g in d.groupby("controller"):
            rows.append(dict(group="cycle", scen=cyc, ctl=ct, n=len(g), env=None, psi=sig(g.psi_tilde_absmax.median()),
                             spd=sig(g.speed_rms.median()), eun=None, hid=None, inv=sig(g.inv_err_rms.median()), tq=None))
    return rows

# ---- 2. mismatch traces from full-rate files -----------------------------------------------------------------------------
RUNS = {"sag": ("fr_battery_sag_awg", "Battery sag, protected controller"),
        "hid": ("fr_plain_hidden_aw", "Hidden voltage limit, anti-windup controller")}
LEDGER = [("ctrl_clip", "Controller's own clip"), ("adv", "Angle advance"), ("delay", "One-step delay"),
          ("rot", "Frame rotation"), ("plant_clip", "Plant voltage limit"), ("inv", "Inverter dead-time + drop")]

def bucket(x, n=700):
    x = np.asarray(x, float); m = len(x) // n
    return [sig(v) for v in x[: m * n].reshape(n, m).mean(1)]

def mismatch():
    out = {}
    for key, (stem, title) in RUNS.items():
        man = json.loads((ROOT / "data/audit" / f"{stem}.json").read_text())
        d = derive(pd.read_parquet(ROOT / "data/audit" / f"{stem}.parquet"), dict(P0, tau=man["tau"]))
        tau = man["tau"]
        led = {k: float(np.sqrt((d[f"d_u_{k}_d"] ** 2 + d[f"d_u_{k}_q"] ** 2).mean())) for k, _ in LEDGER}
        total = float(np.sqrt((d.d_u_d ** 2 + d.d_u_q ** 2).mean()))
        out[key] = dict(
            title=title, tau=tau, event=[4.5, 5.5] if man.get("event") else None,
            t=bucket(d.t), u_req=bucket(np.hypot(d.u_d_cmd, d.u_q_cmd)), u_del=bucket(np.hypot(d.u_d_app, d.u_q_app)),
            u_lim=bucket(d.u_limit), T_req=bucket(d.T_cmd), T_del=bucket(d.T_elec),
            ledger=[dict(k=k, label=l, rms=sig(led[k])) for k, l in LEDGER], total_rms=sig(total),
            peak_gap=sig(float(np.hypot(d.d_u_d, d.d_u_q).max())),
            hidden_clip_max=sig(float(max(d.d_u_plant_d.abs().max(), d.d_u_plant_q.abs().max()))),
            e_shaft_kj=sig(float(d.E_mech_unserved.iloc[-1] / 1000)),
            e_elec_kj=sig(float(d.E_elec_ref_unserved.iloc[-1] / 1000)) if "E_elec_ref_unserved" in d else None,
            e_cu_kj=sig(float((d.dP_cu.sum() * tau) / 1000)),
            torque_gap_rms=sig(float(np.sqrt(((d.T_cmd - d.T_elec) ** 2).mean()))),
            meas_gap=dict(i_q=sig(float(np.sqrt((d.gap_i_q ** 2).mean()))), omega=sig(float(np.sqrt((d.gap_omega ** 2).mean()))),
                          vdc=sig(float(np.sqrt((d.gap_vdc ** 2).mean())))),
            sat_pct=sig(float(100 * d.saturated.mean())))
        print(key, "total rms gap", out[key]["total_rms"], "E shaft", out[key]["e_shaft_kj"], "E elec", out[key]["e_elec_kj"])
    return out

# ---- 3. data trust -------------------------------------------------------------------------------------------------------
def trust():
    a = pd.read_csv(ROOT / "data/audit/dataset_audit.csv")
    fam = []
    for f, g in a.groupby("family"):
        fam.append(dict(family=f, files=len(g), rows=int(g.rows.sum()),
                        monotone=int(g.t_monotone.astype(bool).sum()), finite=int(g.nonfinite_cols.isna().sum()),
                        cat_missing=int((g.undocumented_cols.fillna(0)>0).sum()), derivative_ok=int(g.derivative_valid.astype(bool).sum())))
    return dict(families=fam, files=len(a), catalog=len({c[0] for c in CATALOG}))

payload = dict(headline=headline(), mismatch=mismatch(), trust=trust())
tpl = (ROOT / "dashboard/dash2_template.html").read_text()
html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
(ROOT / "dashboard/dashboard2.html").write_text(html)
print("wrote dashboard2.html", f"{len(html)/1e3:.0f} kB")
