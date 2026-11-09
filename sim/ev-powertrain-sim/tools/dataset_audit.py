"""Audit of the extractable dataset: schema vs catalog, time grid, finiteness, ranges, physics identities, ledger closure.
Run: python tools/dataset_audit.py   -> data/audit/dataset_audit.csv"""
import sys, glob, json, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import telemetry as T
from plant import GEMPlant
from battery import Battery
import run_cycle as RC
from cycles import fit_vehicle_to_motor
import run_cycle_matrix as RM

NAMES = {c[0] for c in T.CATALOG}
PARTS = ["ctrl_clip", "adv", "delay", "rot", "plant_clip", "inv"]
IDENT = ["resid_du_power", "P_balance_resid", "resid_elec_decomp", "resid_req_decomp", "resid_chain", "resid_bus"]
P_DEFAULT = T.params_from_plant(GEMPlant(battery=Battery()))


def cycle_params(kind):
    src, tmax = RM.CYC[kind]
    cyc = RC.ensure_standing_start(RC._load(src)); keep = cyc.t <= tmax
    cyc.t, cyc.v = cyc.t[keep], cyc.v[keep]; cyc.grade = None if cyc.grade is None else cyc.grade[keep]
    veq, _ = fit_vehicle_to_motor(cyc, T_cap=RC.T_CAP)
    return T.params_from_plant(GEMPlant(battery=Battery(), **veq.plant_kwargs()))


def audit(path, family, P):
    raw = pd.read_parquet(path); r = dict(file=pathlib.Path(path).name, family=family, rows=len(raw))
    t = raw.t.values; dt = np.diff(t)
    r.update(t_monotone=bool((dt > 0).all()), dt_med=float(np.median(dt)), dt_dev=float(np.abs(dt - np.median(dt)).max() / np.median(dt)))
    r["nonfinite_cols"] = ",".join(c for c in raw.columns if raw[c].dtype.kind in "fi" and not np.isfinite(raw[c].values).all() and not np.isnan(raw[c].values).all())[:80]
    r["allnan_cols"] = ",".join(c for c in raw.columns if raw[c].dtype.kind in "fi" and np.isnan(raw[c].values).all())[:80]
    try:
        d = T.derive(raw, P)
    except Exception as e:
        r["derive_error"] = str(e)[:80]; return r
    r["derivative_valid"] = bool(d.attrs.get("derivative_channels_valid"))
    r["undocumented_cols"] = len(set(d.columns) - NAMES); r["catalog_missing"] = ",".join(sorted(NAMES - set(d.columns)))[:100]
    r["max_abs_i"] = float(np.hypot(raw.i_d, raw.i_q).max()); r["min_Vdc"] = float(raw.V_dc.min()) if "V_dc" in raw else np.nan
    if "soc" in raw: r["soc_range_ok"] = bool(raw.soc.between(0, 1).all())
    scale = float(np.abs(1.5 * (raw.u_d_app * raw.i_d + raw.u_q_app * raw.i_q)).max())
    for k in IDENT:
        if k in d: r[k + "_rel"] = float(np.nanmax(np.abs(d[k].values)) / max(scale, 1.0))
    if "d_u_inv_d" in raw:
        r["ledger_close"] = float(max(np.abs(sum(raw[f"d_u_{p}_{a}"] for p in PARTS) - raw[f"d_u_{a}"]).max() for a in "dq"))
        f = np.hypot(raw.u_d_app - raw.d_u_inv_d, raw.u_q_app - raw.d_u_inv_q); r["u_bound"] = float((f / raw.u_limit).max())
    return r


def main(extra=()):
    rows = []
    for f in sorted(glob.glob("runs/*.parquet")):
        if f.endswith("_measured.parquet"): continue
        rows.append(audit(f, "legacy", P_DEFAULT))
    for f in extra: rows.append(audit(f, "fullrate", P_DEFAULT))
    for f in sorted(glob.glob("data/matrix/runs/*.parquet")): rows.append(audit(f, "matrix_fw", P_DEFAULT))
    for k in ("udds", "tesla"):
        Pc = cycle_params(k)
        for f in sorted(glob.glob(f"data/cycles/runs/{k}_*.parquet")): rows.append(audit(f, "cycle_fw", Pc))
    A = pd.DataFrame(rows); A.to_csv("data/audit/dataset_audit.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(A.groupby("family").size().to_dict())
    print("time: non-monotone files", int((~A.t_monotone).sum()), "| max grid deviation", A.dt_dev.max().round(4))
    print("non-finite cols:", sorted(set(",".join(A.nonfinite_cols.fillna("")).split(",")) - {""}))
    print("all-NaN cols by family:"); print(A.groupby("family").allnan_cols.agg(lambda s: sorted(set(",".join(s.fillna("")).split(",")) - {""})).to_string())
    print("undocumented columns by family:", A.groupby("family").undocumented_cols.agg(["min", "max"]).to_dict("index"))
    print("catalog columns that derive() does not produce:", sorted(set(",".join(A.catalog_missing.fillna("")).split(",")) - {""}))
    print("derive errors:", A[A.get("derive_error").notna()][["file", "derive_error"]].values.tolist() if "derive_error" in A else "none")
    rel = [c for c in A.columns if c.endswith("_rel")]
    print("identity residual / peak electrical power (max over files) by family:"); print(A.groupby("family")[rel].max().T.round(5).to_string())
    print("ledger closure max:", A.ledger_close.max(), "| u_bound max:", A.u_bound.max())
    print("derivative channels valid:", A.groupby("family").derivative_valid.mean().to_dict())


if __name__ == '__main__':
    main(sys.argv[1:])
