"""Add T_req_data / T_avail / data_ratio to existing firewalled cycle parquets (pure functions of the cycle and the vehicle fit, so
no re-run is needed). Same formulas as firewalled_cycle.py."""
import sys, pathlib, glob; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import run_cycle as RC, run_cycle_matrix as RM
from cycles import fit_vehicle_to_motor, data_demand
from plant import GEMPlant
for kind in ("udds", "tesla"):
    src, tmax = RM.CYC[kind]
    cyc = RC.ensure_standing_start(RC._load(src)); keep = cyc.t <= tmax
    cyc.t, cyc.v = cyc.t[keep], cyc.v[keep]; cyc.grade = None if cyc.grade is None else cyc.grade[keep]
    veq, _ = fit_vehicle_to_motor(cyc, T_cap=RC.T_CAP)
    plant = GEMPlant(**veq.plant_kwargs()); dd = data_demand(cyc, veq, J_rotor=plant.J - veq.mass / veq.k ** 2)
    for f in glob.glob(f"data/cycles/runs/{kind}_*.parquet"):
        d = pd.read_parquet(f)
        k = np.clip(np.searchsorted(cyc.t, d.t.values, side="right") - 1, 0, len(cyc.t) - 2)
        for key, col in (("T_req", "T_req_data"), ("T_avail", "T_avail"), ("ratio", "data_ratio")): d[col] = dd[key + "_seg"][k]
        d.to_parquet(f); print(f, "ok", float(d.data_ratio.max()))
