"""Builds the tables (as pandas frames) shared by the DuckDB and Postgres loaders, from the files that dataset.py indexes.
Tables: catalog, runs, steps (one row per logged step, every catalog column, NULL where a run lacks that subsystem), metrics_<name>, tum_samples."""
import sys, json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import dataset

TUM_COLS = ["time", "speed", "hv_battery_current", "hv_battery_voltage", "hv_soc", "hv_temp_min", "axle_torque", "rear_motor_torque",
            "accelerator_pedal", "elevation", "ambient_air_temp"]


def build(max_runs=None, include_tum=True) -> dict:
    idx = dataset.index()
    if max_runs: idx = idx.iloc[:max_runs]
    idx = idx.reset_index(drop=True); idx.insert(0, "run_id", np.arange(len(idx), dtype="int32"))
    cat = dataset.catalog()
    steps, manifests = [], []
    for _, r in idx.iterrows():
        d = dataset.load_run(ROOT / r.path, derived=True, nan_invalid=True)
        d.insert(0, "k", np.arange(len(d), dtype="int32")); d.insert(0, "run_id", np.int32(r.run_id))
        steps.append(d)
        mp = ROOT / r.manifest if r.manifest else None
        manifests.append(json.dumps(json.loads(mp.read_text()), default=str) if mp and mp.exists() else None)
    steps = pd.concat(steps, ignore_index=True, sort=False)
    front = ["run_id", "k"] + [c for c in cat.name if c in steps.columns]
    steps = steps[front + [c for c in steps.columns if c not in front]].sort_values(["run_id", "k"], kind="stable").reset_index(drop=True)
    runs = idx.drop(columns=["manifest"]).assign(manifest=manifests)
    out = {"catalog": cat, "runs": runs, "steps": steps}
    for name in dataset.METRICS:
        out[f"metrics_{name}"] = dataset.metrics(name)
    if include_tum:
        parts = []
        for f in sorted((ROOT / "data/inputs/ds_tum").glob("track_*.csv")):
            d = pd.read_csv(f, usecols=lambda c: c in TUM_COLS); d.insert(0, "track", f.stem); parts.append(d)
        out["tum_samples"] = pd.concat(parts, ignore_index=True)
    return out
