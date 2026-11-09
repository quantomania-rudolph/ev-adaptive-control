"""One door to every stored piece of data.  `python -c "import dataset; print(dataset.index().head())"`

    import dataset
    dataset.index()                       # table of EVERY stored run file: family, scenario, controller, seed, rows, rate, size, path
    df = dataset.load_run("matrix/sag_awg_s0")          # name from the index (or a path); all 184 catalog columns
    dataset.catalog()                     # data dictionary: name, unit, group, role, description (same as data/catalog.csv)
    dataset.params()                      # true plant, controller beliefs ranges, sensor/inverter/BMS settings (data/params.json)
    dataset.metrics("matrix_plain")       # one row per run, all seeds (matrix_plain, matrix_battery, cycles_udds, cycles_tesla, sensitivity)

Learning note: stored run files were written with only the raw logged columns; the model terms (f, g, d, two-state, residuals, energies) are
computed by telemetry.derive() from the true motor parameters. Cycle runs use a different (fitted) vehicle inertia, so the parameter set
depends on the file; params_for() picks the right one so nobody has to remember. Files written by tools/export_enriched.py already
contain every column, and load_run() then returns them untouched.
"""
import pathlib, warnings
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DERIVATIVE_COLS = ["P_mag", "resid_id", "resid_iq", "resid_w", "P_balance_resid", "resid_elec_decomp", "resid_req_decomp", "resid_bus",
                   "E_mag_store", "ts_res_w", "ts_res_iq"]          # only meaningful when rows are <= 0.4 ms apart
FAMILIES = {   # family -> (glob under data/, description)
    "matrix": ("matrix/runs/*_s0.parquet", "realistic matrix, seed 0 (plain scenarios + battery scenarios x 4 controllers)"),
    "cycles": ("cycles/runs/*_s0.parquet", "UDDS and Tesla drive cycles, seed 0 (naive and awg)"),
    "audit": ("audit/fr_*.parquet", "full-rate (0.1 ms) runs for checking identities"),
}
METRICS = {"matrix_plain": "matrix/plain_metrics.csv", "matrix_battery": "matrix/battery_metrics.csv", "cycles_udds": "cycles/udds_metrics.csv",
           "cycles_tesla": "cycles/tesla_metrics.csv", "sensitivity": "sensitivity/metrics.csv", "clamp_ab": "matrix_clamp/hidden_clamp_metrics.csv",
           "disturbance_bounds": "disturbance_bounds.csv"}


def catalog() -> pd.DataFrame:
    import telemetry
    return pd.DataFrame(telemetry.CATALOG, columns=["name", "unit", "group", "role", "description"])


def metrics(name: str) -> pd.DataFrame:
    if name not in METRICS: raise KeyError(f"unknown metrics table {name!r}; choose from {sorted(METRICS)}")
    return pd.read_csv(DATA / METRICS[name])


def params() -> dict:
    import json
    return json.loads((DATA / "params.json").read_text())


def _parse(path: pathlib.Path, family: str):
    stem = path.stem.replace("_s0", "")
    if family == "audit": return stem.replace("fr_", ""), "", 0
    scen, ctl = stem.rsplit("_", 1) if family == "matrix" else stem.split("_")
    return scen, ctl, 0


def index() -> pd.DataFrame:
    """Every stored run file. `name` is what load_run() accepts. rate_ms is the spacing of stored rows; derivative_valid says if columns in
    DERIVATIVE_COLS can be trusted (rate <= 0.4 ms)."""
    import pyarrow.parquet as pq
    rows = []
    for fam, (pat, desc) in FAMILIES.items():
        for f in sorted(DATA.glob(pat)):
            pf = pq.ParquetFile(f); t = pd.read_parquet(f, columns=["t"]).t.to_numpy()
            scen, ctl, seed = _parse(f, fam); rate = float(np.median(np.diff(t))) * 1e3 if len(t) > 1 else float("nan")
            cols = pf.schema_arrow.names
            rows.append(dict(name=f"{fam}/{f.stem}", family=fam, scenario=scen, controller=ctl, seed=seed, rows=pf.metadata.num_rows, rate_ms=round(rate, 4),
                             derivative_valid=rate <= 0.4001, n_columns=len(cols), has_derived="ts_z1" in cols, size_mb=round(f.stat().st_size / 1e6, 2),
                             path=str(f.relative_to(ROOT)), manifest=str(f.with_suffix(".json").relative_to(ROOT)) if f.with_suffix(".json").exists() else ""))
    return pd.DataFrame(rows)


def params_for(name_or_path) -> dict:
    """Motor/load parameter dict for telemetry.derive(). Cycle runs use the fitted vehicle (different inertia and road load)."""
    from plant import GEMPlant
    from battery import Battery
    import telemetry as T
    s = str(name_or_path)
    if "cycles/" in s or pathlib.Path(s).name.startswith(("udds_", "tesla_")):
        kind = "udds" if "udds_" in s else "tesla"
        import run_cycle as RC, run_cycle_matrix as RM
        from cycles import fit_vehicle_to_motor
        src, tmax = RM.CYC[kind]
        cyc = RC.ensure_standing_start(RC._load(src)); keep = cyc.t <= tmax
        cyc.t, cyc.v = cyc.t[keep], cyc.v[keep]; cyc.grade = None if cyc.grade is None else cyc.grade[keep]
        veq, _ = fit_vehicle_to_motor(cyc, T_cap=RC.T_CAP)
        return T.params_from_plant(GEMPlant(battery=Battery(), **veq.plant_kwargs()))
    return T.params_from_plant(GEMPlant(battery=Battery()))


def load_run(name_or_path, derived: bool = True, nan_invalid: bool = True) -> pd.DataFrame:
    """Load one stored run. derived=True returns all catalog columns (computing any that are missing). nan_invalid=True sets the
    derivative-based columns to NaN when the stored rows are too far apart for them to mean anything."""
    p = pathlib.Path(str(name_or_path))
    if not p.exists():
        cand = DATA / (str(name_or_path) + ("" if str(name_or_path).endswith(".parquet") else ".parquet"))
        if not cand.exists():       # allow family/name from index(): matrix/sag_awg_s0 -> data/matrix/runs/..., audit/fr_x -> data/audit/...
            fam, _, stem = str(name_or_path).partition("/")
            cand = DATA / {"matrix": "matrix/runs", "cycles": "cycles/runs", "audit": "audit"}.get(fam, fam) / (stem + ".parquet")
        p = cand
    if not p.exists(): raise FileNotFoundError(f"{name_or_path!r}: not a path and not in dataset.index()")
    df = pd.read_parquet(p)
    if derived and "ts_z1" not in df.columns:
        import telemetry as T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = T.derive(df, params_for(p))
    if nan_invalid and len(df) > 2 and float(np.median(np.diff(df["t"].to_numpy()))) > 4.001e-4:
        for c in DERIVATIVE_COLS:
            if c in df.columns: df[c] = np.nan
    return df
