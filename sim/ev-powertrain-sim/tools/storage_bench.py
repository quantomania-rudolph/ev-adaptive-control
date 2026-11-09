"""Storage layout benchmark in DuckDB (Postgres/Timescale equivalent lives in db/pg_bench.py, to be run on a machine with Postgres).

Learning note: 'fast recall' is three different jobs: (1) pull one whole run, (2) pull a time window of one run, (3) scan one column
across many runs. A layout that wins one can lose another, so each is timed separately. Run files are replicated with 1e-6 noise to
reach 60 runs; timings are realistic, compression ratios are pessimistic (the noise makes mantissas incompressible).
"""
import sys, time, json, pathlib, shutil, statistics as st
import numpy as np, pandas as pd, duckdb
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
W = pathlib.Path(sys.argv[1]); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
COLS = ["t", "omega_m", "omega_ref", "i_d", "i_q", "u_d_cmd", "u_q_cmd", "u_d_app", "u_q_app", "u_limit", "T_elec", "T_cmd", "T_dist",
        "psi_hat", "V_dc", "d_u_d", "d_u_q", "e_q", "iq_ref", "saturated"]
base = [pd.read_parquet(ROOT / f"data/audit/{s}.parquet") for s in ("fr_battery_sag_awg", "fr_plain_hidden_aw")]
base = [b[[c for c in COLS if c in b.columns]].astype("float64") for b in base]
COLS = list(base[0].columns); N_RUNS, ROWS = 60, len(base[0])
rng = np.random.default_rng(0)
def run_frame(i):
    b = base[i % 2].copy(); v = b.drop(columns=["t"]).to_numpy()
    b.iloc[:, 1:] = v * (1 + 1e-6 * rng.standard_normal(v.shape))
    b.insert(0, "k", np.arange(len(b), dtype="int32")); b.insert(0, "run_id", np.int32(i)); return b
frames = [run_frame(i) for i in range(N_RUNS)]; allf = pd.concat(frames, ignore_index=True)
print(f"{len(allf):,} rows x {allf.shape[1]} cols, raw {allf.memory_usage().sum()/1e9:.2f} GB", flush=True)
res = []; con = duckdb.connect(str(W / "bench.duckdb")); con.execute("PRAGMA threads=2")
def timeit(label, layout, fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    res.append(dict(layout=layout, test=label, ms=1000 * st.median(ts))); print(f"{layout:22s} {label:26s} {1000*st.median(ts):9.1f} ms", flush=True)
def mk(name, order=None):
    t0 = time.perf_counter(); con.register("src", allf)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM src" + (f" ORDER BY {order}" if order else "")); con.unregister("src"); return time.perf_counter() - t0
layouts = {}
layouts["table_unsorted"] = ("table_unsorted", mk("table_unsorted"))
layouts["table_sorted"] = ("table_sorted", mk("table_sorted", "run_id, k"))
layouts["table_sorted+ART"] = ("table_sorted2", mk("table_sorted2", "run_id, k")); t0 = time.perf_counter()
con.execute("CREATE INDEX ix ON table_sorted2(run_id, k)"); layouts["table_sorted+ART"] = ("table_sorted2", layouts["table_sorted+ART"][1] + time.perf_counter() - t0)
for name, ro in (("parquet_1file", 122880), ("parquet_perrun", 122880)):
    t0 = time.perf_counter()
    if name == "parquet_1file":
        con.register("src", allf); con.execute(f"COPY (SELECT * FROM src ORDER BY run_id,k) TO '{W}/one.parquet' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ro})")
        con.unregister("src"); layouts[name] = (f"read_parquet('{W}/one.parquet')", time.perf_counter() - t0)
    else:
        (W / "runs").mkdir()
        for i, f in enumerate(frames): f.to_parquet(W / "runs" / f"run_{i:03d}.parquet", compression="zstd", row_group_size=ro)
        layouts[name] = (f"read_parquet('{W}/runs/*.parquet', filename=true)", time.perf_counter() - t0)
sizes = {"table_unsorted": None}
for lab, (src, ing) in layouts.items(): res.append(dict(layout=lab, test="bulk_load_s", ms=1000 * ing))
con.execute("CHECKPOINT"); db = (W / "bench.duckdb").stat().st_size / 1e6
one = (W / "one.parquet").stat().st_size / 1e6; per = sum(f.stat().st_size for f in (W / "runs").glob("*")) / 1e6
print(f"sizes MB: duckdb(3 tables)={db:.0f} one.parquet={one:.0f} per-run parquet={per:.0f}", flush=True)
rid = 37
for lab, (src, _) in layouts.items():
    if lab.startswith("parquet_perrun"):
        whole = lambda: con.execute(f"SELECT * FROM read_parquet('{W}/runs/run_{rid:03d}.parquet')").fetchnumpy()
        win = lambda: con.execute(f"SELECT * FROM read_parquet('{W}/runs/run_{rid:03d}.parquet') WHERE k BETWEEN 30000 AND 31000").fetchnumpy()
    else:
        whole = lambda s=src: con.execute(f"SELECT * FROM {s} WHERE run_id={rid}").fetchnumpy()
        win = lambda s=src: con.execute(f"SELECT * FROM {s} WHERE run_id={rid} AND k BETWEEN 30000 AND 31000").fetchnumpy()
    colscan = lambda s=src: con.execute(f"SELECT run_id, sqrt(avg(d_u_q*d_u_q)) FROM {s} GROUP BY run_id").fetchall()
    def batch(s=src):                                           # 64 random 256-step windows across runs (training-batch style)
        r = np.random.default_rng(1)
        for _ in range(64):
            i, k0 = int(r.integers(N_RUNS)), int(r.integers(ROWS - 256))
            if lab.startswith("parquet_perrun"): con.execute(f"SELECT omega_m,i_q,u_q_cmd FROM read_parquet('{W}/runs/run_{i:03d}.parquet') WHERE k BETWEEN {k0} AND {k0+255}").fetchnumpy()
            else: con.execute(f"SELECT omega_m,i_q,u_q_cmd FROM {s} WHERE run_id={i} AND k BETWEEN {k0} AND {k0+255}").fetchnumpy()
    timeit("Q1 one whole run", lab, whole); timeit("Q2 1000-row window", lab, win); timeit("Q3 one column, all runs", lab, colscan, 3); timeit("Q4 64 random 256-row windows", lab, batch, 3)
# ingest: append one new run
new = run_frame(N_RUNS); con.register("newrun", new)
for tbl in ("table_unsorted", "table_sorted", "table_sorted2"):
    t0 = time.perf_counter(); con.execute(f"INSERT INTO {tbl} SELECT * FROM newrun"); res.append(dict(layout={"table_sorted2": "table_sorted+ART"}.get(tbl, tbl), test="append_one_run", ms=1000 * (time.perf_counter() - t0)))
t0 = time.perf_counter(); new.to_parquet(W / "runs" / "run_new.parquet", compression="zstd", row_group_size=122880); res.append(dict(layout="parquet_perrun", test="append_one_run", ms=1000 * (time.perf_counter() - t0)))
# TUM tracks: long table, one row per sample
tum = ROOT / "data/inputs/ds_tum"
keep = ["time", "speed", "vehicle_speed", "hv_battery_current", "hv_battery_voltage", "hv_soc", "hv_temp_min", "axle_torque", "rear_motor_torque", "accelerator_pedal", "elevation", "ambient_air_temp"]
t0 = time.perf_counter(); parts = []
for i, f in enumerate(sorted(tum.glob("track_*.csv"))):
    d = pd.read_csv(f, usecols=lambda c: c in keep); d.insert(0, "track", f.stem); parts.append(d)
tum_df = pd.concat(parts, ignore_index=True); t_read = time.perf_counter() - t0
con.register("tumsrc", tum_df); con.execute("CREATE TABLE tum AS SELECT * FROM tumsrc ORDER BY track, time")
res.append(dict(layout="tum_table", test=f"load_{len(tum_df)}_rows", ms=1000 * (time.perf_counter() - t0)))
timeit("TUM slice 60 s of one track", "tum_table", lambda: con.execute("SELECT * FROM tum WHERE track='track_1' AND time BETWEEN 100 AND 160").fetchnumpy())
timeit("TUM mean power per track", "tum_table", lambda: con.execute("SELECT track, avg(hv_battery_current*hv_battery_voltage) FROM tum GROUP BY track").fetchall())
pd.DataFrame(res).to_csv(ROOT / "data/storage_bench_duckdb.csv", index=False)
print(f"TUM rows {len(tum_df):,}; db {db:.0f} MB")
