"""Postgres/TimescaleDB layout benchmark. RUN ON A MACHINE WITH THE DOCKER DATABASE UP. UNTESTED (build sandbox has no Postgres).

    pip install "psycopg[binary]" pandas pyarrow numpy
    set PGURL=postgresql://postgres:<password>@localhost:5432/powertrain      (Windows: setx / $env:PGURL=...)
    python db/pg_bench.py data/audit/fr_battery_sag_awg.parquet data/audit/fr_plain_hidden_aw.parquet

Layouts: A plain table, primary key (run_id,k).  B plain table + BRIN on (run_id,k), loaded in order.
         C Timescale hypertable partitioned by run_id (integer dimension), then compressed (segmentby run_id, orderby k).
Same questions as tools/storage_bench.py (DuckDB), so the two tables of results are comparable.
"""
import os, sys, io, time, statistics as st
import numpy as np, pandas as pd, psycopg
URL = os.environ["PGURL"]; N_RUNS, ROWS_KEEP = 60, None
files = sys.argv[1:]; base = [pd.read_parquet(f) for f in files]
COLS = [c for c in ["t", "omega_m", "omega_ref", "i_d", "i_q", "u_d_cmd", "u_q_cmd", "u_d_app", "u_q_app", "u_limit", "T_elec", "T_cmd", "T_dist",
                    "psi_hat", "V_dc", "d_u_d", "d_u_q", "e_q", "iq_ref", "saturated"] if all(c in b.columns for b in base)]
rng = np.random.default_rng(0)
def run_frame(i):
    b = base[i % len(base)][COLS].astype("float64").copy(); v = b.drop(columns=["t"]).to_numpy()
    b.iloc[:, 1:] = v * (1 + 1e-6 * rng.standard_normal(v.shape))
    b.insert(0, "k", np.arange(len(b), dtype="int32")); b.insert(0, "run_id", np.int32(i)); return b
ddl_cols = ", ".join(f"{c} double precision" for c in COLS)
def create(conn, name, extra=""):
    conn.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
    conn.execute(f"CREATE TABLE {name} (run_id integer NOT NULL, k integer NOT NULL, {ddl_cols}{extra})")
def copy_in(conn, name, df):
    with conn.cursor().copy(f"COPY {name} FROM STDIN (FORMAT binary)") as cp:
        cp.set_types(["int4", "int4"] + ["float8"] * len(COLS))
        for row in df.itertuples(index=False, name=None): cp.write_row(row)
res = []
def timeit(layout, label, fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    res.append((layout, label, 1000 * st.median(ts))); print(f"{layout:12s} {label:28s} {1000*st.median(ts):9.1f} ms", flush=True)
with psycopg.connect(URL, autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    for lay in ("A_pk", "B_brin", "C_hyper"):
        name = f"steps_{lay.lower()}"
        create(conn, name, ", PRIMARY KEY (run_id, k)" if lay == "A_pk" else "")
        if lay == "C_hyper":
            conn.execute(f"SELECT create_hypertable('{name}', 'run_id', chunk_time_interval => 10, if_not_exists => true)")
            conn.execute(f"ALTER TABLE {name} SET (timescaledb.compress, timescaledb.compress_segmentby='run_id', timescaledb.compress_orderby='k')")
        t0 = time.perf_counter()
        for i in range(N_RUNS): copy_in(conn, name, run_frame(i))             # one run per COPY = one episode per transaction
        res.append((lay, "bulk_load_ms", 1000 * (time.perf_counter() - t0)))
        if lay == "B_brin": conn.execute(f"CREATE INDEX ON {name} USING brin (run_id, k)")
        if lay == "C_hyper":
            t0 = time.perf_counter(); conn.execute(f"SELECT compress_chunk(c) FROM show_chunks('{name}') c"); res.append((lay, "compress_ms", 1000 * (time.perf_counter() - t0)))
        conn.execute(f"ANALYZE {name}")
        size = conn.execute(f"SELECT pg_total_relation_size('{name}')" if lay != "C_hyper" else f"SELECT hypertable_size('{name}')").fetchone()[0] / 1e6
        res.append((lay, "size_MB", size)); print(lay, "size MB", round(size))
        rid = 37; r = np.random.default_rng(1)
        timeit(lay, "Q1 one whole run", lambda: conn.execute(f"SELECT * FROM {name} WHERE run_id={rid}").fetchall())
        timeit(lay, "Q2 1000-row window", lambda: conn.execute(f"SELECT * FROM {name} WHERE run_id={rid} AND k BETWEEN 30000 AND 31000").fetchall())
        timeit(lay, "Q3 one column, all runs", lambda: conn.execute(f"SELECT run_id, sqrt(avg(d_u_q*d_u_q)) FROM {name} GROUP BY run_id").fetchall(), 3)
        def batch():
            for _ in range(64):
                i, k0 = int(r.integers(N_RUNS)), int(r.integers(len(base[0]) - 256))
                conn.execute(f"SELECT omega_m,i_q,u_q_cmd FROM {name} WHERE run_id={i} AND k BETWEEN {k0} AND {k0+255}").fetchall()
        timeit(lay, "Q4 64 random 256-row windows", batch, 3)
        new = run_frame(N_RUNS); t0 = time.perf_counter(); copy_in(conn, name, new)
        res.append((lay, "append_one_run_ms", 1000 * (time.perf_counter() - t0))); conn.execute(f"DELETE FROM {name} WHERE run_id={N_RUNS}")
pd.DataFrame(res, columns=["layout", "test", "value"]).to_csv("data/storage_bench_postgres.csv", index=False)
