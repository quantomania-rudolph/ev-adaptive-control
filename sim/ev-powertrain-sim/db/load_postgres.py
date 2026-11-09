"""Load everything into Postgres (optionally TimescaleDB).  UNTESTED against a live server; --dry-run is tested.

    set PGURL=postgresql://postgres:<password>@localhost:5432/powertrain          (bash: export PGURL=...)
    python db/load_postgres.py                # plain Postgres: tables + btree primary key (run_id, k)
    python db/load_postgres.py --timescale    # steps becomes a hypertable on run_id, compressed (segmentby run_id, orderby k)
    python db/load_postgres.py --dry-run      # build the tables and print the SQL, touch no server

Learning note: CSV COPY streams a whole table in one statement (about 100 times faster than row-by-row INSERT). NaN is written as NULL."""
import sys, io, os, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, frames

PG_TYPE = {"f": "double precision", "i": "bigint", "u": "bigint", "b": "boolean", "O": "text"}


def ddl(name: str, df: pd.DataFrame, pk=None) -> str:
    cols = ",\n  ".join(f'"{c}" {PG_TYPE.get(df[c].dtype.kind, "text")}' for c in df.columns)
    return f'CREATE TABLE "{name}" (\n  {cols}' + (f",\n  PRIMARY KEY ({pk})" if pk else "") + "\n);"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--timescale", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int); a = ap.parse_args(argv)
    tables = frames.build(max_runs=a.max_runs)
    # Postgres folds unquoted names to lower case, so mixed-case columns (T_elec, V_dc) would need quotes in every query. Lower-case the
    # column names of steps and the TUM table; catalog.pg_column maps each original name to its Postgres name.
    tables["catalog"]["pg_column"] = tables["catalog"].name.str.lower()
    for n in ("steps", "tum_samples"):
        tables[n].columns = [c.lower() for c in tables[n].columns]
        assert tables[n].columns.is_unique, f"lower-casing made duplicate columns in {n}"
    pks = {"catalog": "name", "runs": "run_id", "steps": "run_id, k"}
    statements = {n: ddl(n, df, pks.get(n)) for n, df in tables.items()}
    if a.dry_run:
        for n, s in statements.items(): print(f"-- {n}: {len(tables[n]):,} rows\n{s[:300]}{' ...' if len(s) > 300 else ''}\n")
        return statements
    import psycopg
    with psycopg.connect(os.environ["PGURL"], autocommit=True) as conn:
        if a.timescale: conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        for n, df in tables.items():
            conn.execute(f'DROP TABLE IF EXISTS "{n}" CASCADE'); conn.execute(statements[n])
            if n == "steps" and a.timescale:
                conn.execute("SELECT create_hypertable('steps', 'run_id', chunk_time_interval => 10)")
                conn.execute("ALTER TABLE steps SET (timescaledb.compress, timescaledb.compress_segmentby='run_id', timescaledb.compress_orderby='k')")
            with conn.cursor().copy(f'COPY "{n}" FROM STDIN WITH (FORMAT csv, NULL \'\')') as cp:
                for i in range(0, len(df), 20000):
                    cp.write(df.iloc[i:i + 20000].to_csv(index=False, header=False, na_rep=""))
            print(f"{n:28s} {len(df):>9,} rows", flush=True)
        conn.execute('CREATE INDEX runs_lookup ON runs (family, scenario, controller, seed)')
        if a.timescale: conn.execute("SELECT compress_chunk(c) FROM show_chunks('steps') c")
        conn.execute("ANALYZE")
    return statements


if __name__ == "__main__":
    main()
