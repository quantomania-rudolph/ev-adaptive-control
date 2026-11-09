"""One-file local database (DuckDB): no server, no password.    python db/build_local_db.py [out.duckdb]
Then:   import duckdb; con = duckdb.connect("data/powertrain.duckdb", read_only=True)
        con.sql("SELECT scenario, controller, count(*) FROM runs GROUP BY ALL").show()
        con.sql("SELECT k, omega_m, i_q FROM steps WHERE run_id = 13 AND k BETWEEN 100 AND 200").df()
Learning note: rows are stored sorted by (run_id, k); that order, not an index, is what makes run and window reads fast (docs/STORAGE_PLAN.md)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import duckdb, frames


def main(out=None, max_runs=None):
    out = pathlib.Path(out or frames.ROOT / "data/powertrain.duckdb"); out.unlink(missing_ok=True)
    tables = frames.build(max_runs=max_runs)
    con = duckdb.connect(str(out))
    for name, df in tables.items():
        con.register("t_src", df); con.execute(f"CREATE TABLE {name} AS SELECT * FROM t_src"); con.unregister("t_src")
        print(f"{name:28s} {len(df):>9,} rows  {df.shape[1]:>4} cols")
    con.execute("CREATE VIEW run_summary AS SELECT r.run_id, r.name, r.scenario, r.controller, r.rows, r.rate_ms, r.derivative_valid FROM runs r")
    con.close(); print("wrote", out, f"{out.stat().st_size/1e6:.0f} MB")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
