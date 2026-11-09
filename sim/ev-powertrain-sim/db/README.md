# db/: use the data from SQL

Three options, easiest first. All read the same files through `dataset.py`, so the tables are identical.

## 1. Local DuckDB file (no server, no password) — TESTED
```bash
pip install duckdb
python db/build_local_db.py                      # writes data/powertrain.duckdb (about 360 MB, 10 s)
python -c "import duckdb; duckdb.connect('data/powertrain.duckdb', read_only=True).sql('select scenario, controller, rows from runs limit 5').show()"
```
Tables: `catalog` (unit and meaning of every column), `runs` (one row per stored run, with the manifest json), `steps` (every logged step of every run, all catalog columns, sorted by `run_id, k`), `metrics_*` (per-run summary tables), `tum_samples`. Example: `select k, omega_m, i_q from steps where run_id = 13 and k between 100 and 200`.

## 2. Postgres or TimescaleDB in Docker — loader dry-run TESTED, live load NOT yet tested
You lost your old password; this makes a NEW database, so you do not need it.
* Windows: `powershell -ExecutionPolicy Bypass -File db\setup.ps1`    Mac/Linux: `bash db/setup.sh`
  It asks for a new password, starts the container (`docker-compose.yml`), loads everything with `load_postgres.py --timescale`.
* Manual: copy `.env.example` to `.env`, set the password, `docker compose -f db/docker-compose.yml --env-file db/.env up -d`, set `PGURL`, run `python db/load_postgres.py [--timescale]`.
* Postgres column names are lower case (`t_elec`, `v_dc`); `catalog.pg_column` gives the mapping. Plain Postgres works without `--timescale`.
* Verify after loading: `select count(*) from steps;` should equal the sum of `runs.rows` (371,512 with the current files).

## 3. Just files
Parquet files plus `data/catalog.csv`: `pd.read_parquet("data/matrix/runs/sag_awg_s0.parquet")` (already has every column). Sharding by run (one file per run, as now) is what to keep; there is no need for sharded weights-style formats at this size.

## Files and benchmarks
`frames.py` builds the tables, `build_local_db.py`, `load_postgres.py`, `schema.sql` (hand-written metadata schema for a long-lived server), `pg_bench.py` (layout benchmark for a live Postgres, untested), `docker-compose.yml`, `.env.example`. Why this design and the DuckDB timings: `docs/STORAGE_PLAN.md`.
