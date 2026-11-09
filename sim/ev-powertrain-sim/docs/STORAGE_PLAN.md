# Storage plan for episodes and training runs (2026-11-09)

## Measured here (DuckDB 2 cores, 4.8 M rows x 22 columns = 60 runs x 80 k steps; `python tools/storage_bench.py <scratch dir>`, results in `data/storage_bench_duckdb.csv`)
| Layout | whole run | 1000-row window | one column, all runs | 64 random 256-row windows | append one 80 k-row run |
|---|---|---|---|---|---|
| table, insertion order | 25 ms | 2.9 ms | 18.6 ms | 80 ms | 61 ms |
| table, sorted by (run_id, k) | 16 ms | 2.1 ms | 18.4 ms | 85 ms | 157 ms |
| table, sorted + ART index | 19 ms | 1.9 ms | 18.4 ms | 82 ms | 80 ms |
| one Parquet file, sorted | 46 ms | 28 ms | 41 ms | 544 ms | – |
| one Parquet file per run | 50 ms | 23 ms | 61 ms | 321 ms | 99 ms |
Sizes: about 660 MB per table copy, 642 MB as one Parquet file (ZSTD). Bulk load of all 4.8 M rows: 2.3 to 3.1 s (Parquet per run: 7.9 s, because 60 small files).
TUM (5 tracks, 35,780 rows): load 0.5 s, 60 s slice 0.7 ms, per-track aggregate 1.4 ms.

## What this says
* At this size nothing is slow. Every access pattern is under 100 ms in a native table, so tuning is about habits for when it grows, not rescue.
* Storing rows sorted by (run_id, k) matters; a secondary index did not help and cost 25 % more load time. Index the small metadata tables, not the step table.
* Parquet is 3 to 10 times slower for many small random reads (each query opens a file and reads footers) but is the same size and is the easiest thing to hand to a teammate or a training job.
* Noise added to replicated runs makes compression ratios pessimistic, not optimistic.
* Limits of this test: 20 data columns not 184; two distinct base runs replicated; DuckDB, not Postgres.

## Recommendation (to revise after the Postgres numbers exist)
1. **Metadata and results in Postgres** (`db/schema.sql`): training_runs, runs (manifest and metrics as jsonb, GIN index), catalog, tum_samples. Small tables, many ways to query them.
2. **Step telemetry as Parquet per run**, written once and never edited, read with DuckDB or pandas. Keep `run_id` and the file path in the `runs` table.
3. **Only if several programs must write and query steps live**, add the Timescale hypertable (`db/pg_bench.py` layout C: partitioned by run_id, compressed, segmentby run_id, orderby k). Expected benefit is compression and SQL access; speed at this scale will not be the reason.

## Not done / untested
* Nothing in `db/` has run against a live server: the build sandbox has no Postgres and no TimescaleDB, and the link to your computer was not connected when I tried. `db/docker-compose.yml` starts a fresh database with a password you choose; `db/pg_bench.py` then produces `data/storage_bench_postgres.csv` in the same shape as the DuckDB table.
* `pg_bench.py` loads with a Python row loop, so its load times include Python overhead; compare layouts with each other, not with DuckDB's load times.
* Docker image tag `2.17.2-pg16` is from memory; check it exists before the first start.
