# tools/: scripts that make or check data
| Script | What it does | When to run |
|---|---|---|
| `export_enriched.py` | adds every derived column to stored run files; derivative columns are NaN when rows are > 0.4 ms apart | after regenerating runs |
| `export_catalog.py` | writes `data/catalog.csv` and `data/params.json` | after editing `telemetry.CATALOG` or any default config |
| `make_assumptions.py` | generates `docs/ASSUMPTIONS.md` from the dataclass defaults and their line comments | after changing any default |
| `dataset_audit.py` | checks every stored file (time axis, finite values, ledger closure, identities) -> `data/audit/dataset_audit.csv` | after any data change |
| `disturbance_bounds.py` | measured maxima of each disturbance piece -> `data/disturbance_bounds.csv` | after regenerating seed-0 runs |
| `regen_seed0.py` | regenerates the stored seed-0 runs (matrix and cycles) | slow (about 30 min on 2 cores) |
| `add_cycle_data_cols.py` | adds the data-demand columns (`T_req_data`, `T_avail`, `data_ratio`) to cycle runs | after `regen_seed0.py` |
| `mutation_check.py` | breaks the code on purpose and checks the tests notice | before trusting the tests |
| `storage_bench.py` | DuckDB layout benchmark -> `data/storage_bench_duckdb.csv` | optional |
Order after regenerating runs: `regen_seed0.py` -> `add_cycle_data_cols.py` -> `export_enriched.py` -> `dataset_audit.py` -> `disturbance_bounds.py` -> `export_catalog.py`.
