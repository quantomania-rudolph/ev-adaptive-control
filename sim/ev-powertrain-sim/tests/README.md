# tests/: what is protected
Run: `pytest -q -m "not slow"` (fast gate), `pytest -q` (everything, several minutes), one file: `pytest -q tests/test_telemetry.py`. Real-track cycle tests read `data/inputs/ds_tum` (set `TUM_DIR` to use another folder). The slow marker is for the drive-cycle runs (about 1 min each).

| File | Protects |
|---|---|
| `test_plant.py` | the GEM facts the project depends on (action space, clipping, termination, determinism, ledger rows) |
| `test_sensors.py`, `test_inverter.py`, `test_bms.py` | unit checks U1 to U19 of each information-path block, plus mutation tests that break the code on purpose |
| `test_realism_firewall.py`, `test_cov_firewall_loop.py` | the information firewall: the controller sees `Measurements` only; hidden state cannot change its output; ledger closes; determinism; unstable d-loop and mismatched time steps are refused |
| `test_cov_controller_fw.py` | every config field and every branch of the controller (modes, governor, clamp, d-gain, BMS-aware), against hand-computed formulas; invalid mode rejected |
| `test_telemetry.py`, `test_cov_telemetry_edges.py` | catalog equals what `derive()` produces; physics identities; two-state identities; edge inputs (short, missing columns, non-uniform time) |
| `test_dataset_access.py`, `test_db_frames.py` | `dataset.py`, `data/catalog.csv`, `data/params.json`, the stored files, and the DuckDB/Postgres table builders |
| `test_bounds.py`, `test_cov_mismatch_gutcheck_bounds.py` | `bounds.py`, `mismatch_map.py`, `data_gutcheck.py` on synthetic frames with known answers |
| `test_cycles.py`, `test_cov_cycles_edges.py`, `test_pipeline.py` | the many ways real speed data goes wrong; request pipeline from cycle data to `P_req_ref` |
| `test_battery_fit.py` | the TUM battery fit and its schema |
| `test_firewalled_cycle.py` | the cycle path through the firewall (slow) |
| `test_cov_tools.py` | `tools/dataset_audit.py` and `tools/disturbance_bounds.py` on tiny synthetic data (run in a throw-away copy of the repo) |
| `test_plan_coverage.py` | fails if any unit test promised in `docs/REALISM_PLAN.md` is missing |

Coverage and what is still untested (plotting scripts, experiment drivers, slow cycle runs): `docs/TEST_COVERAGE.md`. Mutation checks: `python tools/mutation_check.py`.
