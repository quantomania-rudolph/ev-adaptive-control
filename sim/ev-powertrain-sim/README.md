# EV powertrain sim: toy PMSM drive with a realistic information layer

A toy electric-vehicle motor drive (Gym-Electric-Motor PMSM, our own battery and DC link, our own sensors, inverter and BMS imperfections) driven by an adaptive controller that only sees imperfect measurements. Built for ENGR110H (adaptive control under input and state constraints). The product is a **dataset you can extract and trust**, not a perfect controller.

## Get data in 60 seconds
```bash
python -m venv env && env/bin/pip install -r requirements.txt        # pinned versions; add duckdb and "psycopg[binary]" for the database options
python - <<'PY'
import dataset
print(dataset.index())                              # every stored run file: scenario, controller, rows, rate, validity, path
df = dataset.load_run("matrix/sag_awg_s0")          # one run, every catalog column (179 here; 184 for cycle runs)
print(dataset.catalog().head())                     # name, unit, group, role, description for every column
print(dataset.metrics("matrix_battery").head())     # one row per run, all 20 seeds
PY
```
Prefer SQL? `python db/build_local_db.py` writes one DuckDB file (no server). Prefer Postgres/TimescaleDB? See `db/README.md`.

## Where is each piece of information?
| I want | Columns (or file) | Where |
|---|---|---|
| Voltage the controller asked for vs the voltage delivered | `u_d_cmd, u_q_cmd` vs `u_d_app, u_q_app`; gap `d_u_d, d_u_q`; six exact pieces `d_u_{ctrl_clip,adv,delay,rot,plant_clip,inv}_{d,q}` | any run file |
| Shaft torque asked for vs delivered | `T_cmd` vs `T_elec`; gap `torque gap` in metrics (`torque_gap_rms`) | any run file |
| Energy gaps | `E_mech_unserved`, `E_elec_ref_unserved`, `E_req`, `E_del` | any run file |
| Exact two-state model (speed, q-current) | `ts_f1, ts_g1, ts_d1, ts_f2, ts_g2, ts_d2`, residuals `ts_res_*` | `docs/TWO_STATE_MODEL.md` |
| Tracking errors | `ts_z1 = omega_m - omega_ref`, `ts_z2 = i_q - iq_ref` (also `e_speed`, `e_q`) | any run file |
| Uncertain parameters | what the controller believed: `belief_*` columns and each run's `.json` (`beliefs`, `belief_draws`); ranges `data/params.json` -> `belief_ranges`; true values `data/params.json` -> `plant_true` | `data/` |
| Disturbance and its bounds | `T_dist, T_roll, T_visc, T_aero, d_w, d_iq, ts_d1, ts_d2`; measured maxima `data/disturbance_bounds.csv` | `docs/TWO_STATE_MODEL.md` |
| Actuator saturation | `u_limit` (true ceiling), `sat_ctrl`, `sat_plant`, `saturated`, ledger pieces | any run file |
| What the sensors reported vs truth | `meas_*`, `gap_*` | firewalled runs |
| Units, roles, meaning of every column | `data/catalog.csv` (or `dataset.catalog()`) | `data/` |
| Every assumption and default setting | `docs/ASSUMPTIONS.md`, `data/params.json` | `docs/` |
| All seeds, per-run summary numbers | `data/matrix/*_metrics.csv`, `data/cycles/*_metrics.csv`, `data/sensitivity/metrics.csv` | `data/README.md` |
| Real drive inputs (TUM Tesla traces, UDDS) | `data/inputs/` (checksummed) | `data/inputs/README.md` |
| The math | `docs/MATH_REFERENCE.pdf` | `docs/` |
| Storage design, database | `docs/STORAGE_PLAN.md`, `db/` | `db/README.md` |

## Folders
| Folder | What | README |
|---|---|---|
| `*.py` (repo root) | the simulator: plant, battery, sensors, inverter, BMS, controllers, telemetry, run scripts, `dataset.py` | below |
| `data/` | all data products and inputs | `data/README.md` |
| `docs/` | plan, contracts, math, audits, dataset notes | `docs/README.md` |
| `tests/` | regression and coverage tests | `tests/README.md` |
| `tools/` | export, audit and benchmark scripts | `tools/README.md` |
| `db/` | local DuckDB and Postgres/Timescale setup | `db/README.md` |
| `dashboard/` | two HTML dashboards and their audits | `dashboard/README.md` |
| `probes/`, `viewer/`, `runs/`, `runs_fw/` | early evidence scripts, early viewer, legacy raw runs | `data/README.md` (legacy section) |

## Source files at a glance
`plant.py` GEM PMSM with disc voltage limit and ledger rows. `battery.py` pack and DC link. `sensors.py`, `inverter.py`, `bms.py` the imperfect information path. `interfaces.py` frozen config and measurement objects. `control_fw.py` the controller (sees `Measurements` only). `firewalled_loop.py` / `firewalled_cycle.py` run loops. `telemetry.py` the data dictionary (`CATALOG`) and `derive()`. `cycles.py`, `battery_fit.py` TUM and UDDS loading and fitting. `bounds.py`, `mismatch_map.py` measurements of boundedness and mismatch. `run_*.py` experiment runners, `analyze*.py` tables. The original Tier 1 notes are in `docs/LEGACY_TIER1_README.md`.

## Reproduce and verify
```bash
pytest -q -m "not slow"                  # fast gate
pytest -q                               # everything (several minutes)
python tools/dataset_audit.py           # checks every stored file: schema, time axis, ledger closure, energy identities
```
Full regeneration: `python run_matrix_fw.py` (plain and battery matrix), `python run_cycle_matrix.py`, `python tools/regen_seed0.py` (stored seed-0 rows), `python tools/export_enriched.py` then `python tools/export_catalog.py`.

## Honest limits (read before using)
Single-run evidence, not proofs. Battery curves, sensor, inverter and BMS values are assumed, not measured on hardware. The Tesla trace asks for up to 1.5x the toy motor's torque by design. Derivative-based columns are NaN in the stored thinned files and valid only in `data/audit/fr_*` (0.1 ms). Full list: `data/README.md` and `docs/DATASET_README.md`.

## Not in this zip (size limit)
The first-generation `runs/` (386 MB), `runs_fw/` (97 MB) and the superseded `data/cycles_v1_dchatter/` (86 MB, "do not use") are left out to keep the archive under 500 MB. Everything current (`data/matrix`, `data/cycles`, `data/audit`, `data/inputs`) is included. Tests and scripts that read those legacy folders (`analyze*.py`, `plot_*.py`, `tests/test_pipeline.py`, skipped automatically by `conftest.py`) will not work without them; they only reproduce the early plots.
