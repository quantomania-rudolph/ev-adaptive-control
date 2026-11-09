# data/: every stored piece of information

Open everything through `dataset.py` at the repo root (`index()`, `load_run()`, `catalog()`, `params()`, `metrics()`); this page says what is behind it.

## Run files (time series, one row per logged step)
| Folder | Content | Rows per file | Row spacing | Columns | Derivative columns valid? |
|---|---|---|---|---|---|
| `matrix/runs/{scenario}_{controller}_s0.parquet` | 5 scenarios x 4 controllers, seed 0 | 4,706 | 1.7 ms (every 17th step) | 156 (plain: no battery) / 179 (battery) | no (NaN) |
| `cycles/runs/{udds,tesla}_{naive,awg}_s0.parquet` | drive-cycle runs, seed 0 | 26,087 / 32,609 | 4.6 ms (every 23rd step) | 184 | no (NaN) |
| `audit/fr_*.parquet` (+ `.json` manifest) | full-rate runs for checking identities | 80,000 | 0.1 ms | 156 / 179 | yes |
Scenarios: `base` no event; `hidden` unknown 25 V ceiling for t = 4.5 to 5.5 s; `oracle` same event, ceiling known to the controller; `sag` battery current clamp (BMS derate) so the bus voltage falls; `sagbase` battery, no fault. Controllers: `naive` adaptive law only; `robust` sigma-modification and projection; `aw` anti-windup; `awg` anti-windup + current governor + bus-voltage and BMS awareness. The stored files already contain every derived column (model terms, two-state, errors, energies). Why strides 17 and 23: `docs/DATASET_README.md`.

Column counts: plain matrix and audit runs have 156 columns (battery columns absent); battery matrix runs 179; cycle runs 184. Cycle runs add 5 cycle-only columns on top of the battery set: `T_avail`, `T_req_data`, `data_infeasible`, `data_ratio`, `v_ref_mps`. `dataset.load_run` returns only the columns a file has (no NaN padding); the catalog `group` says which family a column belongs to.

Reading a column: its unit and meaning are in `data/catalog.csv`. Role says who may know it: `state` is plant truth (the controller never sees it), `measured` is what the controller received, `error` is measured minus truth, `belief` is the controller's wrong constants, `internal` controller state, `mismatch`/`ledger` requested-vs-delivered gaps.

## Per-run summary tables (one row per run, all seeds)
| File | Rows | Key columns |
|---|---|---|
| `matrix/plain_metrics.csv`, `matrix/battery_metrics.csv` | 20 seeds x scenarios x controllers | `envelope_ok` (current stayed below 275 A and the flux estimate below 0.15 Wb), `ultimate_psi_err`, `speed_rms_post`, `E_mech_unserved_kj`, `hidden_sat_rms`, `inv_err_rms`, `torque_gap_rms`, `gap_iq_rms`, `beliefs` (the draws) |
| `cycles/udds_metrics.csv`, `cycles/tesla_metrics.csv` | 5 seeds x 2 controllers | per run: `cycle` (not `scenario`), `controller`, `seed`, `speed_rms`, `speed_rms_derate`, `speed_max_abs`, `psi_*`, `i_max`, `u_bound_true`, `sat_plant_frac`, `sat_ctrl_frac`, `V_dc_min`, `soc_end`, `gap_omega_rms`, `gap_iq_rms`, `inv_err_rms`, `derate_rows`. No `envelope_ok` or `torque_gap_rms` |
| `sensitivity/metrics.csv` | 3 seeds per level, controllers `naive` and `aw` only (`*_dgl` params use the d-gain-by-L fix) | `param`, `level`: one belief or sensor setting swept at a time |
| `matrix_clamp/hidden_clamp_metrics.csv` | aw with and without the flux clamp | controllers `awc` (aw + flux clamp) and `awgc` (aw + governor + bus-voltage awareness + clamp), scenario `hidden` only |
| `disturbance_bounds.csv` | seed-0 maxima, p99.9, RMS of each disturbance piece | `d1_*`, `d2_*`, `du_q_unmod_V_*` |
| `mismatch_map.csv/.json` | gaps per run as % of natural scale | `mismatch`, `rms_pct`, `peak_pct` |

## Dictionary, parameters, indexes
`catalog.csv` (184 columns: name, unit, group, role, description). `params.json` (true plant values with units, limits, default controller, belief ranges, sensor/inverter/BMS settings, battery defaults). `audit/dataset_audit.csv` (result of the last audit). `gutcheck/` (held-value scan of TUM and sim data, `docs/DATA_GUTCHECK.md`). `fixtures/` (frozen battery-fit and checksum used by tests).

## Inputs
`inputs/` UDDS schedule and five TUM tracks (checksummed, CC BY 4.0). `tum_subset/` the battery-related columns of those tracks. `tum_battery_fit.json` the fitted pack (resistance scale, capacity, current limits). See `inputs/README.md`, `docs/BATTERY_FIT.md`.

## Legacy and superseded (read-only, kept for traceability)
| Folder | What | Use it? |
|---|---|---|
| `../runs/` (53 parquet) and `runs_all_downsampled/`, `runs_500hz/`, `runs_250hz/` (currently empty), `runs_battery_500hz/` | first-generation experiments A to Dg, Eb, Eg and drive-cycle class runs (`H*,M*,T0*,U*,Y*`), from `control.py` before the information firewall; raw columns, derive at load | only to reproduce the early plots; `viewer/` (see `viewer/README.md`) and `probes/` (see `probes/README.md`) are early tooling for them |
| `../runs_fw/` | early firewalled probe runs | no |
| `cycles_v1_dchatter/` | first cycle matrix, contaminated by d-axis chatter (fixed by `d_gain_by_L`) | **no, do not use** |
| `summary_AtoD.txt` | output of `analyze.py` on the first generation | no |
