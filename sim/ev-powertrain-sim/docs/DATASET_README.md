# Extractable dataset: what each file is and what you can trust (audited 2026-10-30; `python tools/dataset_audit.py`)

## Families
| Family | Where | Rate | Notes |
|---|---|---|---|
| Legacy runs (experiments, classes, cycles with the legacy controller) | `runs/*.parquet` (+ `.json` manifest) | 0.1 ms to 5 ms | 30 files audited; only 23% are fine-grained enough for derivative-based identities |
| Realistic matrix, seed 0 | `data/matrix/runs/{scenario}_{controller}_s0.parquet` | 1.7 ms (every 17th step) | derivative-based columns NOT valid |
| Realistic matrix, all seeds | `data/matrix/{plain,battery}_metrics.csv` | one row per run | 20 seeds x scenarios x controllers |
| Realistic cycles, seed 0 | `data/cycles/runs/{udds,tesla}_{naive,awg}_s0.parquet` | 4.6 ms (every 23rd step) | derivative-based columns NOT valid; includes `T_req_data`, `T_avail`, `data_ratio` |
| Realistic cycle metrics | `data/cycles/{udds,tesla}_metrics.csv` | one row per run | 5 seeds; the chatter-affected first version is in `data/cycles_v1_dchatter` and should NOT be used |
| Sensitivity sweep | `data/sensitivity/metrics.csv` | one row per run | 3 seeds |
| Full-rate audit runs | `data/audit/fr_*.parquet` | 0.1 ms | use these to check identities |

## Who knows what (column `role` in `telemetry.CATALOG`, 172 columns)
`state`/truth columns are plant truth and are never given to the controller. `measured` = what the controller received; `error` = measured minus truth; `belief` = the controller's wrong constants; `internal` = controller-side signals; `ledger` = voltage-mismatch pieces (delivered minus requested). `psi_tilde` and `V_theta` use the TRUE flux and are analysis columns only.

## Verified
- Time is strictly increasing on a uniform grid in all 54 audited files; no NaN/Inf in any file.
- Voltage ledger closes: the six pieces sum to the total to 5e-13 V in every firewalled run.
- Pack energy chain (`resid_chain`) closes to numerical zero; actuator-mismatch power identity (`resid_du_power`) closes.
- DC-link balance (`resid_bus`) is 0.1% of peak power on a full-rate battery run.
- Bound on the voltage actually applied: max |u|/true ceiling = 1.0000000000000002.

## Known limits (read before using)
1. **Derivative-based columns are only valid when the log step is <= 0.4 ms** (`P_mag`, `resid_id/iq/w`, `P_balance_resid`, `resid_elec_decomp`, `resid_req_decomp`, `resid_bus`). The seed-0 matrix (2 ms) and cycle (5 ms) files fail this. `derive()` sets an attribute for it but parquet does not keep attributes, so the flag is lost. Use `data/audit/fr_*.parquet` for identity checks.
2. On full-rate runs the motor power-balance identity has RMS error 0.8% of peak power with isolated single-step spikes up to 14% (99.9th percentile about 7%). The cause is **inferred**, not proven: the documented one-step lag in the finite-difference identity plus current ripple from the dead-time model. Legacy full-rate runs show equal or larger spikes.
3. Battery columns (`P_el`, `P_dc`, `V_dc` dynamics, `R0_eff`, `T_batt`, ...) exist only in battery runs; plain runs have a constant 300 V bus.
4. The Tesla track asks for up to 1.50x the torque the toy motor can give (`data_ratio` max 1.50); UDDS peaks at 0.87. So Tesla results include infeasible request, by design of the data.
5. Grade torque (`T_dist` in cycles) is held at its edge value for the first and last 15 s of each cycle (a filter boundary artifact).
6. Battery OCV shape, temperature law and RC branch are generic assumptions; only resistance scale, capacity and current limits come from TUM. Sensor, inverter and BMS values are assumptions (`docs/SENSOR_PARAMS.md`).
7. The dashboard and the zip bundle predate the realism layer and the catalog extension.

## Sampling of stored seed-0 runs (updated 2026-10-30)
Stored seed-0 matrix rows use every 17th step (`STORE_STRIDE`), cycle rows every 23rd (`log_every=23`). The old stride of 20 landed only on speed-loop update instants (every 10 steps), which inflated instantaneous RMS voltage gap about 2x (11.5 V vs 6.1 V full rate). After regeneration the stored sag/awg file gives 6.29 V vs 6.09 V full rate. Energy gaps were never affected. Derivative-based columns (P_mag, resid_*) remain valid only on full-rate files (`data/audit/fr_*`).


## Residual check fixed (2026-11-02)
`resid_id/iq/w` now use a forward difference against same-row delivered voltage (row k = state at start of step, voltage during it). The previous central-difference/previous-row-voltage version overstated resid_iq about 18x (1195 vs 67 A/s on `fr_battery_sag_awg`). Two-state columns `ts_*` added (see `docs/TWO_STATE_MODEL.md`); catalog is now 184 columns. Re-run `python tools/dataset_audit.py` before quoting residual numbers from the audit CSV, which predates this change.

## Audit findings of 2026-11-09 (full report: `docs/AUDIT_PARAMETERS.md`)
Fixed: the cycle runner now defaults to `d_gain_by_L=True` (the default gain made the d-axis loop unstable at tau = 2e-4 s); `run_fw` refuses an unstable d-axis loop and a step-size mismatch between sensor, inverter, controller and plant; `ControllerConfig` rejects an unknown mode; very short cycles no longer crash the grade step; wrong comments and one wrong contract formula corrected.
Kept as documented limits: TUM request hold/quantization (`ref_hold_s`, `ref_quant`) exist but are off in all runs; the BMS discharge limit reported at SOC 0.5 (85 to 100 A) is lower than the pack limit (235 A); the 65 kW pack sizing uses apparent power while the baseline draw is about 13 kW; current limits 240/250/275/400 A are different things (pack, controller, envelope, plant); the naive controller can exceed the +-400 A sensor range (773 A seen); dead-time voltage depends on tau (7.5 V at 1e-4 s, 4.5 V at 2e-4 s); M3 (hidden clipping) is 1.6 V in UDDS, not zero.
Sensitivity: the original `inverter_delay` rows used the unstable gain at delay 2. They are kept for the record; `inverter_delay_dgl` and `nominal_dgl` (stable gain) are the valid ones.

### Design trade-off found by the corrected sweep (CLOSED as D3, see `docs/DECISIONS.md`: default stays off)
The d-axis gain `d_gain_by_L=True` is stable at every step size we use, but it lowers the d-axis feedback gain (k_d = k_i Ld/Lq = 0.93 instead of 3). Under saturation the d current then drifts further. Sag scenario, 3 seeds, protected (aw) controller: current peak 290 to 297 A and `envelope_ok` 0/3 with it, versus 252 A and 3/3 with the default gain at delay 1 (which is stable but only marginally: root magnitude 0.90). Naive ultimate flux error is 0.047 with it versus 0.126 without. A5 (protected below 0.25x naive) holds either way (ratio about 0.03) and A1 holds. So: the drive-cycle runs (tau = 2e-4 s) must use the stable gain; the 1e-4 s matrix runs use the default gain. Choosing one default for both, or a better d-axis design (higher gain with a stable discretisation), is open.
