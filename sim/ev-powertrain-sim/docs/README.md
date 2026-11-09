# docs/: what to read, in order
| Start here | |
|---|---|
| `TWO_STATE_MODEL.md` | the exact two-state nonlinear model (speed, q-current): equations, units, tracking errors, measured disturbance bounds |
| `MATH_REFERENCE.pdf` (+ `.tex`) | all the math in the project: three-state plant, controller, adaptation law, ledger, energy identities |
| `ASSUMPTIONS.md` | every default setting and assumed value, with where it is defined |
| `DATASET_README.md` | what each stored data family is, how it was audited, known limits |
| `INFORMATION_CONTRACT.md` | what the controller is allowed to receive (the firewall) and why |

| Reference | |
|---|---|
| `SENSOR_PARAMS.md` | sensor, inverter and BMS values with provenance tags |
| `BATTERY_FIT.md` | how the TUM data set the pack's resistance, capacity and current limits |
| `DATASET_INVENTORY.md` | TUM columns: units, rate, quality, which feed the disturbance and the speed request |
| `ELECTRICAL_MISMATCH.md` | why the shaft-power gap is not the electrical gap, and the exact rebuild |
| `DATA_GUTCHECK.md` | held-value and glitch scan of the TUM and simulated data |
| `STORAGE_PLAN.md` | how to store episodes and runs; DuckDB benchmark results |
| `AUDIT_PARAMETERS.md`, `TEST_COVERAGE.md` | independent audit of parameters and tests (added 2026-11-09) |

| History | |
|---|---|
| `PLAN.md`, `REALISM_PLAN.md` | original plan with red team and assumption register; the realism-layer plan and build log |
| `LEGACY_TIER1_README.md` | the first README (Tier 1 experiments A to D) |
| `Research_Project_ENGR110H.pdf`, `Powertrain_Sim_Explainer.pdf` | the project notes; a plain-language explainer |
| `figures/`, `make_figures.py`, `make_report.py` | result plots and the scripts that draw them |
| `DECISIONS.md` | closed project decisions D1 to D3 |
