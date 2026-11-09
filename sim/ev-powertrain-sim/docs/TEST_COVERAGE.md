# Test coverage gap analysis

No existing file was edited. Only new files were added: six `tests/test_cov_*.py` files and this report.
All new tests run in about 22 s in total (slowest single test 1.6 s); none needed the `slow` marker.

## Coverage before and after (line coverage, pytest-cov, source modules only)

"Before" is the union of two runs of the existing fast tests (test_bms, inverter, sensors, bounds, plant, realism_firewall without U30
which spawns a nested pytest, plan_coverage, battery_fit, telemetry, cycles, pipeline, firewalled_cycle). "After" adds the new files.
Scripts that tests run in a subprocess (`tools/disturbance_bounds.py`, `dataset_audit.main`) are exercised but not measured.

| Module | Before | After |
|---|---|---|
| bounds.py | 93 % | 100 % |
| control_fw.py | 98 % | 100 % |
| interfaces.py | 97 % | 100 % |
| telemetry.py | 99 % | 100 % |
| cycles.py | 91 % | 99 % |
| mismatch_map.py | 0 % | 99 % |
| data_gutcheck.py | 0 % | 74 % (rest is the `__main__` block) |
| firewalled_loop.py | 82 % | 87 % |
| tools/dataset_audit.py | not imported | 58 % (main runs in a subprocess) |
| tools/disturbance_bounds.py | not imported | exercised in a subprocess, 8 pieces checked against an independent computation |
| All repo modules together | 71 % | 78 % |

## New test files (192 test items: 188 pass, 4 xfail strict)

| File | Tests | What it protects |
|---|---|---|
| test_cov_controller_fw.py | 42 | `ControllerConfig`/`BeliefConfig` defaults and frozenness, `rng_streams`, `rot/clarke/park`, `validate_fit_dict`. Every branch of `FirewalledController` against hand-computed formulas: one-step command, `d_gain_by_L`, `clamp_psi` (aw, robust, custom belief range), robust sigma pull-back, aw `z` dynamics, speed-loop back-calculation and 10-step cadence, `i_lim` saturation, governor ellipse, `vdc_aware`, `bms_aware`, ceiling_hat and oracle, radial clip and `du_q_known`, angle advance, zero-speed safety. |
| test_cov_firewall_loop.py | 26 | Static firewall: exact import set of `control_fw.py`, no eval/getattr/lazy imports, every attribute read from `m`, `b`, `cfg` is a declared contract field, the loop calls `ctl.command(m)` and nothing else on `ctl`. Behavioural firewall: with sensors replaced by a canned stream, changing hidden plant state (road load) leaves every controller column bit-identical while truth differs; changing one measurement does change the output. Ledger: six pieces sum to `d_u` for five configs, a battery run with BMS derate, and the hidden-saturation event lands in piece 5 only. Determinism: same seed is bit-identical (3 configs), different seeds differ, global numpy state is irrelevant, `log_every` only decimates, manifest and parquet output. |
| test_cov_telemetry_edges.py | 24 | `derive` on a hand-checkable frame (every named channel), no input mutation, 1-row/2-row/0-row frames, missing columns, row-spacing vs `P['tau']` warning, validity flag threshold, non-uniform time, battery branch known answers, `data_infeasible` threshold, `req_sign_conflict` rules, catalog hygiene, two-state `ts_*` identities (closed forms, equality with three-state residuals on a short firewalled run). |
| test_cov_mismatch_gutcheck_bounds.py | 48 | `mismatch_map`: M1..M10 statistics, index, owed-energy fractions, data/event/remainder attribution (precedence, windows, zero-error run), legacy class table, tau fallback, `build()` with and without manifests/`_measured` files/empty dir. `data_gutcheck`: `runs_of_equal` known answers (NaN, ties, bit-identical), `scan_sim` verdict for each channel class, moving-only holds, `scan_tum` rates. `bounds`: event keys, window default, flat/zero/growing V, all-saturated split, 1-row and 2-row frames. |
| test_cov_cycles_edges.py | 32 | `cycles.py` error paths and codes (bad units, utf-16, no numeric rows, too few samples, empty table, same column), sparse sampling and strict mode, smoothing, elevation-to-grade (known slope, 15 % cap, downhill, standstill), `VehicleEq` formulas, `data_demand` with grade, road-scale branch and its give-up, TUM adapter fallbacks. |
| test_cov_tools.py | 20 | `dataset_audit.audit` on synthetic parquet (clean, non-monotone time, inf/all-NaN columns, missing columns, broken ledger, no-ledger, battery), `dataset_audit.main` and `disturbance_bounds.py` run inside a throw-away copy of the repo layout (the real `data/` is never touched), output values recomputed independently, markdown table filter, syntax check of heavy scripts (`db/pg_bench.py`, `dashboard/build_dash2.py`, `tools/*`). |

## Bugs found (each is an `xfail(strict=True)` test)

1. `cycles.py:371` (`_add_grade`): any cycle of 30 s or less (300 samples at dt 0.1) that has an elevation channel crashes with a raw numpy
   `ValueError: array of sample points is empty` (`np.interp` on an empty slope array). `min_duration_s` allows 10 s cycles, so this is reachable.
   Test: `test_short_cycle_with_elevation_does_not_crash_the_grade_step`.
2. `telemetry.py:293` vs `telemetry.py:322`: `ts_z2` is written to tolerate a missing `iq_ref` column (NaN), but `P_cu_ref` a few lines later indexes `d["iq_ref"]`
   unguarded, so `derive` raises `KeyError` and the guard is dead code. Test: `test_missing_iq_ref_column_gives_nan_ts_z2_instead_of_a_crash`.
3. `telemetry.py:93` (CATALOG, `ts_z2`): the description claims it is "the same as e_q in firewalled runs". It is not: `e_q` uses the measured `i_q_m`,
   `ts_z2` the true `i_q`; they differ by `gap_i_q` (verified). Documentation bug. Test: `test_catalog_claim_ts_z2_equals_e_q_in_firewalled_runs`.
4. `interfaces.py:166` / `control_fw.py:63`: `ControllerConfig.mode` is unvalidated; `mode="typo"` silently runs as naive, and nothing checks
   `delay_steps` against the inverter's. Test: `test_unknown_controller_mode_is_rejected`.

Characterised, not xfailed (behaviour is defensible but the message is poor): `derive` and `bounds_report` on a 1-row frame raise
`ValueError` from numpy (`np.gradient` / "cannot convert float NaN to integer") instead of a named error; `derive` ignores a time hole
(median spacing is used for the cumulative energies); `cycles._read_text` has an unreachable `file_encoding` error because latin-1 decodes any bytes.

## Modules still untested, and why

| Module | Reason |
|---|---|
| analyze.py, analyze_matrix.py, analyze_sensitivity.py, plot_abcd.py, plot_governor.py | Top-level plotting/reporting scripts, no functions, run on stored data at import. |
| run_clamp_ab.py, run_classes.py, run_matrix_fw.py, run_sensitivity.py, run_cycle_matrix.py | Experiment drivers: minutes of simulation per run and they write into `data/`. Only their building blocks (`run_fw`, loaders) are tested. |
| run_cycle.py (24 %), firewalled_cycle.py (23 %) | Full drive-cycle runs, slow (about 1 min each) and need the TUM data; existing slow tests cover them. `ensure_standing_start`/`peak_power_time` are covered by test_pipeline. |
| run_exp.py `__main__` and `run()` writer | Same as above; the scenario functions are covered. |
| battery_fit.py (95 %) | Remaining lines are fit CLI/printing branches. |
| data_gutcheck.py `__main__`, dataset_audit.py `main()` measured lines | Reads the real `data/` tree and writes CSVs there; exercised only through the isolated subprocess copy. |
| dataset.py | Added by another agent during this task (tests/test_dataset_access.py exists); not analysed here. |
| db/pg_bench.py, dashboard/build_dash2.py, tools/storage_bench.py, mutation_check.py, regen_seed0.py, add_cycle_data_cols.py | Need Postgres, DuckDB, full datasets or minutes of runtime; syntax-compile only. |

Run: `pytest tests/test_cov_*.py -q` (about 22 s on two cores).
