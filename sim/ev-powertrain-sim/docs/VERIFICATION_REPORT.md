# Independent newcomer verification (read-only pass)

Headline numbers
- (a) battery `sag`, envelope_ok: naive 0/20, robust 20/20, aw 20/20, awg 20/20. sagbase: all 20/20. MATCHES the claim of naive failing / protected passing. DATASET_README line 46 quotes 3 seeds (0/3 vs 3/3) for a d_gain_by_L variant; not contradicted.
- (b) RMS sqrt(d_u_d^2+d_u_q^2): audit fr_battery_sag_awg 6.087 V, matrix sag_awg_s0 6.293 V, ratio 1.034 (within 10 %). MATCHES. Caveat: audit manifests (fr_*.json) say seed 1, matrix file is seed 0.
- (c) rows per group: plain 240 (3 scenarios x 4 ctrl x 20), battery 160 (2 x 4 x 20), cycles 5 per (cycle, controller), sensitivity 3 per (param, level, controller) over 20 param/levels x 2 controllers = 120, clamp 20 each. MATCHES. DuckDB runs/steps rows 371,512 as stated.

Discrepancies and gaps
1. README.md table: `ts_z1 = omega - omega_ref` uses `omega`, not a column; stored name is `omega_m`.
2. README.md: "179 here; 184 for cycle runs" correct, but plain runs have 156 (23 battery columns absent); battery matrix runs lack 5 cycle-only columns (T_avail, T_req_data, data_infeasible, data_ratio, v_ref_mps). README never says which columns exist only in some families (plain: E_bus_unserved, soc, V_term, ...). The "any run file" rows are all fine, except none are missing.
3. data/README.md roles: lists `truth`, but catalog roles are state, mismatch, derived, validation, disturbance, margin, controller, measured, error, belief, internal, ledger, proof, input, index. No `truth` role.
4. data/README.md: cycles metrics "same family of columns" is only partly true: key column is `cycle` not `scenario`; no `envelope_ok`, `ultimate_psi_err`, `E_mech_unserved_kj`, `torque_gap_rms`.
5. data/README.md: sensitivity/metrics.csv covers only naive and aw (not stated); hidden_clamp uses controllers `awc`/`awgc`, not defined anywhere in the README.
6. data/runs_250hz/ is an empty folder though listed as legacy content.
7. tools/make_assumptions.py is not in tools/README.md. viewer/ has no README and data/README.md "legacy section" does not mention viewer/ or probes/ (root README says it does). runs_fw/ is mentioned.
8. tests/README.md and root README: bare `pytest -q tests/...` fails with ModuleNotFoundError: dataset in the supplied venv (no conftest, pytest.ini has no pythonpath); `python -m pytest` works (test_dataset_access: 8 passed). Same for `python script.py` run from another directory.
9. README table: "gap torque" wording in the torque row is awkward ("`torque gap` in metrics (`torque_gap_rms`)"); the 'ledger pieces' for saturation are the d_u_* columns, no column named ledger*.
10. dashboard/README.md says audit channel count 118 vs 184: consistent. data/inputs/README.md SHA-256 of track_0.csv matches SOURCE.json (verified).
11. docs/TWO_STATE_MODEL.md tables show only awg/naive; csv has aw/robust too (aw hidden reluctance 95.8, sag aw i_d 58.7) so the "protected runs hold i_d <= 100 A" claim holds (max 100.3 oracle aw; marginal).

Checked OK: all README paths exist (db/.env.example is hidden but present), DuckDB build (6 s, 363 MB, example SQL works, tables as listed), TWO_STATE_MODEL numbers vs data/disturbance_bounds.csv (all 14 rows match to rounding), params.json vs TWO_STATE_MODEL / ASSUMPTIONS (p, R, Ld, Lq, psi, J, b, c, belief ranges, i_lim, 173.2 V, 18.6 = 10/J).
