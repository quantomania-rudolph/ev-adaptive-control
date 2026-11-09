# Data gut-check: what we receive (TUM) and what the sim produces

Scan code: `data_gutcheck.py` (run `python data_gutcheck.py`). Raw tables: `data/gutcheck/sim_holds.csv` (every run in `runs/` and the matrix seed-0 runs), `data/gutcheck/tum_holds.csv`.
A "hold" = consecutive bit-identical values. Labels: **[V]** verified directly in the data/code this session, **[I]** inferred (plausible, not proven).

## 1. TUM tracks (5 local tracks, 10 Hz)

| Channel | Fraction of rows equal to the previous row | Longest hold | Reading |
|---|---|---|---|
| hv_battery_voltage | 4–10 % | 0.4–6.8 s | Mostly fresh. [V] |
| hv_battery_current | 32–66 % | 0.6–6.8 s | **Stale much of the time**: slower underlying update than 10 Hz. [V] |
| hv_soc | 18–49 % equal, but 50–81 % of rows are NaN | up to 44 s | Sparse and slow; unusable as a fast signal. [V] |
| speed | 8–16 % | 0.3–10.1 s | Good; ~10 % duplicates give flat 0.1–0.2 s reference steps. [V] |
| vehicle_speed | 55–87 % equal, 0–31 % NaN | up to 41.8 s | Not a usable second speed channel. [V] |
| hv_temp_min | 95–100 % equal | 59–281 s | Effectively constant; carries no information. [V] |
| ambient / hv_temp_max | 17–58 % equal, 42–83 % NaN | 20–54 s | Sparse. [V] |

Key finding on current holds: they are **stale values, not stopped vehicles**. While moving, current sat at exactly -106.1 A for 1.9 s (track 0), +169.3 A for 2.4 s (track 1) and -690.1 A for 1.2 s (track 103), while voltage changed on every row of those windows (17, 23 and 12 distinct values) and speed kept moving. [V] In track 103 the voltage rose 306 → 328 V during the "held" -690 A, so the current number was not real for that second. [V] The long holds at about -1 to -4 A with speed ≈ 0 are parked aux load: legitimate. [V]
Consequence: any resistance or limit fit from the current-voltage pairs is biased at the peaks. The battery fit already uses windowed estimates and reports a range, not a point, but the stale peaks are the main reason the per-unit R is 8.5–9.9 % and not a single number. [I]
For the controller: a real BMS current is a slow, held signal. The sim's `bms` message (latency buffer, quantization, update period) is the right place for this; the controller must not use it as a fast signal. [V by design, see INFORMATION_CONTRACT]

## 2. Sim runs: legacy runs (`runs/*.parquet`, 1223 ok, 94 flagged by the scan)

Everything flagged was traced to a cause. Nothing was a frozen physical channel.

| Held channel | Longest hold while moving | Cause | Verdict |
|---|---|---|---|
| `T_dist` (T0p, all cycle runs) | 15.00 s at start and 15.00 s at end | The grade is a 30 s-window slope; `np.interp` clamps to the edge value for the first and last half window (15 s). The two holds match the half-window exactly. [V] | Boundary artifact. Legitimate but the first/last 15 s of every grade-bearing cycle use an extrapolated grade. |
| `v_ref_mps`, `T_req_data`, `data_ratio` | 0.1–0.2 s (TUM cycles), 1.0 s (UDDS) | TUM speed duplicates (10 Hz) and the 1 Hz UDDS source. [V] | Source rate. |
| `T_req_data`, Yr runs | 8.0 s | Constant-torque step scenario. [V] | Scripted. |
| `i_batt` | exactly 5.000 A for 3.0 s (naive/protected cycle runs) | Scripted sag event: discharge limit scaled to 5 A for 3 s (`i_batt_max` = 5.0 over the same window, V_dc falls to about 163 V). [V] | Scripted event, not a stuck value. |
| `psi_tilde`, `V_theta` (run C) | 1.85 s | ψ̂ pinned at the 0.100 Wb projection bound, ψ̃ = 0.034 exactly. [V] | Parameter projection; also tells us the bound was active for that stretch. |
| `z` (Hp, T0p, Up, ...) | up to 114 s | Anti-windup integrator decayed to denormal ≈ 1e-321 and stops changing. [V] | Zero in effect. |
| `omega_ref`, `T_cmd`, `iq_ref`, `ceiling_hat`, `u_limit`, `i_lim_eff` | clip/limit plateaus | Saturation/limit values by construction. | Legitimate. |
| ledger rows `d_u_*`, `V_aw`, `i_ref_clip` | zero | Nothing saturating. | Legitimate. |

Not flagged but worth knowing: the speed-PI unsaturated demand `i_ref_clip` reached -9353 A before clipping in one run (Kp_w ≈ 36 A per rad/s). It is clipped before it matters, but anyone plotting it raw will see absurd numbers. [V from earlier scan]

## 3. Sim runs: the new firewalled/noisy runs (matrix seed 0, 12 runs checked)

- No SUSPECT or ZERO-WHILE-MOVING channels. [V]
- Measured current is genuinely quantized: step size 0.1953 A = 800 A / 4096 (12-bit over ±400 A), no clipping in these runs (peaks about ±287 A). [V]
- Measured channels are fresh every sample (equal-to-previous fraction 0.0–0.18 %). [V] There is no stale-data behaviour in the sim's measurement chain yet; real current and SOC are far staler (section 1).
- Constant channels are all by design: `belief_*` (a belief is one draw per seed), `V_dc` in non-battery runs (fixed 300 V), `bms_*` in non-battery runs, ledger rows with no saturation. `meas_vdc` still varies (noise), as it should. [V]

## 4. Honest gaps (do not oversell)

1. TUM current/SOC staleness is not reproduced in the sim's current measurement; only the BMS message has latency. If we want the controller to see realistic stale data, add a sample-and-hold option to the BMS current channel. [I: would take small code]
2. Cycle-run grade has a 15 s extrapolated edge on each end. Either trim the first/last 15 s from metrics or fade grade in. [V]
3. Fresh realistic cycle runs (UDDS / Tesla track) are not yet produced, so the held-value scan of realistic cycles is only covered by the legacy cycle runs above.
4. The plain/battery matrices were still running when this was written; statistics are not in this document.
