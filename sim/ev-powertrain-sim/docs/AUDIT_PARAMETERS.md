# Parameter and assumption audit

Scope: every `.py` at the repo root and in `tools/` (29 + 7 files), the config dataclasses in `interfaces.py`, the GEM 3.0.3 PMSM defaults as read at run time, and the numbers quoted in `docs/`. Source files were not edited. Checks were run with `python -I` on short runs (at most 0.6 s to 3 s of simulated time, under 2 minutes each) and on the stored seed-0 parquet files. The full test suite and the matrix and cycle runners were not run.

## Summary

| Verdict | Count |
|---|---|
| WRONG | 6 (3 change simulated behaviour, 3 are comment or docstring errors) |
| SUSPICIOUS | 14 |
| OK (table below) | 74 rows |

WRONG
* W1 `run_sensitivity.py:20` the `inverter_delay` "hi" level (delay_steps = 2) puts the d-axis current loop into a limit cycle (rms i_d about 51 A against 1.3 A at delay 0). The A5 verdict for that parameter is not a delay sensitivity.
* W2 `firewalled_cycle.py:14` and `interfaces.py:176`: `run_fw_cycle` at TAU_CYCLE = 2e-4 with the default `d_gain_by_L = False` is an unstable d-axis loop (rms i_d about 77 A). Only `run_cycle_matrix.py:22` overrides it; the default (used by `tests/test_firewalled_cycle.py`) does not.
* W3 `interfaces.py:171` comment says `theta_adv = (d + 0.5) omega_m_m p tau`; the code (`control_fw.py:115`) and the contract use `d` (the contract says the 0.5 form double counts). Comment only.
* W4 `battery.py:47` comment says R0 is "x1.9 at 0 C, x0.7 at 45 C"; the formula gives x2.15 and x0.59. Comment only.
* W5 `cycles.py:356` docstring says grade is "capped at +-25 %"; the code (`cycles.py:372`) caps at 0.15. Docstring only.
* W6 `docs/INFORMATION_CONTRACT.md:84` gives the BMS-aware limit as `P_lim / (Kt_n omega eta)`; the code (`control_fw.py:69-70`) uses `eta * V * I_lim / (Kt omega)`. The code is the physically right direction (shaft power is at most eta times electrical power). The doc is wrong.

## WRONG and SUSPICIOUS items with evidence

### W1 inverter delay sensitivity level is an unstable d-axis loop (run_sensitivity.py:20)

Theory (d-axis, `i[k+1] = i[k] - a i[k-d]`, `a = k_i tau / Ld`, Ld = 0.37 mH, k_i = 3 V/A):
```
tau=1e-4 d=0 a=0.811 max|root|=0.189 | d=1 0.900 | d=2 1.082  (unstable)
tau=2e-4 d=0 a=1.622 max|root|=0.622 | d=1 1.273 (unstable) | d=2 1.328 (unstable)
q-axis (Lq = 1.2 mH): a = 0.25 (1e-4) or 0.50 (2e-4): stable for d <= 2
```
Measured, `run_fw` with the exact cell of `run_sensitivity.py` (battery, event, aw, seed 0, 3 s):
```
lo delay 0 ctl delay 0 rms i_d(t>1)  1.296 A  max|i_d|   4.8 A
hi delay 2 ctl delay 2 rms i_d(t>1) 50.94  A  max|i_d| 103.8 A
```
The stored `data/sensitivity/metrics.csv` rows for `inverter_delay` hi/lo look almost identical (speed_rms 6.5 against 6.4) because the event dominates those metrics, so the limit cycle is invisible in them. `docs/REALISM_PLAN.md:186` records the d-axis problem for tau = 2e-4 only. It says the plain matrices (a = 0.8) are "mildly affected"; it does not mention delay 2.

### W2 cycle default d-axis gain

```
tau=2e-4 delay=1 d_gain_by_L=False: max|i_d|=118.9  rms i_d(last 0.1 s)=77.3 A
tau=2e-4 delay=1 d_gain_by_L=True : max|i_d|=  4.3  rms i_d=1.35 A
tau=1e-4 delay=1 d_gain_by_L=False: max|i_d|=  5.7  rms i_d=1.74 A   (stable but |root| = 0.90, oscillatory: 56 deg per step)
```
`ControllerConfig.d_gain_by_L` defaults to False (`tests/test_cov_controller_fw.py:48` pins it). The plain and battery matrices (`run_matrix_fw.py`, tau = 1e-4, d = 1) also run with it False, at a = 0.81. That is stable but marginally damped, and is the reason for the "5.2 V against 2.7 V per step" note in the plan. This is a mismatch between the defaults tests use and the run script that produced the cycle data.

### S1 TUM reference hold and quantisation are never applied (interfaces.py:120-121)

`grep -n "ref_hold\|ref_quant" *.py` outside `sensors.py` and `interfaces.py` returns nothing. The contract (M7b) and `docs/SENSOR_PARAMS.md` say TUM-sourced references use `ref_hold_s = 0.1` and `ref_quant = 0.1`. `firewalled_cycle.py` and `run_cycle_matrix.py` build `SensorConfig(tau=tau)` with both at 0. The Tesla cycle request is therefore linearly interpolated between 10 Hz samples, not held. That gives a smoother request than documented.

### S2 tau is not checked across configs (firewalled_loop.py:64-70)

`SensorConfig.tau`, `InverterConfig.tau`, `ControllerConfig.tau` and the plant tau are four separate copies. Measured:
```
run_fw(tau=2e-4, sensor_cfg=SensorConfig(), inv_cfg=InverterConfig())  -> accepted silently;
manifest: sensor tau 1e-4, inverter tau 1e-4, run tau 2e-4
```
The PLL `dt`, the Vdc filter alpha and the dead-time `K = Td/tau * Vdc` would all use the wrong step. `run_sensitivity.py` is safe because it runs at 1e-4. A one-line assert would remove the trap.

### S3 dead-time error depends on the integration step (inverter.py:61, interfaces.py:133)

`K = dead_time_s / tau * Vdc + 1.5` with Tpwm := tau. Measured:
```
tau=1e-4 K(300 V)=7.5 V  (4/pi K = 9.55)
tau=2e-4 K(300 V)=4.5 V  (4/pi K = 5.73)
```
Cycle runs get 40 % less inverter error than plain runs for the same hardware. `docs/SENSOR_PARAMS.md` states the choice, so it is a design choice, but it is a hidden coupling between solver step and hardware.

### S4 BMS discharge limit reported at the default SOC is far below the pack's own limit (data/tum_battery_fit.json, bms.py:38-39)

`scaling.i_dis_lim_toy_a[0]` (SOC 0.5 bin) is 85.2 A, against `toy_battery_kwargs.i_dis_max` 235.3 A. The default `soc0` is 0.5, the least-supported bin of the envelope (`soc_dependence_identified = False`, one trip per cluster). Stored runs:
```
sag_awg_s0   bms_i_dis_lim msg max 95 A   pack i_batt max 235.3 A
tesla_awg_s0 bms_i_dis_lim msg max 100 A  pack i_batt max 235.3 A
```
Outside SOC [0.5, 1.0] the value is held flat, so the SOC-0.5 value (85 A) applies to every run that dips below 0.5. The message limit and the pack's real limit disagree by 2.5 to 2.8x in the same simulation. The `bms_aware` governor does not bind in the base scenario (23 kW at 85 A against 13 kW demand), so no stored conclusion is clearly affected, but the SOC dependence of the message is an artefact of trip confounding.

### S5 the 65 kW pack sizing uses apparent power, not real power (battery_fit.py:693, docs/BATTERY_FIT.md:47)

`p_base_toy_w = 65000` is 1.5 * 173.2 V * 250 A, the motor's apparent power. Real shaft power at i_lim and 190 rad/s is 74.25 N.m * 190 = 14.1 kW. Stored runs:
```
sagbase_awg_s0 P_el max 12.7 kW, i_batt max 38.7 A (pack limit 235 A)
tesla_awg_s0   P_el max 31.2 kW, udds_awg_s0 52.7 kW (spikes, include reactive terms)
```
The pack's 235 A discharge limit and 0.12 ohm resistance are scaled for a load about 4.6x the real power this toy motor draws at rated conditions. The BMS limit only binds inside the injected derate (idis_scale = 0.02). Per-unit resistance is preserved, but a 38.7 A baseline draw on 0.125 ohm gives about 5 V of sag, much smaller than the "6 to 11 %" per-unit figure that motivated the fit.

### S6 current limits: 240, 250, 275, 400 and up to 773 A (several files)

GEM nominal 240 A, controller `i_lim` 250, envelope 275 (`run_matrix_fw.py:19`), GEM/ADC/`plant.limits["i"]` 400. In stored `hidden_naive_s0` the true phase-current magnitude reaches 772.8 A: above the plant's 400 A limit (constraints are switched off), above the sensor full scale (`i_range_a = 400`, so the measurement clips), and above the 275 A envelope. Documented as the naive failure mode, but the ADC clip means the "measured" current saturates in exactly the runs where it matters. `bounds.py:27` default `i_lim = 250`, `telemetry` `dist_i_limit` uses 400.

### S7 R and L belief ranges (interfaces.py:156-157)

`dR in [-0.15, +0.40]` is justified by "winding 20 to 120 C is +39 %", but that is the truth moving up against a fixed-temperature belief, i.e. a negative belief error. The belief range has the opposite sign and is asymmetric. `dL in [0, +0.25]` is one-sided by construction (belief above truth only). Beliefs cannot make the controller unstable: I checked the worst case by reasoning (R error 0.0072 ohm against k_i = 3 ohm; L error adds at most 43 V cross-coupling at 250 A and 570 rad/s, absorbed by k_i = 3 into about 14 A of i_d). I did not simulate the corners.

### S8 symmetric current clip uses the discharge limit for regeneration (control_fw.py:66-70)

`bms_aware` clips `iq` to `+-p_lim/(Kt w)` with `p_lim` from `i_dis_lim`; `i_chg_lim` is carried in the message but never read. In a derate (idis_scale = 0.02) regeneration is also cut to about 5 A of pack current, although the fault only lowers the discharge limit. In a plain regen run (`firewalled_loop`, ramp 150 to 10 rad/s) the pack saw -28.6 A (limit 63.4) with `iq` reaching -251 A, so the charge limit did not bind.

### S9 speed loop and stride hard-code 10 steps (control_fw.py:50,86; control.py:49,80; run_matrix_fw.py:20)

`dt_w = 10 * tau` with `n % 10`. The docs call it a 1 kHz speed loop; at the cycle step it is 500 Hz. The anti-windup gain `5.0` (control_fw.py:80, units 1/s, equals the integrator time constant 4/speed_bw) is an unnamed constant. The matrix `STORE_STRIDE = 17` and cycle `log_every = 23` are tied to that cadence by comment only.

### S10 hidden clipping M3 is not always 0 (plant.py, mismatch_map.py:30)

The inverter error is added after the clip (`u_gem = u_app + u_extra`), so at the hexagon edge the GEM duty can exceed 1 and GEM clips silently:
```
udds_awg_s0   M3 max 1.59 V  rms 0.052 V  duty_peak max 1.009, 0.23 % of rows > 1
udds_naive_s0 M3 max 1.57 V  rms 0.058 V  duty_peak max 1.009
all matrix runs and tesla: M3 = 0.00
```
The label "must be 0" in `SCALES["M3"]` is false for UDDS (small).

### S11 quantiser levels and ADC range (sensors.py:20, interfaces.py:105-106)

Measured `LSB_i = 0.1953125 A`, 4097 levels from -2048 to +2048 LSB, `LSB_v = 0.1220703 V`. 400/0.1953125 = 2048, so the code range is 4097 levels (13-bit count) rather than 4096; the +2048 code is unreachable for a 12-bit converter. A 12-bit ADC over +-400 A is otherwise right (LSB = 800/4096). The voltage ADC hard-codes `2 ** 12` (`sensors.py:70`) and ignores `adc_bits`.

### S12 battery docstring time constant (battery.py:15)

Says "C_dc*Rs ~ 0.3 ms". Measured: default Battery 0.17 ms ((0.08 + 0.005) * 2e-3), fit kwargs 0.25 ms. Close enough not to matter, but the numbers differ. Explicit Euler at 1e-4 would be at 0.6 to 0.4 of the step, marginal as claimed, so the implicit update is justified.

### S13 Battery() default differs from the fitted toy pack (battery.py:30-34, run_exp.py:36, params.json)

Default: R0 = 0.08 ohm, i_dis 250 A, i_chg 150 A. Fit kwargs: R0 = 0.1195 ohm, i_dis 235.3 A, i_chg 63.4 A. `run_fw(battery=True)` without `fit_path` and `run_exp --battery` use the default; `run_matrix_fw`, `run_sensitivity` and the cycle runners use the fit. `data/params.json` documents the default under "battery_default". Runs of the two families are not comparable on regen limit (2.4x) or resistance (1.5x).

### S14 other small items

* `bounds.py:27` parameter `u_nom` is never used.
* `cycles.py:408` `fit_vehicle_to_motor(omega_target=190)` is above the base speed at i_lim (computed 188 rad/s: `(173.2 / sqrt((1.2e-3 * 250)^2 + 0.066^2)) / 3`), so "below the field-weakening speed" is marginal. The governor (factor 0.75) covers it.
* `cycles.py:150ff` `required_torque` (mass sizing) omits rotor inertia while `data_demand` includes it (0.0388); small.
* `run_cycle.py:153` the CLI computes the derate time on the full cycle before `--t-max` truncation, so with `--t-max` the derate can fall outside the run (the `run_classes.py` and `firewalled_cycle.py` paths compute it inside the window).
* `firewalled_loop.py:111` the rotation by the angle error uses `delta` of step k although the delayed command was built at step k-d. For constant offset and slow ripple this is exact; for the 0.3 deg white noise it is mis-aligned (statistically equivalent, not sample-exact).
* `telemetry.py:121` derivative channels are valid at `tau <= 4e-4` but `mismatch_map.py` flags `decimated` at `tau > 4.5e-4`; the thresholds differ (no stored run sits between them).
* `data_demand` (`cycles.py:486`) hard-codes `psi = 0.066, Lq = 1.2e-3, i_lim = 250, u_max = 173.2, J_rotor = 0.0388, p = 3`; `mismatch_map.py:28` and `run_cycle.py:27` hard-code `0.066` and `3`. All match the plant today but none is read from it.
* `run_fw` has no divergence guard (`run_cycle` has one); a NaN current makes `round()` in `sensors._quant_i` raise rather than propagate.
* `RNG_STREAMS` includes "events" (`interfaces.py:42`), which no code draws from.

## Things checked and found consistent

* V_dc = 300 V, ceiling 173.205 V: `plant.py` (`u_sup` limit 300, `u_max_hw` 173.2051), `interfaces.py` `V_SCALE_GEM`/`U_CAP`, `bounds.py` 173.2051, `mismatch_map.py` 173.2051 and 300. `cycles.py` uses the rounded 173.2 (0.003 % off).
* Frame/sample alignment: `park(i_abc, epsilon_prev)` reproduces the plant's `i_d, i_q` to 6.0e-13 A; using the current angle gives 0.97 A error (3000 steps, random drive). Matches the contract.
* Torque constant: `T_elec = 1.5 p (psi + (Ld - Lq) i_d) i_q` to 2.3e-13 N.m on `hidden_naive_s0`; `control_fw.py:35` and `run_fw` `T_cmd` use `1.5 p psi`.
* Regen sign: ramp 150 to 10 rad/s gave P_el min -9.9 kW, i_batt min -28.6 A, V_dc max 305.9 V, SOC 0.5000 to 0.4999 (consistent with a discharge-positive pack current, charging negative).
* Hidden clipping with no inverter error: random drive, M3 max 4e-14 V.
* Seeds: every random draw comes from `rng_streams(seed)` (`SeedSequence.spawn`, fixed stream order), per run. `sensors.reset` always draws 8 uniforms, `measure` always draws 5 normals, `draw_beliefs` always draws 4 uniforms, `BMSModel` one normal per message. GEM `env.reset(seed=0)` is fixed in `plant.py:78`, which only matters if GEM randomises initial state (omega = 0, no random init: not an issue). `battery_fit.py` bootstrap seed 20240611 is fixed.
* Docs against code (grep): `TWO_STATE_MODEL.md` table (R 0.018, Ld 0.37, Lq 1.2, psi 0.066, J 0.5388 against measured 0.53883, b 0.02, c 1e-4, p 3, belief ranges, 250/275) matches. `BATTERY_FIT.md` toy kwargs match `data/tum_battery_fit.json`. `SENSOR_PARAMS.md` values match `interfaces.py` dataclass defaults.
* Hard-coded 1e-4 in code: only `plant.py:53-54` (defaults) and `interfaces.py:22`. `dataset.py:105` hard-codes the 4.001e-4 threshold for the same derivative-validity rule.
* Stale path: `run_cycle_matrix.py` pointed at a scratchpad directory when I first read it; it was changed during this audit (12:13) to `data/inputs`. `tests/test_cycles.py:247` still defaults to a scratchpad path.

## Full OK table

Provenance codes: T = measured from TUM data, G = taken from GEM, A = assumed typical class value, D = design choice.

| file:line | name | value | unit | provenance | verdict / evidence |
|---|---|---|---|---|---|
| plant.py (GEM) | p | 3 | - | G | OK; same in SensorConfig.p, ControllerConfig.p |
| plant.py (GEM) | Rs | 0.018 | ohm | G | OK |
| plant.py (GEM) | Ld, Lq | 0.37e-3, 1.2e-3 | H | G | OK |
| plant.py (GEM) | psi_p | 0.066 | Wb | G | OK; `T_cap = 1.5*3*0.066*250 = 74.25` consistent |
| plant.py (GEM) | j_rotor + j_load | 0.03883 + 0.5 = 0.5388 | kg m^2 | G + D | OK |
| plant.py:54 | tau | 1e-4 | s | D | OK |
| plant.py:53 | b, c, a | 0.02, 1e-4, 0 | N.m.s, N.m.s^2, N.m | D | OK; matches TWO_STATE_MODEL.md |
| plant.py:57 | omega limit | 418.879 | rad/s | G | OK (4000 rpm) |
| plant.py:72 | limits["i"] | 400 | A | G | OK (see S6) |
| plant.py | V_dc, u_max_hw | 300, V_dc/sqrt(3) | V | G, D | OK; hexagon inscribed disc, duty peak 1.000 at the edge |
| plant.py:125 | min-max zero-sequence injection | - | - | D | OK |
| plant.py:134 | p_el | 1.5 (u_d i_d + u_q i_q) | W | D | OK; amplitude-invariant, consistent with `clarke` (2/3) |
| plant.py:130 | `t += tau` before fault_fn(t) | - | s | D | OK; one-step offset is negligible (event times 4.5 s, 5.5 s) |
| battery.py:21-22 | OCV table | 3.00 to 4.20 | V/cell | A | OK; 302.4 V at SOC 0.5 for 80s |
| battery.py:30 | n_series | 80 | - | A | OK |
| battery.py:30 | q_ah | 60 | Ah | A | OK; 18.1 kWh |
| battery.py:30 | r0_cell | 1e-3 (fit 1.494e-3) | ohm | A / T | OK; R0 25 C = 0.08 (default) |
| battery.py:30 | r1_cell, tau1 | 0.6e-3, 20 | ohm, s | A | OK; fit says R1/tau not identified |
| battery.py:31 | r_bus | 5e-3 | ohm | A | OK |
| battery.py:31 | c_dc | 2e-3 | F | A | OK; units consistent |
| battery.py:31 | i_dis_max, i_chg_max | 250, 150 (fit 235.3, 63.4) | A | A / T | OK values; see S13 |
| battery.py:34 | r_inv, p_inv_idle | 8e-3, 30 | ohm, W | A | OK |
| battery.py:34 | c_th, r_th | 4e5, 0.05 | J/K, K/W | A | OK; T_batt moves 4e-4 K in 150 s, so thermal part is inert |
| battery.py:34 | v_min | 20 | V | D | OK; guards the division by V_dc |
| battery.py:48 | Arrhenius B | 2500 | K | A | formula OK; comment wrong (W4) |
| battery.py | SOC update | `soc -= i_b dt / q_as`, i_b > 0 discharge | - | D | OK; checked in the regen run |
| bms.py (config) | period 0.05, latency 0.02, quant 1.0, rel_err 0.05 | s, s, A, - | | D / A | OK; causal, floor(t/period + 1e-9) |
| inverter.py | delay FIFO | cfg.delay_steps | steps | D | OK; identity mode 0 |
| inverter.py | R(phi) convention | [[c,-s],[s,c]] | | D | OK; delay rotation `-d omega_el tau` is in firewalled_loop.py:111 |
| interfaces.py:105-106 | i_range_a, adc_bits | 400, 12 | A, bits | D | OK (S11) |
| interfaces.py:107 | i_gain_err | 0.01 | - | A (LEM datasheet class) | OK |
| interfaces.py:108 | i_offset_a | 2.0 | A | A (LEM class) | OK |
| interfaces.py:109 | i_noise_lsb | 1.0 | LSB = 0.195 A | A | OK |
| interfaces.py:111-113 | theta offset 1 deg, noise 0.3 deg, ripple 0.2 deg | rad | A | OK, flagged "no source" in SENSOR_PARAMS |
| interfaces.py:114-115 | pll_wn 300, zeta 0.707 | rad/s, - | D | OK; settling 4/(zeta wn) = 18.9 ms; wn*dt = 0.03 |
| interfaces.py:116 | vdc_range_v | 500 | V | D | OK; LSB 0.122 V |
| interfaces.py:117-119 | vdc gain 0.005, noise 0.5 V, corner 500 Hz | | A / D | OK; alpha = 0.27 at 1e-4 |
| interfaces.py:133-136 | dead_time 2e-6 s, drop 1.5 V, band 1 A, comp 0.7 | | A / D | OK (S3 for tau coupling) |
| interfaces.py:132 | delay_steps | 1 | steps | D | OK for 0 and 1; 2 unstable at k_i = 3 (W1) |
| interfaces.py:156-161 | belief ranges, psi_start 0.7, psi_range (0.03, 0.10) | | D | OK (S7); psi range brackets 0.066 |
| interfaces.py:177 | gamma | 2e-5 | Wb/(A rad) | D | OK; 0.011 Wb/s per amp of error at 570 rad/s |
| interfaces.py:177 | k_i | 3 | V/A | D | OK for q axis; marginal on d (W1, W2) |
| interfaces.py:177 | sigma | 2 | 1/s | D | OK |
| interfaces.py:178-180 | i_lim, speed_bw, gov_factor | 250, 20, 0.75 | A, rad/s, - | D | OK |
| interfaces.py:183 | eta | 0.9 | - | D | OK; applied as multiply (W6) |
| control_fw.py:35 | Kt | 1.5 p psi_name | N.m/A | D | OK |
| control_fw.py:36-37 | Kp_w, Ki_w | J bw/Kt, Kp bw/4 | A.s/rad, A/rad | D | OK dimensions |
| control_fw.py:48 | governor ellipse | `(u/w)^2 - psi^2`, Lq | | D | OK; matches the plant steady state with i_d = 0 and R ignored |
| control_fw.py:112 | `ceiling_hat = min(vdc_m/sqrt3, U_CAP)` | V | D | OK; clip order is controller clip, FIFO, rotation, plant clip, inverter error |
| control_fw.py:80 | anti-windup gain | 5.0 | 1/s | D | OK, equals 4/speed_bw (S9) |
| control.py:37 | psi_nom | 0.7 psi_p | Wb | D | OK |
| run_exp.py:22-24 | T_END, T_EVENT, U_EVENT | 8, (4.5, 5.5), 25 | s, s, V | D | OK; back-EMF at 200 rad/s = 39.6 V, so 25 V makes the command infeasible as stated |
| run_exp.py:34 | bms derate | 0.02 | - | D | OK; 5 A against 12 A cruise draw |
| run_exp.py:28 | omega_ref ramp 200 in 4 s, grade 10 N.m at 1.5 s | | D | OK |
| firewalled_cycle.py:32 | derate 0.02 for 3 s at peak power - 0.5 s | | D | OK |
| firewalled_loop.py | RNG stream order | sensors, inverter, bms, beliefs, events | | D | OK |
| run_matrix_fw.py:19 | ENV_PSI, ENV_I | 0.15, 275 | Wb, A | D | OK; documented |
| run_matrix_fw.py:20 | STORE_STRIDE | 17 | steps | D | OK (S9) |
| run_matrix_fw.py:CTRL | awg = aw + governor + vdc_aware (+ bms_aware on battery) | | D | OK |
| run_sensitivity.py | levels 0.5x, 2x, delay 0/2 | | D | OK except delay (W1) |
| run_cycle_matrix.py | log_every 23, tmax 120/150, d_gain_by_L True | | D | OK; differs from library defaults (W2) |
| cycles.py:385-388 | crr 0.010, CdA 0.62, rho 1.2, drivetrain_b 0.02 | | A | OK; CdA is high for a Model 3 (about 0.5) |
| cycles.py:200 | a_max 6, v_max 80, dt 0.1, max_gap 5 | | D | OK |
| cycles.py:408 | omega_target 190, headroom 0.85, quantile 0.99, mass range 20 to 2500 kg | | D | OK (S14) |
| cycles.py:372 | grade cap | 0.15 | - | D | OK value; docstring W5 |
| run_cycle.py:27,35 | I_LIM 250, a_ramp 1.5 | A, m/s^2 | D | OK |
| mismatch_map.py:28-32 | scales: 190 rad/s, 173.2 V, 250 A, T_CAP 74.25, P_RATED 14.1 kW, 300 V | | D | OK; P_RATED is shaft power, not the 65 kW apparent power (S5) |
| mismatch_map.py | attribution windows 2 s data, event + 3 s | | D | OK |
| telemetry.py | all identities (`1.5 R (id^2 + iq^2)`, `P_mag`, `P_balance`, two-state split) | | D | OK; dimensions consistent, tau taken from the row spacing |
| bounds.py | tolerance, 10 % window, 2 % settle | | D | OK; no unguarded division |
| battery_fit.py:60-75 | windows 0.1 to 1 s, REST_A 5 A, TAU 1 to 30 s, sentinel -20 C, soc jump 5 pts | | D | OK |
| battery_fit.py:693 | p_base 65000 W, v_toy 300 V, motor 250 A | | D | OK arithmetic (216.67 A, 235.3 A, R mid 0.1257 ohm); see S5 |
| data/tum_subset | sign: discharge negative | | T | OK; corr(I, accel) = -0.83, -0.67, -0.80 on tracks 0, 1, 10 |
| data/tum_subset | v 305 to 360 V, SOC glitches +-51.2, temp sentinel -25 | | T | OK, matches the filter rules |
| tools/*.py, run_classes.py, analyze*.py, dataset.py, data_gutcheck.py, plot_*.py | presentation or bookkeeping constants only | | D | OK; no physics constant not already listed |

## What I could not verify

* Stability of the controller at the corners of the belief ranges was argued, not simulated (S7).
* I did not re-run the matrix, cycle or sensitivity runners, so claims about stored results come from the stored parquet and CSV files as they were. The sensitivity cell for delay 2 was reproduced for 3 s of the 8 s run only.
* The sign of the delay rotation (`-d omega_el tau`) and the angle-advance sign were not independently re-derived; I relied on `tests/test_realism_firewall.py` (lines near 311) and the contract.
* The TUM-derived fit quality (leave-one-out RMSE gain of only 0.76 V with 2.7 V spread, status "ok") was read from the JSON, not recomputed.
* Cd*A, rolling resistance, thermal, inverter loss and OCV values are assumed class values with no external source checked.
* GEM internals (solver order, state clipping) were only probed through behaviour (states above the 400 A limit are not clipped; torque formula residual 2e-13).
