# Realism build plan: sensors, inverter, BMS, TUM battery, firewalled controller

Status: v2 (after one independent red-team; log in section 9). NOT built yet. Read `INFORMATION_CONTRACT.md` first. This file says how we build it, how we prove it, and what "done" means.

## 0. What the TUM data does and does not say about the battery (corrected twice)

Two earlier statements of mine were wrong and are retracted:
1. "The generic battery is 5x too resistive": invalid, it ignored that the toy motor is about 7.5x smaller than the Tesla's.
2. "Per-unit it is within 1 to 1.5x of the real pack (1.5 to 3.1 %)": also wrong. It used a 100 ms regression that is biased low by held samples.

Measured here (5 local tracks [T1]); the apparent pack resistance depends on the difference window, because the CAN current is held for several samples while the voltage is not:

| difference window | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---|---|---|---|
| R_eff, tracks 0 / 1 / 10 / 100 / 103 (mOhm) | 7.5 to 16 | 22 to 43 | 33 to 52 | 42 to 60 |

At 0.5 to 1 s the value mixes ohmic and polarization resistance, so it is an upper bound on a fast resistance and a fair measure of the sag a 1 s event sees. In per-unit (R x 667 A / 348 V) that is roughly **6 % to 11 %**, against **2.8 % to 4.4 %** for the generic toy pack. **Unresolved: the toy pack may be 2 to 3 times too low in resistance.** Agent B's first deliverable is a held-sample-aware estimate with uncertainty, before any scaling.

**No cold data exists in these five tracks.** The "-25 C" in track 1 is 10 consecutive rows (index 4711 to 4720) of a sentinel value; the rest of the data sits at 20 to 34 C, one temperature band per track. So R0(T), the data-grounded cold-pack event, and any temperature-monotonicity test are NOT supported. The fit is over current and SOC only; R0(T) stays the generic Arrhenius curve, labelled an assumption. Agent B will try to find cold tracks (best effort, cap 15 tracks / 1 GB, "cold" = `hv_temp_min` below 10 C for at least 60 s); if none, we say so.

Other data facts: current held on 33 to 66 % of rows, voltage 4 to 10 %; SOC glitches exist (track 10 jumps by +51.2, track 1 holds 9.2 for 240 rows); discharge current is negative in the files, so the fit must state its sign convention; regen-to-discharge ratio is **0.16 to 0.40** by current (0.19 to 0.43 by power), not "a quarter", and the toy's existing limits give 0.6.

## 1. Architecture and interfaces (frozen before any agent starts)

| File | Owner | Contents |
|---|---|---|
| `interfaces.py` | me, FIRST | frozen dataclasses `Measurements`, `BMSMsg`; configs; `CONTRACT_FIELDS`; **truth-dict key list**; **fit-JSON schema**; the shared `park()` and rotation functions; RNG helper using `SeedSequence.spawn` (one stream each for sensors, inverter, BMS, beliefs, so changing the controller never changes the noise) |
| `sensors.py` | Agent S | `SensorSuite`: current sensors, resolver, PLL, Vdc divider and filter, request hold; identity-bypass flags |
| `inverter.py` | Agent S | delay FIFO, angle-error rotation, dead-time and drop; returns `u_extra` for the plant hook |
| `battery_fit.py`, `bms.py`, `data/tum_battery_fit.json` | Agent B | fit (current and SOC only), limit envelope, scaling, BMS message model |
| `control_fw.py`, `firewalled_loop.py`, `battery.py` hook, `plant.py` hook `apply_voltage(u_extra=None)` | me | the controller on `Measurements`; loop; logging with truth, `meas_*`, `est_*`, `belief_*` separated |

I write a fixture fit-JSON by hand so my battery hook is not blocked on Agent B (the one real dependency).

## 2. The battery adaptation (Agent B)

Method:
1. Filter sentinels and glitches (temperature below -20, SOC jumps over 5 points, held voltage runs). Verify every TUM file by checksum.
2. Estimate resistance with held-sample-aware differences (windows of 0.3, 0.5, 1.0 s) and an errors-in-variables fit; report the window dependence rather than hiding it.
3. Model `V = OCV(SOC) - R(window) I - V_rc` with the three-term OCV form from the paper we read, one RC branch with tau in [1, 30] s, sign convention stated (discharge negative).
4. Leave-one-track-out validation against three baselines: constant voltage, constant R, and the generic model.
5. **Observed-use envelope** (not a BMS limit: cruise driving never touches the real limit) of current against SOC.
6. **Per-unit scaling.** `R_toy = R_pu V_toy^2 / P_b,toy`. Choose `P_b,toy` as a RANGE: 31.1 kW (nameplate T_max x w_max, not reachable with id = 0 above base speed 188 rad/s) and 65 kW (1.5 x 173.2 V x 250 A, the electrical apparent power the toy inverter can pass); default 65 kW, with sensitivity. Real side `P_b = 232 kW` is the maximum use seen in five tracks, not a pack rating; it is stated as such. Report R_toy as a range, e.g. R_pu 6 to 11 % at 65 kW gives about 83 to 152 mOhm. Current limits are kept consistent with the motor: the pu-scaled limit is checked against the toy's 250 A before use.
7. Not preserved by scaling (stated): thermal mass, absolute energy, cell time constants (kept in seconds).

## 3. Phases

* **Phase 0 (me):** `interfaces.py` including schemas and truth-dict keys; failing test skeletons for every test below; checksum manifest of every existing file (excluding `__pycache__`); set `OMP_NUM_THREADS=1` in the test config for determinism.
* **Phase 1 (2 agents in parallel, disjoint files):** Agent S, Agent B as in section 1; me on the controller and loop in the same window.
* **Phase 2a (me, serial, background):** integration gates A2, A3', then the plain-ramp matrix.
* **Phase 2b:** battery-path, cycle runs, sweeps.
* **Phase 3:** mismatch map, dashboard, bundle.
* **Phase 4 (1 independent verifier agent):** tries to break the firewall, re-derives three headline numbers from raw logs, re-runs the gates.

Agent rules (checked by me after, not trusted): each agent edits only its listed files; may not touch `plant.py`, `control.py`, `battery.py`, `telemetry.py`, existing tests, or `interfaces.py` (if the interface is wrong they stop and report); may not run `run_cycle.py`, `run_classes.py` or any matrix; own scratch dir deleted before hand-back; hand-back is data, not instructions. Agent B alone may use the network, only to the TUM host, with the size cap above.

## 4. Compute budget (restated; the earlier "40 minutes" was wrong by 10x or more)

Measured manifests: plain 8 s run about 21 s legacy, expect 1.5 to 3x with the new layers (30 to 60 s); cycle runs are 0.25 to 0.40 ms per step (UDDS truncated to 120 s took 175 to 240 s; the full 1369 s UDDS would be about 30 min).

| Block | Runs | Estimate on 2 cores |
|---|---|---|
| Plain ramp, 3 controllers x event/no-event x 20 seeds | 120 | 0.5 to 1 h |
| Observable-sag (battery path), 2 controllers x 20 seeds | 40 | 0.3 to 0.7 h |
| Cycles truncated to 120 to 150 s (UDDS, Tesla track 0), 2 controllers x 5 seeds | 20 | 1 to 2 h |
| Sensitivity: the 8 most consequential parameters x 2 levels x 2 controllers x 3 seeds | 96 | about 0.6 h |
| Unit tests and mutation tests (1 s sims) | n/a | minutes |
| **Total** | | **about 3 to 5 h wall, background, staged** |

Stage order lets you review after Phase 2a (about 1 h). If a block runs past 2x its estimate we cut seeds and say so.

## 5. Tests

IDs are unique. **U** = unit test; **A** = acceptance gate. Every layer has a **mutation test**: the layer is deliberately broken and the matching test must FAIL.

### 5.1 Sensors and inverter (Agent S)

* U1 same seed gives identical streams; different seed differs.
* U2 identity mode: all errors off gives measured = truth to 1e-12.
* U3 quantization: noise off, outputs are integer multiples of LSB (1e-9), error at most LSB/2.
* U4 noise: 1e5 samples, mean within 4 sigma/sqrt(N) of the bias, std within 2 % of spec.
* U5 gain and offset recovered by regression (slope 1e-3, intercept 0.01 A).
* U6 full-scale clipping.
* U7 equal offsets on three phases cancel in dq with the full Clarke; do NOT cancel with the two-sensor form; one-phase offset gives a spectral peak at the electrical frequency within 10 % of the analytic amplitude.
* U8 angle error: with `delta = theta_m - theta_true`, `i_dq_m = R(-delta) i_dq` to 1e-9 and `i_d_m = +i_q sin(delta)` (numeric value in the test).
* U9 PLL (second order, damping 0.707): overshoot within 20 % (relative) of the analytic value (**build correction:** 4.3 % is the SPEED response to a speed step; an ANGLE step gives 20.8 % overshoot in the angle estimate; both are tested), zero error at constant speed, noise within 25 % of the analytic prediction var(omega_hat_el) = sigma^2 dt wn^3 / (4 zeta).
* U10 Vdc filter: step time constant within 5 %, exact discretization at both tau values; DC gain 1 + g.
* U11 request hold: analytic reference unchanged (M7a); data reference equals the held, quantized samples (M7b).
* U12 delay: impulse at step k appears at k + d exactly for d = 0, 1, 2.
* U13 dead-time: sign follows the current; DC part of the dq error within 1 % of `(4/pi)(Td/Tpwm)Vdc` (ripple excluded); zero when Td = 0; scales with tau.
* U14 near zero current the error is bounded and continuous (max step below 10 % of the amplitude).
* U15 rotation: `u_true = R(delta) u_cmd` to 1e-9.
* U16 order of operations: delivered voltage after the plant clip never exceeds the true ceiling (property test over 1e5 random and adversarial commands); the final voltage with inverter error stays within the stated dead-time bound.
* U17 the zero-sequence part from unequal gain errors does not enter dq with the full Clarke.

### 5.2 BMS and battery (Agent B)

* U18 BMS cadence on 50 ms boundaries; `age_s` in [0, period + latency]; **causality**: perturbing truth after t leaves the message at t unchanged.
* U19 the message does not reflect an injected fault until the latency has passed.
* U20 fit deterministic; parameters positive; OCV monotone in SOC; sign convention test (discharge negative gives positive resistance).
* U21 held-out RMSE (leave-one-track-out) beats the best of three baselines (constant V, constant R, generic) by a margin reported with its spread; if it does not, the fit file says `status: "not_better_than_baseline"` and the toy keeps the generic pack. Reported, never hidden.
* U22 observed-use envelope covers at least 99 % of observed discharge and regen current (documented as true by construction, so a negative control: a shifted trace must fall outside it).
* U23 independent scaling check: simulate a real-scale pack and the toy pack through the SAME per-unit power step, compute per-unit sag in the test WITHOUT calling the scaling function, and compare within 2 %.
* U24 provenance: every TUM file used has a recorded checksum that verifies; the fit hash reproduces from the data.
* U25 glitch filter removes the known sentinel (track 1 index 4711 to 4720) and the +51.2 SOC jump (track 10) and nothing else.

### 5.3 Firewall and integration (me)

* U26 static: `control_fw.py` imports nothing from plant, battery, BMS or ledger modules; its constructor takes only config dataclasses (`inspect.signature`); `command` takes `Measurements`; CI grep finds none of `plant.`, `self.pl.`, `led`, `ledger`.
* U27 **non-interference, three parts**: (a) replay: identical measurement streams with different hidden truth (R, L, flux, disturbance, battery state) give bit-identical commands over 2000 steps; (b) poison: every truth field not in the bundle set to NaN, including objects reachable from module globals, output finite and identical; (c) mutation: inject `plant.psi_p` into the controller and require (a) and (b) to FAIL.
* U28 `Measurements` frozen; its fields equal `CONTRACT_FIELDS`; the contract table is checked against the constant.
* U29 **legacy equivalence** (identity mode, exact beliefs, delay 0, advance off): trajectories equal the legacy controller within 1e-9 for naive and aw, with and without governor, 1 s run; the stored run A reproduces its logged energy gap. Legacy `control.py` is also run unchanged against the new plant hook.
* U30 the existing 283-test suite passes unchanged in the final tree.
* U31 ledger decomposition: `d_u = d_u_ctrl_clip + d_u_delay + d_u_rot + d_u_plant_clip + d_u_inv` to 1e-9 in every logged step; `telemetry.derive` identity `dP_elec = 1.5 d_u . i` still exact.
* U32 per-channel gaps: with errors on, EVERY measured channel (including PLL speed, `vdc_m`, each `bms_msg` field) has a gap with nonzero variance within 20 % of the prediction; a gap that is identically zero fails.
* U33 logging separation: truth columns keep names and values; `meas_*`, `est_*`, `belief_*` exist.
* U34 beliefs: draws in contract ranges, seeded, in the manifest, from their own RNG stream.
* U35 **seed independence**: noise streams are identical between naive and aw runs at equal seed (common random numbers); belief draws do not share a generator with sensor draws.
* U36 **tau independence**: sensors and inverter behave the same in seconds at tau = 1e-4 and 2e-4.
* U37 **BMS wiring**: poisoning `bms_msg` changes the controller output in the battery-derate scenario; zeroing the sensors' `vdc_m` changes the ceiling.
* U38 stability versus delay: the protected controller stays inside the demo envelope at delay 0, 1, 2, with and without angle advance (and the result, positive or negative, is reported).
* U39 determinism: two runs at equal seed are bit-identical on 2 cores.

### 5.4 Acceptance gates (fixed before any result)

* A1 bounded input: `|u_app| / u_ceiling <= 1 + 1e-9` in every matrix run against the TRUE ceiling; the final voltage with inverter error is reported against its own bound.
* A2 U27, U29 and U30 pass. Hard gate: no result is shown if any fails.
* A3 wiring is visible: with errors on, each layer (sensor, inverter, belief, BMS) alone moves at least one mismatch row, shown per layer, so a layer cannot be unwired silently. (The earlier "monotone index" test is dropped: dithering near saturation can make it non-monotone.)
* A4 statistics: 20 seeds per condition; rates with Wilson 95 % intervals; comparisons by exact test. With n = 20 the Wilson lower bound is 0.839 for 20 of 20, 0.764 for 19, 0.640 for 17, so an absolute lower-bound threshold of 0.80 is not used.
* A5 **comparison, not absolute thresholds:** under imperfect information, in the observable-sag event, the protected controller's ultimate error is below 0.25x the naive controller's median, Mann-Whitney p < 0.01 at n = 20. "Ultimate" is defined once: the maximum of `|psi_tilde|` over the last 1.0 s of an 8 s run. If naive no longer fails, that is reported as a result (the failure depended on perfect information), not a test to fix.
* A6 the demo envelope (`|psi_hat| <= 0.15 Wb`, `|i| <= 275 A`) survival rate is REPORTED with Wilson intervals for each controller; the 0.033 Wb absolute threshold is dropped (the existing robust controller peaks at 0.034 Wb even with perfect information). Thresholds for any absolute claim are set from the perfect-information pilot by a pre-registered rule and signed off by Theory (decision D3).
* A7 sensitivity on the 8 most consequential [T2]/[T3] parameters at 0.5x and 2x; conclusions A5 must hold or the parameter is listed as conclusion-deciding.
* A8 energy identities on truth columns close with the previous tolerances.
* A9 hidden saturation: `d_u_plant_clip` is exactly zero when every imperfection is off and the ceiling is known; it is nonzero in the sag event, and its size is reported (it is lag-limited in the observable event, 100 V or more in the oracle-ceiling event).
* A10 mutation meta-tests for every layer (noise x10, delay bypassed, dead-time sign flipped, angle error dropped, BMS made non-causal, controller reading truth) each make the matching unit test fail.
* A11 dashboard and bundle: both audits pass, new sections show measured vs true, the zip is rebuilt.
* A12 the plain-event finding is documented: `aw` equals `naive` when the ceiling is not observable, and the oracle-ceiling reproduction reproduces experiments A to D within 1e-6 of the logged values.

## 6. Open risks

1. Sensor and inverter magnitudes are [T2]; TUM is post-ECU CAN data and cannot calibrate phase-sensor noise.
2. Per-unit scaling preserves per-unit sag and limits, not thermal mass or absolute energy; `P_b` choices move `R_toy` by 2x, so it is reported as a range.
3. The BIBS proof was written for the true shortfall; `du_known` may change it (D1).
4. Belief errors model the effect of R, L, J error on the loop, not heating or saturation physics.
5. If matrix time runs more than 2x over, we cut seeds and say so.

## 7. Decisions needed

* D1 (Theory): BIBS/UUB for the true shortfall, for `du_known`, or for the hidden remainder. We build on `du_known`.
* D2: EKF flux-estimator comparison (EKF for flux, RLS for Ld, d-axis pulse for R, the structure MathWorks documents)? Not in Phase 1.
* D3 (Theory): sign off the pre-registered rule behind A5 and A6.

## 8. What each agent returns

Files created, test counts, any deviation from the contract, and every [T2] number with the source it found or the label kept.

## 9. Build log (v2.1)

* Agent S: 39 tests pass (17 U-tests, 18 mutation tests). Sensor current parameters now cited to a LEM datasheet; dead-time, angle errors and Vdc errors stay [T2] with no source. Bound on dead-time error is (4/3)(1+c)K, not (4/pi)K (the latter is only the DC part).
* Agent B: 24 tests pass. Fit status ok but weak (leave-one-track-out RMSE 2.98 V vs 3.82 V constant-R baseline, 4 of 5 folds). Not identified: temperature, RC branch, OCV shape.
* Lead: 31 firewall tests pass, including bit-level legacy equivalence (8 parametrized cases, a battery case, and the stored run A) and the existing suite.
* Build corrections to the contract: angle advance is `d` not `d+0.5`; compensation lives in the inverter; Park is in the measurement chain.

## 10. Red-team log (one independent reviewer, 19 tool calls)

Verified by me against the data: the "-25 C" sentinel (10 rows), and the held-sample bias (resistance grows with the difference window, table in section 0).
Adopted: all eight top items. Cold-data claim removed; per-unit battery claim retracted and restated as a range; plain-event consequence (`aw` equals `naive` without an observable ceiling) added with two event types; legacy equivalence redefined with identity mode and the verified one-step GEM angle offset; full leak checklist added to the contract; `ceiling_hat` defined with the 300 V regen cap and the BMS consumer defined; absolute thresholds replaced by comparisons with a defined "ultimate" window; compute restated in hours.
Also adopted: sign-convention fix (`i_d_m = +i_q sin delta`), second-order PLL test, dead-time numbers, full Clarke vs two-sensor, tau independence, seed independence, bms wiring test, stability versus delay, legacy suite gate, replacement of circular scaling tests, unique test IDs, frozen schemas.
Not adopted: nothing rejected. One item is deferred: a hardware-in-the-loop style test of the PLL against a real encoder trace (no such data).

## Build log: sensitivity sweep (A7), 2026-10-28
8 parameters x {0.5x, 2x} (inverter delay: 0 and 2 steps) x {naive, aw} x 3 seeds, observable-sag battery scenario (`run_sensitivity.py`, `analyze_sensitivity.py`). Rule at n=3: worst aw seed < 0.25x best naive seed (complete separation; an exact test cannot reach p<0.01 with 3 per arm).
Result: A5 holds at every level (ratio 0.023 to 0.040 vs 0.25 threshold); no conclusion-deciding parameter; A1 holds (max 1.0). aw stays inside the envelope 3/3 everywhere; naive 0/3 everywhere.
Wiring check: sensor-error metrics move with the parameters (e.g. i_q measurement gap 0.66 -> 1.75 A rms over the current-offset range). `bms_latency` has NO effect on aw because aw does not read the BMS message (only awg does), so that row did not test the controller that uses it; rerun with awg if BMS timing matters.

## Build log: aw range clamp A/B (hidden step, plain, 20 seeds), 2026-10-28
Option `ControllerConfig.clamp_psi` (default False): in aw mode also project psi_hat onto the prior range (0.03, 0.10 Wb), as robust already does. `run_clamp_ab.py`.
aw: psi_hat peak 0.309 Wb, i peak 288 A, envelope 0/20, ultimate flux error 0.0102.  aw+clamp: 0.100, 256 A, 20/20, 0.0047.
awg: 0.307, 240 A, 0/20, 0.0124.  awg+clamp: 0.100, 209 A, 20/20, 0.0055.  robust: 0.100, 256 A, 20/20, 0.0340.
Voltage bound holds (1.0). Speed error after the event: aw 12.8 -> 9.8 rad/s; awg 11.0 -> 12.4 (slightly worse). Cost: the clamp assumes the prior range brackets the true flux; if the magnet were outside 0.03-0.10 Wb the clamp would bias the estimate. The default stays off; adopting it is the team's call (Theory, decision D1).

## Build log: mutation meta-tests (A10), 2026-10-28
`tools/mutation_check.py` breaks one layer in a copy of the repo and requires a test failure. Unmutated baseline passes. 7 of 7 mutations caught (noise x10, delay bypassed, dead-time sign flipped, angle error dropped, BMS without latency, controller importing the plant, plant ignoring the inverter error). Weak spots to note: "angle error dropped" was first caught by the noise-statistics test, not a dedicated angle test, and "plant ignores the inverter error" was caught only by the legacy-suite test (U30). Both are caught, but a dedicated test for each would be stronger.

## Build log: d-axis chatter in cycle runs, 2026-10-28
At the cycle step (tau = 2e-4) with a one-step computation delay, the single feedback gain k_i = 3 V/A made the d-axis current loop unstable (tau*k_i/Ld = 1.6 > 1): the d command flipped about +/-310 V every few steps (step-to-step change 194 V rms; identity mode 0.6 V). Plain/battery matrices (tau = 1e-4, ratio 0.8) were mildly affected (5.2 V vs 2.7 V per step, no chatter) and are unchanged.
Option `ControllerConfig.d_gain_by_L` (default False): k_d = k_i * Ld/Lq so both loops have the same bandwidth. 30 s UDDS test: chatter 194 -> 21 V, speed rms 6.0 -> 0.12 rad/s, plant-saturation fraction 5.2% -> 0.1%. Cycle runs re-done with the fix (old kept in data/cycles_v1_dchatter).
Cycle results after the fix (5 seeds, awg vs naive): UDDS speed rms 3.2 vs 4.3, derate-window rms 17.7 vs 23.7 rad/s, flux peak 0.071 vs 0.185 Wb; Tesla 8.2 vs 9.6, 52.9 vs 61.0, 0.071 vs 0.169. Plant saturation 15% -> about 1%. Voltage bound holds.
