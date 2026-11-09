# Two-state nonlinear model (speed and q-current) — v1, 2026-11-02

State x = [ω, i_q] (ω = mechanical speed, rad/s; i_q = torque-producing current, A). Input u = u_q (V, the voltage actually delivered).
Everything below is an exact regrouping of the three-state dq plant in `docs/MATH_REFERENCE.tex`; nothing is dropped. The d-axis terms that
the usual i_d = 0 reduction throws away stay in the disturbance, so the equations hold for every row of every run.

## Equations
```
ω̇   = f1 + g1·i_q + d1                f1 = −T_fric(ω)/J                    g1 = 1.5·p·ψ / J
i̇_q = f2 + g2·u_q + d2                f2 = (−R·i_q − p·ω·ψ)/Lq             g2 = 1 / Lq

d1 = ( −T_dist  +  1.5·p·(Ld−Lq)·i_d·i_q ) / J         road load        + reluctance torque (zero when i_d = 0)
d2 = ( δu_q  −  p·ω·Ld·i_d ) / Lq                      actuator mismatch + d-axis cross-coupling (zero when i_d = 0)
δu_q = u_q(delivered) − u_q(requested)                  exact, from the six-piece voltage ledger
T_fric = a·sgn(ω) + b·ω + c·ω·|ω|                      a = 0, b = 0.02 N·m·s/rad, c = 1e-4 N·m·s²/rad²
```
Bilinear coupling: the term p·ω·ψ in f2 (speed times flux). Saturation: |u_q| ≤ sqrt(u_max² − u_d²), u_max = V_dc/√3 capped at 173.2 V (radial disc).

## Tracking errors (backstepping form, columns `ts_z1`, `ts_z2`)
* z1 = ω − ω_ref (rad/s), the velocity tracking error. (`e_speed` is the same with the opposite sign.)
* z2 = i_q − iq_ref (A), the second error: the current minus its virtual control, the controller's speed-loop output. Equals `e_q` in firewalled runs.

## Nominal numbers and units (true plant; controller beliefs are drawn per seed from the ranges)
| Symbol | Meaning | True value | Unit | Controller belief range (per seed) |
|---|---|---|---|---|
| p | pole pairs | 3 | – | known |
| R | stator resistance | 0.018 | Ω | −15 % to +40 % |
| Ld, Lq | inductances | 0.37, 1.2 | mH | 0 to +25 % |
| ψ | magnet flux | 0.066 | Wb | start at 70 % of true, learned online, range 0.03 to 0.10 |
| J | inertia (rotor + load) | 0.5388 | kg·m² | ±20 % |
| b, c, a | viscous, quadratic, constant load | 0.02, 1e-4, 0 | see above | not adapted |
| u_max | voltage ceiling | 173.2 (V_dc 300 V) | V | belief from sensed V_dc; BMS-aware option |
| i_lim | current limit used by the controller | 250 (275 envelope) | A | – |
Source of every row: `plant.py`, `telemetry.params_from_plant`, `interfaces.BeliefConfig`; run-specific draws are in each run's `.json` (`belief_draws`, `beliefs`).

## How exact, and how big is the reduction? (checked on full-rate runs, 0.1 ms, `tests/test_telemetry.py`)
* Algebra: the two-state residuals equal the three-state residuals row by row (max difference 1e-11).
* Discretization: forward-difference residual is 0.4 rad/s² (1 % of dω/dt rms) for ω and 67 A/s (1.7 %) for i_q.
* Timing note: row k holds the state at the start of a step and the voltage delivered during it, so the check is a forward difference against same-row voltage. The old check (central difference, previous-row voltage) overstated the i_q residual about 18x. Fixed in `telemetry.derive`.
* Size of the dropped i_d terms: battery sag, protected controller: |ts_d1_red| is 17 % of |d1| rms, |ts_d2_red| 13 % of |d2| rms. Hidden voltage limit: i_d reaches 27.6 A rms and the dropped terms are 77 % of d1. So the i_d terms are NOT negligible under saturation; any bound d̄ must include them.

## Measured disturbance bounds (stored seed-0 runs; `python tools/disturbance_bounds.py`, full data in `data/disturbance_bounds.csv`)
Single-run maxima (cycle rows use the fitted vehicle inertia; an earlier version of this table used the motor's own inertia for them and overstated the cycle reluctance terms about 3x), not envelopes over seeds (only seed 0 is stored). Unmodeled δu_q = delivered − requested, EXCLUDING the controller's own known clip (ledger pieces: advance, delay, rotation, plant clip, inverter).
| Run | controller | max i_d (A) | road load |d_w| max (rad/s²) | reluctance max (rad/s²) | unmodeled δu_q max (V) | cross-coupling max (A/s) |
|---|---|---|---|---|---|---|---|
| tesla | awg | 25 | 0.5 | 10.0 | 141.0 | 3442 |
| tesla | naive | 206 | 0.5 | 86.7 | 141.2 | 26928 |
| udds | awg | 51 | 0.0 | 9.6 | 207.1 | 8239 |
| udds | naive | 238 | 0.0 | 44.3 | 207.5 | 24750 |
| base | awg | 6 | 18.6 | 6.4 | 27.3 | 1081 |
| base | naive | 6 | 18.6 | 6.4 | 27.3 | 1081 |
| hidden | awg | 100 | 18.6 | 34.8 | 150.7 | 18452 |
| hidden | naive | 731 | 18.6 | 1515.1 | 150.8 | 103929 |
| oracle | awg | 100 | 18.6 | 24.1 | 27.3 | 18510 |
| oracle | naive | 727 | 18.6 | 1513.8 | 27.3 | 103570 |
| sag | awg | 24 | 18.6 | 11.7 | 27.1 | 3729 |
| sag | naive | 250 | 18.6 | 390.5 | 27.1 | 43320 |
| sagbase | awg | 6 | 18.6 | 6.1 | 27.1 | 1020 |
| sagbase | naive | 6 | 18.6 | 6.1 | 27.1 | 1020 |

Reading it:
* d1 road load: |d_w| ≤ 18.6 rad/s² (10 N·m / J) in all plain and battery scenarios; cycle grade torque is smaller.
* Unmodeled δu_q: 27 V with no fault (delay, rotation, inverter). With a fault it grows to about the size of the fault: 141 V (Tesla), 207 V (UDDS), 150 V (hidden limit). That is the actuator-saturation disturbance the research notes describe.
* Reluctance and cross-coupling bounds exist only if i_d is bounded. Protected runs hold i_d ≲ 100 A in the worst case (hidden limit; measured max 100.3 A, oracle aw), 6 A with no fault. The naive adapter lets i_d reach 731 A, so no bound holds for it. The bound is a property of the controller, not of the plant.
* The d-axis loop gives q-axis voltage priority when saturated, so i_d is the least well controlled state under saturation. Worth a design decision.

## Not yet done
* Bounds over many seeds (only seed 0 is stored; the all-seed metrics CSVs do not carry these maxima).
* Column `ts_d2` includes δu_q, which depends on the controller and saturation, so d̄2 is state-dependent, not a constant.
* Stored seed-0 rows are every 17th/23rd step: `ts_res_*` are valid only on full-rate files (`data/audit/fr_*`).
