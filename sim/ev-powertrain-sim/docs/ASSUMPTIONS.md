# Assumptions register (generated; do not edit by hand: `python tools/make_assumptions.py`)

Provenance tags: **[T1]** measured from the TUM traces by a script in this repo; **[T2]** typical class value, UNVERIFIED; **[T3]** design choice (swept in the sensitivity study where it matters). Anything without a tag is a setting of the simulation itself.
Independent audit of these numbers: `docs/AUDIT_PARAMETERS.md` (6 wrong, 14 suspicious items found 2026-11-09, listed there).

## 1. The toy scale
The simulator is a TOY: 3 pole pairs, 300 V bus (ceiling 173.2 V), motor from Gym-Electric-Motor 3.0.3 defaults. "Realistic" means the controller receives the same KINDS of imperfect information a production drive does, at toy scale; it is not a digital twin of the Tesla.

## 2. True plant (units in `data/params.json`)
| Symbol | Value | Unit | Source |
|---|---|---|---|
| p | 3 |  | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| Ld | 0.00037 | H | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| Lq | 0.0012 | H | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| R | 0.018 | ohm | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| psi | 0.066 | Wb | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| J | 0.53883 | kg m^2 | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| a_load | 0 | N m | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| b_load | 0.02 | N m s/rad | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| c_load | 0.0001 | N m s^2/rad^2 | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |
| tau | 0.0001 | s | plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults |

Limits: current 400 A (plant termination), speed 418.9 rad/s. Voltage limit is a disc of radius V_dc/sqrt(3) (our own radial projection). Cycle runs replace J and the load terms with a fitted vehicle (`cycles.fit_vehicle_to_motor`).

## 3. Settings of the information path and controller
### Sensors (`SensorConfig`)


| Setting | Default | Source comment |
|---|---|---|
| `tau` | `0.0001` |  |
| `p` | `3` | pole pairs (nameplate constant; electrical angle = p x mechanical) |
| `identity` | `False` | True: measured == truth exactly; bypasses PLL, filter, quantization (legacy equivalence) |
| `i_range_a` | `400.0` | full scale of each current sensor, +/- [T3] |
| `adc_bits` | `12` | [T3] |
| `i_gain_err` | `0.01` | half-width of uniform per-phase gain error [T2] |
| `i_offset_a` | `2.0` | half-width of uniform per-phase offset, A [T2] |
| `i_noise_lsb` | `1.0` | white noise sigma in LSB [T2] |
| `two_sensor` | `False` | sensitivity case |
| `theta_offset_rad` | `0.017453292519943295` | half-width of uniform per-run offset [T2] |
| `theta_noise_rad` | `0.005235987755982988` | white noise sigma [T2] |
| `theta_ripple_rad` | `0.003490658503988659` | 1x and 2x mechanical harmonic amplitude [T2] |
| `pll_wn` | `300.0` | rad/s natural frequency [T3] |
| `pll_zeta` | `0.707` | [T3] |
| `vdc_range_v` | `500.0` | ADC full scale [T3] |
| `vdc_gain_err` | `0.005` | half-width of uniform per-run gain error [T2] |
| `vdc_noise_v` | `0.5` | white noise sigma [T2] |
| `vdc_corner_hz` | `500.0` | first-order low-pass corner [T3] |
| `ref_quant` | `0.0` | request quantization (rad/s); 0 = off. 0.1 for TUM-sourced references |
| `ref_hold_s` | `0.0` | request sample-and-hold period; 0 = off. 0.1 for TUM-sourced references |

### Inverter (`InverterConfig`)


| Setting | Default | Source comment |
|---|---|---|
| `tau` | `0.0001` |  |
| `delay_steps` | `1` | extra computation delay d in {0,1,2}; total (0.5 + d) tau [T3] |
| `dead_time_s` | `2e-06` | [T2] |
| `device_drop_v` | `1.5` | [T2] |
| `smooth_band_a` | `1.0` | current band around zero where the distortion is smoothed [T3] |
| `compensation` | `0.7` | fraction of the distortion the controller removes [T3] |
| `enabled` | `True` |  |

### BMS (`BMSConfig`)


| Setting | Default | Source comment |
|---|---|---|
| `period_s` | `0.05` | [T3] |
| `latency_s` | `0.02` | [T3] |
| `quant_a` | `1.0` | [T3] |
| `rel_err` | `0.05` | sigma of multiplicative error [T2] |
| `enabled` | `True` |  |

### Controller beliefs about the plant (`BeliefConfig`): drawn per seed, relative to truth


| Setting | Default | Source comment |
|---|---|---|
| `dR_range` | `(-0.15, 0.4)` |  |
| `dL_range` | `(0.0, 0.25)` |  |
| `dJ_range` | `(-0.2, 0.2)` |  |
| `dpsi_range` | `(-0.1, 0.1)` | error of the NAMEPLATE flux used for Kt and the governor (not the adaptive psi_hat) |
| `psi_start_frac` | `0.7` | existing design: start 30 % low |
| `psi_range` | `(0.03, 0.1)` | existing projection prior for the robust mode |
| `exact` | `False` | True: beliefs equal truth (legacy equivalence; psi_hat0 still 0.7 psi) |

### Controller (`ControllerConfig`)


| Setting | Default | Source comment |
|---|---|---|
| `mode` | `naive` | naive / robust / aw |
| `governor` | `False` |  |
| `vdc_aware` | `False` | governor uses vdc_m rather than the nominal ceiling |
| `bms_aware` | `False` | governor also limits iq from the BMS message |
| `angle_advance` | `True` | theta_adv = d * omega_m_m * p * tau [T3]  (d = delay_steps; code and contract use d, not d + 0.5) |
| `delay_steps` | `1` | must equal InverterConfig.delay_steps (the controller knows its own delay) |
| `gamma` | `2e-05` |  |
| `k_i` | `3.0` |  |
| `sigma` | `2.0` |  |
| `d_gain_by_L` | `False` | True: d-axis feedback gain scaled by Ld/Lq so both current loops have the same bandwidth |
| `clamp_psi` | `False` | True: also project psi_hat onto the prior range in aw mode (robust always does) |
| `i_lim` | `250.0` |  |
| `speed_bw` | `20.0` |  |
| `gov_factor` | `0.75` |  |
| `tau` | `0.0001` |  |
| `p` | `3` | pole pairs: a nameplate constant the controller legitimately knows |
| `eta` | `0.9` | drivetrain efficiency belief used by the BMS-aware limit |

## 4. Battery
Pack numbers come from `data/tum_battery_fit.json` ([T1]: resistance scale, capacity, current limits; `docs/BATTERY_FIT.md`). The open-circuit-voltage curve, the temperature law of R0, the RC branch and the thermal mass are generic NMC assumptions [T2]. `Battery()` with no fit file uses defaults that differ from the fitted pack (audit item).

## 5. Run design choices
* Event: ceiling forced to 25 V for t = 4.5 to 5.5 s (`hidden`, `oracle`); BMS derate for the battery scenarios; cycle runs derate to 0.02 for 3 s at peak power minus 0.5 s.
* Integration step tau = 1e-4 s (plain and battery runs), 2e-4 s (cycle runs). The speed loop runs every 10 steps.
* Seeds: 20 per scenario and controller (matrix), 5 (cycles), 3 (sensitivity). Beliefs, sensor offsets and noise are drawn per seed from separate streams.
* Stored seed-0 rows: every 17th step (matrix), every 23rd (cycles); full rate only in `data/audit/fr_*`.
* Tesla trace: scaled to the toy motor (vehicle fitted to the motor); asks for up to 1.5x its torque by design.
* Envelope test `envelope_ok`: current below 275 A and flux estimate below 0.15 Wb.
* Statistics: Wilson 95 % intervals; Mann-Whitney for the pass/fail rule A5; n = 3 sweep uses complete separation.

## 6. Known limits that are assumptions, not bugs
Single-run evidence, not proofs. Sensor, inverter and BMS values are assumed. TUM request holding/quantization (`ref_hold_s`, `ref_quant`) is available but off in the runs. Current-limit numbers (240/250/275/400 A) have different roles (pack, controller, envelope, plant) and are not one parameter.
