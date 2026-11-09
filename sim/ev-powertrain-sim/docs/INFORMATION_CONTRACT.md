# Information contract: what the controller is allowed to receive

Status: v2.1 (red-teamed; adjusted during the build, see the notes marked Build). Nothing here is built yet. Every number carries a provenance tag:

* **[T1] measured here** from the TUM Tesla Model 3 traces by a script in this repo.
* **[T2] typical class value, UNVERIFIED**: from memory, no fetched source gives these numbers. Agent S must cite a source for each or keep the label. Each is swept (gate A7).
* **[T3] design choice**, with a sweep.

The sim stays a TOY (3 pole pairs, 300 V bus scale). "Realistic" means the controller sees the same KINDS of imperfect information a production drive sees, at toy scale. It is not a digital twin of the Tesla.

All time constants are in SECONDS and all PWM quantities follow the integration step `tau` (1e-4 s for the plain runs, 2e-4 s for the cycle runs), never "10 kHz" hard-coded. Tests run at both.

## 1. The principle: an information firewall

The plant owns the truth. The controller module receives one frozen object per step, `Measurements`, plus its own previous outputs, and nothing else.

```
 PLANT (truth) -> SENSORS (imperfect) -> Measurements -> CONTROLLER -> clip vs ceiling_hat
      ^                                                                   |
      |  true clip -> dead-time/drop error added AFTER the clip  <- rotation by angle error <- delay FIFO
```

Truth is still logged (separate columns). The firewall controls only what the controller reads.

## 2. What the controller RECEIVES

| # | Channel | Real source | Imperfections | Params |
|---|---|---|---|---|
| M1 | `i_a_m, i_b_m, i_c_m` | three Hall or shunt sensors + ADC | per-phase gain and offset (drawn once per run), white noise, 12-bit quantization over the range, clipping at full scale. Note LSB 0.195 A is 1.5 % of a 12.6 A cruise current and ±2 A offsets are 16 % of it: this matters at low load | range ±400 A [T3]; gain ±1 % [T2]; offset ±2 A [T2]; noise 1 LSB [T2] |
| M2 | `theta_m` electrical angle | resolver or encoder + decoder | constant offset per run, white noise, 1x and 2x harmonic ripple, quantization | offset ±1.0 deg el [T2]; noise 0.3 deg el [T2]; ripple 0.2 deg el [T2] |
| M3 | `omega_m_m` speed | second-order PLL on M2, never the true speed | lag, noise from M2. Used EVERYWHERE the controller needs speed: speed loop, decoupling terms, adaptation (it replaces `omega_el`) | natural frequency 300 rad/s, damping 0.707 [T3] |
| M4 | `i_d_m, i_q_m` | Park transform of the measured phase currents (done in the measurement chain by `sensors.py`; the controller receives the result). Frame angle is `epsilon_prev + delta_prev` (verified against GEM to 7e-13) | inherits M1 and M2. One shared function `park()` in `interfaces.py` is used by controller and tests | none new |
| M5 | `vdc_m` | divider + ADC + low-pass | gain, noise, quantization, filter lag | gain ±0.5 % [T2]; noise 0.5 V [T2]; 12-bit over 0 to 500 V; corner 500 Hz [T3], exact discretization |
| M6 | `bms_msg` every 50 ms | BMS over CAN | cadence, latency, quantization, estimation error, staleness. It carries `i_dis_lim, i_chg_lim, v_pack, age_s`. **It does NOT reflect the injected fault scale `idis_scale` before the fault acts** (the BMS message is built from what the BMS knows, which lags the physical cut by the latency); a BMS-derate event is therefore partly a surprise | period 50 ms, latency 20 ms, quantization 1 A [T3]; error 5 % [T2] |
| M7a | `omega_ref` analytic reference | cycle generator | exact, no hold (a smooth ramp has no 10 Hz hold) | none |
| M7b | `omega_ref` from TUM data | CAN speed channel | 10 Hz sample, held between samples; repeats measured at 10 to 22 % for the speed channels (the 33 to 66 % figure is for the battery CURRENT channel, not speed) [T1] | quantize 0.1 rad/s |
| M8 | motor beliefs `R_n, Ld_n, Lq_n, psi_n, J_n, Kt_n` | commissioning tables | wrong by construction, section 4 | section 4 |

Timing: delay `d` extra steps (config 0, 1, 2; default 1). **Build correction (v2.1):** GEM converts the dq command to abc with the angle at the moment it is APPLIED, so a delay by itself does not rotate the vector in the plant's frame; a real inverter holds abc computed at the earlier angle. The loop therefore applies the physical rotation `-d omega_el tau` explicitly, and the controller carries an **angle advance** `theta_adv = d * omega_m_m * p * tau` for its OWN known delay (default ON; sweep OFF). GEM's built-in half-step hold is already in the legacy results, so for `d = 0` there is nothing to compensate and legacy equivalence is exact. (The red-team's `(d+0.5)` form double counted the half step.)

**Frame facts, verified against GEM (red team):** GEM's `i_sd, i_sq` states are computed with the electrical angle ONE STEP EARLIER than the `epsilon` reported with `i_a, i_b, i_c`. The sensor layer therefore builds the measurement frame from `epsilon_prev` so that the zero-error measurement equals the plant's dq current. Identity mode (all errors off) bypasses the PLL, the filter and Park and reads the plant dq states, so legacy equivalence holds to 1e-9.

**Sign convention, stated once:** `delta = theta_m - theta_true`. Then `i_dq_m = R(-delta) i_dq` and `u_true = R(delta) u_cmd`, so `i_d_m = +i_q sin(delta)` (a positive angle error leaks torque current into +d). R(phi) = [[cos, -sin],[sin, cos]].

Phase-current transform: full three-phase Clarke, `alpha = (2/3)(ia - ib/2 - ic/2)`, which cancels equal (common-mode) offsets. A two-sensor variant (`ic = -ia - ib`) does NOT cancel them; it is run as a sensitivity case.

## 3. The migration checklist: every truth leak in today's `control.py` (all must be closed)

Line numbers re-checked against the 113-line file (the red-team report quoted line numbers from a longer file that does not exist here; its leak list is otherwise accurate).

| Where | Leak | Replacement |
|---|---|---|
| `control.py` 34, 36, 37 | `plant.V_dc, Rs, Ld, Lq, p`, `psi_nom = 0.7 plant.psi_p` | `BeliefConfig` constants (section 4) |
| 42, 43 | speed-loop gains from true `psi_p` and true `J` (so even a "30 % wrong" flux does not touch the speed loop) | `Kt_n = 1.5 p psi_n`, `J_n` belief |
| 63, 64 | governor reads `u_max_hw` and true `psi_p` | belief flux and `ceiling_hat` |
| 79, 85 | `st["V_dc"]` is true | `vdc_m` |
| 83 | `st["omega_el"]` is true | `p * omega_m_m` (PLL) |
| 89 | `prev_ledger["d_u_q"]` is the TRUE shortfall | `du_known` (section 5) |
| 110 | `psi_tilde = psi_hat - plant.psi_p` computed inside the controller | removed from the controller; computed by the logger from truth |
| 49, 84 | `self.pl.tau` | config constant |
| run scripts | the whole `led` dict and whole `st` (T_elec, T_dist, T_fric, `u_limit`, `soc`...) are passed in | `command(meas)` takes only `Measurements`; the controller constructor takes only config dataclasses, not plant objects (checked with `inspect.signature`) |
| `psi_range=(0.03,0.10)` | brackets the true 0.066 | kept as a design prior, declared in `BeliefConfig` |

A CI grep rejects `plant.`, `self.pl.`, `led`, `ledger` in `control_fw.py`.

## 4. What the controller believes about the motor

Controller-side relative errors (the plant stays fixed; the effect on the loop is the same). Static draws per run, from their own RNG stream.

| Belief | Draw | Why (tag) |
|---|---|---|
| R_n | x(1 + dR), dR ~ U[-0.15, +0.40] | winding 20 to 120 deg C at about 0.39 %/K is +39 % [T2] |
| Ld_n, Lq_n | x(1 + dL), dL ~ U[0, +0.25] | saturation lowers true L; belief offset only. **The plant has constant L (GEM), so this is a belief error, not saturation physics** [T3] |
| psi_n | 0.70 x truth start (design: gives the adapter something to learn) | existing choice |
| J_n | x(1 + dJ), dJ ~ U[-0.2, +0.2] | passenger or cargo load [T3] |
| Kt_n | derived from psi_n | follows psi_n |

The naive adapter keeps adapting only flux. R, L, J stay wrong. That is the point.

## 5. The saturation signal (most important design decision)

* The controller clips its own command against `ceiling_hat = min(vdc_m / sqrt(3), 173.2 V)`. The 173.2 V cap is the GEM 300 V hexagon, a config constant, because the plant caps its own ceiling there during regeneration (`plant.py:106`). Without it `du_hidden` is nonzero in regen.
* `du_known = u_sent - u_cmd` (what the controller itself removed). This replaces the ledger's `d_u_q` as the adaptive law and anti-windup input.
* A BMS current limit does NOT give a voltage ceiling. Its consumer is the **governor current limit**: `iq_limit = min(i_lim, eta * P_lim / (Kt_n omega_m_m))` with `P_lim = vdc_m * i_dis_lim` and `eta = 0.9` belief (the efficiency multiplies the pack power; corrected 2026-11-09 to match control_fw.py). Gate: poisoning `bms_msg` must change controller output in the battery-derate scenario.
* **Consequence, found by the red team:** in the plain event (`run_exp.py:48`) the 25 V ceiling is injected inside the plant and `vdc_m` still reads about 300 V. Then `du_known = 0` for the whole event, anti-windup state `z` stays 0, and the `aw` controller is identical to `naive`. Physically that is also what a real drive would see if a limit appeared with no voltage signature. So we run two event types and label them:
  1. **Oracle-ceiling event** (legacy reproduction): the controller is given the true ceiling through an explicit `ceiling_telemetry` channel. Clearly non-physical, used only to reproduce experiments A to D.
  2. **Observable sag event** (the realistic one): the bus collapses through the battery path, so `vdc_m` falls within the filter lag and `du_known` captures most of it. The hidden part is the lag, sensor error and the part of the cut that is a current limit.
  For Theory this is a real finding: a fast DC-link measurement makes most voltage saturation KNOWN, so the unmodeled part is smaller than the simulation assumed. **Decision D1 for Theory (Nolan, Kai):** is the BIBS/UUB argument written for the true shortfall, for `du_known`, or for the hidden remainder?

## 6. Ledger definition after the change

Legacy `d_u = u_applied - u_cmd`. New definition: **`d_u := u_true_frame_delivered - u_ctrl_cmd`**, where `u_ctrl_cmd` is the controller's own pre-clip command. It decomposes EXACTLY into logged columns: `d_u_ctrl_clip` (controller's own clip, equals `du_known`), `d_u_delay`, `d_u_rot` (angle error), `d_u_plant_clip` (true ceiling, the hidden saturation), `d_u_inv` (dead-time and device drop). The sum is a gate (A9). `telemetry.derive` keeps consuming `d_u_*`, so `dP_elec = 1.5 d_u . i` stays exact. A small backward-compatible hook in `plant.apply_voltage(u_extra=None)` adds the inverter error AFTER the true clip.

Order of operations: controller clip -> delay FIFO -> rotation by angle error -> plant true clip -> add dead-time and device-drop voltage. No second clip. Bounded-input gate A1 is checked on the voltage the plant clipped (`u_app`, true ceiling, by construction) AND the final voltage including inverter error is reported with its own bound (dead-time error at most `(4/pi)(Td/Tpwm)Vdc + Vdrop`).

## 7. Inverter nonlinearity

Dead-time `Td = 2 us`, `Tpwm = tau`, device drop 1.5 V [T2]. Fundamental error `(4/pi)(Td/Tpwm) Vdc` = 7.6 V at 300 V and tau = 1e-4 (about 4 % of the 173 V ceiling, larger than R i = 1.8 V at 100 A, so this is the dominant new voltage error). Its dq image is opposite the current direction with a 6th-harmonic ripple, not a constant. Smoothed over a ±1 A band around zero current. Compensation lives in the INVERTER firmware (as in real drives), computed from the MEASURED current sign at 70 % of the true value, so it is imperfect and wrong near zero crossings. Build note: Agent S placed it in `InverterModel.post_clip_error`; the controller does not know about it.

## 8. What is not modelled, and why it is acceptable

iron loss, cogging and winding-harmonic torque ripple (affect efficiency and ripple, not boundedness); motor thermal dynamics (temperature is a drawn parameter); CAN jitter and message loss; resolver-converter dynamics beyond the PLL; sensor faults. Also not modelled: real saturation of L in the plant.

## 9. What new mismatch rows result

Measurement gap (measured minus true, per channel), hidden saturation (`d_u_plant_clip`), inverter error (`d_u_inv`), belief gap (belief minus truth). The shaft/electrical identities are computed on truth columns and must close exactly as before.
