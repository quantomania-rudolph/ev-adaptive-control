# EV Powertrain Build Plan

Oct 26, 2026 · @Rudolph

Build Tier 1 on Gym-Electric-Motor (GEM) with real drive-cycle data, and log every request-versus-delivered gap as a first-class signal, so the Tier 2 full car swaps in without changing controllers, logging or analysis.

## Build vs reuse

We reuse the whole electrical drive and the drive-cycle data, and write only the glue that makes saturation visible.

| Need | Reuse | Why this one |
| --- | --- | --- |
| Supply, inverter, PMSM, load | [Gym-Electric-Motor](https://github.com/upb-lea/gym-electric-motor) v3.0.3 (`pip install gym-electric-motor`) | Already wired as supply, converter, motor and load; the PMSM's d/q currents and speed are the f(x) + g(x)u + d plant in our notes |
| Drive-cycle speed profiles | [FASTSim](https://www.nrel.gov/fastsim) cycles (UDDS and HWFET are bundled; other cycles from EPA) | Standard speed-vs-time cycles become the road-load disturbance d(x,t) |
| Real measured EV behaviour | Argonne D3 and TUM i3 trips (see Data sets) | Validates the road-load model and, in Tier 2, the battery |
| Numerics, logs, plots | SciPy, NumPy, pandas, Parquet, Matplotlib | Nothing custom needed |

We write three things ourselves, and they are the project's actual contribution:

1. **Plant adapter.** One interface (`state_names`, `input_names`, `limits()`, `step()`) that wraps GEM today and the full car later.
2. **Controllers that expose their internals.** Naive adaptive, σ-modification, projection, anti-windup. Each returns e, θ̃, z and P so V(t) can be computed offline.
3. **Ledger logger.** Records every request-vs-delivered pair at every step (next section).

## Mismatch ledger (mission-critical)

Every interface where one layer asks and the next layer delivers gets a ledger row, logged at every time step, so Δ is measured at the point of saturation and never inferred afterwards.

| ID | Interface | Requested | Delivered | Δ = delivered − requested | Unit | Limit that can bind |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Driver → governor | r (driver demand) | r\_safe | Δr | N·m or m/s | Governor constraint set |
| M2 | Controller → inverter | u\_c = (u\_d, u\_q) commanded | u applied after our clip | Δu (the Δu in the notes) | V | DC-link voltage |
| M3 | Inverter → motor (plant-side check) | u after our clip | voltage GEM reports as applied | Δu\_plant | V | GEM's per-leg duty clip (should never fire) |
| M4 | Current demand → motor | i\_q,cmd | i\_q measured | Δi | A | Current limit I\_max |
| M5 | Torque demand → shaft | T\_cmd | T\_elec | ΔT | N·m | Voltage, current, speed (back-EMF) |
| M6 | Power demand → supply | P\_req = u·i terms + auxiliaries | P\_avail | ΔP | W | Battery or supply power limit |
| M7 | Wheel torque → road (Tier 2) | F\_tractive request | F\_delivered | ΔF | N | Tyre grip, gearbox |
| M8 | Energy, cumulative | ∫P\_req dt | ∫P\_delivered dt | ΔE | J | Any of the above |

Rules that keep the ledger trustworthy:

1. **Compute Δ where the clip happens.** Our wrapper saturates the controller output before calling GEM, so M2 is exact; the saturation is a disc of radius V\_dc/√3 in the d/q plane (red team R3). M3 then compares that value against what GEM reports as applied, which catches clipping hidden inside the converter model.
2. **One sign convention.** Δ = delivered − requested everywhere, so the notes' example (450 requested, 300 delivered) logs as Δ = −150 N·m.
3. **Log which limit bound.** Each row also records a `saturated` flag and a `limit_id` (voltage, current, torque, power, SOC), so analysts can slice failures by cause.
4. **Log it normalised too.** Δ divided by its limit makes rows comparable across layers and across Tier 1 and Tier 2.
5. **Distinct from tracking error.** e = x − x\_ref and the anti-windup-compensated e\_a = e − Cz are separate columns. The whole thesis is telling these apart from Δ.
6. **Ledger is configuration.** Rows live in one `ledger.yaml` (request signal, delivered signal, limit signal, unit). Tier 2 adds rows; the logger code does not change.

## State-vector contract

Every plant, Tier 1 or Tier 2, must publish the signals below under these names and units; a signal a plant lacks is logged as missing, never silently zero. GEM's own state names get mapped to ours in one adapter file (the names were checked against an installed GEM 3.0.3: \`omega\`, \`torque\`, \`i\_sd\`, \`i\_sq\`, \`u\_sd\`, \`u\_sq\`, \`epsilon\`, \`u\_sup\` and six more).

| Group | Signal (our name) | Unit | Tier 1 source | Tier 2 source |
| --- | --- | --- | --- | --- |
| Mechanical | `omega_m` motor speed | rad/s | GEM state | Gearbox + wheel model |
| Mechanical | `T_elec` electrical torque | N·m | GEM state | Motor model |
| Mechanical | `T_load` load torque (the d term) | N·m | Road-load from drive cycle | Full vehicle road load |
| Electrical | `i_d`, `i_q` | A | GEM state | Motor model |
| Electrical | `u_d`, `u_q` applied | V | GEM state | Motor model |
| Supply | `V_dc` DC-link voltage | V | Constant supply (GEM) | Battery model with sag |
| Supply | `P_batt`, `I_batt` | W, A | Computed from u·i | Battery model, compared with TUM trips |
| Vehicle | `v`, `a`, grade | m/s, m/s², % | Drive cycle (input only) | Longitudinal dynamics (state) |
| Battery | `SOC` | % | Not modelled | Battery model |
| Thermal | `T_motor`, `T_batt` | °C | Not modelled | Simple lumped thermal states |
| Reference | `x_ref`, `r`, `r_safe` | per signal | Governor + reference generator | Same |
| Controller internals | `e`, `e_a`, `theta_hat`, `theta_tilde`, `z` | per signal | Controller | Same |
| Proof quantities | `V`, `V_dot_est` | – | Computed offline from internals | Same |
| Constraints | `dist_to_limit_k` for every active constraint | normalised | Computed from limits() | Same |
| Ledger | `req_*`, `del_*`, `delta_*`, `saturated_*`, `limit_id_*` | per M-row | Logger | Logger |

The controller internals and proof quantities are what the notes call for in every run: u\_c, u, Δu, tracking error, states, parameter estimates, V(t), estimated V-dot and distance to each constraint. Each run writes one Parquet file plus a `manifest.json` (plant, controller, dataset, seed, commit hash, limits), so any two runs can be compared by the data team without asking who ran them.

## Tier 1 core

Tier 1 is a speed-tracking PMSM drive under a drive-cycle load, with the notes' six-step ladder (naive, robust, anti-windup, barrier) run on it. It must work before any Tier 2 effort starts.

**Why the PMSM fits the notes' toy plant.** With field-oriented control holding i\_d near 0, the PMSM reduces to two states, speed ω and torque-producing current i\_q, with a bilinear coupling through back-EMF. That is the second-order nonlinear plant the notes ask for, and we get it from GEM instead of writing it. In d/q form the voltages u\_d, u\_q are the input u, and the road load is d(x,t). **Limit:** the reduction only holds below base speed. At 314 rad/s, GEM's own nominal speed, i\_d = 0 allows 42.5 N·m against 210.6 N·m with field weakening (red team R6), so the Tier 1 experiments stay at or below 200 rad/s.

**Saturation we study first.** Voltage saturation: the DC-link voltage sets a ceiling on the d/q voltage. GEM clips each phase leg silently and in a direction-dependent way, so our wrapper applies a disc-shaped limit of radius V\_dc/√3 itself and logs the gap (red team R2, R3). Torque saturation (the 450 → 300 N·m example in the notes) appears as a consequence: when voltage or current binds, T\_elec falls below T\_cmd, and ledger row M5 shows it. A literal torque clip would be one extra line in the wrapper.

**Checked on day one:** GEM clips out-of-range duty cycles silently and ends the episode when current exceeds its limit. Our wrapper does its own saturation and switches the termination off (red team R2, R4), and tests pin both behaviours.

**Experiment matrix (from the notes), with what each must show:**

| Exp | Setup | Evidence it must produce |
| --- | --- | --- |
| A | No saturation, baseline controller | Tracking works; Δu = 0 for the whole run |
| B | Saturation + naive adaptive law | θ-hat drifts while Δu ≠ 0; poor recovery. This is the Milestone 3 failure figure |
| C | σ-modification or projection | θ-hat stays bounded during the same saturation event |
| D | Anti-windup z-dot = −A\_z z + B\_zΔu, e\_a = e − Cz | Less overshoot and faster recovery after saturation clears |
| E | Barrier function on one state (current or speed error) | Constrained state never crosses its bound; `dist_to_limit` stays positive |
| F | Same architecture on the Tier 2 car | Same plots, same ledger, new plant |

Experiments B to D use the same plant, the same drive-cycle segment and the same saturation event, so the only difference is the controller. Plots saved for every run: u\_c vs u, Δu, tracking error, states, θ-hat, V(t), estimated V-dot and distance to each constraint.

**Parameter to learn.** One unknown parameter, chosen in the open decisions below (an effective inertia, friction or load coefficient are the natural candidates, all of which appear in ω-dot).

## Data sets

Use three sources: standard cycles to drive Tier 1, and two measured-EV sets to check the model against reality. Each gets one loader that returns our signal names, so the data team works from a single schema.

| Dataset | What it is | Feeds | Access |
| --- | --- | --- | --- |
| [FASTSim drive cycles](https://www.nrel.gov/fastsim) | Standard speed-vs-time cycles, version 3.1.0 bundles only UDDS and HWFET; loads in Python, e.g. `Cycle.from_resource("udds.csv")` | Tier 1 road-load disturbance d(x,t); Tier 2 vehicle input | Free; 3.1.0 installed on Python 3.13 (older 2.x docs said 3.8 to 3.10, and the API changed), so pin the version |
| [Argonne Downloadable Dynamometer Database (D3)](https://www.anl.gov/taps/downloadable-dynamometer-database) | Public chassis-dynamometer tests of advanced-technology vehicles, including battery EVs | Validating road-load torque and current/voltage behaviour of a real EV drive | Free; the site requires a stated attribution when you use the data. Channels differ per test file, so confirm that torque, motor voltage and HV current are present before relying on them |
| [TUM "Battery and Heating Data in Real Driving Cycles"](https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles) | 72 real trips in a BMW i3 (60 Ah): speed, throttle, elevation, ambient temperature, battery voltage, current, temperature and SOC, plus heating-circuit data | Tier 2 battery and thermal validation; a real-world speed profile for Tier 1 | Free IEEE DataPort login (IEEE membership not required); DOI 10.21227/6jr9-5235 |

**How the data connects to the ledger.** A cycle gives the requested motion (speed, grade). The simulation then reports what the drive delivered, and D3 or TUM give a real vehicle's delivered torque, current and battery power for the same kind of cycle. Comparing simulated and measured delivered quantities is how the team shows that our Δ values are realistic and not artefacts of the simulator.

**Limits to expect.** None of these sets contains a deliberately saturated actuator with a controller we designed; saturation events come from our simulation, and the measured data checks the plant and load models only. The TUM trips are one vehicle in one city, so they validate model structure, not a general claim.

**Owners.** Data analysis (Saleheen, Taehwan) writes the three loaders, the unit and sampling-rate checks, and the plot templates. The coding team consumes only the loaders' output.

## Tier 2: full car

Tier 2 is required and is built on top of Tier 1, so the plant changes while the controllers, ledger and analysis stay put. The minimum viable car is a longitudinal vehicle, a single-speed gear, the GEM motor, a battery with voltage sag and an SOC state, and lumped thermal states.

| Layer | Tier 1 | Tier 2 change | New ledger rows |
| --- | --- | --- | --- |
| Vehicle | Drive cycle gives load torque as an input | Longitudinal dynamics with mass, drag, rolling resistance and grade; speed v becomes a state | M7 (wheel force vs road) |
| Drivetrain | Motor shaft only | Gear ratio and wheel; torque is requested at the wheel and converted back to the motor | M1, M5 re-expressed at the wheel |
| Supply | Constant DC link | Battery equivalent circuit: V\_dc falls with current and SOC, plus a power limit | M6 (power demand vs available) |
| Thermal | None | Motor and battery temperature with derating limits | New `limit_id` values, no new code path |
| Data | Cycles | TUM trips as input and as validation of battery current and SOC | None |

**Rules that protect Tier 1 while Tier 2 is built**

1. **Limits become functions of state.** `limits(state)` is part of the adapter from day one, returning constants in Tier 1. In Tier 2 the voltage limit follows V\_dc and the current limit follows temperature. This directly answers the notes' question about whether the admissible input set changes with speed, SOC, temperature or DC-link voltage: in a car, yes.
2. **Controllers see only the adapter.** No controller imports GEM, so replacing the plant cannot break it.
3. **Regression gate.** Experiments A to E are rerun on every commit against stored results; a Tier 2 change that alters a Tier 1 result is rejected.
4. **Governor arrives with Tier 2.** The power and thermal limits are what make the reference governor necessary, so M1 starts carrying real values there.

**Time scales, measured.** GEM steps at 100 µs and ran about 3,900 steps per second with our controller, so UDDS (1,369 s, 13.7 million steps) would take about 58 minutes per run on one core. Use windows of 10 s or less for saturation studies, a reduced motor model (first-order torque lag with the same voltage and current limits) for full cycles, and re-measure the speed on each laptop (red team R13). The ledger and state contract are identical either way.

**Stretch, not scope.** A high-fidelity battery such as PyBaMM's DFN model is heavy. A [preprint](https://www.preprints.org/manuscript/202608.1536) already couples PyBaMM with GEM through a vehicle-dynamics layer, which shows the route exists, but it is not peer reviewed, and the equivalent circuit is enough for our proofs.

**Gate to start Tier 2:** the Experiment B failure figure exists and Experiments A to D run end to end on Tier 1.

## Tier 2 work breakdown

Verdict: moderate to hard. The parts that look scary (a bigger motor, a variable DC link) turned out to be cheap, and the part that looks routine (higher speed) is the hard one, because above about 1,900 rpm the notes' "hold i\_d at 0" reduction stops working (see Prototype results). Effort figures are my estimates in person-days, roughly plus or minus 50 percent; nobody on the team has validated them.

| Work package | What it takes | Effort (person-days) | Risk | Needs |
| --- | --- | --- | --- | --- |
| 1. EV-scale plant | GEM accepted a 350 V supply and an EV-scale motor as pure parameters (tested with illustrative numbers). Make limits, gear ratio and wheel radius adapter parameters instead of hard-coded constants | 2 to 3 | Low | Plant adapter |
| 2. Vehicle layer | Longitudinal dynamics (mass, drag, rolling resistance, grade), a driver model that turns a cycle into a torque request, and friction-brake blending when regeneration cannot deliver the braking demand (a new actuator, ledger row M7) | 3 to 4 | Medium | 1 |
| 3. Battery and DC link | Equivalent-circuit pack with SOC and power limit. V\_dc(t) only needs to set the voltage-disc radius V\_dc/√3 in our wrapper, which is exactly how the Tier 1 event works, so GEM itself is not modified | 3 to 4 | Medium | 2 |
| 4. Field weakening and governor | A current reference from the voltage ellipse and current circle (MTPA below base speed, field weakening above), replacing the fixed 250 A clamp. This is the notes' reference governor made concrete | 5 to 8 | **High** | 1, 3 |
| 5. Theory extension | Adaptive law and Lyapunov function for the coupled d/q loop. Recommended scope: prove UUB inside the governor's feasible set, and treat field weakening as simulation evidence, not theorem | 5 to 10 | **High** | Tier 1 proof, 4 |
| 6. Thermal states | Lumped motor and battery temperature, derating of the current limit | 2 to 3 | Medium | 3 |
| 7. Data loaders and validation | Argonne D3 and TUM loaders in our schema; compare simulated battery current, SOC and temperature with TUM trips and with FASTSim's Bolt model | 4 to 6 | Medium (TUM needs a login; D3 channels unknown until a file is opened) | 2, 3 |
| 8. Ledger and regression | Add rows M6 and M7 to `ledger.yaml`, Tier 1 regression gate on every commit | 1 to 2 | Low | 2, 3 |
| 9. Compute | Reduced-order motor for full cycles, window selection, run orchestration on 2 cores | 2 to 3 | Medium | 1 |

Total: roughly 18 to 27 person-days of coding, 5 to 10 of theory and 4 to 6 of data work. Spread over three teams that fits Weeks 6 to 8 only if Tier 1 is frozen by Nov 6, and Week 9 (Thanksgiving) is used for unattended runs. Cut order if time runs out: thermal first, then the PyBaMM stretch, then full-cycle runs; keep field weakening and the ledger rows.

**What does not need rework:** the controllers' structure, the logger, `ledger.yaml` format, the Parquet and manifest layout, the regression tests and all analysis code. That is the payoff of the adapter and the ledger.

**What does:** `plant.py` hard-codes the speed limit and uses a scalar road load, and `control.py` assumes i\_d = 0 and a single voltage ceiling.

## Prototype results (measured Oct 26)

The Tier 1 prototype runs Experiments A to D on the real GEM drive and reproduces the failure your notes predict: under an unmodeled voltage ceiling, the naive adaptive law winds ψ̂ up to 41 times the true flux and the current overshoots its 400 A limit by 47 percent. Scenario: speed ramps to 200 rad/s, a 10 N·m load steps in at 1.5 s, then the voltage ceiling drops from 173 V to 25 V for one second (4.5 to 5.5 s). The unknown parameter is the flux linkage ψ (true 0.066 Wb), started 30 percent low.

&#91;image: Experiments A to D: speed, current, voltage mismatch, torque mismatch, estimate and V(t)\]

| Run | Controller | Peak ψ̂ (Wb) | Peak current (A, limit 400) | Speed error RMS, 6.5 to 8 s (rad/s) |
| --- | --- | --- | --- | --- |
| A | Naive, no event | 0.066 | 151 | 0.00 |
| B | Naive, event | **2.74** | **589** | 64.1 |
| C | σ-modification + projection, event | 0.10 (at the bound) | 255 | 5.98 |
| D | Anti-windup, event | 0.0665 | 250 | 0.81 |
| Bg | B + feasibility governor | **2.23** | **516** | 46.5 |
| Cg | C + governor | 0.10 | 229 | 0.06 |
| Dg | D + governor | 0.0662 | 224 | 0.07 |

Two separate mechanisms showed up, and they need separate fixes:

1. **Estimator windup.** While the voltage is clipped, the error stays large and the naive law keeps adapting. Anti-windup (D) removes it; the governor alone (Bg) does not.
2. **Infeasible current command.** After the ceiling is restored, C and D settled into a trap: i\_q\* = 250 A requested, i\_q about 230 A achieved, but i\_d stuck near +59 A and only 18 N·m of torque, because the d-axis cross-coupling voltage ω·L\_q·i\_q (about 166 V) eats the voltage budget. A speed-dependent current limit from the voltage ellipse (the Cg and Dg runs) removes it. This is the notes' reference governor, and it is the first evidence for why the admissible input set depends on speed.

&#91;image: Separating estimator windup from the voltage-limited trap\]

**What this does and does not show.** It shows the ledger works end to end (M2 and M5 carry the story; M3, the hidden-clipping check, was exactly 0 in every run) and that the failure and both fixes are reproducible: reruns are bit-identical. It does not show a proof: V(t) is plotted from the logged internals but V-dot ≤ −λV + c is not verified, and C's result depends on the projection bound I chose (0.03 to 0.10 Wb). One scenario, one parameter, no sensor noise.

**Corrections found on the way.** A sign error in my first adaptive law made even the no-saturation baseline diverge; the correct law is ψ̂-dot = −Γ·ω\_el·e, as in your notes. Anyone re-deriving the Lyapunov step should check the sign by simulation. Code, tests, pinned versions and a README are in `ev_bibs/` (sent in chat).

## Red team: where this can go wrong

Each row says how I know. **Measured** = I ran it in a sandbox this session (scripts in `ev_bibs/probes/`). **Computed** = worked out from GEM's motor parameters. **Source** = read on a web page. **Reasoned** = my judgment, untested. Severity is my call.

| # | What can go wrong | How I know | Severity | Fix, and who |
| --- | --- | --- | --- | --- |
| R1 | **Novelty is already partly taken.** A 2026 IEEE Transactions on Power Electronics paper, [Anti-Windup Composite Adaptive Speed Control for PMSMs](https://scholarx.skku.edu/item/f073a646-4805-47d7-9d18-6ec6b355f56f), combines adaptive control and anti-windup for PMSM speed with a UUB result. A second title, "Antiwindup compensation for model reference adaptive control schemes", appeared in search but its page was blocked to me | Source (summary only; I did not read either paper). The summary says current saturation was tested and does not discuss voltage saturation | High | Read both before Week 6. Candidate gaps: voltage-ceiling saturation, the measured mismatch ledger, the governor + barrier combination, EV-scale validation. Theory |
| R2 | **Saturation is invisible in GEM.** It clips out-of-range duty cycles silently, so a controller can ask for 2,000 V, get 173 V, and nothing is logged | Measured: action of 3.0 delivered exactly 150 V per leg with no flag | Critical | Saturate in our wrapper before GEM and log both sides (done in `plant.py`); test pins it. Coding |
| R3 | **GEM's clip is direction dependent**, so "sat(u)" in the proof would not match the simulator | Measured: a 160 V command along d delivered 153.3 V; the same size at 30° passed untouched | High | Project onto a disc of radius V\_dc/√3 after zero-sequence injection (173.2 V; verified exact in all tested directions). Theory should adopt the disc as the saturation model. Theory + Coding |
| R4 | **GEM ends the episode above its current limit**, so the very failure we want to show (589 A) would have stopped the run with no data | Measured: a run ended at step 11 with 403 A; B peaked at 589 A | High | `constraints=()` plus our own distance-to-limit column; test pins it. Coding |
| R5 | **Default load fixes the speed**, so speed is not a state and any speed-tracking result is meaningless | Measured: GEM's default is a constant-speed load at 100 rad/s | High | `RoadLoad` subclass with a real inertia (done). Coding |
| R6 | **"Hold i\_d at 0" stops working at modest speed.** The notes' second-order plant is only valid below base speed | Computed: with i\_d = 0 the motor gives 42.5 N·m at 314 rad/s, versus 210.6 N·m with field weakening; base speed at 240 A is 195 rad/s (1,866 rpm) | **Critical** for Tier 2 | State the theorem for the governor's feasible set; treat field weakening as simulation evidence; build WP4. Theory + Coding |
| R7 | **An infeasible current limit traps the drive**, even with good anti-windup | Measured: runs C and D sat at 230 A, i\_d = +59 A, 18 N·m after the ceiling returned | High | Speed-dependent current limit from the voltage ellipse (Cg, Dg removed it). Coding |
| R8 | **A wrong sign in the adaptive law** makes even the unsaturated baseline diverge | Measured: it happened to me; ψ̂ ran to −6.8 Wb in run A | High | Regression test: run A must converge ψ̂ to within 5 percent of the truth, and a second person re-derives V-dot. Theory + Coding |
| R9 | **Anti-windup may only be exact for this plant.** For the scalar q-axis loop, z removes the saturation-induced error exactly; with coupled d/q, uncertain R or L, or a different saturation shape it is only approximate | Reasoned (algebra). Run D is consistent: ψ̂ stayed at 0.0665 | High | Repeat D with R and L\_q wrong by ±20 percent; extend the derivation to the coupled loop. Theory |
| R10 | **The simulation is too kind.** R, L and J are known exactly, there is no sensor noise, and GEM applies the command in the same step it is computed (no delay) | Measured for the delay (probe 5); Reasoned for the rest | Medium | Add options for parameter error, current-sensor noise and a one-step delay; rerun A to D. Coding |
| R11 | **Results lean on tuning.** C ended pinned at the projection bound (0.10 Wb) with a steady speed error; Γ, σ, the governor factor (0.75), the 25 V ceiling and the ramp are all my choices | Measured for C; Reasoned for the rest | Medium | Sweep each; report sensitivity, not one number. Data analysis |
| R12 | **One scenario proves little, and chaos theory (a listed course topic) is not used anywhere.** The windup transient may depend sharply on the ceiling depth, its timing and ψ̂ at the start | Untested | Medium | A grid over ceiling (10 to 60 V) × start time × initial ψ̂, mapping recovery time. That turns the chaos topic into evidence. About 45 s per run, 2 cores here, so 25 runs take roughly 10 minutes. Data analysis |
| R13 | **Full drive cycles are too slow in GEM.** UDDS is 1,369 s = 13.7 million steps | Measured: about 3,900 steps/s with the controller, so about 58 minutes per run; this sandbox has 2 cores | Medium | Windows of 10 s or less; reduced motor for full cycles; teammates' laptops for overnight runs. Re-benchmark on each laptop. Coding |
| R14 | **Library drift.** GEM only requires gymnasium ≥ 0.29.1 and numpy ≥ 1.16.4; I ran Python 3.13, numpy 2.5.4, gymnasium 1.4.0. NumPy 2 already broke 2-D `np.cross` in my own test, and FASTSim's API changed between versions 2 and 3 | Measured | Medium | One pinned `requirements.txt` for everyone (included); CI runs `pytest` on every push. Coding |
| R15 | **FASTSim is thinner than advertised.** Version 3.1.0 bundles only two cycles (UDDS, HWFET), and its battery has no voltage state, so it cannot supply V\_dc sag | Measured (installed, listed resources and fields) | Medium | Other cycles from EPA (the UDDS text file downloaded fine); our own battery equivalent circuit; FASTSim only as a power/SOC/temperature cross-check. Data analysis |
| R16 | **The measured data may not contain what Tier 2 validation needs.** D3's page lists no channels or file formats, the ANL host returned HTTP 403 to my sandbox shell, and TUM needs an IEEE login. I opened no D3 or TUM file | Measured (page text, HTTP status) | **High** | One teammate downloads one D3 battery-EV test and the TUM trips this week and writes a channel inventory. Go/no-go for Tier 2 validation at Week 6. Data analysis |
| R17 | **Measured data cannot validate our saturation story.** No real vehicle was driven with our controller into a voltage ceiling | Reasoned | Medium | Say so in the paper: data checks the plant, load and battery models, not Δu. Theory + Data analysis |
| R18 | **The proof and the plots could become circular.** If c and λ are estimated from the logs, the "bound" just restates the simulation | Reasoned | Medium | Derive c and λ from parameters first, then compare with logs. Theory |
| R19 | **Process risk.** The prototype was written by me, not the team; Rudolph and Nolan have no GitHub handle in the notes; Tier 1 freezes Nov 6 | Reasoned | High | Each team reviews its own files line by line this week (tasks below); no result goes in a slide until a teammate has re-run it. All |
| R20 | **Regeneration and friction braking are untested**, so negative-torque windup and the battery charge limit are unknown | Untested | Medium | Add a deceleration scenario in WP2. Coding |
| R21 | **Voltage headroom is left on the table.** We stop at 173.2 V; overmodulation reaches the hexagon vertex at 200 V | Measured (probe 6: 200 V reached at the vertices) | Low | Keep the disc for the proof; mention it as conservative. Theory |

## Assumption register

Every assumption the plan rests on, with its status. **Verified** means I observed it directly (a run, a test or an installed package). **Read** means I saw it on a web page. **Open** means nobody has checked it. The script that checks each item is named so a teammate can re-run it on their own machine.

| # | Assumption | Status | Evidence, and how to re-check |
| --- | --- | --- | --- |
| A1 | GEM's action is three phase duty cycles in \[−1, 1\] | Verified | `probes/probe1.py` printed `Box(-1, 1, (3,))` |
| A2 | GEM's PMSM state names are `omega, torque, i_a, i_b, i_c, i_sd, i_sq, u_a, u_b, u_c, u_sd, u_sq, epsilon, u_sup` | Verified | `probes/probe1.py` |
| A3 | GEM's PMSM follows the textbook d/q equations, with p = 3, L\_d = 0.37 mH, L\_q = 1.2 mH, R\_s = 18 mΩ, ψ = 0.066 Wb | Verified | `probes/probe7.py`: currents matched an independent model to 2e-10 A over 0.3 s; torque matched 1.5·p·(ψ + (L\_d − L\_q)i\_d)·i\_q |
| A4 | A zero-sequence offset does not change the reported d/q voltage, and the reachable disc is V\_dc/√3 = 173.2 V | Verified | `probes/probe6.py` (exact up to 170 V in every direction tested) |
| A5 | GEM silently clips duty cycles beyond ±1 | Verified | `probes/probe4.py` |
| A6 | GEM ends an episode when current exceeds its limit unless `constraints=()` | Verified | `probes/probe4.py` (ended at step 11, 403 A) |
| A7 | A time-varying load torque can be injected because `mechanical_ode` receives the time | Verified | Source read, then `RoadLoad` ran in all experiments |
| A8 | GEM runs about 5,800 raw steps/s, about 3,900 with our controller | Verified here only | `probes/probe2.py`; this sandbox has 2 cores, so every laptop must re-measure |
| A9 | Reruns are bit-identical | Verified | `tests/test_plant.py::test_rerun_is_bit_identical` |
| A10 | GEM accepts an EV-scale motor and a 350 V supply as parameters | Verified with made-up numbers | One-off check; real vehicle parameters not tried, and the field-weakening behaviour of that motor is untested |
| A11 | FASTSim 3.1.0 installs on Python 3.13, bundles UDDS and HWFET, and includes Leaf, Bolt, Model 3 and Zoe models with thermal states but no battery voltage | Verified | Installed and listed; the Bolt model ran UDDS in 0.04 s and exposed 139 columns |
| A12 | The EPA UDDS speed trace is downloadable | Verified | 1,372 lines fetched from epa.gov. I did not compare it with FASTSim's copy |
| A13 | Argonne D3 has battery-EV tests and requires attribution | Read | The D3 page. It does **not** list channels or formats; Open until someone opens a file |
| A14 | TUM i3 data has 72 trips with speed, battery voltage, current, SOC and temperature | Read | The IEEE DataPort page. Sampling rate (0.1 s) comes from a third-party paper. Login needed; Open until someone opens a file |
| A15 | For the q-axis loop, the auxiliary state z removes the saturation-induced error exactly | Derived, observed | Algebra in `control.py`; run D kept ψ̂ at 0.0665. Not proven for the coupled d/q loop (R9) |
| A16 | V-dot ≤ −λV + c holds for the proposed controller | **Open** | Not derived or checked. Plotted V(t) is a candidate only |
| A17 | Leaking ψ̂ toward its nominal value instead of toward zero is acceptable σ-modification | Design choice | The notes write −σθ̂; Theory must confirm the variant |
| A18 | Voltage saturation is the right first actuator limit | Judgment | GEM limits voltage natively and the notes' torque example follows from it (row M5); the group can overrule |
| A19 | Week 1 began Sep 28 | Open | Inferred from Thanksgiving falling in Week 9; confirm against the term calendar |
| A20 | Tier 2 effort estimates | Judgment | Person-days are mine, ±50 percent |
| A21 | What counts as "done" for the Tier 2 full car | **Open** | The notes do not define it. This plan assumes: longitudinal vehicle, gear, GEM motor, battery equivalent circuit with V\_dc sag and SOC, one drive cycle, ledger rows M6 and M7 |

## Week 5 task list (Oct 26 to 30)

Every task has a check a teammate can run or read, so "done" is not a matter of opinion.

| # | Owner | Task | Done when |
| --- | --- | --- | --- |
| T1 | Everyone | Create one Python environment from `requirements.txt`; run `pytest -q tests/` | 5 tests pass on your own machine; paste the pass line in the team chat |
| T2 | Coding | Put `ev_bibs/` in the team GitHub repo; add Rudolph and Nolan as collaborators; turn on a CI job that runs the tests | A push shows a green check; all six people have access |
| T3 | Coding | Read `plant.py` and `control.py` line by line against the equations in the notes; list every disagreement | A written list (even "none") in the repo, reviewed by one Theory member |
| T4 | Coding | Benchmark GEM steps per second on three laptops, using `probes/probe2.py` | Three numbers recorded in `docs/benchmarks.md`; decide window length for full-cycle runs |
| T5 | Coding | Make speed limit, supply voltage and the saturation ceiling constructor arguments of `GEMPlant` (they are hard-coded today) | `GEMPlant(limits=...)` works and the 5 tests still pass; Experiments A and D reproduce the numbers in this doc |
| T6 | Theory | Re-derive the adaptive law and its sign from the Lyapunov function; write V-dot for run D including Δu | One page with the sign, c and λ as formulas in the physical parameters; a plot of V-dot from `runs/D.parquet` against your bound |
| T7 | Theory | Read the 2026 IEEE TPEL anti-windup adaptive PMSM paper and the MRAC anti-windup paper; list what each does not cover | A half-page "gap" note; propose the novelty claim at the Week 6 meeting |
| T8 | Theory | Decide the saturation model for the proof (disc of radius V\_dc/√3) and the leak variant in σ-modification | Written decision in the Open decisions table |
| T9 | Data analysis | Download one Argonne D3 battery-EV test and the TUM trips; write a channel inventory (name, unit, sampling rate, missing values) | `docs/data_inventory.md` lists every channel, and states whether motor torque, voltage and battery current exist in D3 (go or no-go for Tier 2 validation) |
| T10 | Data analysis | Write the EPA UDDS loader and the road-load function; compare with FASTSim's `udds.csv` | Maximum speed difference between the two traces printed; a figure of road-load torque vs time |
| T11 | Data analysis | Design the sensitivity sweep in R12 (ceiling depth, start time, initial ψ̂) | A grid table and an estimate of total run time on 2 cores |
| T12 | All | Confirm or overrule the Open decisions table and A18, A19, A21 | Each row has a name and a date |

Nothing in a Week 6 slide should come from a result that one teammate has not re-run.

## Schedule and owners

The Week 6 presentation needs Tier 1 failure evidence, so Tier 1 comes first and Tier 2 starts once that figure exists. Dates assume Week 1 began on Sep 28, which puts Thanksgiving in Week 9 as your timeline shows; correct me if the term calendar differs.

| Week | Dates | Course milestone (from the timeline) | Deliverable from this plan |
| --- | --- | --- | --- |
| 5 | Oct 26 to 30 | Explore articles, prepare first presentation | Repo and plant adapter; GEM smoke test and step-time benchmark; `ledger.yaml` v1 with M2 and M3 logging; FASTSim cycle loader; Experiment A running |
| 6 | Nov 2 to 6 | Present and synthesize findings | Experiment B failure figure (u\_c vs u, Δu, θ-hat drift); Week 6 deck |
| 7 | Nov 9 to 13 | Explore a solution | Experiments C and D; Lyapunov derivation with Δu explicit; D3 loader and road-load check |
| 8 | Nov 16 to 20 | Prepare to present solution | Experiment E (barrier); Tier 2 vehicle and battery equivalent circuit; TUM loader; interface frozen |
| 9 | Nov 23 to 27 | Thanksgiving, no studio | Asynchronous Tier 2 runs (Experiment F) and TUM validation |
| 10 | Nov 30 to Dec 4 | Final presentation | Tier 1 and Tier 2 results, ledger plots, predicted vs observed bound |

**Week 6 presentation sections, and what feeds each**

| Required section | Supplied by |
| --- | --- |
| Topic introduction | Notes' one-sentence project description; the powertrain chain from GEM |
| Problem statement | Saturation mismatch Δu and the Experiment B figure |
| Approach | Governor → adaptive controller → anti-windup → saturation → plant, plus the ledger |
| Proposal objective | A UUB result under actuator saturation, V-dot ≤ −λV + c |
| How we will collaborate | Owner table below and the shared signal schema |
| Conclusion | What Tier 1 showed and what Tier 2 will test |
| References (APA) | The notes' five references, the GEM paper, and the three datasets |

**Owners (from your notes)**

| Team | People | Owns |
| --- | --- | --- |
| Coding | Rudolph, Liam | Plant adapter, saturating wrapper, controllers, ledger logger, regression gate |
| Theoretical | Nolan, Kai | Definitions of e, e\_a, z and V; the Lyapunov proof; checking the predicted bound c/λ against logged runs |
| Data analysis | Saleheen, Taehwan | Loaders, schema and unit checks, run comparison, paper-quality plots |

The ledger also gives Theory something concrete: the largest logged ‖Δu‖ and ‖d‖ per run feed the c in c/λ, and the maximum state norm after the transient is the number to compare with it. The notes list GitHub handles for Sal, Liam, Kai and Taehwan only, so Rudolph and Nolan need repo access added.

## Open decisions

These are the questions from your notes that the plan depends on. Each has a default so work can start; the group confirms or overrides in Week 5.

| Question from the notes | Default in this plan | Decide by |
| --- | --- | --- |
| Smallest EV model worth using | GEM PMSM with road-load disturbance (Tier 1); add gear, vehicle and battery equivalent circuit (Tier 2) | Week 5 |
| Which saturation first | Voltage saturation; torque shortfall appears as a consequence (M5) | Week 5 |
| Main theorem | UUB with Δu explicit in V-dot; forward invariance of a safe set via the barrier function as the second result | Week 6 |
| Does the admissible input set change | Constant in Tier 1; depends on V\_dc and temperature in Tier 2 through `limits(state)` | Week 8 |
| Learnable parameter vs bounded disturbance | Learn one parameter (effective inertia, friction or load coefficient); treat grade, aerodynamic load and auxiliaries as bounded d | Week 5 |
| Baseline to beat | Experiment B naive adaptive law, plus a fixed-gain PI with clamp-and-back-calculation anti-windup as the classical reference | Week 6 |
| Where the novelty sits | Not decided; candidates are saturation-gated adaptation using Δu, the barrier combination, and the EV application. A first search found an overlapping 2026 paper (red team R1); Theory does the full literature pass in Week 5 | Week 6 |

Two checks that can change the plan on day one: how GEM treats a state limit (clip or episode end), and how fast it steps relative to a drive cycle. Both are in the Tier 1 and Tier 2 sections.

## Sources

Checked on Oct 26, 2026. I installed GEM 3.0.3 and FASTSim 3.1.0 and ran them (scripts in \`ev\_bibs/probes/\`); I opened no D3 or TUM data file, so their channels and formats are unverified.

- [Gym-Electric-Motor, GitHub](https://github.com/upb-lea/gym-electric-motor): building blocks (supply, converter, motor, load) and install
- [GEM v3.0.3 release notes](https://github.com/upb-lea/gym-electric-motor/releases/tag/v3.0.3): current release, tested motors and examples
- [NREL FASTSim](https://www.nrel.gov/fastsim): drive cycles and Python version
- [Argonne D3](https://www.anl.gov/taps/downloadable-dynamometer-database): public dynamometer data and attribution requirement
- [DOE merit review, Argonne benchmarking (2016)](https://www.energy.gov/sites/prod/files/2016/06/f33/vs030_stutenberg_2016_o_web.pdf): which vehicles Argonne tests and the instrumentation it lists
- [TUM "Battery and Heating Data in Real Driving Cycles", IEEE DataPort](https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles): trip contents and access
- [PyBaMM + GEM digital twin preprint](https://www.preprints.org/manuscript/202608.1536): precedent for coupling battery and motor models (not peer reviewed)
- Project notes (Research\_Project\_ENGR110H.pdf): problem statement, experiments, roles and timeline

* [EPA UDDS speed trace](https://www.epa.gov/sites/default/files/2015-10/uddscol.txt): downloaded and parsed (1,372 lines)
* [Anti-Windup Composite Adaptive Speed Control for PMSMs, IEEE Transactions on Power Electronics 2026](https://scholarx.skku.edu/item/f073a646-4805-47d7-9d18-6ec6b355f56f): summary page only, full paper not read
* `ev_bibs/` in the chat attachments: the code, tests and probe scripts behind every item marked Verified
