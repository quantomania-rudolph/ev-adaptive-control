# Dataset inventory: disturbance d(t) and speed reference omega_ref(t)

Scope: which real data can feed the plant's external disturbance and the speed request. Local files were profiled with pandas (this session). Web items were read from the cited page; anything not on a page is marked UNVERIFIED. Nothing in the repo was changed.

## 1. Local data

### 1.1 TUM Tesla Model 3 CAN traces (CC BY 4.0, doi:10.14459/2024mp1735471)

Source facts (mediaTUM page, MDPI paper doi:10.3390/wevj15060268): 228 tracks, 6.3 GB in 2 files, CSV and MAT, one 2020 Model 3 Standard Range Plus, CAN bus polled. Empty mass measured 1755 kg (1861 kg per Certificate of Conformity). The paper says motor torque is estimated by the control unit, not measured, and that two software versions differ in torque behaviour (420 N.m peak vs 320.8 N.m). Local tracks: 0 (283 s, 3.2 km), 1 (1796 s, 33 km), 10 (841 s, 6.4 km), 100 (293 s, 6.3 km), 103 (365 s, 3.0 km).

Two schema generations were found. This explains the "rear_motor_torque is untrustworthy" observation:

| Property | Tracks 0, 1 | Tracks 10, 100, 103 |
|---|---|---|
| Wheel speed column | `engine_rpm` (can be slightly negative, min -26) | `axle_rpm` |
| `rear_motor_torque` range | 0 to 2100, never negative | -125 to +324 N.m, 16 to 33% negative |
| Corr. of torque with measured accel | -0.57 and -0.49 (wrong sign) | +0.64 (T10), +0.89 (T100), +0.75 (T103) |
| `regen_active` | absent | present (mean accel while active: -0.4 to -1.0 m/s^2) |
| `rear_inverter_power` | absent | T100, T103 only; corr. with battery power 0.996 and 0.978 |
| `axle_torque / rear_motor_torque` median | 0.61 and 0.53 (inconsistent) | 8.9 (T100), 8.7 (T103), i.e. about the gear ratio; T10 gives 5.0 (axle_torque is coarse there) |

Verdict: torque is distrusted for tracks 0 and 1 only. In 100 and 103 it is signed, correlates with acceleration and with inverter power; in 10 it is plausible but weaker.

All tracks: time is epoch ms, uniform 100 ms, no gaps, no non-monotonic steps. Many channels are sample-and-hold on that grid, so the true update rate is lower than 10 Hz (see "Held" below).

### 1.2 Column table (all five tracks; `-` means not a disturbance or reference)

Unit is as measured or inferred. "Held" is the fraction of consecutive identical samples.

| Column | Unit | Rate / quality | Role |
|---|---|---|---|
| `time` | ms epoch | 10 Hz uniform | time base |
| `speed` | m/s (GPS) | 10 Hz, held 7 to 16%; T1 has 885 NaN (4.9%, GPS dropout); 4 to 35 samples per track with \|a\|>6 m/s^2 (glitches) | omega_ref (through wheel radius and gear); inertia and aero term |
| `cal_speed_combined` | km/h | equals `speed`*3.6 in 0, 10, 100, 103; in T1 it is 0 during GPS dropouts (1643 samples differ by >1 m/s from GPS) | omega_ref only after masking those zeros |
| `vehicle_speed` | km/h (CAN) | integer-valued, 33 to 935 unique values, 1 to 2 Hz effective; T100 min -4.2 (signed) | backup speed; coarse |
| `engine_rpm` (T0, T1) / `axle_rpm` (T10, 100, 103) | wheel rpm | implied rolling radius 0.329 to 0.332 m (median) from rpm and GPS speed | omega_wheel; omega_m = omega_wheel * G |
| `rear_motor_torque` | N.m | held 82 to 93%; semantics see 1.1 | measured motor torque (T10/100/103 only) |
| `rear_motor_torque_request` | N.m | T100, T103 only; corr. with accel 0.93 and 0.89 | controller demand; best torque-vs-accel match |
| `axle_torque`, `axle_torque_request` | N.m at axle | T0/T1 non-negative like rear torque; T10/100/103 signed; 56 to 92 unique values in 10/100/103 (coarse) | wheel force estimate (T100, T103) |
| `accelerator_pedal` | % | 0 to 100 | input, not a disturbance |
| `brake_pedal` (T0, 1, 10) | coarse | values only {0,4,8,12} in T0/T1, {0,1} in T10, 14 to 60 changes per track | brake flag, not torque |
| `pos_break_pedal` (T100, 103; sic) | % | continuous, 0 to 12 | brake position, no torque scale |
| `regen_active` | 0/1 | T10, 100, 103 | regen flag |
| `acc_x`, `acc_y`, `acc_z` | m/s^2 | 10 Hz, noisy. `acc_z` is +9.8 in T0, 1, 10 and -9.8 in T100, 103 (mounting differs). `acc_x` median offset: -0.1, 0.0, 1.3, -1.2, -2.1; slope vs d(speed)/dt: 0.90, 0.85, 0.38, 0.20, 0.57 | not usable for grade without per-track calibration |
| `gyro_x/y/z` | rad/s | noisy | neither |
| `altitude_baro` | m | held 80 to 82% (about 2 Hz); steps up to 9 m in T1 | best grade source (see 1.3) |
| `altitude_gps` | m | held 18 to 29%; T103 min 445 vs 476 to 488 typical (glitch) | glitchy, avoid |
| `elevation` | m, integer | 9 to 66 unique values | quantised, grade too coarse |
| `pressure` | Pa | held 80%, 93.4 to 96.4 kPa | air density rho = p/(R T) |
| `ambient_air_temp` | deg C | present on 21 to 58% of rows; T1 max 58.5 (glitch) | air density; fill-forward and reject outliers |
| `temperature` | deg C (battery?) | UNVERIFIED meaning | neither |
| `course`, `hdop` | deg, - | `hdop` up to 12.6 in T1 | GPS quality flag only |
| `cal_distance_covered` | km | matches integral of `speed` within about 1% | cross-check |
| `mileage` | km | integer steps | neither |
| `hv_battery_current` | A, discharge negative | 10 Hz | validation, not a disturbance |
| `hv_battery_voltage`, `hv_soc`, `hv_temp_*`, `battery_voltage`, `dcdc_*`, `ptc*`, `cell_voltage_*`, `total_energy_*`, `temperature`, `evap_power`, `brake_temp_*`, `temp_*`, `range_display`, `temp_ac_*`, `3171`, `3172` | various | cell columns are only 9 to 16% populated; `hv_soc` 0.1% steps, T0 has 2 values; `brake_temp_*` constant in T103 (20 deg C); `temp_rear_motor_stator` constant 40 | neither (battery/thermal validation) |
| `bms_max_drive_power`, `bms_max_regen_power`, `sys_max_*_power`, `bms_power_limit_state` | W / state | T100, T103; `bms_power_limit_state` constant 1; `bms_max_regen_power` constant 85 kW in T100 | useful as the real saturation limits for the regen and drive side |
| `rear_inverter_power` | W | T100, T103 | validation of electrical power |
| `num_sats` | - | all NaN in all five tracks | neither |
| Unnamed `3171`, `3172` (T100, T103) | UNVERIFIED | 3171 about 287 to 363 (looks like pack voltage), 3172 2 to 1536 | ignore |

### 1.3 Which disturbance terms can be built per track

Forces at the wheel, then to the shaft: T_dist = (r/G) * F_load / eta, with r = 0.33 m (measured above) and G about 9 (inferred from T100/T103 torque ratio; confirm the exact ratio).

| Term | Needs | From data | Assumed | Per-track note |
|---|---|---|---|---|
| Rolling, m g Crr | m, Crr | speed gate only (sign(v), zero at standstill) | m (1755 kg empty + driver and payload; payload unknown), Crr (not in data) | all 5 tracks, constant torque |
| Aero, 0.5 rho CdA v^2 | rho, CdA, v_air | `speed`; rho from `pressure` and `ambient_air_temp` (both measured) | CdA (not in data); wind = 0, so v_air = v | all 5; T1 needs GPS gaps filled from `vehicle_speed` |
| Grade, m g sin(theta) | altitude, distance | `altitude_baro` (10 s median, 100 m window) | m | all 5. Grade percentiles (50/95/99% of \|grade\|): baro T0 0.6/1.9/2.8 %, T1 0.6/3.0/4.1 %, T10 0.7/5.4/5.5 %, T100 0.9/3.2/4.3 %, T103 0.6/2.0/2.9 %. GPS gives T1 99th = 50 % and T103 = 20 %, so GPS grade is unusable there |
| Inertia, m a (+ rotating inertia) | a, m | a = d(`speed`)/dt, smoothed 1 s | m, wheel/rotor inertia | all 5; not a disturbance if the plant J already includes vehicle inertia. Only add it if J is motor-only |
| Brake / regen (friction part) | wheel torque, brake torque | residual: F_demand - F_motor - road load | m, r, G | T10, 100, 103 only; no track has brake torque or pressure |
| Residual road load (measured) | motor torque, accel | T_load = G*T_motor*eta/r - m*a (all in wheel N) | m, r, G, eta | T100 and T103 best; lumps rolling, aero, grade and friction brake with no coefficient guesses |
| Wind | anemometer | none | zero | not in any column |
| Payload / mass | scale | none | constant | not in any column |
| Tyre slip | wheel speed vs ground speed | one wheel rpm vs GPS speed, GPS-lag limited | - | not separable; note: rpm radius spread 0.326 to 0.335 m (IQR) is slip plus noise |

Not trusted: `acc_x` for grade, `altitude_gps`, `elevation`, `rear_motor_torque` in T0 and T1.

## 2. Disturbance vector definition

Plant convention (plant.py): J dw/dt = T_elec - T_fric(w) - T_dist(t); T_dist > 0 opposes motion.

- T_dist(t) [N.m, motor shaft] = (r / (G * eta)) * [ F_roll + F_aero + F_grade (+ F_brake) (+ m a, if J is motor-only) ] when the vehicle is moving forward, with eta applied the other way when braking.
- d_w(t) [rad/s^2] = -T_dist(t) / J_total.

| Component | Filled from data | Must be assumed |
|---|---|---|
| F_grade = m g sin(theta) | theta from `altitude_baro` and integrated `speed` | m |
| F_aero = 0.5 rho CdA (v - v_wind)^2 | v, rho (`pressure`, `ambient_air_temp`) | CdA, v_wind = 0 |
| F_roll = m g Crr sign(v) | v gate | m, Crr |
| m a | `speed` derivative | m (plus rotating inertia if needed) |
| F_brake (friction) | residual in T10/100/103 | m, r, G, eta |
| Mass / payload | measured empty mass in paper (1755 kg) | driver, payload |
| Wind, tyre slip, road surface | none | all |

## 3. External datasets

Verdicts: use now / use later / skip. "UNVERIFIED" means the page did not state it.

| Dataset | URL | Licence | Size | Channels that matter | Direct use | Friction | Verdict |
|---|---|---|---|---|---|---|---|
| TUM Tesla Model 3, other 223 tracks | https://mediatum.ub.tum.de/1735471 | CC BY 4.0 | 6.3 GB, 2 files | speed, torque, battery, baro, rpm (as 1.2); availability varies by track | omega_ref and d(t) as in section 1 | HTTP, FTP or rsync, no login | use now |
| Argonne D3 Downloadable Dynamometer Database | https://www.anl.gov/taps/downloadable-dynamometer-database | Attribution required, no other licence stated | UNVERIFIED | Page lists no channels; EV list runs from 2012 Nissan Leaf to 2023 Hyundai Ioniq5 (EV testing page). A DOE slide set lists direct axle torque, HV current/voltage, CAN for the i3 REx test car; not confirmed for D3 files | Possible torque and dyno speed for known cycles | No login stated. Earlier plan notes an HTTP 403 from the sandbox shell. Contact d3info@anl.gov | use later (open one EV file first) |
| nuScenes CAN bus expansion (Renault Zoe) | https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md | UNVERIFIED (not on the page; check nuScenes terms) | UNVERIFIED | Zoe vehicle info 100 Hz: wheel speeds, meanEffTorque, requestedTorqueAfterProc, regen, pedal_cc, longitudinal_accel; vehicle monitor 2 Hz: brake pressure (bar), throttle, vehicle_speed km/h; pose 50 Hz; IMU 100 Hz | Torque + regen + brake pressure + speed on a real EV; no grade channel listed | nuScenes download may need registration (UNVERIFIED; the nuScenes page fetch returned no content) | use later |
| NREL / NLR DriveCAT and Fleet DNA | https://nrel.gov/transportation/drive-cycle-tool, https://nrel.gov/fleetdna | Not stated; asks for citation and notice to NREL | UNVERIFIED | Time, speed; road grade only on Fleet DNA cycles; vocations incl. 19 transit buses (472 days), 39 refuse trucks, 70 tractors | omega_ref plus grade for heavy-duty | No login stated; file formats not stated | use later (heavy-duty case) |
| DriveCAT+ (python package) | https://pypi.org/project/drivecatplus | BSD | small | tool takes time, speed, elevation; a yanked 0.2.0 release | cycle metrics only, no data | pip | skip (tool, not data) |
| FASTSim | https://www.nrel.gov/fastsim | Not stated | UNVERIFIED | 20+ vehicles and standard US/EU/JP cycles; grade support UNVERIFIED | omega_ref | Excel and Python downloads | skip |
| EPA dynamometer schedules (UDDS, FTP, HWFET, US06, SC03, NYCC, LA-92) | https://epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules | Not stated on page | each file 1 to 98 KB | speed vs time, tab-delimited; 10 Hz versions of FTP and HWFET | omega_ref only | none | use now |
| WLTC (class 3b, 1800 s, 131.3 km/h max phase) | https://dieselnet.com/standards/cycles/wltp.php | Not stated | small | time-speed table link on page; sampling interval not stated | omega_ref only | page shows Log in/Subscribe prompt; data link appeared visible | use later (verify the table, or take the UNECE GTR 15 text) |
| Mendeley 7vdkzpnjgj, Real-world EV data: driving and charging | https://data.mendeley.com/datasets/7vdkzpnjgj/2 | CC BY NC 3.0 | not stated | Only current, voltage, SOC, temperature, epoch time (no speed) | none | file list not visible; .mat in Drive/Charge folders | skip |
| Mendeley mm9rvfvmrb, BEV on actual roads 2018 and 2019 | https://data.mendeley.com/datasets/mm9rvfvmrb/3 | CC BY 4.0 | not stated | page names no channels | UNVERIFIED | Download All button; login UNVERIFIED | skip until channels known |
| IEEE DataPort, EV speed, pedal, accel, V, I on four roads (Zhidou ZD2) | https://ieee-dataport.org/documents/electric-vehicle-speed-pedal-accel-voltage-and-current-data-over-four-different-roads | Not stated | 5 files, 0.4 to 1.5 MB | SPEED km/h, PEDAL, ACCEL, V, I; no grade | omega_ref | Requires IEEE DataPort subscription | skip |
| Vehicle Energy Dataset (VED), Ann Arbor | https://github.com/gsoh/VED | Apache 2.0 | UNVERIFIED (7z of weekly CSV) | Vehicle speed km/h, GPS, engine rpm, HV battery I/V/SOC; static "Generalized Weight (lb)"; no elevation; 27 PHEV/EV of 383 | omega_ref and mass class | none | skip for d(t) |
| comma2k19 | https://github.com/commaai/comma2k19 | MIT | about 100 GB, 10 chunks | CAN car speed, wheel speeds, raw CAN, IMU, GNSS; torque and grade not listed | omega_ref; grade only if GNSS altitude is present (UNVERIFIED) | Academic Torrents; RAV4 and Civic, not EVs | skip |
| AEGIS CAN example (Zenodo 3267184) | https://zenodo.org/record/3267184 | Not stated | 922 MB, 21 HDF files | WheelSpeed FL, accel, yaw rate, GPS lat/lon at 20 Hz | weak | none | skip |
| Ford AV dataset | https://avdata.ford.com/ | CC BY-NC-SA 4.0 | not stated | page does not mention vehicle CAN | UNVERIFIED | none stated | skip |
| Realistic Driving Profile Generator (Zenodo 8205773) | https://zenodo.org/records/8205773 | MIT | 1.5 MB installer | generates speed profile and slope profile for a predefined trip | synthetic omega_ref plus grade | Windows installer UNVERIFIED | use later (needs checking of its data source) |

## 4. Ranked recommendation: next 3 to bring in

1. **TUM, remaining tracks (same dataset).** No new loader, CC BY 4.0, anonymous download. Select tracks that have `axle_rpm`, signed `rear_motor_torque`, `regen_active`, and ideally `rear_inverter_power` (like 100, 103). This gives measured torque and a residual road-load d(t) with only m, r, G assumed.
2. **nuScenes CAN bus (Zoe).** Real EV torque request and delivered torque, regen, brake pressure and wheel speeds at 100 Hz, which TUM lacks (TUM has no brake torque). Check licence and download steps first.
3. **Argonne D3 EV tests.** Dynamometer data for standard cycles with a known vehicle, useful to calibrate Crr, CdA, eta. Open one file before committing; channels are unconfirmed.

Standard cycles (EPA, WLTC) are zero-cost omega_ref additions and need no ranking; they carry no disturbance.

## 5. Sources

- TUM Tesla Model 3 data: https://mediatum.ub.tum.de/1735471 (doi:10.14459/2024mp1735471)
- Paper: https://www.mdpi.com/2032-6653/15/6/268
- https://www.anl.gov/taps/downloadable-dynamometer-database
- https://www.anl.gov/taps/electric-vehicle-testing
- https://www.energy.gov/sites/prod/files/2016/06/f33/vs030_stutenberg_2016_o_web.pdf
- https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md
- https://nrel.gov/transportation/drive-cycle-tool
- https://nrel.gov/fleetdna
- https://pypi.org/project/drivecatplus
- https://www.nrel.gov/fastsim
- https://epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
- https://dieselnet.com/standards/cycles/wltp.php
- https://data.mendeley.com/datasets/7vdkzpnjgj/2
- https://data.mendeley.com/datasets/mm9rvfvmrb/3
- https://ieee-dataport.org/documents/electric-vehicle-speed-pedal-accel-voltage-and-current-data-over-four-different-roads
- https://github.com/gsoh/VED
- https://github.com/commaai/comma2k19
- https://zenodo.org/record/3267184
- https://avdata.ford.com/
- https://zenodo.org/records/8205773
