# Sensor and inverter parameters: values, provenance, sources

Written by Agent S. Tags follow `INFORMATION_CONTRACT.md`: [T1] measured here, [T1-lit] backed by a fetched published source (upgraded from [T2] only when a source was actually opened and the number checked), [T2] typical value, no source found, [T3] design choice. No citation below was written from memory: each was fetched in this session. Where a source supports only the same order of magnitude, that is stated.

## SensorConfig

| Field | Value | Tag | Note |
|---|---|---|---|
| `tau` | 1e-4 s (plain), 2e-4 s (cycle) | [T3] | all time constants below are in seconds |
| `p` | 3 | nameplate | toy motor |
| `identity` | False | switch | True = measured equals truth exactly |
| `i_range_a` | 400 A | [T3] | LEM HO 200-S/SP33 measures +/-500 A (see below); 400 A is of the same class |
| `adc_bits` | 12 | [T3] | LSB = 800/4096 = 0.1953125 A |
| `i_gain_err` | +/-1 % uniform | [T1-lit] | LEM HO-S/SP33 datasheet: "Sensitivity error @ IPN" -0.5 to +0.5 % and "Accuracy @ IPN" -1 to +1 % of IPN at 25 C (-4.08 to +4.08 % at 85 C). The 1 % half-width equals the 25 C accuracy limit. |
| `i_offset_a` | +/-2 A uniform | [T1-lit] | same datasheet: "Electrical offset current, referred to primary" +/-1.087 A (HO 100-S) and +/-2.174 A (HO 200-S, +/-500 A range). 2 A matches the HO 200-S limit. |
| `i_noise_lsb` | 1 LSB (0.195 A) sigma | [T1-lit] | same datasheet, HO 200-S: output noise 2.7 mV peak-to-peak DC to 10 kHz at 2.3 mV/A = 1.17 A pk-pk. Assuming Gaussian noise and pk-pk about 6.6 sigma, sigma is about 0.18 A = 0.9 LSB. The Gaussian assumption is mine. |
| `two_sensor` | False | switch | sensitivity case |
| `theta_offset_rad` | +/-1.0 deg el uniform | [T2] | no source found. For scale: the AD2S1210 resolver-to-digital converter alone is specified at +/-2.5 arc-min (+ 1 LSB), about 0.04 deg (Analog Devices product page, https://www.analog.com/en/products/ad2s1210.html). The 1 deg is therefore NOT a converter figure; it stands for resolver, mounting and zero-calibration error, for which no number was found. |
| `theta_noise_rad` | 0.3 deg el sigma | [T2] | no source found |
| `theta_ripple_rad` | 0.2 deg el amplitude | [T2] | no source found |
| `pll_wn` | 300 rad/s | [T3] | |
| `pll_zeta` | 0.707 | [T3] | |
| `vdc_range_v` | 500 V | [T3] | 12-bit, LSB 0.122 V |
| `vdc_gain_err` | +/-0.5 % uniform | [T2] | no source found |
| `vdc_noise_v` | 0.5 V sigma | [T2] | no source found |
| `vdc_corner_hz` | 500 Hz | [T3] | exact discretization alpha = 1 - exp(-2 pi fc dt) |
| `ref_quant` | 0 (0.1 rad/s for TUM references) | [T1] per contract | not touched by Agent S |
| `ref_hold_s` | 0 (0.1 s for TUM references) | [T1] per contract | not touched by Agent S |

## InverterConfig

| Field | Value | Tag | Note |
|---|---|---|---|
| `tau` | as above | [T3] | Tpwm = tau |
| `delay_steps` | 1 | [T3] | total latency (0.5 + d) tau |
| `dead_time_s` | 2 us | [T2] | no source found for the 2 us itself. Infineon FF900R12IP4D datasheet (preliminary), turn-on delay 0.20 us and turn-off delay 0.70 to 0.85 us at 900 A, 600 V, Rg 1.6 ohm, shows the switching-delay spread a dead time must cover (about 0.5 to 0.65 us). It does not give a dead-time setting. https://www.infineon.com/assets/row/public/documents/60/49/infineon-ff900r12ip4d-ds-en-jp.pdf |
| `device_drop_v` | 1.5 V | [T1-lit] (class only) | same Infineon datasheet: IGBT VCE(sat) typ 1.70 V (25 C) to 2.10 V (150 C) and diode VF typ 1.65 V to 1.50 V, all at 900 A rated current. 1.5 V is inside that class; at the toy's lower currents the real drop is smaller, and the model uses one constant. |
| `smooth_band_a` | 1 A | [T3] | |
| `compensation` | 0.7 | [T3] | lives in the inverter model (deviation, below) |
| `enabled` | True | switch | |

Sources opened: LEM HO-S/SP33 series datasheet https://www.lem.com/sites/default/files/products_datasheets/ho-s_sp33_series.pdf ; Analog Devices AD2S1210 page https://www.analog.com/en/products/ad2s1210.html ; Infineon FF900R12IP4D datasheet (above). A dead-time compensation paper was also opened (Jung et al., "Simple On-line Observation and Compensation of Dead-time Effects", WIT Transactions on The Built Environment 135, 2014, doi 10.2495/CR140031) and quotes no dead-time or drop value, so it supports nothing numerical here.

## Things in the contract or plan that I believe are physically or mathematically wrong

1. **The (4/pi)(Td/Tpwm)Vdc + Vdrop "bound" is not a bound.** It is the DC (fundamental) part of the error for a rotating sinusoidal current. The instantaneous dq error has a 6th-harmonic ripple on top. For a per-phase square wave of amplitude K = Td/Tpwm Vdc + Vdrop the instantaneous dq magnitude reaches (4/3) K (1.333 K against 1.273 K for the DC part). With the compensation taken from a wrong measured sign the exact bound is (4/3)(1 + c) K, attained when the measured current has the opposite sign of the true one; for matched measurement it is (4/3)|1 - c| K. `InverterModel.max_error_dq(vdc)` returns the first, `dc_fundamental(vdc)` the contract's number; tests/test_inverter.py (U16) checks both are tight. Gate A1's "own bound" should use max_error_dq.
2. **U9 as written is ill-posed.** The 4.3 % overshoot (exp(-pi zeta/sqrt(1 - zeta^2)) at zeta 0.707) is the response of the PLL SPEED output to a step in speed (a ramp in angle). A step in ANGLE makes the PLL phase estimate overshoot by about 20.8 % because the loop has a zero (closed loop (2 zeta wn s + wn^2)/D). The test checks both against their own analytic values (measured in the simulation: 4.04 % and 20.99 %).
3. **theta_mech = epsilon_el / p is not a mechanical angle.** GEM wraps epsilon_el to +/-pi, so the 1x/2x mechanical ripple would jump at every wrap. I accumulate the unwrapped electrical angle before dividing by p (identical until the first wrap). The initial mechanical phase is unknown and taken as epsilon_0 / p.
4. **Contract M2 lists angle quantization, but SensorConfig has no field for it.** Not modelled.
5. **Compensation placement.** The contract puts the 70 % dead-time compensation in the controller; it is in `InverterModel.post_clip_error` (inverter firmware), so the controller never needs Td or Vdc. Same net voltage on the plant.
6. Contract arithmetic checked and fine: LSB 0.195 A is 1.55 % of 12.6 A; 2 A is 15.9 % of 12.6 A; (4/pi)(2/100)(300) = 7.64 V.
7. `interfaces.py` comment above `TRUTH_KEYS` says the truth also carries "the true battery bus current for the BMS model", but the tuple has no such key (the sensor suite does not need it; BATTERY_TRUTH_KEYS covers the BMS). Not an obstacle for sensors.py or inverter.py.
