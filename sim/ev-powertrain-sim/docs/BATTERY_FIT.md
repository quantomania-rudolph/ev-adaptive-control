# Battery fit from the TUM Tesla Model 3 traces (Agent B)

Files: `battery_fit.py` (fit, envelope, scaling, CLI), `bms.py` (BMS message model), `data/tum_battery_fit.json` (output, deterministic),
`tests/test_battery_fit.py`, `tests/test_bms.py`. Regenerate with `python battery_fit.py` (about 3 s; same input gives a byte-identical file).

Sign convention: TUM files store discharge as NEGATIVE; the loader multiplies by -1, so inside the fit and the BMS discharge is POSITIVE.

## What the data can and cannot say

* Five 10 Hz tracks, 20 to 34 C, one temperature band per track. **No temperature dependence is fitted.** The toy keeps the generic Arrhenius R0(T) of `battery.py` (assumption).
* The current is a repeated (held) value on 33 to 66 % of rows, the voltage on 4 to 10 %. Evidence that long repeated-current runs at |I| > 20 A are stale frames, not constant current: during them the voltage moves by 2 to 20 V (track 1: 8.1 V mean range over 70 runs) against about 1 V in windows where the current updates (`pack_real.r_eff_methods.long_hold_evidence`).
* **Held-sample-aware** = repeated values are treated as missing and both signals are linearly interpolated between their update rows before differencing (identical runs longer than 1 s at |I| < 5 A stay plateaus). Checked on synthetic data with the real hold probabilities: the estimator recovers a known 45 mOhm within 5 %, the naive regression does not (test `test_estimator_recovers_known_resistance...`).
* Timing is not recorded. Voltage responds 0.1 to 0.3 s after the recorded current on tracks 0, 1, 10 (best lag -3, -2, -2 samples; -1 on tracks 100, 103). That is physics (fast polarization) and/or CAN timing; **the data cannot separate them**, so the window dependence below is reported as is.

## Resistance by difference window (mOhm, pooled over tracks, 95 % block-bootstrap CI)

| window | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---|---|---|---|
| naive raw differences | 10.0 | 27.1 | 37.9 | 49.9 |
| **held-aware OLS (primary)** | 38.7 [33.8, 43.6] | 44.4 [39.0, 49.2] | 47.4 [42.3, 52.3] | 51.6 [47.1, 56.3] |
| EIV (Deming, noise-ratio from rest) | 40.0 | 45.6 | 48.5 | 52.5 |
| lag-aligned (upper bound if the lag is timing) | 44.0 | 51.4 | 54.3 | 57.3 |

* Held-aware per track (0.5 s): 45.2, 43.2, 52.9, 40.4, 58.5 mOhm (tracks 0, 1, 10, 100, 103).
* The EIV estimate is a Deming regression through the origin; the error-variance ratio comes from rest-time sample noise. It is about equal to OLS because sample noise is tiny against driving dI. The dominant errors-in-variables effect is timing, shown by the lag-aligned row. Neither removes the timing error.
* **The 0.1 to 0.5 s values depend on how long identical-current runs are treated** (`hold_rule_sensitivity_ohm`): if runs longer than 1.0 s were plateaus the 0.1 s value would be 18.9 mOhm (11.9 for runs > 0.4 s), the 0.5 s value 41 (38). The 1.0 s value (50 to 52) does not move. So: 1 s is robust, 0.1 to 0.5 s are method-dependent.
* Per unit (R x 667.2 A / 348.16 V): 7.4 % (0.1 s), 8.5 % (0.3 s), **9.1 % (0.5 s, mid)**, 9.9 % (1.0 s); outer 95 % interval over 0.3 to 1.0 s: 7.5 % to 10.8 %. `r_eff_pu` lo/mid/hi = 8.5 / 9.1 / 9.9 %.
* Bases: v_nom = 348.16 V (median of the per-track median bus voltages), p_base = 232.3 kW = largest discharge power seen (track 10), i_base = 667.2 A. A use maximum, not a pack rating.

## OCV, RC, validation

* OCV `V = K0 + K1 ln s + K2 ln(1-s)`, rest-like points (|I| < 5 A, 7400 points, each track equal weight), bounds K1 >= 0, K2 <= 0: **K = [353.02, 6.43, 0.00]** (K2 sits on its bound; 95 % bootstrap K1 in [5.3, 7.7], K2 in [-0.35, 0]). OCV(0.5) = 348.6 V, OCV(0.6) 349.7, OCV(0.8) 351.6, OCV(0.98) 352.9. **Flat and weakly identified**: rest points sit on four SOC clusters (about 0.53, 0.55 to 0.63, 0.80, 0.985), each cluster is one trip, rest voltage still contains unrelaxed polarization, nothing below SOC 0.5 is observed. Dropping any one track leaves the 0.58-to-0.98 rise at 3.3 to 3.5 V, but that is four clusters, not a curve. The generic toy OCV rises about 30 V over the same SOC range; the real one about 4 V (cannot be set through `Battery(...)` kwargs).
* RC branch (one branch, tau in [1, 30] s, per-track offset and linear drift removed, equal track weight): R0 = 56.1 mOhm, R1 = 64.0 mOhm, **tau = 30 s = the upper bound**. Dropping track 1 gives tau = 1 s (the lower bound) and R1 = 27 mOhm. **R1 and tau are NOT identified** (`rc.identified = false`); the toy therefore keeps its generic RC (r1_cell 0.6 mOhm, tau 20 s). The fitted slow part says the long-time resistance may be about 120 mOhm (23 % pu), well above the 1 s value; this is unsupported by an interior optimum, treat it as a hint only. Allowing tau below 1 s does not help (the in-sample error is lowest at the 30 s bound).
* Leave-one-track-out voltage RMSE (V), overall = sqrt(mean fold MSE): **fit 2.98**, const_v 6.17, const_r 3.82, generic (scaled) 13.21. **status = "ok"** (beats the best baseline const_r). Margin: -0.76 V mean over folds, fold SD 2.69 V, the fit wins 4 of 5 folds (it loses on track 100: 4.8 against 1.06). A weak, noisy win. Folds (fit): 1.45, 3.55, 1.31, 4.81, 2.20 (tracks 0, 1, 10, 100, 103). Error is dominated by the track-to-track OCV level (SOC cluster and unrelaxed polarization), not by the resistance.
* Baselines: const_v = training mean voltage; const_r = a - R_eff(0.5 s) I; generic = `battery.Battery` OCV, R0+Rbus, R1, tau scaled to the real pack (one voltage gain from the training mean, per-unit resistance at the 65 kW toy base). Every model sees the same hold-aware current.

## Observed-use envelope (NOT a BMS limit)

Per SOC bin (grid 0.50 to 1.00 step 0.05, half-width 0.025), the 99.5 % quantile of discharge current and of regen magnitude, inflated by 5 % so that after interpolation 99.5 % of discharge and 100 % of regen samples lie inside (checked on the data, negative control: x1.2 falls outside). Bins with too few samples (0.65 to 0.75, 0.85 to 0.95) use the pooled quantile (485.5 A discharge, 184.7 A regen before inflation). **SOC dependence is confounded with the trip** (each cluster is one drive), so it is not a SOC law; a flat alternative (`i_dis_lim_flat_a`, `i_chg_lim_flat_a`) is provided. Regen/discharge ratio 0.27 (limits) which is inside the measured 0.16 to 0.40.

## Scaling to the toy

`R_toy = R_pu * v_toy^2 / p_base_toy`, `I_base_toy = p_base_toy / v_toy`, OCV x `v_toy / v_nom`, current limits x `I_base_toy / I_base_real`.

| toy base | I_base_toy | R_toy lo / mid / hi (mOhm) | scaled max discharge / regen limit | vs toy motor (250 A) |
|---|---|---|---|---|
| **65 kW (default)** | 216.7 A | 117.7 / 125.7 / 136.9 | 235.3 A / 63.4 A | consistent |
| 31.1 kW | 103.7 A | 246.1 / 262.7 / 286.2 | 112.6 A / 30.3 A | throttles the motor to 45 %: not used |

* Generic toy pack in the same units: ohmic (R0 + bus) 6.1 % at 65 kW and 2.9 % at 31.1 kW; with R1 9.6 % and 4.6 %. Against the real 8.5 to 9.9 % (0.3 to 1.0 s) the generic pack is about 1.4 to 1.6 times too low at the 65 kW base, but 3 times too low at 31.1 kW. The "2 to 3 times" statement of the plan depends on the toy base.
* `toy_battery_kwargs` (65 kW): n_series 80, q_ah 60, r0_cell 1.494 mOhm, r1_cell 0.6 mOhm, tau1 20 s, r_bus 5 mOhm, i_dis_max 235.3 A, i_chg_max 63.4 A. r0_cell is set so that series R0 + R_bus + R1 (1 - exp(-0.5/tau)) equals the mid total at the 0.5 s window; i_dis_max is the largest scaled discharge envelope value (kept below the motor's 250 A; the existing default i_dis_max = 250 A is not binding for it). The fixed q_ah is not preserved.
* Not preserved: thermal mass, absolute energy, temperature dependence (generic Arrhenius), the real OCV flatness, cell time constants (kept in seconds).

## BMS message model

`BMSModel(cfg, fit, rng).step(t, battery_truth) -> BMSMsg`. Messages are built on multiples of `period_s` from a causal history buffer (the truth at `t - latency_s`, earliest sample before enough history exists), held in between, `age_s = t - t_sample`. `e ~ N(0, rel_err)` is drawn once per message (one rng draw), applied to both limits, then quantized to `quant_a`. `i_batt_max` is part of the sampled truth, so an injected `idis_scale` fault is invisible until `latency_s` after it happens. `enabled=False` returns the unquantized, error-free, zero-latency truth with age 0. The age bound is `period + latency` up to one simulation step.

## Not done

* Cold data: none exists in these tracks; not fitted.
* Extra tracks: the files of dataset 1735471 are served from `dataserv.ub.tum.de` (linked from the mediaTUM page), not from `mediatum.ub.tum.de`; I stayed inside the host named in the task and fetched nothing. `data/tum_extra/` does not exist.
