# Electrical mismatch: what we give the BIBS / UUB team, and why

Convention everywhere: **delta = delivered − requested**. All numbers below were measured on the stored 8 s runs
(10 kHz, full rate) with `telemetry.derive`. Every identity is a column you can check yourself.

## 1. The short answer

*Can the shaft-power mismatch be turned back into an electrical-power mismatch?* **Yes, exactly, but not by relabelling it.**
Shaft power is only one of three places electrical power goes. The exact bookkeeping is

    P_del  =  P_cu  +  P_mag  +  P_mech                      (physics: volts x amps = heat + stored field + shaft)
    P_req_ref = P_cu(i*) + T_cmd * w_m                       (what a steady motor needs to deliver the requested torque)
    dP_elec_ref = P_del − P_req_ref = dP_mech + dP_cu + P_mag
                  (shaft gap)        (copper-loss gap)  (magnetic storage rate)

* `dP_mech = T_elec w_m − T_cmd w_m` is the shaft gap we already logged (M6b).
* `dP_cu = P_cu − P_cu_ref`, with `P_cu = 1.5 R (i_d² + i_q²)` at the actual current and `P_cu_ref = 1.5 R i_q*²` at the requested one.
* `P_mag = 1.5 (L_d i_d di_d/dt + L_q i_q di_q/dt)` is the rate at which field energy is stored. It integrates to a bounded stored energy (`E_mag_store`), so it cannot drift.

So **reconstructing electrical mismatch from the shaft mismatch is correct only if the copper and magnetic terms are added back.**
Using the shaft number alone as the electrical number is wrong by 2 to 18 % of the energy in the event runs
(B 18.2 %, D 4.7 %, Dg 7.9 %, Eb 2.3 %) and the copper term can even have the opposite sign (run B: shaft −32.65 kJ, copper +4.96 kJ).

Closure, measured: energy of `dP_mech + dP_cu + P_mag` against `dP_elec_ref` is −27.62 / −27.62 kJ (B), −39.47 / −39.48 kJ (D),
−12.45 / −12.46 kJ (Dg), −34.92 / −34.92 kJ (Eb). The instantaneous residual is 0.2 to 0.6 % of the electrical gap in event runs and about 11 to 27 W
rms in every run, which is the same finite-difference noise as `P_balance_resid`.

## 2. Three electrical requests, and which one to use for what

| Name | Channel | Definition | Depends on | Use it for |
|---|---|---|---|---|
| Reference-implied (real power) | `P_req_ref`, `dP_elec_ref`, `E_elec_ref_unserved` | `1.5 R i_q*² + T_cmd w_m` | the request and the speed only | "the power the driver asked for", energy owed, comparing runs |
| Actuator (u-space) | `dP_elec` = `1.5 (Δu_d i_d + Δu_q i_q)` | `u_app·i − u_cmd·i` | Δu and the actual current | BIBS theory written in voltage and current coordinates. The identity is exact: `resid_du_power` is 1e-12 W. |
| Voltage-command | `P_req_elec` = `1.5 u_cmd·i` | what the voltage command would transfer at the present current | the command and the delivered current | diagnosis only (see the caution) |

**Why the voltage-command request looks counterfactual.** It splits exactly into
`P_req_elec = P_cu + P_that + P_fb`, where `P_that` is the shaft power the controller *believes* it is commanding (it uses the estimate ψ̂) and
`P_fb = −1.5 k (i_d² + e_q i_q)` is feedback-effort power. We verified that last identity to 3e-11 W. In run D the feedback term has an rms of 9.5 kW,
far larger than the torque-producing power, because the command grows huge while the actuator is saturated. That is why `P_req_elec` swings
negative while the motor is being asked to accelerate (`req_sign_conflict` flags it, 3 to 14 % of samples in the fault runs, 0 % in the clean ones).
It is a legitimate quantity in the theory's u-space, and a poor measure of how much power was *asked for*. Use `P_req_ref` for that.

**Where the parameter error enters.** Exactly,
`dP_elec = −1.5 w_el ψ̃ i_q + P_mag − P_fb`, i.e. `P_psi_err + P_mag − P_fb` (channels exist for each, plus `resid_req_decomp`).
This is the term the adaptive law is trying to drive out, written as a power.

## 3. Do we use the shaft mismatch for energy?

Both, for different jobs.

* **Shaft energy owed** (`E_mech_unserved`, M8b) is the physically cleanest number: it is energy the vehicle did not receive at the wheels. Use it for "how much did the driver lose".
* **Electrical energy owed** (`E_elec_ref_unserved`, M8c) = shaft energy owed + copper-loss energy + the bounded stored-field term. Use it when the claim is about the electrical side (BIBS on electrical parameters).
* Over a long window the stored-field term is bounded, so electrical energy mismatch ≈ shaft mismatch + copper mismatch. At any one instant the magnetic rate also matters.

## 4. Limits you should know about

* `P_req_ref` assumes the d-axis request `i_d* = 0` (what our controller uses). A controller with field weakening needs `i_d*` logged (`params["id_ref"]`).
* It is a quasi-steady request: it ignores the power needed to change the field. That is what the `P_mag` term in the decomposition accounts for.
* On decimated logs (the cycle runs are logged at 200 Hz) the finite-difference `P_mag` is not meaningful. The mismatch map therefore uses the quasi-steady gap `dP_mech + dP_cu` (M6c) and the pointwise actuator gap (M6d), both valid at any logging rate.
* The decomposition check (`resid_elec_decomp`) equals `P_balance_resid` by construction. The independent evidence is that `P_balance_resid` (voltage times current against the plant's own torque and our copper and magnetic terms) stays near zero.
