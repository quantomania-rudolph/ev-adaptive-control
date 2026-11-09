"""FROZEN interface for the realism layer. Written before any agent starts; agents may NOT edit this file.

If something here is wrong, stop and report it. The contract is docs/INFORMATION_CONTRACT.md, the plan is
docs/REALISM_PLAN.md. Everything below is checked against those documents by tests/test_plan_coverage.py.

Sign conventions (stated once, used everywhere):
  * delta = theta_m - theta_true (measured minus true electrical angle).
  * i_dq_m = R(-delta) i_dq, u_true = R(delta) u_cmd,  R(phi) = [[cos, -sin], [sin, cos]].
  * delta = delivered - requested for every mismatch.
  * TUM files: battery discharge current is NEGATIVE. Inside the fit and BMS code discharge is POSITIVE (I_dis > 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

SQRT3 = float(np.sqrt(3.0))
V_SCALE_GEM = 300.0                       # the GEM converter is scaled to a fixed 300 V
U_CAP = V_SCALE_GEM / SQRT3               # 173.205 V: hexagon-inscribed disc at the GEM scale (plant.py caps its ceiling here)
TAU_PLAIN = 1.0e-4                        # plain experiments
TAU_CYCLE = 2.0e-4                        # cycle experiments (all time constants below are in SECONDS, never in steps)

# --------------------------------------------------------------------------------------------- what the controller may see
# EXACT list of fields of Measurements. tests/test_realism_firewall.py checks the dataclass against this constant and the
# contract markdown against this constant. Adding a field here without updating the contract table must fail a test.
CONTRACT_FIELDS = (
    "t", "i_a_m", "i_b_m", "i_c_m", "i_d_m", "i_q_m", "theta_m", "omega_m_m", "vdc_m", "bms", "omega_ref", "ceiling_telemetry",
)
BMS_FIELDS = ("i_dis_lim", "i_chg_lim", "v_pack", "age_s")

# Keys of the TRUTH dict handed to SensorSuite.measure(truth). Exactly the keys GEMPlant.state() already returns, plus the
# previous step's electrical angle (maintained by the loop) and the true battery bus current for the BMS model.
TRUTH_KEYS = (
    "t", "omega_m", "omega_el", "i_a", "i_b", "i_c", "i_d", "i_q", "epsilon_el", "epsilon_prev", "V_dc",
)
# Keys of the battery truth dict handed to BMSModel.step(t, battery_truth) (battery.Battery.step() output + t).
BATTERY_TRUTH_KEYS = ("V_dc", "i_batt", "soc", "T_batt", "V_term", "i_batt_max", "bms_limited", "i_dc")

# Streams (SeedSequence.spawn order is part of the contract: adding a stream goes at the END).
RNG_STREAMS = ("sensors", "inverter", "bms", "beliefs", "events")


def rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """One independent generator per layer: changing the controller can never change the sensor noise (common random numbers)."""
    ss = np.random.SeedSequence(int(seed))
    return {name: np.random.default_rng(child) for name, child in zip(RNG_STREAMS, ss.spawn(len(RNG_STREAMS)))}


# --------------------------------------------------------------------------------------------- shared math (one implementation)
def rot(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def clarke(i_a: float, i_b: float, i_c: float) -> tuple[float, float]:
    """Full three-phase amplitude-invariant Clarke: cancels equal (common-mode) offsets on the three phases."""
    return (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c), (2.0 / 3.0) * (SQRT3 / 2.0) * (i_b - i_c)


def clarke_two_sensor(i_a: float, i_b: float) -> tuple[float, float]:
    """Two-sensor variant (i_c = -i_a - i_b assumed). Does NOT cancel common-mode offsets; used as a sensitivity case."""
    return i_a, (i_a + 2.0 * i_b) / SQRT3


def park(i_a: float, i_b: float, i_c: float, theta: float, two_sensor: bool = False) -> tuple[float, float]:
    """abc -> dq at electrical angle theta. The ONLY Park implementation: sensors, controller and tests all call it."""
    al, be = clarke_two_sensor(i_a, i_b) if two_sensor else clarke(i_a, i_b, i_c)
    c, s = np.cos(theta), np.sin(theta)
    return c * al + s * be, -s * al + c * be


# --------------------------------------------------------------------------------------------- what the controller receives
@dataclass(frozen=True)
class BMSMsg:
    i_dis_lim: float      # A, pack discharge current limit as the BMS last reported it (toy units)
    i_chg_lim: float      # A, pack charge (regen) current limit (positive number)
    v_pack: float         # V, pack voltage as the BMS last measured it
    age_s: float          # s since the BMS computed this message (staleness)


@dataclass(frozen=True)
class Measurements:
    t: float
    i_a_m: float
    i_b_m: float
    i_c_m: float
    i_d_m: float          # Park of the measured phase currents in the measurement frame (shared park())
    i_q_m: float
    theta_m: float        # measured electrical angle, rad
    omega_m_m: float      # PLL speed, rad/s mechanical (NEVER the true speed)
    vdc_m: float          # filtered, noisy DC-link voltage, V
    bms: BMSMsg
    omega_ref: float      # the REQUEST (speed reference), rad/s mechanical
    ceiling_telemetry: Optional[float] = None   # None in realistic runs; true ceiling (V) ONLY in the labelled oracle-ceiling event


# --------------------------------------------------------------------------------------------- configs (all errors can be switched off)
@dataclass(frozen=True)
class SensorConfig:
    tau: float = TAU_PLAIN
    p: int = 3                             # pole pairs (nameplate constant; electrical angle = p x mechanical)
    identity: bool = False                 # True: measured == truth exactly; bypasses PLL, filter, quantization (legacy equivalence)
    i_range_a: float = 400.0               # full scale of each current sensor, +/- [T3]
    adc_bits: int = 12                     # [T3]
    i_gain_err: float = 0.01               # half-width of uniform per-phase gain error [T2]
    i_offset_a: float = 2.0                # half-width of uniform per-phase offset, A [T2]
    i_noise_lsb: float = 1.0               # white noise sigma in LSB [T2]
    two_sensor: bool = False               # sensitivity case
    theta_offset_rad: float = np.deg2rad(1.0)    # half-width of uniform per-run offset [T2]
    theta_noise_rad: float = np.deg2rad(0.3)     # white noise sigma [T2]
    theta_ripple_rad: float = np.deg2rad(0.2)    # 1x and 2x mechanical harmonic amplitude [T2]
    pll_wn: float = 300.0                  # rad/s natural frequency [T3]
    pll_zeta: float = 0.707                # [T3]
    vdc_range_v: float = 500.0             # ADC full scale [T3]
    vdc_gain_err: float = 0.005            # half-width of uniform per-run gain error [T2]
    vdc_noise_v: float = 0.5               # white noise sigma [T2]
    vdc_corner_hz: float = 500.0           # first-order low-pass corner [T3]
    ref_quant: float = 0.0                 # request quantization (rad/s); 0 = off. 0.1 for TUM-sourced references
    ref_hold_s: float = 0.0                # request sample-and-hold period; 0 = off. 0.1 for TUM-sourced references

    @staticmethod
    def identity_mode(tau: float = TAU_PLAIN) -> "SensorConfig":
        return SensorConfig(tau=tau, identity=True, i_gain_err=0.0, i_offset_a=0.0, i_noise_lsb=0.0, theta_offset_rad=0.0,
                            theta_noise_rad=0.0, theta_ripple_rad=0.0, vdc_gain_err=0.0, vdc_noise_v=0.0)


@dataclass(frozen=True)
class InverterConfig:
    tau: float = TAU_PLAIN
    delay_steps: int = 1                   # extra computation delay d in {0,1,2}; total (0.5 + d) tau [T3]
    dead_time_s: float = 2.0e-6            # [T2]
    device_drop_v: float = 1.5             # [T2]
    smooth_band_a: float = 1.0             # current band around zero where the distortion is smoothed [T3]
    compensation: float = 0.7              # fraction of the distortion the controller removes [T3]
    enabled: bool = True

    @staticmethod
    def identity_mode(tau: float = TAU_PLAIN) -> "InverterConfig":
        return InverterConfig(tau=tau, delay_steps=0, dead_time_s=0.0, device_drop_v=0.0, compensation=0.0, enabled=False)


@dataclass(frozen=True)
class BMSConfig:
    period_s: float = 0.05                 # [T3]
    latency_s: float = 0.02                # [T3]
    quant_a: float = 1.0                   # [T3]
    rel_err: float = 0.05                  # sigma of multiplicative error [T2]
    enabled: bool = True


@dataclass(frozen=True)
class BeliefConfig:
    """What the controller BELIEVES about the motor, as relative errors against the plant truth (drawn once per run)."""
    dR_range: tuple = (-0.15, 0.40)
    dL_range: tuple = (0.0, 0.25)
    dJ_range: tuple = (-0.2, 0.2)
    dpsi_range: tuple = (-0.10, 0.10)      # error of the NAMEPLATE flux used for Kt and the governor (not the adaptive psi_hat)
    psi_start_frac: float = 0.7            # existing design: start 30 % low
    psi_range: tuple = (0.03, 0.10)        # existing projection prior for the robust mode
    exact: bool = False                    # True: beliefs equal truth (legacy equivalence; psi_hat0 still 0.7 psi)


@dataclass(frozen=True)
class ControllerConfig:
    mode: str = "naive"                    # naive | robust | aw
    governor: bool = False
    vdc_aware: bool = False                # governor uses vdc_m rather than the nominal ceiling
    bms_aware: bool = False                # governor also limits iq from the BMS message
    angle_advance: bool = True             # theta_adv = d * omega_m_m * p * tau [T3]  (d = delay_steps; code and contract use d, not d + 0.5)
    delay_steps: int = 1                   # must equal InverterConfig.delay_steps (the controller knows its own delay)
    gamma: float = 2.0e-5
    k_i: float = 3.0
    sigma: float = 2.0
    d_gain_by_L: bool = False              # True: d-axis feedback gain scaled by Ld/Lq so both current loops have the same bandwidth
    clamp_psi: bool = False                # True: also project psi_hat onto the prior range in aw mode (robust always does)
    i_lim: float = 250.0
    speed_bw: float = 20.0
    gov_factor: float = 0.75
    tau: float = TAU_PLAIN
    p: int = 3                             # pole pairs: a nameplate constant the controller legitimately knows
    eta: float = 0.9                       # drivetrain efficiency belief used by the BMS-aware limit

    def __post_init__(self):
        if self.mode not in ("naive", "robust", "aw"):
            raise ValueError(f"ControllerConfig.mode must be 'naive', 'robust' or 'aw', got {self.mode!r}")
        if self.delay_steps < 0 or self.tau <= 0 or self.i_lim <= 0:
            raise ValueError("ControllerConfig needs delay_steps >= 0, tau > 0 and i_lim > 0")


def d_loop_spectral_radius(k_d: float, tau: float, Ld: float, delay_steps: int) -> float:
    """Largest root magnitude of the discrete d-axis current loop z^(N+1) - z^N + a = 0, a = k_d*tau/Ld, N = delay in steps.
    Below 1 is stable. Found by the 2026-11-09 audit: with k_d = k_i and tau = 2e-4 the default loop is unstable even with one step of delay."""
    a = k_d * tau / Ld; N = max(int(delay_steps), 0)
    return float(np.max(np.abs(np.roots([1.0, -1.0] + [0.0] * (N - 1) + [a])))) if N >= 1 else abs(1.0 - a)


# --------------------------------------------------------------------------------------------- the battery-fit file schema
FIT_SCHEMA_VERSION = 1
FIT_REQUIRED = {
    "schema_version": int,
    "status": str,                          # "ok" | "not_better_than_baseline"
    "sign_convention": str,
    "source": dict,                         # {"files": {name: sha256}, "n_tracks": int, "fit_hash": str, "filters": {...}}
    "pack_real": dict,                      # {"v_nom","p_base_w","i_base_a","r_eff_by_window_ohm":{"0.1","0.3","0.5","1.0"},
                                            #  "r_eff_pu":{"lo","mid","hi"}, "ocv_k":[K0,K1,K2], "rc":{"r1_pu","tau1_s"}}
    "envelope_real": dict,                  # {"soc_grid":[..],"i_dis_lim_a":[..],"i_chg_lim_a":[..],"note": "observed-use envelope"}
    "scaling": dict,                        # {"p_base_toy_w","v_toy","i_base_toy","default_r_pu","r_total_toy_ohm":{"lo","mid","hi"},
                                            #  "i_dis_lim_toy_a":[..],"i_chg_lim_toy_a":[..]}
    "toy_battery_kwargs": dict,             # kwargs accepted by battery.Battery(...) (n_series,q_ah,r0_cell,r1_cell,tau1,r_bus,i_dis_max,i_chg_max)
    "validation": dict,                     # {"loo_rmse_v":{"fit","const_v","const_r","generic"},"status_reason": str, "glitches_removed": {...}}
}


def validate_fit_dict(d: dict) -> list[str]:
    """Return a list of schema problems (empty list = valid)."""
    bad = []
    for k, typ in FIT_REQUIRED.items():
        if k not in d:
            bad.append(f"missing key {k}")
        elif not isinstance(d[k], typ):
            bad.append(f"{k} should be {typ.__name__}, is {type(d[k]).__name__}")
    if d.get("schema_version") != FIT_SCHEMA_VERSION:
        bad.append("schema_version mismatch")
    if d.get("status") not in ("ok", "not_better_than_baseline"):
        bad.append("status must be ok or not_better_than_baseline")
    return bad
