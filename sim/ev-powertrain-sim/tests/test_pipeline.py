"""Red-team tests of the request pipeline: cycle data -> omega_ref -> speed PI -> iq_ref -> T_cmd -> P_mech_req -> P_req_ref,
plus the data-feasibility ledger. Every test says in its docstring what failure it catches.

Cheap by design: only one 0.3 s live simulation (3000 steps); everything long reads the stored runs/*.parquet.
Tests named *BUG* assert the CORRECT behaviour and are xfail(strict=True): they currently fail because of a real defect in the
source (documented in the docstring). When the defect is fixed they XPASS, which strict mode turns into a failure, so the
marker gets removed deliberately.
"""
import sys, json, pathlib, glob
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pytest

from plant import GEMPlant
from control import CascadeController
from battery import Battery
from telemetry import derive, params_from_plant
from cycles import Cycle, CycleError, VehicleEq, data_demand, fit_vehicle_to_motor, synthetic_cycle
from run_cycle import ensure_standing_start, T_CAP

RUNS = ROOT / "runs"
PSI, P_POLE = 0.066, 3
KT = 1.5 * P_POLE * PSI
LONG_RUNS = ["A", "B", "C", "D", "Dg", "Eb", "Eg"]                       # 8 s at 10 kHz, log every step
ALL_RUNS = sorted(p.stem for p in RUNS.glob("*.parquet") if "measured" not in p.stem)
Y_RUNS = [n for n in ALL_RUNS if n.startswith("Y")]


# ----------------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def plant():
    return GEMPlant()


@pytest.fixture(scope="module")
def P_batt():
    return params_from_plant(GEMPlant(battery=Battery()))


_cache = {}


def derived(name, P):
    """derive() of a stored run, cached for the whole session (derive on 80k rows is the expensive part)."""
    if name not in _cache:
        _cache[name] = derive(pd.read_parquet(RUNS / f"{name}.parquet"), P)
    return _cache[name]


def manifest(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def ctl_for(plant, **kw):
    return CascadeController(plant, mode="aw", **kw)


@pytest.fixture(scope="module")
def live(P_batt):
    """One 0.3 s closed-loop run (3000 steps, ~1 s wall) on a battery-less plant with a road disturbance."""
    pl = GEMPlant(disturbance_fn=lambda t: 3.0 if t > 0.1 else 0.0)
    ctl = CascadeController(pl, mode="aw")
    rows, led = [], {}
    for _ in range(3000):
        st = pl.state()
        u, it = ctl.command(min(st["t"] * 50.0, 40.0), st, led)
        led = pl.apply_voltage(u)
        T = KT * it["iq_ref"]
        rows.append({**st, **it, **led, "T_cmd": T, "d_T": st["T_elec"] - T, "dist_i_limit": 1.0, "V": 0.0})
    return derive(pd.DataFrame(rows), params_from_plant(pl))


# =========================================================================================== 1. speed loop unit tests
def test_speed_loop_sign_follows_error(plant):
    """Catches a flipped sign in e_w = omega_ref - omega (the loop would accelerate away from the request)."""
    for om_ref, om in [(50.0, 10.0), (10.0, 50.0), (-30.0, 20.0), (20.0, -30.0), (-5.0, -40.0)]:
        c = ctl_for(plant)
        c._speed_loop(om_ref, om)
        assert np.sign(c.iq_ref) == np.sign(om_ref - om), (om_ref, om)


def test_speed_loop_first_call_is_pure_proportional(plant):
    """Catches Kp_w being ignored/mis-scaled: on the first call the integrator is 0 so iq_ref must be exactly Kp_w*e."""
    c = ctl_for(plant)
    assert c.Kp_w == pytest.approx(plant.J * 20.0 / KT)
    c._speed_loop(52.0, 50.0)
    assert c.iq_ref == pytest.approx(c.Kp_w * 2.0, rel=1e-12)
    assert c.i_ref_clip == 0.0 and c.int_w == pytest.approx(c.dt_w * c.Ki_w * 2.0)


def test_speed_loop_zero_error_from_rest_gives_zero_request(plant):
    """Catches a bias/offset in the speed loop: omega_ref = omega = 0 must request exactly 0 A, and keep it at 0."""
    c = ctl_for(plant)
    for _ in range(50):
        c._speed_loop(0.0, 0.0)
        assert c.iq_ref == 0.0 and c.int_w == 0.0 and c.i_ref_clip == 0.0


def test_speed_loop_clips_at_i_lim_and_reports_clip(plant):
    """Catches a missing/wrong clip: a huge error must give exactly +-i_lim, and i_ref_clip must equal clipped minus unsaturated."""
    for e in (1000.0, -1000.0):
        c = ctl_for(plant, i_lim=250.0)
        unsat = c.Kp_w * e + c.int_w
        c._speed_loop(e, 0.0)
        assert c.iq_ref == pytest.approx(np.sign(e) * 250.0)
        assert c.i_ref_clip == pytest.approx(c.iq_ref - unsat, rel=1e-12)
        assert np.sign(c.i_ref_clip) == -np.sign(e)                    # clip always removes request, never adds
    c = ctl_for(plant, i_lim=100.0)
    c._speed_loop(1000.0, 0.0)
    assert c.iq_ref == 100.0 and c.i_lim_eff == 100.0


def test_speed_loop_governor_limit_is_used_and_below_i_lim(plant):
    """Catches the governor limit not being applied (or exceeding i_lim): at 150 rad/s the voltage-ellipse limit is < 250 A."""
    c = ctl_for(plant, governor=True)
    om = 150.0
    w_el = plant.p * om
    expected = 0.75 * np.sqrt((plant.u_max_hw / w_el) ** 2 - plant.psi_p ** 2) / plant.Lq
    assert expected < 250.0
    c._speed_loop(om + 500.0, om)
    assert c.i_lim_eff == pytest.approx(expected, rel=1e-12)
    assert c.iq_ref == pytest.approx(expected, rel=1e-12)
    assert c.i_ref_clip == pytest.approx(expected - c.Kp_w * 500.0, rel=1e-12)


def test_speed_loop_governor_never_raises_limit_above_i_lim_at_low_speed(plant):
    """Catches the governor granting more than i_lim at standstill (room is huge there): limit must be min(i_lim, governor)."""
    c = ctl_for(plant, governor=True, i_lim=250.0)
    c._speed_loop(100.0, 0.0)
    assert c.i_lim_eff <= 250.0 and abs(c.iq_ref) <= 250.0


def test_speed_loop_antiwindup_integrator_stops_growing_when_clipped(plant):
    """Catches missing back-calculation: with a persistent huge error the clipped integrator must saturate, while the same
    loop with no clip (i_lim=1e9) keeps integrating linearly."""
    cl, free = ctl_for(plant, i_lim=250.0), ctl_for(plant, i_lim=1e9)
    hist = []
    for _ in range(3000):
        cl._speed_loop(1000.0, 0.0)
        free._speed_loop(1000.0, 0.0)
        hist.append(cl.int_w)
    assert free.int_w == pytest.approx(3000 * free.dt_w * free.Ki_w * 1000.0, rel=1e-9)   # pure integration
    assert abs(cl.int_w) < 0.01 * free.int_w
    assert abs(hist[-1] - hist[-500]) < 0.01                 # plateaued
    assert cl.iq_ref == 250.0


def test_speed_loop_antiwindup_recovers_quickly(plant):
    """Catches a wound-up integrator that keeps the request railed after the error reverses: after the error flips sign the
    clipped loop must leave the +i_lim rail within a few calls."""
    c = ctl_for(plant, i_lim=250.0)
    for _ in range(3000):
        c._speed_loop(1000.0, 0.0)
    n = 0
    while c.iq_ref > 0 and n < 200:
        c._speed_loop(0.0, 100.0)
        n += 1
    assert n < 50, n


def test_speed_loop_antisymmetric_in_error(plant):
    """Catches any asymmetry between accel and brake requests (clip, integrator, governor): negating (omega_ref, omega) must
    negate iq_ref on every call of a sequence, including through saturation."""
    for kw in (dict(), dict(governor=True)):
        a, b = ctl_for(plant, **kw), ctl_for(plant, **kw)
        for k in range(300):
            om = 100.0 + 0.2 * k
            ref = 100.0 + (400.0 if 100 <= k < 200 else 8.0)
            a._speed_loop(ref, om)
            b._speed_loop(-ref, -om)
            assert b.iq_ref == pytest.approx(-a.iq_ref, rel=1e-12, abs=1e-12)
            assert b.i_ref_clip == pytest.approx(-a.i_ref_clip, rel=1e-12, abs=1e-12)


def test_speed_loop_called_only_every_ten_steps(plant):
    """Catches the 1 kHz speed loop being run at the 10 kHz current-loop rate (dt_w would then be wrong by 10x)."""
    c = ctl_for(plant)
    assert c.dt_w == pytest.approx(10 * plant.tau)
    st = plant.state()
    c.command(50.0, st, {})
    first = c.iq_ref
    for _ in range(9):
        c.command(0.0, st, {})                                          # omega_ref changed, but loop must hold its output
        assert c.iq_ref == first


# ============================================================================================ 2. request chain identities
@pytest.mark.parametrize("name", ALL_RUNS)
def test_chain_identities_exact(name, P_batt):
    """Catches any break in T_cmd = 1.5 p psi iq_ref, P_mech_req = T_cmd w_m, P_req_ref = P_cu_ref + P_mech_req, the finiteness
    of P_req_ref, and the disturbance split T_roll+T_visc+T_aero == T_fric."""
    d = derived(name, P_batt)
    assert np.allclose(d.T_cmd, KT * d.iq_ref, rtol=1e-12, atol=1e-12)
    assert np.allclose(d.P_mech_req, d.T_cmd * d.omega_m, rtol=1e-12, atol=1e-9)
    assert np.allclose(d.P_req_ref, d.P_cu_ref + d.P_mech_req, rtol=1e-12, atol=1e-9)
    assert np.allclose(d.P_cu_ref, 1.5 * P_batt["R"] * d.iq_ref ** 2, rtol=1e-12, atol=1e-12)
    assert np.isfinite(d.P_req_ref.to_numpy()).all()
    assert np.abs(d.T_roll + d.T_visc + d.T_aero - d.T_fric).max() < 1e-9


@pytest.mark.parametrize("name", ALL_RUNS)
def test_req_power_never_below_shaft_power_and_sign_consistent(name, P_batt):
    """Catches a negative copper-loss term or a sign error that makes the electrical request disagree with the torque request:
    P_req_ref >= P_mech_req always, and the two can differ in sign only where copper loss exceeds the regen shaft power."""
    d = derived(name, P_batt)
    assert (d.P_req_ref - d.P_mech_req).min() >= -1e-9
    opp = (d.P_req_ref * d.P_mech_req) < 0
    assert (d.P_mech_req[opp].abs() <= d.P_cu_ref[opp] + 1e-9).all()


@pytest.mark.parametrize("name", LONG_RUNS)
def test_elec_decomposition_closes_on_10khz_runs(name, P_batt):
    """Catches a broken M6c identity dP_elec_ref = dP_mech + dP_cu + P_mag: residual must be < 1 % of P_mech rms on runs logged
    at the integration step (so the finite-difference tau in derive() is right)."""
    d = derived(name, P_batt)
    assert rms(d.resid_elec_decomp) / rms(d.P_mech) < 0.01


@pytest.mark.parametrize("name", ALL_RUNS)
def test_actuator_power_identity_is_exact(name, P_batt):
    """Catches a broken M2->M6 link: dP_elec must equal 1.5 (du_d i_d + du_q i_q) to float precision (< 1e-6 W) on every run."""
    d = derived(name, P_batt)
    assert d.resid_du_power.abs().max() < 1e-6


@pytest.mark.parametrize("name", LONG_RUNS)
def test_req_decomposition_residual_small(name, P_batt):
    """Catches a broken voltage-command decomposition (psi-error + magnetic - feedback): rms residual must be < 1 % of P_mech rms
    on 10 kHz runs, the same order as the balance residual."""
    d = derived(name, P_batt)
    assert rms(d.resid_req_decomp) / rms(d.P_mech) < 0.01
    assert rms(d.P_balance_resid) / rms(d.P_mech) < 0.01


def test_live_run_chain_and_ledger(live):
    """Catches regressions on a fresh simulation (not stored data): identities, finiteness, bounded input, no hidden clipping."""
    d = live
    assert np.allclose(d.T_cmd, KT * d.iq_ref) and np.allclose(d.P_mech_req, d.T_cmd * d.omega_m)
    assert np.allclose(d.P_req_ref, d.P_cu_ref + d.P_mech_req)
    assert np.isfinite(d.P_req_ref).all()
    assert d.resid_du_power.abs().max() < 1e-6
    assert d.d_u_plant_d.abs().max() < 1e-6 and d.d_u_plant_q.abs().max() < 1e-6
    assert (np.hypot(d.u_d_app, d.u_q_app) <= d.u_limit * (1 + 1e-9)).all()
    s = d.iloc[1000:]                                                   # settled part; short low-power run has a constant ~20 W discretisation floor
    assert rms(s.resid_elec_decomp) / rms(s.P_mech) < 0.07


# ====================================================================================== 3. feasible data: controller floor
def test_run_A_electrical_mismatch_is_controller_floor(P_batt):
    """Catches a controller/plant energy mismatch on clean, event-free data: run A dP_elec_ref rms must be < 2 % of P_mech rms
    (measured 0.89 %) and the actuator must never saturate."""
    d = derived("A", P_batt)
    assert rms(d.dP_elec_ref) / rms(d.P_mech) < 0.02
    assert d.saturated.max() == 0.0
    assert d.d_u_d.abs().max() == 0.0 and d.d_u_q.abs().max() == 0.0


def test_run_A_speed_tracks_request(P_batt):
    """Catches a speed loop that silently fails to track feasible data: run A rms speed error must be small vs the request."""
    d = derived("A", P_batt)
    assert rms(d.e_speed) < 0.05 * rms(d.omega_ref)


@pytest.mark.parametrize("name", ["Yrb", "Ysb"])
def test_feasible_baselines_have_small_mismatch(name, P_batt):
    """Catches data-induced mismatch in the feasible baselines (no event): no saturation, no infeasible flag, tiny speed error."""
    if not (RUNS / f"{name}.parquet").exists():
        pytest.skip("run not written yet")
    d = derived(name, P_batt)
    assert d.saturated.max() == 0.0 and d.data_infeasible.sum() == 0
    assert rms(d.e_speed) < 0.01 * d.omega_ref.max()


# =========================================================================================== 4. counterfactual flag
def test_sign_conflict_flag_zero_on_A_positive_on_B(P_batt):
    """Catches a dead or always-on counterfactual flag: req_sign_conflict must be 0 on clean run A and fire on saturated run B."""
    assert derived("A", P_batt).req_sign_conflict.sum() == 0
    assert derived("B", P_batt).req_sign_conflict.sum() > 0


def test_sign_conflict_implies_a_large_opposite_sign_pair(P_batt):
    """Catches a flag that fires on noise: wherever flagged, both powers exceed the 10 W floor and have opposite sign."""
    d = derived("B", P_batt)
    f = d[d.req_sign_conflict == 1]
    assert len(f) > 0
    assert (f.P_req_elec * f.P_req_ref < 0).all() and (f.P_req_elec.abs() > 10).all() and (f.P_req_ref.abs() > 10).all()


# ==================================================================================================== 5. data ledger
@pytest.fixture(scope="module")
def fitted():
    out = {}
    for kind in ("ramp", "sine"):
        c = synthetic_cycle(kind)
        veq, _ = fit_vehicle_to_motor(c, T_cap=T_CAP)
        out[kind] = (c, veq, data_demand(c, veq, J_rotor=0.0388))
    from scipy.ndimage import uniform_filter1d
    c = synthetic_cycle("step")
    sm = Cycle(t=c.t.copy(), v=uniform_filter1d(c.v, size=40, mode="nearest"), dt=c.dt)       # same recipe as run class Yt (fit_smooth 4 s)
    veq, _ = fit_vehicle_to_motor(sm, T_cap=T_CAP)
    out["step"] = (c, veq, data_demand(c, veq, J_rotor=0.0388))
    return out


@pytest.mark.parametrize("kind", ["ramp", "sine"])
def test_feasible_synthetic_cycles_have_ratio_below_one(fitted, kind):
    """Catches fit_vehicle_to_motor producing a vehicle the motor cannot drive: smooth cycles must give data_ratio <= 1 everywhere
    (measured max 0.88 = the 0.85 headroom plus rotor inertia)."""
    _, _, dd = fitted[kind]
    assert dd["ratio"].max() <= 1.0 and np.isfinite(dd["ratio"]).all()


def test_step_cycle_flags_exactly_the_jump_samples(fitted):
    """Catches a ledger that misses jumps or smears the flag: ratio > 1 only at the sample before and at each jump (central difference)."""
    c, _, dd = fitted["step"]
    jumps = np.flatnonzero(np.abs(np.diff(c.v)) > 1.0) + 1                  # index of first sample after each jump
    assert len(jumps) == 4
    expect = sorted(set(jumps) | set(jumps - 1))
    assert list(np.flatnonzero(dd["ratio"] > 1.0)) == expect
    assert dd["ratio"].max() > 5.0


def test_T_avail_bounded_and_decreases_above_base_speed():
    """Catches a torque envelope above the physical cap or one that does not roll off: T_avail <= 74.25 N.m, flat below ~188 rad/s
    and monotonically decreasing (strictly beyond) above it."""
    w = np.linspace(0.0, 400.0, 2001)
    c = Cycle(t=np.arange(w.size) * 0.1, v=w, dt=0.1)
    dd = data_demand(c, VehicleEq(k=1.0, mass=50.0))
    Ta = dd["T_avail"]
    assert Ta.max() <= 1.5 * 3 * 0.066 * 250.0 + 1e-9 and Ta.max() == pytest.approx(74.25)
    assert np.all(np.diff(Ta[w > 190.0]) < 0)
    assert np.allclose(Ta[w < 186.0], 74.25)
    assert Ta[-1] < 0.7 * 74.25


def test_data_ratio_invariant_to_time_shift(fitted):
    """Catches an absolute-time dependence in data_demand (e.g. gradient w.r.t. index or t=0 anchoring)."""
    c, veq, dd = fitted["step"]
    c2 = Cycle(t=c.t + 123.4, v=c.v.copy(), dt=c.dt)
    dd2 = data_demand(c2, veq, J_rotor=0.0388)
    assert np.allclose(dd["ratio"], dd2["ratio"], rtol=1e-9, atol=1e-12)
    assert np.allclose(dd["T_req"], dd2["T_req"], rtol=1e-9, atol=1e-9)


def test_inertia_demand_scales_linearly_with_mass():
    """Catches a mass term that is not linear (inertia torque = m/k^2 * dw/dt): with road load and rotor inertia removed,
    doubling the mass must exactly double data_ratio."""
    c = synthetic_cycle("ramp")
    kw = dict(k=7.6, crr=0.0, cda=0.0, drivetrain_b=0.0)
    r1 = data_demand(c, VehicleEq(mass=50.0, **kw), J_rotor=0.0)["ratio"]
    r2 = data_demand(c, VehicleEq(mass=100.0, **kw), J_rotor=0.0)["ratio"]
    assert r1.max() > 0.05
    assert np.allclose(r2, 2.0 * r1, rtol=1e-9, atol=1e-12)


def test_synthetic_cycle_unknown_kind_raises():
    """Catches silent fallback for a misspelled cycle class: must raise CycleError with code 'bad_kind'."""
    with pytest.raises(CycleError) as e:
        synthetic_cycle("triangle")
    assert e.value.code == "bad_kind"


@pytest.mark.parametrize("kind", ["ramp", "sine", "step"])
def test_synthetic_cycles_deterministic_and_clean(kind):
    """Catches randomness/NaN/negatives in the generated data: two calls identical, finite, v >= 0, uniform dt, starts at t=0."""
    a, b = synthetic_cycle(kind), synthetic_cycle(kind)
    assert np.array_equal(a.v, b.v) and np.array_equal(a.t, b.t)
    assert np.isfinite(a.v).all() and (a.v >= 0).all()
    assert a.t[0] == 0.0 and np.allclose(np.diff(a.t), a.dt)
    assert a.v.max() <= 25.0 + 1e-9


def test_data_demand_sees_jump_acceleration_of_interpolated_request():
    """FIXED (was a bug found by this test): omega_ref reaches the controller through np.interp (piecewise linear), so a speed jump of dv over one sample dt has
    acceleration dv/dt inside that interval. data_demand uses np.gradient (dv/2dt), so it under-reports the torque demand by ~1.9x.
    Evidence: a 1.0 m/s jump in 0.1 s gives data_ratio 0.785 (not flagged) while the demand of the interpolated request is 1.455
    (infeasible for any controller). The ledger thus misses requests that are infeasible (false negatives in data_infeasible)."""
    veq = VehicleEq(k=7.6, mass=73.4)
    t = np.arange(0, 10.001, 0.1)
    c = Cycle(t=t, v=np.where(t < 5, 10.0, 11.0), dt=0.1)
    reported = data_demand(c, veq)["ratio"].max()
    tf = np.arange(0, 10.0001, 0.001)
    cf = Cycle(t=tf, v=np.interp(tf, c.t, c.v), dt=0.001)
    true = data_demand(cf, veq)["ratio"].max()
    assert true > 1.0                       # really infeasible for the request as the controller sees it
    assert reported > 1.0                   # ... so the ledger must say so (it says 0.785)


# ======================================================================================== 6. omega_ref data-quality gates
def _moving_cycle(v0, n=200, dt=0.1):
    t = np.arange(n) * dt
    v = v0 + 5.0 * np.sin(np.pi * t / t[-1])
    return Cycle(t=t, v=v, dt=dt)


def test_omega_ref_finite_and_starts_at_zero_for_all_kinds():
    """Catches NaN/inf or a non-zero first request reaching the controller (the motor starts at rest, so a jump at t=0 is a data fault)."""
    for kind in ("ramp", "sine", "step"):
        c = ensure_standing_start(synthetic_cycle(kind))
        veq = VehicleEq(k=7.6, mass=60.0)
        wref = np.array([veq.k * float(np.interp(t, c.t, c.v)) for t in np.linspace(0, c.duration, 500)])
        assert np.isfinite(wref).all()
        assert abs(wref[0]) < 1e-9 or kind == "sine"                    # the sine cycle starts at its minimum, 0.15*v_peak: see next test


def test_sine_cycle_does_not_start_at_standstill_BUG_candidate():
    """Documents (passes) a data-quality fact: synthetic 'sine' starts at 0.15*v_peak = 3.75 m/s, and ensure_standing_start prepends a
    ramp from rest for it, so the controller still sees omega_ref(0) = 0. Catches the standing-start fix not being applied."""
    c = synthetic_cycle("sine")
    assert c.v[0] == pytest.approx(0.15 * 25.0)
    c2 = ensure_standing_start(synthetic_cycle("sine"))
    assert c2.v[0] == 0.0 and "standing_start_added" in c2.codes() and c2.v.size > c.v.size


def test_standing_start_ramp_added_and_gentle():
    """Catches a missing/too-steep standing-start ramp: a cycle beginning at 12 m/s must get a ramp from 0 with slope <= 1.5 m/s^2,
    ending exactly at the original first sample, with t uniform and a report entry."""
    c0 = _moving_cycle(12.0)
    n0, v_first = c0.v.size, c0.v[0]
    c = ensure_standing_start(_moving_cycle(12.0))
    n_add = c.v.size - n0
    assert n_add > 0 and c.v[0] == 0.0 and c.v[n_add] == v_first
    assert np.all(np.diff(c.v[: n_add + 1]) >= 0) and np.max(np.diff(c.v[: n_add + 1])) / c.dt <= 1.5 + 1e-9
    assert np.allclose(np.diff(c.t), c.dt) and c.t[0] == 0.0 and "standing_start_added" in c.codes()


def test_standing_start_not_added_when_already_at_rest():
    """Catches a spurious ramp on cycles that already start (near) at rest: the data must be left untouched."""
    c = _moving_cycle(0.0)
    v = c.v.copy()
    out = ensure_standing_start(c)
    assert np.array_equal(out.v, v) and "standing_start_added" not in out.codes()


def test_standing_start_ramp_carries_grade_and_measured_channels():
    """Catches length mismatch between v, grade and measured channels after the ramp is prepended (would misalign the replay)."""
    c = _moving_cycle(8.0)
    c.grade = np.full(c.v.size, 0.02)
    c.measured = {"torque": np.ones(c.v.size)}
    out = ensure_standing_start(c)
    assert out.grade.size == out.v.size == out.t.size == out.measured["torque"].size
    assert np.isnan(out.measured["torque"][0]) and out.grade[0] == 0.02


def test_fit_maps_peak_speed_to_190_rad_s():
    """Catches a wrong gear ratio: after fit_vehicle_to_motor the maximum omega_ref = k*max(v) must be 190 rad/s (below base speed)."""
    for kind in ("ramp", "sine"):
        c = ensure_standing_start(synthetic_cycle(kind))
        veq, info = fit_vehicle_to_motor(c, T_cap=T_CAP)
        assert (veq.k * c.v).max() == pytest.approx(190.0, rel=1e-12) and info["omega_peak"] == pytest.approx(190.0)


def test_piecewise_linear_omega_ref_never_overshoots_data():
    """Catches an interpolation that rings/overshoots: between samples omega_ref must stay inside [min, max] of its two neighbours
    and globally inside the data range (np.interp is linear; this guards against swapping in a spline)."""
    c = ensure_standing_start(synthetic_cycle("sine"))
    k = 8.0
    tf = np.linspace(0, c.duration, 4001)
    wf = k * np.interp(tf, c.t, c.v)
    idx = np.clip(np.searchsorted(c.t, tf, side="right") - 1, 0, c.v.size - 2)
    lo = k * np.minimum(c.v[idx], c.v[idx + 1]) - 1e-9
    hi = k * np.maximum(c.v[idx], c.v[idx + 1]) + 1e-9
    assert np.all((wf >= lo) & (wf <= hi))
    assert wf.max() <= k * c.v.max() + 1e-9 and wf.min() >= k * c.v.min() - 1e-9


@pytest.mark.parametrize("name", Y_RUNS)
def test_stored_runs_omega_ref_clean(name, P_batt):
    """Catches bad data in the stored cycle runs: omega_ref finite, starts at 0, peaks at 190 rad/s (to 1e-6) and never exceeds it."""
    d = derived(name, P_batt)
    assert np.isfinite(d.omega_ref.to_numpy()).all()
    assert abs(d.omega_ref.iloc[0]) < 1e-9
    assert d.omega_ref.max() == pytest.approx(190.0, abs=1e-6)
    assert np.isfinite(d.data_ratio.to_numpy()).all() and (d.data_ratio >= 0).all()
    assert d.T_avail.max() <= 74.25 + 1e-9 and d.T_avail.min() > 0
    assert (d.data_infeasible == (d.data_ratio > 1.0)).all()


# ========================================================================================================= 7. attribution
def _needs(*names):
    missing = [n for n in names if not (RUNS / f"{n}.parquet").exists()]
    if missing:
        pytest.skip(f"runs not written yet: {missing}")


def _err_frac_after_infeasible(d, window=2.0):
    t, e = d.t.to_numpy(), d.e_speed.to_numpy() ** 2
    inf = d.data_infeasible.to_numpy() > 0
    near = np.zeros(t.size, bool)
    for ti in t[inf]:
        near |= (t >= ti) & (t <= ti + window)
    return e[near].sum() / e.sum(), near.mean()


def test_attribution_feasible_ramp_baseline_has_no_infeasible_flag(P_batt):
    """Catches a false data-infeasible flag on the feasible ramp baseline Yrb (must be 0 everywhere, and ratio max < 1)."""
    _needs("Yrb")
    d = derived("Yrb", P_batt)
    assert d.data_infeasible.sum() == 0 and d.data_ratio.max() < 1.0
    assert manifest("Yrb")["data_demand"]["frac_ratio_gt_1"] == 0.0


def test_attribution_step_baseline_error_sits_on_infeasible_samples(P_batt):
    """Catches mis-attribution in the no-event step run Ytb: its speed-error energy must be >= 95 % within 2 s after an infeasible
    sample, with no saturation (so the mismatch is the data's fault by construction)."""
    _needs("Ytb")
    d = derived("Ytb", P_batt)
    frac, _ = _err_frac_after_infeasible(d)
    assert frac >= 0.95 and d.saturated.max() == 0.0
    assert rms(d.e_speed) > 10 * rms(derived("Yrb", P_batt).e_speed) if (RUNS / "Yrb.parquet").exists() else True


@pytest.mark.parametrize("name", ["Ytn", "Ytp"])
def test_attribution_step_with_event_error_enriched_near_infeasible(name, P_batt):
    """Catches a flag with no explanatory power in the step+derate classes: error energy within 2 s of an infeasible sample must
    exceed the share of time that window covers (measured 0.58 vs 0.45 for Ytn; weak, see BUG test below)."""
    _needs(name)
    d = derived(name, P_batt)
    frac, tfrac = _err_frac_after_infeasible(d)
    assert frac > tfrac


@pytest.mark.parametrize("name", ["Ytn", "Ytp"])
def test_step_class_mismatch_is_concentrated_where_data_is_infeasible(name, P_batt):
    """FIXED by moving the fault onto a flat plateau (run_classes.py, class Yt). Before, the BMS cut landed on a jump, so event error and
    data error overlapped and the flag could not separate them. Original finding: the quality rule says mismatch must be explainable by the data flag, but in Ytn only 58 % of the
    speed-error energy lies within 2 s of an infeasible sample (those windows cover 45 % of the run), and the biggest error
    (t = 8..12 s, 96 % of the energy) is the same place where the FEASIBLE ramp with the same derate (Yrn) puts 98 % of its error.
    So the data_infeasible flag cannot discriminate the event-caused (plant) error from data-caused error. Expect >= 80 % to hold."""
    _needs(name)
    frac, _ = _err_frac_after_infeasible(derived(name, P_batt))
    assert frac >= 0.80


def test_attribution_event_error_in_feasible_ramp_has_zero_data_share(P_batt):
    """Catches data blame being assigned to a feasible request: Yrn/Yrp have a huge speed error (rms > 10 rad/s) yet data_infeasible
    must be 0 everywhere, so all of it is controller/plant by the ledger's own rule."""
    _needs("Yrn", "Yrp")
    for n in ("Yrn", "Yrp"):
        d = derived(n, P_batt)
        assert rms(d.e_speed) > 10.0 and d.data_infeasible.sum() == 0


def test_manifest_ratio_flag_matches_stored_rows():
    """Catches an inconsistency between the manifest data_demand summary and the logged flag: the logged infeasible fraction
    (after interpolating the 0.1 s ratio to the log grid) may be larger than the sample fraction, but never smaller, and zero
    exactly when the manifest says zero."""
    _needs("Ytb", "Yrb")
    for n in ("Ytb", "Yrb"):
        d = pd.read_parquet(RUNS / f"{n}.parquet")
        m = manifest(n)["data_demand"]["frac_ratio_gt_1"]
        f = float((d.data_ratio > 1).mean())
        assert (f == 0.0) == (m == 0.0) and f >= m - 1e-9


# ============================================================================================ 8. bounded input by construction
@pytest.mark.parametrize("name", ALL_RUNS)
def test_applied_voltage_inside_disc_and_no_hidden_clipping(name, P_batt):
    """Catches a saturation that is not a projection onto the disc: |u_app| <= u_limit*(1+1e-9) on every row, u_limit <= 173.2051 V
    (300/sqrt3), and the plant never clips behind our back (|d_u_plant| < 1e-6 V)."""
    d = pd.read_parquet(RUNS / f"{name}.parquet")
    mag = np.hypot(d.u_d_app, d.u_q_app)
    assert (mag <= d.u_limit * (1 + 1e-9)).all()
    assert d.u_limit.max() <= 173.2051 and d.u_limit.min() > 0
    assert d.d_u_plant_d.abs().max() < 1e-6 and d.d_u_plant_q.abs().max() < 1e-6
    sat = d.saturated > 0
    if sat.any():                                                       # saturated rows must sit ON the circle, unsaturated strictly inside it
        assert np.allclose(mag[sat], d.u_limit[sat], rtol=1e-9)
    assert (np.hypot(d.u_d_cmd, d.u_q_cmd)[~sat] <= d.u_limit[~sat] * (1 + 1e-9)).all()


# ====================================================================================== 9. telemetry timebase (suspicious)
@pytest.mark.parametrize("name", [n for n in Y_RUNS if (RUNS / f"{n}.json").exists()])
def test_finer_tau_gives_smaller_balance_residual_on_decimated_runs(name, P_batt):
    """Documents the cause of the BUG below: on runs logged every 25 steps, using the true row spacing (tau_log) as derive()'s tau
    shrinks the energy-balance residual relative to using the plant's integration step (which is what params_from_plant returns)."""
    df = pd.read_parquet(RUNS / f"{name}.parquet")
    m = manifest(name)
    assert m["log_every"] == 25 and m["tau"] == pytest.approx(2e-4)
    good = derive(df, dict(P_batt, tau=m["tau_log"]))
    with pytest.warns(UserWarning, match="row spacing"):
        auto = derive(df, P_batt)                      # wrong tau in P: derive() must notice and use the time axis, and say so
    assert np.allclose(good.P_balance_resid, auto.P_balance_resid, rtol=1e-9, atol=1e-9)
    assert auto.attrs["tau_used"] == pytest.approx(m["tau_log"])
    assert auto.attrs["derivative_channels_valid"] is False


@pytest.mark.xfail(strict=True, reason="LIMITATION: even with the right row spacing, 5 ms rows cannot resolve 0.4 ms current dynamics, so the "
                   "derivative-based energy books do not close on decimated logs. derive() flags this in attrs['derivative_channels_valid']")
def test_LIMITATION_decimated_logs_cannot_close_the_derivative_energy_books(P_batt):
    """KNOWN LIMITATION (the wrong-tau part of the original bug is fixed): derive() takes dt = P['tau'] (plant.tau, 1e-4 s by default, and not even the run's 2e-4 s) while stored cycle runs are
    logged every 25 steps (5 ms). np.gradient(i, tau) is then wrong by ~50x, so P_mag, P_balance_resid and resid_elec_decomp are garbage
    (Yrb: resid_elec_decomp = 17 % of P_mech rms, Ytn 565 %, T0n 224 %) with no warning. derive() should infer dt from the t column."""
    _needs("Yrb")
    d = derived("Yrb", P_batt)
    assert rms(d.resid_elec_decomp) / rms(d.P_mech) < 0.01
    assert rms(d.P_balance_resid) / rms(d.P_mech) < 0.01
