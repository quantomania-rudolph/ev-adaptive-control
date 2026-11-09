"""Coverage gap tests: interfaces.py configs and every branch of control_fw.FirewalledController.

All expectations are recomputed here from the formulas in the module docstring, never read back from the controller.
"""
import dataclasses
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import interfaces as I
from control_fw import Beliefs, FirewalledController

R_, LD, LQ, PSI, J_, PSI0 = 0.018, 0.00037, 0.0012, 0.066, 0.54, 0.0462
KT = 1.5 * 3 * PSI
KP_W = J_ * 20.0 / KT
KI_W = KP_W * 20.0 / 4.0


def mk(**over):
    base = dict(t=0.0, i_a_m=0.0, i_b_m=0.0, i_c_m=0.0, i_d_m=0.0, i_q_m=10.0, theta_m=0.0, omega_m_m=100.0, vdc_m=300.0,
                bms=I.BMSMsg(237.0, 61.0, 300.0, 0.01), omega_ref=100.5, ceiling_telemetry=None)
    base.update(over)
    return I.Measurements(**base)


def bel(**over):
    b = dict(R=R_, Ld=LD, Lq=LQ, psi_name=PSI, J=J_, psi_hat0=PSI0)
    b.update(over)
    return Beliefs(**b)


def ctl(**over):
    kw = dict(delay_steps=0, angle_advance=False)
    kw.update(over)
    return FirewalledController(I.ControllerConfig(**kw), bel())


# ----------------------------------------------------------------------------------------------- interfaces defaults
def test_controller_config_defaults_are_the_documented_values():
    c = I.ControllerConfig()
    assert (c.mode, c.governor, c.vdc_aware, c.bms_aware, c.angle_advance, c.delay_steps) == ("naive", False, False, False, True, 1)
    assert (c.gamma, c.k_i, c.sigma, c.i_lim, c.speed_bw, c.gov_factor, c.p, c.eta) == (2.0e-5, 3.0, 2.0, 250.0, 20.0, 0.75, 3, 0.9)
    assert (c.d_gain_by_L, c.clamp_psi, c.tau) == (False, False, I.TAU_PLAIN)


def test_belief_config_defaults_are_the_documented_values():
    b = I.BeliefConfig()
    assert b.dR_range == (-0.15, 0.40) and b.dL_range == (0.0, 0.25) and b.dJ_range == (-0.2, 0.2) and b.dpsi_range == (-0.10, 0.10)
    assert b.psi_start_frac == 0.7 and b.psi_range == (0.03, 0.10) and b.exact is False


@pytest.mark.parametrize("cls", [I.ControllerConfig, I.BeliefConfig, I.SensorConfig, I.InverterConfig, I.BMSConfig])
def test_configs_are_frozen(cls):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cls(), "tau" if hasattr(cls(), "tau") else "exact", 1)


def test_u_cap_is_hexagon_inscribed_disc_at_gem_scale():
    assert I.U_CAP == pytest.approx(300.0 / np.sqrt(3.0), abs=1e-12) and 173.205 < I.U_CAP < 173.206


def test_identity_modes_switch_every_error_source_off():
    s, v = I.SensorConfig.identity_mode(2e-4), I.InverterConfig.identity_mode(2e-4)
    assert s.identity and s.tau == 2e-4 and (s.i_gain_err, s.i_offset_a, s.i_noise_lsb, s.theta_offset_rad, s.theta_noise_rad, s.theta_ripple_rad,
                                           s.vdc_gain_err, s.vdc_noise_v) == (0.0,) * 8
    assert (v.enabled, v.delay_steps, v.dead_time_s, v.device_drop_v, v.compensation) == (False, 0, 0.0, 0.0, 0.0)


def test_rng_streams_are_named_independent_and_reproducible():
    a, b = I.rng_streams(7), I.rng_streams(7)
    assert tuple(a) == I.RNG_STREAMS
    for k in a:
        assert np.array_equal(a[k].standard_normal(5), b[k].standard_normal(5))
    c = I.rng_streams(8)
    assert not np.array_equal(I.rng_streams(7)["sensors"].standard_normal(5), c["sensors"].standard_normal(5))
    draws = [I.rng_streams(7)[k].standard_normal(3).tobytes() for k in I.RNG_STREAMS]
    assert len(set(draws)) == len(I.RNG_STREAMS)


def test_rot_clarke_park_known_values():
    assert np.allclose(I.rot(np.pi / 2) @ [1.0, 0.0], [0.0, 1.0], atol=1e-12)
    al, be = I.clarke(1.0, -0.5, -0.5)
    assert al == pytest.approx(1.0) and be == pytest.approx(0.0)
    assert I.clarke(5, 5, 5) == pytest.approx((0.0, 0.0))                       # common mode cancels
    assert I.clarke_two_sensor(5, 5)[0] == 5                                    # but not in the two-sensor variant
    d, q = I.park(1.0, -0.5, -0.5, np.pi / 2)                                   # alpha axis seen from a 90 degree frame is -q
    assert d == pytest.approx(0.0, abs=1e-12) and q == pytest.approx(-1.0)


def test_validate_fit_dict_reports_missing_and_wrong_type():
    assert "missing key pack_real" in I.validate_fit_dict({})
    good = {k: (1 if t is int else "ok" if t is str else {}) for k, t in I.FIT_REQUIRED.items()}
    assert I.validate_fit_dict(good) == []
    assert any("status must be" in s for s in I.validate_fit_dict({**good, "status": "bad"}))
    assert any("schema_version mismatch" in s for s in I.validate_fit_dict({**good, "schema_version": 2}))
    assert any("source should be dict" in s for s in I.validate_fit_dict({**good, "source": []}))


def test_unknown_controller_mode_is_rejected():
    with pytest.raises(ValueError):
        FirewalledController(I.ControllerConfig(mode="typo"), bel())


# ----------------------------------------------------------------------------------------------- constructor constants
def test_constructor_derives_gains_from_beliefs():
    c = ctl()
    assert c.Kt == pytest.approx(KT) and c.Kp_w == pytest.approx(KP_W) and c.Ki_w == pytest.approx(KI_W)
    assert c.dt_w == pytest.approx(10 * I.TAU_PLAIN) and c.psi_hat == PSI0 and c.psi_nom == PSI0 and c.i_lim_eff == 250.0


# ----------------------------------------------------------------------------------------------- one-step exact command
def test_unsaturated_naive_command_matches_hand_formula():
    c = ctl()
    m = mk(i_d_m=2.0, i_q_m=10.0, omega_m_m=100.0, omega_ref=100.5)
    u, info = c.command(m)
    iq_ref = KP_W * 0.5
    e = 10.0 - iq_ref
    w_el = 300.0
    psi_new = PSI0 + 1e-4 * (-2.0e-5 * w_el * e)               # the law updates psi_hat BEFORE the voltage is formed
    u_q = R_ * 10 + w_el * (LD * 2.0 + psi_new) - 3.0 * e
    u_d = R_ * 2.0 - w_el * LQ * 10.0 - 3.0 * 2.0
    assert info["iq_ref"] == pytest.approx(iq_ref) and info["e_q"] == pytest.approx(e)
    assert info["u_q_ctrl"] == pytest.approx(u_q) and info["u_d_ctrl"] == pytest.approx(u_d)
    assert np.allclose(u, [u_d, u_q]) and info["theta_adv"] == 0.0 and info["du_q_known"] == 0.0
    # psi_hat law: dpsi = -gamma w_el e
    assert info["psi_hat"] == pytest.approx(PSI0 + 1e-4 * (-2.0e-5 * 300.0 * e))


def test_psi_hat_stays_put_when_current_error_is_zero():
    c = ctl()
    iq = KP_W * 0.5
    _, info = c.command(mk(i_q_m=iq))
    assert info["e_q"] == pytest.approx(0.0, abs=1e-12) and info["psi_hat"] == pytest.approx(PSI0, abs=1e-15)


# ----------------------------------------------------------------------------------------------- d_gain_by_L
def test_d_gain_by_L_scales_d_feedback_by_Ld_over_Lq_only_in_the_d_axis():
    m = mk(i_d_m=5.0, i_q_m=0.0)
    off, on = ctl(d_gain_by_L=False), ctl(d_gain_by_L=True)
    _, a = off.command(m)
    _, b = on.command(m)
    assert a["u_d_ctrl"] == pytest.approx(R_ * 5.0 - 3.0 * 5.0)
    assert b["u_d_ctrl"] == pytest.approx(R_ * 5.0 - 3.0 * (LD / LQ) * 5.0)
    assert b["u_d_ctrl"] - a["u_d_ctrl"] == pytest.approx(3.0 * 5.0 * (1 - LD / LQ))
    assert a["u_q_ctrl"] == b["u_q_ctrl"]


# ----------------------------------------------------------------------------------------------- clamp_psi and modes
def _drive(c, n, **mkw):
    info = None
    for k in range(n):
        _, info = c.command(mk(t=k * 1e-4, **mkw))
    return info


def test_clamp_psi_projects_onto_prior_range_in_aw_mode_but_not_without_it():
    kw = dict(mode="aw", gamma=2e-3)
    up = dict(i_q_m=-200.0, omega_ref=150.0)           # e << 0 -> psi_hat grows
    free = _drive(ctl(**kw), 800, **up)
    clamped = _drive(ctl(**kw, clamp_psi=True), 800, **up)
    assert free["psi_hat"] > 0.10                       # without the clamp it leaves the prior
    assert clamped["psi_hat"] == 0.10                   # with the clamp it sits exactly on the upper bound
    dn = dict(i_q_m=300.0, omega_ref=0.0)              # e >> 0 -> psi_hat falls
    assert _drive(ctl(**kw), 800, **dn)["psi_hat"] < 0.03
    assert _drive(ctl(**kw, clamp_psi=True), 800, **dn)["psi_hat"] == 0.03


def test_robust_mode_always_clamps_without_clamp_psi_flag():
    info = _drive(ctl(mode="robust", gamma=2e-3), 800, i_q_m=-200.0, omega_ref=150.0)
    assert 0.03 <= info["psi_hat"] <= 0.10


def test_clamp_uses_the_beliefs_range_not_a_hardcoded_one():
    c = FirewalledController(I.ControllerConfig(mode="aw", clamp_psi=True, gamma=2e-3, delay_steps=0, angle_advance=False), bel(psi_range=(0.04, 0.05)))
    for k in range(800):
        _, info = c.command(mk(i_q_m=-200.0, omega_ref=150.0))
    assert info["psi_hat"] == 0.05


def test_robust_sigma_pulls_psi_hat_back_to_nominal_but_aw_and_naive_do_not():
    iq = KP_W * 0.5                                       # e = 0, so only the sigma term acts
    for mode, pulls in (("robust", True), ("aw", False), ("naive", False)):
        c = ctl(mode=mode)
        c.psi_hat = 0.05
        _, info = c.command(mk(i_q_m=iq))
        expect = 0.05 - 1e-4 * 2.0 * (0.05 - PSI0) if pulls else 0.05
        assert info["psi_hat"] == pytest.approx(expect, abs=1e-15), mode


# ----------------------------------------------------------------------------------------------- anti-windup
def test_aw_auxiliary_state_integrates_the_known_shortfall_and_naive_never_moves_z():
    sat = dict(i_q_m=0.0, omega_ref=150.0, vdc_m=300.0)   # iq_ref=250 -> huge u_q -> the controller's own clip bites
    aw, nv = ctl(mode="aw"), ctl(mode="naive")
    _, a1 = aw.command(mk(**sat))
    _, n1 = nv.command(mk(**sat))
    assert a1["du_q_known"] < 0 and a1["z"] == 0.0                     # z is updated from the PREVIOUS step's shortfall
    du1 = a1["du_q_known"]
    _, a2 = aw.command(mk(**sat))
    assert a2["z"] == pytest.approx(1e-4 * (-3.0 * 0.0 + du1) / LQ)
    assert a2["e_qa"] == pytest.approx(a2["e_q"] - a2["z"])
    _, n2 = nv.command(mk(**sat))
    assert n1["z"] == 0.0 and n2["z"] == 0.0 and n2["e_qa"] == n2["e_q"]


def test_aw_z_decays_toward_zero_when_nothing_is_clipped():
    c = ctl(mode="aw")
    c.z = 5.0
    _, info = c.command(mk(i_q_m=KP_W * 0.5))
    assert info["z"] == pytest.approx(5.0 + 1e-4 * (-3.0 * 5.0) / LQ)


def test_speed_loop_integrator_back_calculates_the_clip():
    c = ctl(i_lim=50.0)
    c.command(mk(omega_ref=110.0))                       # e_w = 10, unsat = Kp*10 = 363 > 50
    unsat = KP_W * 10.0
    assert c.iq_ref == 50.0 and c.i_ref_clip == pytest.approx(50.0 - unsat)
    assert c.int_w == pytest.approx(1e-3 * (KI_W * 10.0 + 5.0 * (50.0 - unsat)))


def test_i_lim_saturation_is_symmetric_and_reported():
    for sign in (1.0, -1.0):
        c = ctl(i_lim=40.0)
        _, info = c.command(mk(omega_ref=100.0 + sign * 5.0))
        assert info["iq_ref"] == sign * 40.0 and info["i_lim_eff"] == 40.0
        assert info["i_ref_clip"] == pytest.approx(sign * 40.0 - KP_W * sign * 5.0)


def test_speed_loop_runs_every_tenth_call_only():
    c = ctl()
    _, i0 = c.command(mk(omega_ref=100.5))
    first = i0["iq_ref"]
    for k in range(1, 10):                                # the request changes, the held iq_ref must not
        _, ik = c.command(mk(omega_ref=105.0))
        assert ik["iq_ref"] == first
    _, i10 = c.command(mk(omega_ref=105.0))
    assert i10["iq_ref"] != first and c.n == 11


# ----------------------------------------------------------------------------------------------- governor
def _lim(w_m, u_nom=I.U_CAP, i_lim=250.0, gf=0.75):
    w_el = max(abs(3 * w_m), 1.0)
    room = (u_nom / w_el) ** 2 - PSI ** 2
    return min(i_lim, gf * np.sqrt(max(room, 0.0)) / LQ)


def test_governor_limit_follows_the_ellipse_formula_and_falls_with_speed():
    vals = []
    for w in (50.0, 150.0, 190.0, 220.0):
        c = ctl(governor=True)
        _, info = c.command(mk(omega_m_m=w, omega_ref=w + 100.0))
        assert info["i_lim_eff"] == pytest.approx(_lim(w)), w
        vals.append(info["i_lim_eff"])
    assert vals[0] == 250.0 and vals == sorted(vals, reverse=True) and vals[-1] < vals[-2] < 250.0


def test_governor_off_keeps_i_lim_even_where_the_ellipse_would_bind():
    _, info = ctl(governor=False).command(mk(omega_m_m=220.0, omega_ref=320.0))
    assert info["i_lim_eff"] == 250.0 and info["iq_ref"] == 250.0


def test_governor_collapses_to_zero_when_back_emf_exceeds_the_ceiling():
    c = ctl(governor=True)
    _, info = c.command(mk(omega_m_m=1000.0, omega_ref=1100.0))   # w_el psi = 198 V > 173 V: no room for any current
    assert info["i_lim_eff"] == 0.0 and info["iq_ref"] == 0.0


def test_vdc_aware_governor_uses_the_measured_bus_not_the_nominal_ceiling():
    m = mk(omega_m_m=150.0, omega_ref=250.0, vdc_m=200.0)
    _, plain = ctl(governor=True).command(m)
    _, aware = ctl(governor=True, vdc_aware=True).command(m)
    assert plain["i_lim_eff"] == pytest.approx(_lim(150.0))
    assert aware["i_lim_eff"] == pytest.approx(_lim(150.0, u_nom=200.0 / np.sqrt(3.0)))
    assert aware["i_lim_eff"] < plain["i_lim_eff"]
    # with a healthy 300 V bus both are identical to 1e-3 (vdc/sqrt3 = 173.205 = U_CAP)
    h = mk(omega_m_m=150.0, omega_ref=250.0, vdc_m=300.0)
    assert ctl(governor=True, vdc_aware=True).command(h)[1]["i_lim_eff"] == pytest.approx(ctl(governor=True).command(h)[1]["i_lim_eff"], rel=1e-5)


def test_bms_aware_governor_limits_iq_through_the_power_balance():
    m = mk(omega_m_m=100.0, omega_ref=200.0, vdc_m=300.0, bms=I.BMSMsg(10.0, 61.0, 300.0, 0.0))
    _, info = ctl(governor=True, bms_aware=True).command(m)
    assert info["i_lim_eff"] == pytest.approx(300.0 * 10.0 * 0.9 / (KT * 100.0))      # ~90.9 A
    assert info["iq_ref"] == pytest.approx(info["i_lim_eff"])
    _, loose = ctl(governor=True, bms_aware=True).command(mk(omega_m_m=100.0, omega_ref=200.0, bms=I.BMSMsg(1e6, 61.0, 300.0, 0.0)))
    assert loose["i_lim_eff"] == 250.0


def test_bms_limit_is_ignored_unless_the_governor_is_on():
    m = mk(omega_m_m=100.0, omega_ref=200.0, bms=I.BMSMsg(10.0, 61.0, 300.0, 0.0))
    _, info = ctl(governor=False, bms_aware=True).command(m)
    assert info["i_lim_eff"] == 250.0


def test_bms_aware_limit_at_zero_speed_does_not_divide_by_zero():
    _, info = ctl(governor=True, bms_aware=True).command(mk(omega_m_m=0.0, omega_ref=5.0, bms=I.BMSMsg(10.0, 61.0, 300.0, 0.0)))
    assert np.isfinite(info["i_lim_eff"]) and info["i_lim_eff"] == pytest.approx(min(250.0, 300 * 10 * 0.9 / (KT * 1.0), 0.75 * np.sqrt(max((I.U_CAP / 1.0) ** 2 - PSI ** 2, 0)) / LQ))


# ----------------------------------------------------------------------------------------------- ceiling and clip
def test_ceiling_hat_is_min_of_bus_and_cap_and_oracle():
    c = ctl()
    assert c._ceiling_hat(mk(vdc_m=300.0)) == pytest.approx(I.U_CAP)
    assert c._ceiling_hat(mk(vdc_m=600.0)) == pytest.approx(I.U_CAP)               # never above the hexagon-inscribed disc
    assert c._ceiling_hat(mk(vdc_m=100.0)) == pytest.approx(100.0 / np.sqrt(3.0))
    assert c._ceiling_hat(mk(vdc_m=300.0, ceiling_telemetry=40.0)) == 40.0           # labelled oracle event
    assert c._ceiling_hat(mk(vdc_m=100.0, ceiling_telemetry=170.0)) == pytest.approx(100.0 / np.sqrt(3.0))


def test_own_clip_is_radial_and_reports_the_q_shortfall():
    c = ctl()
    u, info = c.command(mk(i_q_m=0.0, omega_ref=150.0, vdc_m=100.0))
    ceil = 100.0 / np.sqrt(3.0)
    raw = np.array([info["u_d_ctrl"], info["u_q_ctrl"]])
    assert np.hypot(*raw) > ceil
    assert np.hypot(*u) == pytest.approx(ceil)
    assert np.allclose(u, raw * ceil / np.hypot(*raw))                               # direction preserved
    assert info["du_q_known"] == pytest.approx(u[1] - raw[1]) and info["du_q_known"] < 0
    assert info["u_d_clip"] == pytest.approx(u[0]) and info["u_q_clip"] == pytest.approx(u[1])


def test_no_clip_leaves_command_untouched_and_du_zero():
    c = ctl()
    u, info = c.command(mk())
    assert info["du_q_known"] == 0.0 and np.allclose(u, [info["u_d_ctrl"], info["u_q_ctrl"]])


# ----------------------------------------------------------------------------------------------- angle advance
def test_angle_advance_rotates_by_delay_times_electrical_step_and_keeps_magnitude():
    base = ctl()
    adv = FirewalledController(I.ControllerConfig(delay_steps=2, angle_advance=True), bel())
    m = mk(i_d_m=3.0)
    u0, _ = base.command(m)
    u1, info = adv.command(m)
    theta = 2 * 300.0 * I.TAU_PLAIN
    assert info["theta_adv"] == pytest.approx(theta)
    assert np.allclose(u1, I.rot(theta) @ u0) and np.hypot(*u1) == pytest.approx(np.hypot(*u0))


def test_angle_advance_is_zero_when_disabled_or_delay_is_zero():
    assert ctl(delay_steps=2, angle_advance=False).command(mk())[1]["theta_adv"] == 0.0
    assert ctl(delay_steps=0, angle_advance=True).command(mk())[1]["theta_adv"] == 0.0


def test_advance_sign_follows_speed_sign():
    c = FirewalledController(I.ControllerConfig(delay_steps=1, angle_advance=True), bel())
    _, info = c.command(mk(omega_m_m=-100.0, omega_ref=-99.5))
    assert info["theta_adv"] == pytest.approx(-300.0 * I.TAU_PLAIN)


# ----------------------------------------------------------------------------------------------- misc
def test_controller_is_deterministic_and_state_isolated_between_instances():
    seq = [mk(t=k * 1e-4, i_q_m=10 + 0.1 * k, omega_ref=120.0) for k in range(60)]
    a, b = ctl(mode="aw", governor=True), ctl(mode="aw", governor=True)
    ua = np.array([a.command(m)[0] for m in seq])
    ub = np.array([b.command(m)[0] for m in seq])
    assert np.array_equal(ua, ub) and a is not b and a.last is not b.last


def test_last_dict_has_the_documented_keys():
    _, info = ctl().command(mk())
    assert set(info) == {"iq_ref", "e_q", "e_qa", "z", "psi_hat", "omega_ref", "i_ref_clip", "i_lim_eff", "u_d_ctrl", "u_q_ctrl", "u_d_clip",
                         "u_q_clip", "ceiling_hat", "du_q_known", "theta_adv"}


def test_zero_speed_is_handled_without_nan():
    _, info = ctl(governor=True, vdc_aware=True).command(mk(omega_m_m=0.0, omega_ref=0.0, i_q_m=0.0, vdc_m=300.0))
    assert all(np.isfinite(v) for v in info.values())
