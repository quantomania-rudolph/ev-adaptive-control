"""Unit tests U12-U17 for inverter.py (U17 also exercises sensors.py) plus mutation tests (test_MUT_*).

check_Uxx(...) functions raise AssertionError; test_U* run them on the real model, test_MUT_* run them on a broken variant and
require the assertion to FAIL.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import replace

import numpy as np
import pytest

from interfaces import InverterConfig, SensorConfig, BMSMsg, park, rot, TAU_PLAIN, TAU_CYCLE
from inverter import InverterModel
from sensors import SensorSuite

BMS = BMSMsg(100.0, 60.0, 300.0, 0.0)


def mk(cfg=None, cls=InverterModel, **kw):
    return cls(replace(cfg or InverterConfig(), **kw), np.random.default_rng(0))


def must_fail(check, *a, **kw):
    with pytest.raises(AssertionError):
        check(*a, **kw)


def abc_of(i_d, i_q, eps):
    """dq -> abc (amplitude invariant, same convention as interfaces.park / GEM)."""
    al, be = math.cos(eps) * i_d - math.sin(eps) * i_q, math.sin(eps) * i_d + math.cos(eps) * i_q
    return (al, -0.5 * al + math.sqrt(3) / 2 * be, -0.5 * al - math.sqrt(3) / 2 * be)


# ============================================================================== U12
def check_U12(factory=mk):
    for d in (0, 1, 2):
        inv = factory(delay_steps=d)
        k0 = 5
        for k in range(12):
            u = np.array([7.0, -3.0]) if k == k0 else np.zeros(2)
            out = inv.pre_clip(u, 0.0)
            want = np.array([7.0, -3.0]) if k == k0 + d else np.zeros(2)
            assert np.array_equal(out, want), (d, k, out)
            # exact decomposition
            assert np.allclose(u + inv.last_delay_dv + inv.last_rot_dv, out, atol=1e-12, rtol=0)
            assert np.allclose(inv.last_rot_dv, 0.0, atol=1e-12)
        # initial content is zeros
        inv.reset()
        assert np.array_equal(inv.pre_clip(np.array([1.0, 2.0]), 0.0), np.array([1.0, 2.0]) if d == 0 else np.zeros(2))


def test_U12_delay_impulse_appears_at_k_plus_d():
    check_U12()


def test_MUT_delay_bypassed_fails_U12():
    class M(InverterModel):
        def __init__(self, cfg, rng):
            super().__init__(replace(cfg, delay_steps=0), rng)
    must_fail(check_U12, lambda **kw: mk(cls=M, **kw))


# ============================================================================== U13
def _rotating_mean_error(inv, I, phi, vdc, N=600, meas_equals_true=True):
    """Mean over exactly ONE electrical period (N uniformly spaced samples) of the dq error for i_abc = I cos(th + phi - 2 pi x/3)
    with epsilon_true = th, i.e. a constant dq current (I cos phi, I sin phi).  A full period of uniform samples averages the
    6th-harmonic ripple (and all other harmonics below N) to zero, so the mean is the DC (fundamental) part.  Also returns the
    peak-to-peak ripple so the test can show the exclusion is not vacuous."""
    errs = []
    for k in range(N):
        th = 2 * math.pi * k / N
        i = [I * math.cos(th + phi - 2 * math.pi * x / 3) for x in range(3)]
        errs.append(inv.post_clip_error(i, i, vdc, th))
    errs = np.array(errs)
    return errs.mean(axis=0), np.ptp(errs, axis=0)


def check_U13(factory=mk):
    vdc, I = 300.0, 100.0
    # (a) sign follows the current: the dq error opposes the current vector (compensation 0 and default 0.7 - both leave a net opposing error)
    for comp in (0.0, 0.7):
        inv = factory(compensation=comp)
        for iv in ((30.0, -15.0, -15.0), (-30.0, 15.0, 15.0), (0.0, 40.0, -40.0)):
            e = inv.post_clip_error(iv, iv, vdc, 0.0)
            al, be = park(*iv, 0.0)
            cross = e[0] * be - e[1] * al                     # 2-D cross product: error exactly antiparallel to the current vector
            assert np.dot(e, [al, be]) < 0.0 and abs(cross) < 1e-9 * np.linalg.norm(e) * np.hypot(al, be), (comp, iv, e)
        # a current along +d gives a negative d error, along -d a positive one
        e = inv.post_clip_error(abc_of(50.0, 0.0, 0.0), abc_of(50.0, 0.0, 0.0), vdc, 0.0)
        assert e[0] < 0
        e = inv.post_clip_error(abc_of(-50.0, 0.0, 0.0), abc_of(-50.0, 0.0, 0.0), vdc, 0.0)
        assert e[0] > 0
    # (b) DC part vs -(1-c) (4/pi)(Td/Tpwm Vdc + Vdrop) along the current direction, relative to that magnitude, within 1 %
    for comp in (0.0, 0.7):
        for phi in (0.0, 0.3, 2.0, -1.2):
            inv = factory(compensation=comp)
            mean, ptp = _rotating_mean_error(inv, I, phi, vdc)
            dc = (1 - comp) * 4.0 / math.pi * (inv.cfg.dead_time_s / inv.cfg.tau * vdc + inv.cfg.device_drop_v)
            want = -dc * np.array([math.cos(phi), math.sin(phi)])
            assert np.linalg.norm(mean - want) < 0.01 * dc, (comp, phi, mean, want)
            assert ptp.max() > 0.05 * dc, "the 6th-harmonic ripple should be visible (so excluding it is meaningful)"
    # (c) zero when Td = 0 and drop = 0 (exactly), and when disabled
    z = factory(dead_time_s=0.0, device_drop_v=0.0)
    assert np.array_equal(z.post_clip_error([50.0, -20.0, -30.0], [49.0, -20.0, -29.0], vdc, 1.0), np.zeros(2))
    z = factory(enabled=False)
    assert np.array_equal(z.post_clip_error([50.0, -20.0, -30.0], [49.0, -20.0, -29.0], vdc, 1.0), np.zeros(2))
    # (d) scales with tau: dead-time term (drop off, uncompensated) is 2x at tau = 2e-4 -- no wait, Td/Tpwm halves: error ~ 1/tau
    m1, _ = _rotating_mean_error(factory(tau=TAU_PLAIN, device_drop_v=0.0, compensation=0.0), I, 0.3, vdc)
    m2, _ = _rotating_mean_error(factory(tau=TAU_CYCLE, device_drop_v=0.0, compensation=0.0), I, 0.3, vdc)
    assert abs(np.linalg.norm(m1) / np.linalg.norm(m2) - TAU_CYCLE / TAU_PLAIN) < 0.01 * 2
    assert abs(np.linalg.norm(m1) - 4 / math.pi * 2e-6 / TAU_PLAIN * vdc) < 0.01 * np.linalg.norm(m1)
    assert abs(np.linalg.norm(m2) - 4 / math.pi * 2e-6 / TAU_CYCLE * vdc) < 0.01 * np.linalg.norm(m2)
    # drop-only term does not depend on tau
    a, _ = _rotating_mean_error(factory(tau=TAU_PLAIN, dead_time_s=0.0, compensation=0.0), I, 0.3, vdc)
    b, _ = _rotating_mean_error(factory(tau=TAU_CYCLE, dead_time_s=0.0, compensation=0.0), I, 0.3, vdc)
    assert np.allclose(a, b, atol=1e-12)


def test_U13_dead_time_sign_dc_part_zero_and_tau_scaling():
    check_U13()


def test_MUT_dead_time_sign_flipped_fails_U13():
    class M(InverterModel):
        def post_clip_error(self, *a, **k):
            return -super().post_clip_error(*a, **k)
    must_fail(check_U13, lambda **kw: mk(cls=M, **kw))


# ============================================================================== U14
def check_U14(factory=mk):
    vdc = 300.0
    for kw in ({}, {"compensation": 0.0}):
        inv = factory(**kw)
        amp = inv.dc_fundamental(vdc)                         # (4/pi) K, the amplitude of the error
        band = inv.cfg.smooth_band_a
        di = 0.01
        sweep = np.arange(-10.0, 10.0 + di / 2, di)
        for meas_shift in (0.0, 0.7, -2.5):                   # matched, and a measured current that is wrong near zero
            errs = []
            for x in sweep:
                i = (x, -0.5 * x, -0.5 * x)
                im = (x + meas_shift, -0.5 * x + meas_shift, -0.5 * x - meas_shift)
                errs.append(inv.post_clip_error(i, im, vdc, 0.4))
            errs = np.array(errs)
            steps = np.linalg.norm(np.diff(errs, axis=0), axis=1)
            assert np.linalg.norm(errs, axis=1).max() <= inv.max_error_dq(vdc) * (1 + 1e-12)       # bounded
            assert steps.max() < 0.10 * amp, (kw, meas_shift, steps.max(), amp)                    # continuous
            # Lipschitz bound of the smoothed model: per phase K(1+c) di/band, dq image at most 4/3 of the phase value
            assert steps.max() <= 4.0 / 3.0 * (1 + inv.cfg.compensation) * inv.distortion_amplitude(vdc) * di / band * 1.0001 + 1e-12


def test_U14_bounded_and_continuous_through_zero_current():
    check_U14()


def test_MUT_hard_sign_fails_U14():
    must_fail(check_U14, lambda **kw: mk(smooth_band_a=0.0, **kw))


# ============================================================================== U15
def check_U15(factory=mk):
    rng = np.random.default_rng(3)
    inv = factory(delay_steps=0)
    for _ in range(2000):
        u = rng.uniform(-200, 200, 2)
        dl = rng.uniform(-math.pi, math.pi)
        out = inv.pre_clip(u, dl)
        z = complex(*u) * cmath.exp(1j * dl)                    # independent: complex multiplication = R(delta) u
        assert abs(out[0] - z.real) < 1e-9 and abs(out[1] - z.imag) < 1e-9
        assert np.allclose(out, rot(dl) @ u, atol=1e-9, rtol=0)  # the shared rotation matrix
        assert abs(np.linalg.norm(out) - np.linalg.norm(u)) < 1e-9
        assert np.allclose(u + inv.last_delay_dv + inv.last_rot_dv, out, atol=1e-9, rtol=0)
    # positive delta rotates a +d command towards +q
    assert factory(delay_steps=0).pre_clip(np.array([10.0, 0.0]), 0.1)[1] > 0
    # with a delay the rotation uses the CURRENT delta on the delayed command
    inv = factory(delay_steps=1)
    inv.pre_clip(np.array([5.0, 1.0]), 0.0)
    out = inv.pre_clip(np.array([99.0, 99.0]), 0.3)
    assert np.allclose(out, rot(0.3) @ np.array([5.0, 1.0]), atol=1e-12)


def test_U15_rotation_u_true_equals_R_delta_u_cmd():
    check_U15()


def test_U15b_park_of_voltage_matches_gem_dq_to_abc():
    from plant import GEMPlant
    ps = GEMPlant().ps
    rng = np.random.default_rng(9)
    for _ in range(20):
        u = rng.uniform(-100, 100, 2)
        e = rng.uniform(-math.pi, math.pi)
        v = ps.dq_to_abc_space(u, e)
        assert np.allclose(park(*v, e), u, atol=1e-9)
        assert np.allclose(v, abc_of(u[0], u[1], e), atol=1e-9)


def test_MUT_rotation_sign_flipped_fails_U15():
    class M(InverterModel):
        def pre_clip(self, u, delta):
            return super().pre_clip(u, -delta)
    must_fail(check_U15, lambda **kw: mk(cls=M, **kw))


# ============================================================================== U16
def _pipeline_correct(inv, u_cmd, delta, lim, i_true, i_meas, vdc, eps):
    u_pre = inv.pre_clip(u_cmd, delta)
    m = float(np.hypot(*u_pre))
    u_app = u_pre if m <= lim else u_pre * (lim / m)       # stand-in for the plant's disc projection
    return u_app, u_app + inv.post_clip_error(i_true, i_meas, vdc, eps)


def check_U16(pipeline=_pipeline_correct, factory=mk, n=100_000):
    rng = np.random.default_rng(16)
    inv = factory(delay_steps=1)
    exceeded = 0
    worst_err = 0.0
    for k in range(n):
        lim = rng.choice([173.205, rng.uniform(1.0, 173.2), 25.0])
        kind = k % 5
        if kind == 0:
            u = rng.normal(0, 150, 2)
        elif kind == 1:
            u = rng.normal(0, 1, 2) * 10 ** rng.uniform(-3, 9)          # tiny to huge
        elif kind == 2:
            ang = rng.uniform(0, 2 * math.pi); u = lim * np.array([math.cos(ang), math.sin(ang)]) * (1 + rng.choice([-1e-12, 0, 1e-12, 1e-9]))
        elif kind == 3:
            u = np.zeros(2)
        else:
            u = rng.choice([-1, 1], 2) * lim * rng.choice([1.0, 0.7071067811865476, 2.0])
        delta = rng.uniform(-math.pi, math.pi)
        vdc = rng.uniform(0.0, 500.0)
        eps = rng.uniform(-math.pi, math.pi)
        it = rng.uniform(-500, 500, 3) * rng.choice([1.0, 1e-3, 0.0])
        im = it * rng.choice([1.0, -1.0, 0.0]) + rng.normal(0, 3, 3)
        u_app, fin = pipeline(inv, u, delta, lim, it, im, vdc, eps)
        assert np.hypot(*u_app) <= lim * (1 + 1e-12), "clipped voltage above the ceiling"
        err = fin - u_app
        bound = inv.max_error_dq(vdc)
        assert np.hypot(*err) <= bound * (1 + 1e-12) + 1e-12, (np.hypot(*err), bound)
        worst_err = max(worst_err, np.hypot(*err) / bound if bound > 0 else 0.0)
        exceeded += np.hypot(*fin) > lim * (1 + 1e-9)
    # the error is added AFTER the clip, so the final voltage can legitimately exceed the ceiling (by at most the bound)
    assert exceeded > 0, "final voltage never exceeds the ceiling: the error is not being added after the clip"
    # exact bound is tight (adversarial: measured current has the opposite sign of the true one, large magnitude)
    inv2 = factory()
    for vdc in (0.0, 120.0, 300.0, 500.0):
        iv = (60.0, -30.0, -30.0)
        e = inv2.post_clip_error(iv, tuple(-x for x in iv), vdc, 0.7)
        b = inv2.max_error_dq(vdc)
        assert np.hypot(*e) <= b * (1 + 1e-12) + 1e-12
        if b > 0:
            assert np.hypot(*e) >= 0.999 * b, "bound (4/3)(1+c)K should be attained"
        e = inv2.post_clip_error(iv, iv, vdc, 0.7)             # matched measurement: (4/3)(1-c)K
        b = 4 / 3 * abs(1 - inv2.cfg.compensation) * inv2.distortion_amplitude(vdc)
        assert np.hypot(*e) <= b * (1 + 1e-12) + 1e-12
        if b > 0:
            assert np.hypot(*e) >= 0.999 * b
    # the contract's (4/pi)K is the DC part, which sits BELOW the instantaneous peak (4/3)K (6th-harmonic ripple)
    assert inv2.dc_fundamental(300.0) < 4 / 3 * inv2.distortion_amplitude(300.0)


def test_U16_order_of_operations_ceiling_and_dead_time_bound():
    check_U16()


def test_MUT_error_before_clip_fails_U16():
    def pipe(inv, u_cmd, delta, lim, it, im, vdc, eps):    # error injected BEFORE the clip (wrong order)
        u_pre = inv.pre_clip(u_cmd, delta) + inv.post_clip_error(it, im, vdc, eps)
        m = float(np.hypot(*u_pre))
        u_app = u_pre if m <= lim else u_pre * (lim / m)
        return u_app - inv.post_clip_error(it, im, vdc, eps) * 0.0, u_app
    must_fail(check_U16, pipe, mk, 3000)


def test_MUT_error_twice_too_large_fails_U16():
    class M(InverterModel):
        def post_clip_error(self, *a, **k):
            return 2.0 * super().post_clip_error(*a, **k)
    must_fail(check_U16, _pipeline_correct, lambda **kw: mk(cls=M, **kw), 3000)


# ============================================================================== U17
def check_U17(two_sensor=False):
    cfg = SensorConfig(i_gain_err=0.0, i_offset_a=0.0, i_noise_lsb=0.0, theta_offset_rad=0.0, theta_noise_rad=0.0,
                       theta_ripple_rad=0.0, vdc_gain_err=0.0, vdc_noise_v=0.0, two_sensor=two_sensor)
    s = SensorSuite(cfg, np.random.default_rng(1))
    s.gain_i = np.array([0.03, -0.02, 0.01])                  # unequal per-phase gain errors
    worst_i0 = 0.0
    for k in range(200):
        eps = 0.05 * k
        i = abc_of(20.0 * math.cos(0.1 * k), 80.0, eps)       # balanced true currents (zero sequence exactly 0)
        assert abs(sum(i)) < 1e-12
        d, q = park(*i, eps)
        tr = dict(t=0.0, omega_m=0.0, omega_el=0.0, i_a=i[0], i_b=i[1], i_c=i[2], i_d=d, i_q=q, epsilon_el=eps + 0.05,
                  epsilon_prev=eps, V_dc=300.0)
        m = s.measure(tr, 0.0, BMS)
        im = (m.i_a_m, m.i_b_m, m.i_c_m)
        i0 = sum(im) / 3.0
        worst_i0 = max(worst_i0, abs(i0))
        # remove the zero-sequence part by hand and take the full Clarke: identical to the measured dq (the sensor ignores i0)
        d_ref, q_ref = park(im[0] - i0, im[1] - i0, im[2] - i0, eps)
        assert abs(m.i_d_m - d_ref) < 1e-9 and abs(m.i_q_m - q_ref) < 1e-9
    assert worst_i0 > 0.5, "the unequal gains must create a real zero-sequence current"
    # the gain errors still make a (positive-/negative-sequence) dq error, so this is not a trivial pass
    assert abs(m.i_d_m - d) + abs(m.i_q_m - q) > 0.1


def test_U17_unequal_gains_zero_sequence_ignored_by_full_clarke():
    check_U17()
    # numerically, for the record: the zero-sequence part is non-trivial and the two-sensor form does NOT ignore it
    i = (101.0, -50.0, -50.0)
    i0 = sum(i) / 3
    assert abs(i0 - 0.3333333333) < 1e-9
    assert np.allclose(park(*i, 0.7), park(i[0] - i0, i[1] - i0, i[2] - i0, 0.7), atol=1e-12)
    assert not np.allclose(park(*i, 0.7, two_sensor=True), park(i[0] - i0, i[1] - i0, i[2] - i0, 0.7, two_sensor=True), atol=1e-6)


def test_MUT_two_sensor_fails_U17():
    must_fail(check_U17, True)
