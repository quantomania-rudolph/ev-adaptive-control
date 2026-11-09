"""Unit tests U1-U11 for sensors.py plus mutation tests (test_MUT_*).

Each U check is a plain function check_Uxx(...) that raises AssertionError; the test_U* function calls it on the real
SensorSuite and the matching test_MUT_* calls it on a deliberately broken subclass and requires the assertion to FAIL.
"""
from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import pytest

from interfaces import BMSMsg, SensorConfig, park, TAU_PLAIN, TAU_CYCLE
from sensors import SensorSuite, wrap_pi

BMS = BMSMsg(100.0, 60.0, 300.0, 0.01)
ZERO = dict(i_gain_err=0.0, i_offset_a=0.0, i_noise_lsb=0.0, theta_offset_rad=0.0, theta_noise_rad=0.0,
            theta_ripple_rad=0.0, vdc_gain_err=0.0, vdc_noise_v=0.0)


def zero_cfg(**kw):
    return SensorConfig(**{**ZERO, **kw})


def mk(cfg, seed=1, cls=SensorSuite):
    return cls(cfg, np.random.default_rng(seed))


def synth(k, tau=TAU_PLAIN, fe=100.0, amp=50.0, phi=0.4, vdc=300.0, omega_m=None):
    """Balanced rotating three-phase truth at step k; dq = Park of abc at epsilon_prev (the verified plant frame)."""
    w = 2 * math.pi * fe
    eps, epsp = wrap_pi(w * k * tau), wrap_pi(w * (k - 1) * tau)
    th = w * k * tau + phi
    ia, ib, ic = (amp * math.cos(th - 2 * math.pi * x / 3) for x in range(3))
    d, q = park(ia, ib, ic, epsp)
    return dict(t=k * tau, omega_m=w / 3 if omega_m is None else omega_m, omega_el=w, i_a=ia, i_b=ib, i_c=ic, i_d=d, i_q=q,
                epsilon_el=eps, epsilon_prev=epsp, V_dc=vdc)


def flat(eps=0.0, epsp=None, ia=0.0, ib=0.0, ic=0.0, t=0.0, vdc=300.0, om=0.0):
    d, q = park(ia, ib, ic, eps if epsp is None else epsp)
    return dict(t=t, omega_m=om, omega_el=3 * om, i_a=ia, i_b=ib, i_c=ic, i_d=d, i_q=q, epsilon_el=eps,
                epsilon_prev=eps if epsp is None else epsp, V_dc=vdc)


def mutant(post):
    """Subclass of SensorSuite whose output is altered by post(self, m, truth, omega_ref)."""
    class M(SensorSuite):
        def measure(self, truth, omega_ref, bms, ceiling_telemetry=None):
            m = super().measure(truth, omega_ref, bms, ceiling_telemetry)
            return post(self, m, truth, omega_ref)
    return M


def must_fail(check, *a, **kw):
    with pytest.raises(AssertionError):
        check(*a, **kw)


# ============================================================================== U1
def check_U1(cls=SensorSuite):
    def stream(seed):
        s = mk(SensorConfig(), seed, cls)
        out = [(s.gain_i.copy(), s.offset_i.copy(), s.theta_offset, s.vdc_gain)]
        for k in range(300):
            m = s.measure(synth(k), 10.0, BMS)
            out.append((m.i_a_m, m.i_b_m, m.i_c_m, m.i_d_m, m.i_q_m, m.theta_m, m.omega_m_m, m.vdc_m))
        return out

    a, b, c = stream(7), stream(7), stream(8)
    flat_ = lambda st: np.concatenate([np.ravel(np.asarray(x, dtype=float)) if not isinstance(x, tuple) else
                                       np.concatenate([np.ravel(np.asarray(y, dtype=float)) for y in x]) for x in st])
    fa, fb, fc = flat_(a), flat_(b), flat_(c)
    assert np.array_equal(fa, fb), "same seed must give a bit-identical stream"
    assert not np.array_equal(fa, fc), "different seed must differ"
    assert np.max(np.abs(fa - fc)) > 1e-3


def test_U1_same_seed_identical_different_seed_differs():
    check_U1()


def test_U1b_does_not_touch_global_rng():
    np.random.seed(123)
    ref = np.random.random()
    np.random.seed(123)
    s = mk(SensorConfig(), 3)
    for k in range(50):
        s.measure(synth(k), 1.0, BMS)
    assert np.random.random() == ref


def test_MUT_seed_ignored_fails_U1():
    class M(SensorSuite):
        def __init__(self, cfg, rng):
            super().__init__(cfg, np.random.default_rng(0))
    must_fail(check_U1, M)


# ============================================================================== U2
@functools.lru_cache(maxsize=1)
def plant_rollout():
    from plant import GEMPlant
    pl = GEMPlant()
    eps_prev = pl.state()["epsilon_el"]
    out = []
    for k in range(500):
        pl.apply_voltage(np.array([-0.3, 2.5 + 0.8 * math.sin(k / 25.0)]))
        s = pl.state()
        s["epsilon_prev"] = eps_prev
        eps_prev = s["epsilon_el"]
        out.append(s)
    return out


def check_U2(cls=SensorSuite):
    tr = plant_rollout()
    assert max(abs(s["i_q"]) for s in tr) > 20.0, "rollout should carry real current"
    # (a) identity mode: every channel equals truth to 1e-12
    ident = mk(SensorConfig.identity_mode(), 1, cls)
    for k, s in enumerate(tr):
        m = ident.measure(s, 3.0, BMS, None)
        for got, want in ((m.i_a_m, s["i_a"]), (m.i_b_m, s["i_b"]), (m.i_c_m, s["i_c"]), (m.i_d_m, s["i_d"]),
                          (m.i_q_m, s["i_q"]), (m.theta_m, s["epsilon_el"]), (m.omega_m_m, s["omega_m"]),
                          (m.vdc_m, s["V_dc"]), (m.omega_ref, 3.0), (m.t, s["t"])):
            assert abs(got - want) <= 1e-12
        assert m.bms is BMS and m.ceiling_telemetry is None
    # (b) identity False, every error zero (quantization stays ON: adc_bits cannot be removed through existing fields)
    z = mk(zero_cfg(), 1, cls)
    lsb = z.LSB_i
    for k, s in enumerate(tr):
        m = z.measure(s, 3.0, BMS)
        if k == 0:
            continue
        i_m = (m.i_a_m, m.i_b_m, m.i_c_m)
        # measured dq is exactly the shared Park of the measured abc at epsilon_prev (frame check, no angle error)
        dd, qq = park(*i_m, s["epsilon_prev"])
        assert abs(m.i_d_m - dd) < 1e-12 and abs(m.i_q_m - qq) < 1e-12
        # that frame reproduces the plant dq of the TRUE currents (lead's verified fact)
        d0, q0 = park(s["i_a"], s["i_b"], s["i_c"], s["epsilon_prev"])
        assert abs(d0 - s["i_d"]) < 1e-9 and abs(q0 - s["i_q"]) < 1e-9
        # only quantization separates measured and true: abc error <= LSB/2 each, Clarke gain bound 1 => |dq error| <= LSB
        assert max(abs(a - b) for a, b in zip(i_m, (s["i_a"], s["i_b"], s["i_c"]))) <= lsb / 2 + 1e-12
        assert math.hypot(m.i_d_m - s["i_d"], m.i_q_m - s["i_q"]) <= lsb + 1e-9
        assert m.theta_m == s["epsilon_el"]
        assert abs(m.vdc_m - s["V_dc"]) <= z.LSB_v / 2 + 1e-9
    # (c) with the quantizer exactly satisfied (truth on the LSB grid) the zero-error dq equals the plant dq to 1e-9
    rng = np.random.default_rng(5)
    z = mk(zero_cfg(), 1, cls)
    for k in range(200):
        a, b, c = (rng.integers(-1500, 1500, 3) * lsb)
        ep, e = rng.uniform(-math.pi, math.pi, 2)
        d, q = park(a, b, c, ep)
        m = z.measure(dict(t=k * 1e-4, omega_m=0.0, omega_el=0.0, i_a=a, i_b=b, i_c=c, i_d=d, i_q=q, epsilon_el=e,
                           epsilon_prev=ep, V_dc=300.0), 0.0, BMS)
        assert abs(m.i_d_m - d) < 1e-9 and abs(m.i_q_m - q) < 1e-9


def test_U2_identity_and_zero_error_modes_match_truth():
    check_U2()


def test_MUT_wrong_frame_fails_U2():
    def post(self, m, tr, ref):          # builds dq with the CURRENT angle instead of epsilon_prev
        d, q = park(m.i_a_m, m.i_b_m, m.i_c_m, tr["epsilon_el"])
        return replace(m, i_d_m=d, i_q_m=q)
    must_fail(check_U2, mutant(post))


# ============================================================================== U3
def check_U3(cls=SensorSuite):
    s = mk(zero_cfg(), 2, cls)
    lsb = s.LSB_i
    assert abs(lsb - 0.1953125) < 1e-15
    rng = np.random.default_rng(11)
    for k in range(3000):
        ia, ib, ic = rng.uniform(-390, 390, 3)
        m = s.measure(flat(ia=ia, ib=ib, ic=ic), 0.0, BMS)
        for got, true in ((m.i_a_m, ia), (m.i_b_m, ib), (m.i_c_m, ic)):
            assert abs(got / lsb - round(got / lsb)) * lsb <= 1e-9, "not a multiple of LSB"
            assert abs(got - true) <= lsb / 2 + 1e-9, "quantization error above LSB/2"


def test_U3_quantization_multiples_of_lsb_and_half_lsb_error():
    check_U3()


def test_MUT_no_quantization_fails_U3():
    def post(self, m, tr, ref):
        return replace(m, i_a_m=tr["i_a"], i_b_m=tr["i_b"], i_c_m=tr["i_c"])
    must_fail(check_U3, mutant(post))


# ============================================================================== U4
NOMINAL = SensorConfig()


@functools.lru_cache(maxsize=4)
def _noise_run(factory_key):
    factory = {"real": SensorSuite, "x10": None}[factory_key]
    return None


def check_U4(cls=SensorSuite, n=100_000):
    """Noise isolated: gain, offset, angle offset, ripple and vdc gain zero; offsets then set explicitly as known biases."""
    cfg = replace(NOMINAL, i_gain_err=0.0, i_offset_a=0.0, theta_offset_rad=0.0, theta_ripple_rad=0.0, vdc_gain_err=0.0)
    s = mk(cfg, 4, cls)
    bias = np.array([1.234, -0.7, 2.5])
    s.offset_i = bias.copy()
    s.theta_offset = 0.01
    s.vdc_gain = 0.003
    lsb = s.LSB_i
    cur = np.empty((n, 3)); ang = np.empty(n); vd = np.empty(n)
    base = flat(eps=0.3, epsp=0.3)
    for k in range(n):
        m = s.measure(base, 0.0, BMS)
        cur[k] = (m.i_a_m, m.i_b_m, m.i_c_m); ang[k] = m.theta_m - 0.3; vd[k] = m.vdc_m
    # currents: mean within 4 sigma/sqrt(N) of the bias (spec sigma), std vs sqrt(sigma^2 + LSB^2/12) (Sheppard; exact to 1e-8
    # relative for sigma >= 1 LSB), which is within 2 % of the bare sigma only when the quantizer is included
    sig = cfg.i_noise_lsb * lsb
    for x in range(3):
        assert abs(cur[:, x].mean() - bias[x]) < 4 * sig / math.sqrt(n)
        assert abs(cur[:, x].std() / math.sqrt(sig ** 2 + lsb ** 2 / 12) - 1) < 0.02
    # angle: mean = offset, std = spec
    assert abs(ang.mean() - 0.01) < 4 * cfg.theta_noise_rad / math.sqrt(n)
    assert abs(ang.std() / cfg.theta_noise_rad - 1) < 0.02
    # vdc: filtered white noise, var_out = var_in * alpha / (2 - alpha), var_in = sigma^2 + LSB_v^2/12 (12-bit)
    al = 1 - math.exp(-2 * math.pi * cfg.vdc_corner_hz * cfg.tau)
    sd = math.sqrt((cfg.vdc_noise_v ** 2 + s.LSB_v ** 2 / 12) * al / (2 - al))
    n_eff = n * (1 - (1 - al)) / (1 + (1 - al))
    assert abs(vd[200:].mean() - 1.003 * 300.0) < 4 * sd / math.sqrt(n_eff)
    assert abs(vd[200:].std() / sd - 1) < 0.02


def test_U4_noise_statistics_1e5_samples():
    check_U4()
    # large-sigma variant: quantization negligible, so the BARE spec sigma is tested to 2 %
    cfg = replace(NOMINAL, i_gain_err=0.0, i_offset_a=0.0, i_noise_lsb=8.0)
    s = mk(cfg, 9)
    x = np.array([s.measure(flat(), 0.0, BMS).i_a_m for _ in range(100_000)])
    assert abs(x.std() / (8.0 * s.LSB_i) - 1) < 0.02


def test_MUT_noise_x10_fails_U4():
    class M(SensorSuite):
        def __init__(self, cfg, rng):
            super().__init__(replace(cfg, i_noise_lsb=10 * cfg.i_noise_lsb, theta_noise_rad=10 * cfg.theta_noise_rad,
                                     vdc_noise_v=10 * cfg.vdc_noise_v), rng)
    must_fail(check_U4, M, 20_000)


# ============================================================================== U5
def check_U5(cls=SensorSuite):
    s = mk(replace(NOMINAL, theta_offset_rad=0.0), 6, cls)
    n = 10_000
    x = np.linspace(-300.0, 300.0, n)
    meas = np.empty((n, 3))
    for k in range(n):
        m = s.measure(flat(ia=x[k], ib=x[k], ic=x[k]), 0.0, BMS)
        meas[k] = (m.i_a_m, m.i_b_m, m.i_c_m)
    assert np.max(np.abs(s.gain_i)) > 1e-3 and np.max(np.abs(s.offset_i)) > 0.1, "draws should be non-trivial"
    for p in range(3):
        slope, icpt = np.polyfit(x, meas[:, p], 1)
        assert abs(slope - (1 + s.gain_i[p])) < 1e-3, (p, slope, 1 + s.gain_i[p])
        assert abs(icpt - s.offset_i[p]) < 0.01, (p, icpt, s.offset_i[p])


def test_U5_gain_and_offset_recovered_by_regression_on_a_ramp():
    check_U5()


def test_MUT_gain_applied_twice_fails_U5():
    def post(self, m, tr, ref):
        return replace(m, i_a_m=m.i_a_m * 1.01, i_b_m=m.i_b_m * 1.01, i_c_m=m.i_c_m * 1.01)
    must_fail(check_U5, mutant(post))


# ============================================================================== U6
def check_U6(cls=SensorSuite):
    s = mk(zero_cfg(), 3, cls)
    lsb, r = s.LSB_i, s.cfg.i_range_a
    for v in (400.5, 450.0, 1000.0, 1e5):
        m = s.measure(flat(ia=v, ib=-v, ic=v), 0.0, BMS)
        assert abs(m.i_a_m - r) <= lsb and abs(m.i_b_m + r) <= lsb and abs(m.i_c_m - r) <= lsb
        assert abs(m.i_a_m) <= r + 1e-12
    # clipping happens after gain and offset: a true 399 A with +5 A offset must read full scale
    s2 = mk(zero_cfg(), 3, cls)
    s2.offset_i = np.array([5.0, -5.0, 0.0])
    m = s2.measure(flat(ia=399.0, ib=-399.0, ic=0.0), 0.0, BMS)
    assert abs(m.i_a_m - r) <= lsb and abs(m.i_b_m + r) <= lsb and abs(m.i_c_m) <= lsb
    # below range nothing is clipped
    m = s.measure(flat(ia=399.0), 0.0, BMS)
    assert abs(m.i_a_m - 399.0) <= lsb / 2 + 1e-9


def test_U6_full_scale_clipping():
    check_U6()


def test_MUT_no_clipping_fails_U6():
    def post(self, m, tr, ref):
        return replace(m, i_a_m=tr["i_a"], i_b_m=tr["i_b"], i_c_m=tr["i_c"])
    must_fail(check_U6, mutant(post))


# ============================================================================== U7
def _grid_truth(rng, lsb):
    a, b, c = (rng.integers(-1000, 1000, 3) * lsb)
    return float(a), float(b), float(c), float(rng.uniform(-math.pi, math.pi))


def check_U7(full_two_sensor=False, cls=SensorSuite):
    lsb = 2 * 400.0 / 2 ** 12
    off = 7 * lsb
    rng = np.random.default_rng(21)
    # (1) equal offsets on all three phases cancel in dq with the FULL Clarke
    for _ in range(100):
        a, b, c, ep = _grid_truth(rng, lsb)
        ref_s = mk(zero_cfg(two_sensor=full_two_sensor), 1, cls)
        off_s = mk(zero_cfg(two_sensor=full_two_sensor), 1, cls)
        off_s.offset_i = np.array([off, off, off])
        tr = flat(eps=ep, epsp=ep, ia=a, ib=b, ic=c)
        m0, m1 = ref_s.measure(tr, 0.0, BMS), off_s.measure(tr, 0.0, BMS)
        assert abs(m1.i_d_m - m0.i_d_m) < 1e-9 and abs(m1.i_q_m - m0.i_q_m) < 1e-9, "common-mode offset leaked into dq"
    # (2) ... and do NOT cancel with the two-sensor form: dq shift has magnitude exactly 2*offset (alpha=b, beta=sqrt3*b)
    for _ in range(100):
        a, b, c, ep = _grid_truth(rng, lsb)
        ref_s = mk(zero_cfg(two_sensor=True), 1, cls)
        off_s = mk(zero_cfg(two_sensor=True), 1, cls)
        off_s.offset_i = np.array([off, off, off])
        tr = flat(eps=ep, epsp=ep, ia=a, ib=b, ic=c)
        m0, m1 = ref_s.measure(tr, 0.0, BMS), off_s.measure(tr, 0.0, BMS)
        assert abs(math.hypot(m1.i_d_m - m0.i_d_m, m1.i_q_m - m0.i_q_m) - 2 * off) < 1e-9
    # (3) one-phase offset b_a, zero true current: i_d_m = (2/3) b cos(eps_prev), i_q_m = -(2/3) b sin(eps_prev), a tone at f_e
    tau, fe, N, ncyc = TAU_PLAIN, 100.0, 2000, 20
    w = 2 * math.pi * fe
    s = mk(zero_cfg(), 1, cls)
    s.offset_i = np.array([off, 0.0, 0.0])
    d = np.empty(N); q = np.empty(N)
    for k in range(N):
        m = s.measure(flat(eps=wrap_pi(w * k * tau), epsp=wrap_pi(w * (k - 1) * tau), t=k * tau), 0.0, BMS)
        d[k], q[k] = m.i_d_m, m.i_q_m
    for sig in (d, q):
        spec = np.abs(np.fft.rfft(sig)) * 2.0 / N
        assert int(np.argmax(spec[1:])) + 1 == ncyc, "spectral peak not at the electrical frequency"
        assert abs(spec[ncyc] / (2.0 / 3.0 * off) - 1) < 0.10
        assert np.max(np.delete(spec, [0, ncyc])) < 0.02 * spec[ncyc], "leakage / other tones"


def test_U7_common_mode_offset_cancels_full_clarke_not_two_sensor_and_tone():
    check_U7()


def test_MUT_two_sensor_instead_of_full_clarke_fails_U7():
    must_fail(check_U7, True)


# ============================================================================== U8
def check_U8(cls=SensorSuite):
    delta = 0.02
    # truth: eps_prev = -pi/2, i_abc = (100, -50, -50) -> alpha = 100, beta = 0 -> plant dq = (0, 100), all on the LSB grid
    tr = flat(eps=-math.pi / 2, epsp=-math.pi / 2, ia=100.0, ib=-50.0, ic=-50.0)
    assert abs(tr["i_d"]) < 1e-12 and abs(tr["i_q"] - 100.0) < 1e-12
    s = mk(zero_cfg(), 1, cls)
    s.theta_offset = delta
    for _ in range(3):
        m = s.measure(tr, 0.0, BMS)
        # numeric expected values (independent of the code): 100 sin(0.02), 100 cos(0.02)
        assert abs(m.i_d_m - 1.9998666693333) < 1e-9
        assert abs(m.i_q_m - 99.98000066666) < 1e-9
        # general relation i_dq_m = R(-delta) i_dq
        c, sn = math.cos(delta), math.sin(delta)
        assert abs(m.i_d_m - (c * tr["i_d"] + sn * tr["i_q"])) < 1e-9
        assert abs(m.i_q_m - (-sn * tr["i_d"] + c * tr["i_q"])) < 1e-9
        assert abs(m.theta_m - (tr["epsilon_el"] + delta)) < 1e-12 and abs(s.last_delta - delta) < 1e-15
        assert abs(s.last_gaps["theta_m"] - delta) < 1e-15
    # with noise and ripple on, the dq current uses the delta of the PREVIOUS step's angle: i_dq_m = R(-delta_prev) park(abc_m, eps_prev)
    s = mk(replace(zero_cfg(), theta_noise_rad=0.01, theta_ripple_rad=0.02, theta_offset_rad=0.02), 2, cls)
    prev = None
    for k in range(300):
        t = synth(k)
        m = s.measure(t, 0.0, BMS)
        dprev = s.last_delta if prev is None else prev
        dd, qq = park(m.i_a_m, m.i_b_m, m.i_c_m, t["epsilon_prev"])
        c, sn = math.cos(dprev), math.sin(dprev)
        assert abs(m.i_d_m - (c * dd + sn * qq)) < 1e-9 and abs(m.i_q_m - (-sn * dd + c * qq)) < 1e-9
        prev = s.last_delta
    assert np.std([s.measure(synth(k), 0.0, BMS).theta_m - synth(k)["epsilon_el"] for k in range(300, 400)]) > 1e-3


def test_U8_angle_error_rotates_dq_currents_by_minus_delta():
    check_U8()


def test_MUT_angle_error_dropped_fails_U8():
    def post(self, m, tr, ref):
        d, q = park(m.i_a_m, m.i_b_m, m.i_c_m, tr["epsilon_prev"])
        return replace(m, i_d_m=d, i_q_m=q, theta_m=tr["epsilon_el"])
    must_fail(check_U8, mutant(post))


# ============================================================================== U9
def _overshoot_angle_step(zeta):
    """Analytic overshoot of theta_hat for a step in theta_m: Y = (2 z s + 1)/(s (s^2 + 2 z s + 1)), wn = 1:
    y = 1 - e^{-z t}[cos(wd t) - (z/wd) sin(wd t)]  (z < 1); returns max(y) - 1."""
    wd = math.sqrt(1 - zeta ** 2)
    t = np.linspace(0, 40, 400_001)
    y = 1 - np.exp(-zeta * t) * (np.cos(wd * t) - zeta / wd * np.sin(wd * t))
    return y.max() - 1.0


def _overshoot_speed_step(zeta):
    """Analytic overshoot of omega_hat for a step in speed (theta_m ramp): wn^2/D, = exp(-pi z / sqrt(1 - z^2))."""
    return math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))


def check_U9(cls=SensorSuite, noise_n=60_000):
    cfg = zero_cfg()
    wn, z, dt, p = cfg.pll_wn, cfg.pll_zeta, cfg.tau, cfg.p
    # (a) speed (frequency) step: the 4.3 % figure of the plan belongs to THIS response
    w1 = 200.0
    s = mk(cfg, 1, cls)
    om = []
    for k in range(1500):
        m = s.measure(flat(eps=wrap_pi(w1 * k * dt), epsp=wrap_pi(w1 * (k - 1) * dt), t=k * dt, om=0.0), 0.0, BMS)
        om.append(m.omega_m_m * p)
    om = np.array(om)
    ov = om.max() / w1 - 1
    an = _overshoot_speed_step(z)
    assert abs(an - 0.0432) < 1e-4
    assert abs(ov / an - 1) < 0.20, (ov, an)
    # zero steady-state error at constant speed (after ~30 time constants)
    # (theta_hat is a one-step predictor: after the update it sits omega*dt ahead of the last sample, so the detector error is 0)
    assert abs(om[-1] - w1) < 1e-6 * w1 and abs(wrap_pi(s.pll_theta_hat - (w1 * 1499 * dt + w1 * dt))) < 1e-9
    # (b) angle step: theta_hat overshoots by ~20.8 % (the loop has a zero), NOT 4.3 %
    s = mk(cfg, 1, cls)
    step = 0.1
    th = []
    for k in range(2500):
        eps = 0.0 if k < 200 else step
        s.measure(flat(eps=eps, epsp=eps, t=k * dt), 0.0, BMS)
        th.append(s.pll_theta_hat)
    th = np.array(th)
    ov_a = (th.max() - 0.0) / step - 1
    an_a = _overshoot_angle_step(z)
    assert 0.20 < an_a < 0.22
    assert abs(ov_a / an_a - 1) < 0.20, (ov_a, an_a)
    assert abs(th[-1] - step) < 1e-9
    # (c) noise on omega_m_m for white angle noise: var(omega_hat_el) = sigma^2 dt wn^3 / (4 zeta)
    sig = np.deg2rad(0.3)
    s = mk(replace(zero_cfg(), theta_noise_rad=sig), 5, cls)
    base = flat(eps=0.0, epsp=0.0)
    x = np.array([s.measure(base, 0.0, BMS).omega_m_m for _ in range(noise_n)])[3000:]
    pred = math.sqrt(sig ** 2 * dt * wn ** 3 / (4 * z)) / p
    assert abs(x.std() / pred - 1) < 0.25, (x.std(), pred)


def test_U9_pll_overshoot_zero_steady_error_and_noise():
    check_U9()


def test_MUT_pll_reads_true_speed_fails_U9():
    def post(self, m, tr, ref):
        return replace(m, omega_m_m=tr["omega_m"])
    must_fail(check_U9, mutant(post), 3000)


# ============================================================================== U10
def check_U10(cls=SensorSuite):
    for tau in (TAU_PLAIN, TAU_CYCLE):
        cfg = zero_cfg(tau=tau)
        tc = 1.0 / (2 * math.pi * cfg.vdc_corner_hz)
        s = mk(cfg, 1, cls)
        s.vdc_gain = 0.004
        v0, v1, n0, N = 50.0, 450.0, 5, 200
        y = []
        for k in range(N):
            y.append(s.measure(flat(vdc=v0 if k < n0 else v1, t=k * tau), 0.0, BMS).vdc_m)
        y = np.array(y)
        # DC gain 1 + gv (quantizer only)
        assert abs(y[0] - 1.004 * v0) <= s.LSB_v / 2 + 1e-9 and abs(y[-1] - 1.004 * v1) <= s.LSB_v / 2 + 1e-9
        # step: samples n0.. follow y0 + (yf - y0)(1 - (1-alpha)^(m+1)); fit ln(residual fraction) over residual > 3 %
        yf, y0 = y[-1], y[n0 - 1]
        res = 1 - (y[n0:] - y0) / (yf - y0)
        use = np.where(res > 0.03)[0]
        assert len(use) >= 3
        slope = np.polyfit(use, np.log(res[use]), 1)[0]       # per sample
        tc_est = -tau / slope
        assert abs(tc_est / tc - 1) < 0.05, (tau, tc_est, tc)
        # exact discretization: first-sample response equals 1 - exp(-dt/Tc)
        assert abs((y[n0] - y0) / (yf - y0) - (1 - math.exp(-tau / tc))) < 2e-3


def test_U10_vdc_filter_time_constant_and_dc_gain_at_both_tau():
    check_U10()


def test_MUT_euler_alpha_fails_U10():
    class M(SensorSuite):
        def reset(self):
            super().reset()
            self._alpha = 2 * math.pi * self.cfg.vdc_corner_hz * self.dt     # forward-Euler instead of exact
    must_fail(check_U10, M)


# ============================================================================== U11
def check_U11(cls=SensorSuite):
    tau = TAU_PLAIN
    f = lambda t: 37.0123 * t + 0.5 * math.sin(40 * t)
    # analytic reference: unchanged, bit for bit
    s = mk(zero_cfg(), 1, cls)
    for k in range(3000):
        m = s.measure(flat(t=k * tau), f(k * tau), BMS)
        assert m.omega_ref == f(k * tau)
    # TUM-style: 10 Hz hold and 0.1 rad/s quantization
    s = mk(zero_cfg(ref_hold_s=0.1, ref_quant=0.1), 1, cls)
    n = 5000
    got = [s.measure(flat(t=k * tau), f(k * tau), BMS).omega_ref for k in range(n)]
    for k in range(n):
        kk = int(round(k * tau / 0.1 - 1e-9 + 1e-6)) if False else math.floor(k * tau / 0.1 + 1e-6)
        want = np.round(f(kk * 0.1) / 0.1) * 0.1
        assert abs(got[k] - want) < 1e-12, (k, got[k], want)
    assert len(set(got)) == 5
    # hold only (quant 0) and quant only (hold 0)
    s = mk(zero_cfg(ref_hold_s=0.1), 1, cls)
    h = [s.measure(flat(t=k * tau), f(k * tau), BMS).omega_ref for k in range(1500)]
    assert h[999] == f(0.0) and h[1000] == f(0.1) and h[1499] == f(0.1)
    s = mk(zero_cfg(ref_quant=0.1), 1, cls)
    assert abs(s.measure(flat(), 3.14159, BMS).omega_ref - 3.1) < 1e-12


def test_U11_request_hold_and_quantization():
    check_U11()


def test_MUT_hold_bypassed_fails_U11():
    def post(self, m, tr, ref):
        return replace(m, omega_ref=ref)
    must_fail(check_U11, mutant(post))


# ============================================================================== extras
def test_U2b_tau_independence_of_per_run_draws_and_gaps_keys():
    a = mk(SensorConfig(tau=TAU_PLAIN), 5)
    b = mk(SensorConfig(tau=TAU_CYCLE), 5)
    assert np.array_equal(a.gain_i, b.gain_i) and np.array_equal(a.offset_i, b.offset_i)
    m = a.measure(synth(3), 1.0, BMS)
    assert set(a.last_gaps) == {"i_a_m", "i_b_m", "i_c_m", "i_d_m", "i_q_m", "theta_m", "omega_m_m", "vdc_m", "omega_ref"}
    assert m.bms is BMS


def test_U4b_ripple_is_continuous_across_epsilon_wrap():
    s = mk(replace(zero_cfg(), theta_ripple_rad=0.01), 1)
    prev = None
    for k in range(4000):
        t = synth(k, fe=150.0)
        s.measure(t, 0.0, BMS)
        if prev is not None:
            # max slope: ripple*(1+2)*d(th_mech) = 0.01*3*(2 pi 150 1e-4/3) = 9.4e-4; a wrap jump of epsilon/p would be ~1.7e-2
            assert abs(s.last_delta - prev) < 1.0e-3
        prev = s.last_delta
