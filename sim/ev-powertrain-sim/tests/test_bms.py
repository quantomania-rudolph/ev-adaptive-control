"""BMS message model tests (plan U18, U19) and mutation tests. Uses a small SYNTHETIC fit dict so the tests do not depend
on the fitted numbers. Run: pytest -q tests/test_bms.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from interfaces import BMSConfig, BMSMsg, BATTERY_TRUTH_KEYS
from bms import BMSModel

DT = 1.0e-3
FIT = {"envelope_real": {"soc_grid": [0.0, 0.5, 1.0]},
       "scaling": {"i_dis_lim_toy_a": [100.0, 200.0, 200.0], "i_chg_lim_toy_a": [30.0, 60.0, 60.0]}}


def truth_fn(t, fault_t=None, fault_val=60.0, perturb_after=None, seed=0):
    """Smoothly varying synthetic truth; optional fault (i_batt_max drops) and an arbitrary perturbation AFTER a time."""
    soc = 0.6 + 0.05 * np.sin(2.0 * np.pi * t / 1.7)
    v = 300.0 + 8.0 * np.sin(2.0 * np.pi * t / 0.37)
    imax = 250.0
    if fault_t is not None and t >= fault_t:
        imax = fault_val
    if perturb_after is not None and t > perturb_after:
        r = np.random.default_rng(int(round(t / DT)) + 7919 * seed)
        soc, v, imax = soc + r.normal(0, 0.05), v + r.normal(0, 20), imax * r.uniform(0.2, 1.5)
    d = {k: 0.0 for k in BATTERY_TRUTH_KEYS}
    d.update({"V_dc": v, "soc": soc, "i_batt_max": imax, "i_batt": 10.0, "i_dc": 10.0, "V_term": v, "T_batt": 300.0})
    return d


_SCENARIO = {}                                                     # truth scenario of the current run (read only by the oracle MUTANT)


def run(factory, T, **tkw):
    _SCENARIO.clear(); _SCENARIO.update(tkw)
    m = factory()
    ts = np.arange(0, int(round(T / DT)) + 1) * DT
    return ts, [m.step(float(t), truth_fn(float(t), **tkw)) for t in ts]


def real_factory(cfg=None, seed=1):
    cfg = cfg or BMSConfig()
    return lambda: BMSModel(cfg, FIT, np.random.default_rng(seed))


def _emissions(ts, msgs):
    """Indices where a NEW message appears: the age drops (a held message only gets older)."""
    return [j for j in range(1, len(msgs)) if msgs[j].age_s < msgs[j - 1].age_s - 1e-12]


# ---------------------------------------------------------------------------------------------- causality harness
def causality_violations(factory, t0s=(0.611, 0.731, 0.9), T=1.2):
    """Run twice with identical truth up to t0 and DIFFERENT truth afterwards; count times t <= t0 where the messages differ.
    A causal BMS gives 0."""
    ts, a = run(factory, T, perturb_after=None)
    n = 0
    for t0 in t0s:
        _, b = run(factory, T, perturb_after=t0)
        n += sum(1 for t, x, y in zip(ts, a, b) if t <= t0 and x != y)
    return n


class NonCausalBMS(BMSModel):
    """MUTANT: ignores the truth it is handed and reads the truth `lead` seconds into the FUTURE (an oracle)."""
    lead = 0.08

    def step(self, t, battery_truth):
        return super().step(t, truth_fn(t + self.lead, **_SCENARIO))


def noncausal_factory(_unused=None):
    return lambda: NonCausalBMS(BMSConfig(), FIT, np.random.default_rng(1))


# ---------------------------------------------------------------------------------------------- U18
def test_U18_cadence_age_and_causality():
    cfg = BMSConfig()                                              # period 0.05, latency 0.02
    ts, msgs = run(real_factory(cfg), 2.0)
    # cadence: a new message appears only when t crosses a multiple of the period, and one appears at each such crossing
    em = _emissions(ts, msgs)
    for j in em:
        r = ts[j] / cfg.period_s
        assert abs(r - round(r)) < 1e-6, f"message changed off-boundary at t={ts[j]}"
    assert len(em) == int(round(2.0 / cfg.period_s))              # 40 boundaries after t = 0 in 2 s
    # content only changes at emissions; the held message is identical (same limits and v_pack) in between
    for j in range(1, len(msgs)):
        if j not in em:
            a, b = msgs[j - 1], msgs[j]
            assert (a.i_dis_lim, a.i_chg_lim, a.v_pack) == (b.i_dis_lim, b.i_chg_lim, b.v_pack)
    # age bounds
    ages = np.array([m.age_s for m in msgs])
    assert ages.min() >= -1e-12 and ages.max() <= cfg.period_s + cfg.latency_s + 1e-9
    # after the first latency the age right after an emission equals the latency
    assert abs(msgs[em[3]].age_s - cfg.latency_s) < 1e-9
    # causality
    assert causality_violations(real_factory(cfg)) == 0
    # the harness has power: it flags a BMS that reads the future (also asserted as MUT test)
    assert causality_violations(noncausal_factory()) > 0


def test_MUT_noncausal_bms_is_flagged_by_causality_check():
    assert causality_violations(noncausal_factory()) > 0
    assert causality_violations(real_factory()) == 0


# ---------------------------------------------------------------------------------------------- U19
def fault_first_seen(factory, t_f, T=1.0):
    ts, msgs = run(factory, T, fault_t=t_f, fault_val=60.0)
    em = _emissions(ts, msgs)
    seen = [ts[j] for j in em if msgs[j].i_dis_lim < 120.0]      # normal limits are about 190-210 A, faulted about 60 A
    return t_f, (seen[0] if seen else None), ts, msgs, em


def test_U19_fault_not_reflected_before_latency():
    cfg = BMSConfig(period_s=0.05, latency_s=0.02)
    t_f = 0.541                                                   # NOT on a boundary; the 0.55 s message (sampled 0.53) must not show it
    _, first, ts, msgs, em = fault_first_seen(real_factory(cfg), t_f)
    assert first is not None, "the fault never reaches the message"
    # no message built before t_f + latency may show the fault
    assert first >= t_f + cfg.latency_s - DT - 1e-9, f"fault visible at {first} < t_f+latency={t_f + cfg.latency_s}"
    # and the first message built after that moment does show it (messages every 50 ms)
    assert first <= t_f + cfg.latency_s + cfg.period_s + DT
    # a long latency pushes the first visible message later by the same amount
    cfg2 = BMSConfig(period_s=0.05, latency_s=0.12)
    _, first2, *_ = fault_first_seen(real_factory(cfg2), t_f, T=1.2)
    assert first2 >= t_f + 0.12 - DT - 1e-9
    # truth DID drop at t_f: a zero-latency instantaneous reader would see it right away
    assert truth_fn(t_f, fault_t=t_f)["i_batt_max"] == 60.0


def test_MUT_zero_latency_bms_reflects_fault_early():
    t_f, cfg = 0.541, BMSConfig(period_s=0.05, latency_s=0.02)
    mutant = real_factory(BMSConfig(period_s=0.05, latency_s=0.0))     # latency bypassed
    _, first, *_ = fault_first_seen(mutant, t_f)
    assert first is not None and first < t_f + cfg.latency_s - DT - 1e-9   # the U19 bound is violated -> U19 would fail


# ---------------------------------------------------------------------------------------------- further properties
def test_message_construction_quantization_and_error_statistics():
    cfg = BMSConfig(period_s=0.05, latency_s=0.02, quant_a=1.0, rel_err=0.05)
    ts, msgs = run(real_factory(cfg, seed=5), 40.0)
    em = _emissions(ts, msgs)
    lim = np.array([msgs[j].i_dis_lim for j in em])
    assert np.allclose(lim, np.round(lim))                         # quantized to 1 A
    # reference: min(envelope, 250) at the sampled truth, relative error N(0, 5 %)
    soc_s = np.array([truth_fn(ts[j] - cfg.latency_s)["soc"] for j in em])
    ref = np.interp(soc_s, FIT["envelope_real"]["soc_grid"], FIT["scaling"]["i_dis_lim_toy_a"])
    rel = lim / ref - 1.0
    assert abs(rel.mean()) < 4 * 0.05 / np.sqrt(len(rel)) + 0.5 / ref.mean()
    assert 0.8 * 0.05 < rel.std() < 1.2 * 0.05
    # v_pack is the sampled V_dc (no extra noise)
    j = em[10]
    assert abs(msgs[j].v_pack - truth_fn(ts[j] - cfg.latency_s)["V_dc"]) < 2 * 8.0 * 2 * np.pi / 0.37 * DT


def test_deterministic_and_rng_drawn_once_per_message():
    ts, a = run(real_factory(seed=3), 1.0)
    _, b = run(real_factory(seed=3), 1.0)
    assert a == b
    _, c = run(real_factory(seed=4), 1.0)
    assert a != c
    rng = np.random.default_rng(3)
    m = BMSModel(BMSConfig(), FIT, rng)
    for t in ts:
        m.step(float(t), truth_fn(float(t)))
    n_msgs = int(np.floor(1.0 / 0.05 + 1e-9)) + 1
    ref = np.random.default_rng(3)
    ref.normal(size=n_msgs)
    assert rng.normal() == ref.normal()                            # exactly one draw per message


def test_disabled_message_equals_truth():
    m = BMSModel(BMSConfig(enabled=False, quant_a=5.0, rel_err=0.5, latency_s=1.0), FIT, np.random.default_rng(0))
    for t in (0.0, 0.0123, 0.4):
        tr = truth_fn(t)
        msg = m.step(t, tr)
        assert msg.age_s == 0.0 and msg.v_pack == tr["V_dc"]
        assert msg.i_dis_lim == min(float(np.interp(tr["soc"], [0, .5, 1], [100, 200, 200])), tr["i_batt_max"])
        assert msg.i_chg_lim == float(np.interp(tr["soc"], [0, .5, 1], [30, 60, 60]))


def test_before_enough_history_uses_earliest_sample_and_time_cannot_go_backwards():
    cfg = BMSConfig(period_s=0.05, latency_s=0.3)
    m = BMSModel(cfg, FIT, np.random.default_rng(0))
    m0 = m.step(0.0, truth_fn(0.0))
    assert abs(m0.age_s) < 1e-12 and abs(m0.v_pack - truth_fn(0.0)["V_dc"]) < 1e-9
    m1 = m.step(0.05, truth_fn(0.05))                              # target t - 0.3 < 0: still the earliest sample
    assert abs(m1.v_pack - truth_fn(0.0)["V_dc"]) < 1e-9 and abs(m1.age_s - 0.05) < 1e-9
    try:
        m.step(0.01, truth_fn(0.01))
        raise AssertionError("expected ValueError for time going backwards")
    except ValueError:
        pass
    m.reset()
    assert m.step(0.0, truth_fn(0.0)).age_s == 0.0


def test_message_type_and_fields():
    msg = real_factory()().step(0.0, truth_fn(0.0))
    assert isinstance(msg, BMSMsg) and msg.i_dis_lim >= 0 and msg.i_chg_lim >= 0
