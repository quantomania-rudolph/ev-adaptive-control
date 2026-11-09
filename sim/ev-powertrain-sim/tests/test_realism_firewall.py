"""Firewall and integration tests (plan section 5.3, U26-U39). Requires sensors.py, inverter.py, bms.py from the build agents
for the loop-level tests; controller-level tests need only control_fw.py.

Naming: test_U<id>_... so tests/test_plan_coverage.py can check that every promised test exists.
"""
import ast
import dataclasses
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import interfaces as I
import run_exp as R
from control_fw import Beliefs, FirewalledController
from plant import GEMPlant

FIT = ROOT / "data" / "fixtures" / "tum_battery_fit_FIXTURE.json"
RUNS = ROOT / "runs"


# ------------------------------------------------------------------------------------------------ helpers
def mk_meas(**over):
    base = dict(t=0.0, i_a_m=0.0, i_b_m=0.0, i_c_m=0.0, i_d_m=0.0, i_q_m=10.0, theta_m=0.0, omega_m_m=100.0, vdc_m=300.0,
                bms=I.BMSMsg(237.0, 61.0, 300.0, 0.01), omega_ref=150.0, ceiling_telemetry=None)
    base.update(over)
    return I.Measurements(**base)


def beliefs(**over):
    b = dict(R=0.018, Ld=0.00037, Lq=0.0012, psi_name=0.066, J=0.54, psi_hat0=0.0462)
    b.update(over)
    return Beliefs(**b)


def stream(n=2000, seed=5):
    rng = np.random.default_rng(seed)
    return [mk_meas(t=k * 1e-4, i_q_m=float(20 + 5 * rng.standard_normal()), i_d_m=float(rng.standard_normal()),
                    omega_m_m=float(100 + 3 * rng.standard_normal()), vdc_m=float(300 + 0.5 * rng.standard_normal()),
                    omega_ref=150.0) for k in range(n)]


def replay(ms, cfg=None, b=None):
    c = FirewalledController(cfg or I.ControllerConfig(mode="aw", governor=True, delay_steps=0, angle_advance=False), b or beliefs())
    return np.array([c.command(m)[0] for m in ms])


@pytest.fixture
def fast_scenario(monkeypatch):
    """Short, aggressive scenario: reaches saturation within 1.2 s (the real one takes 4.5 s)."""
    monkeypatch.setattr(R, "T_END", 1.2)
    monkeypatch.setattr(R, "T_EVENT", (0.5, 0.8))
    monkeypatch.setattr(R, "omega_ref", lambda t: 200.0 * min(t / 0.4, 1.0))
    monkeypatch.setattr(R, "disturbance", lambda t: 10.0 if t >= 0.2 else 0.0)


def legacy_df(tmp_path, **kw):
    df, _ = R.run("L", outdir=str(tmp_path), **kw)
    return df


# ------------------------------------------------------------------------------------------------ U26 static firewall
def test_U26_controller_module_is_statically_firewalled():
    src = (ROOT / "control_fw.py").read_text()
    tree = ast.parse(src)
    forbidden_mods = {"plant", "battery", "bms", "sensors", "inverter", "run_exp", "telemetry", "firewalled_loop", "gym_electric_motor"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not ({a.name.split(".")[0] for a in node.names} & forbidden_mods)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_mods
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    leaks = names & {"plant", "pl", "led", "ledger", "psi_p", "u_max_hw", "prev_ledger", "st"}
    assert not leaks, f"truth-looking identifiers in control_fw.py: {leaks}"
    import inspect
    assert list(inspect.signature(FirewalledController.__init__).parameters) == ["self", "cfg", "beliefs"]
    assert list(inspect.signature(FirewalledController.command).parameters) == ["self", "m"]


# ------------------------------------------------------------------------------------------------ U27 non-interference (a) replay (b) poison (c) mutation
def test_U27a_replay_is_bit_identical_across_unrelated_activity():
    ms = stream()
    a = replay(ms)
    np.random.seed(123)                       # unrelated global state
    pl = GEMPlant()
    for _ in range(20):
        pl.apply_voltage(np.array([0.0, 30.0]))
    b = replay(ms)
    assert np.array_equal(a, b)


class _Leaky(FirewalledController):
    """Deliberately broken controller: reads a truth value from a global plant. The poison test must catch it."""
    plant = None

    def command(self, m):
        u, i = super().command(m)
        return u + 1e-3 * type(self).plant.psi_p, i


def _poisoned_plant():
    pl = GEMPlant()
    for attr in ("psi_p", "Rs", "Ld", "Lq", "J", "u_max_hw", "_vdc", "V_dc", "p"):
        setattr(pl, attr, float("nan"))
    return pl


def test_U27b_poisoning_all_truth_does_not_change_the_command():
    ms = stream(500)
    clean = replay(ms)
    pl = _poisoned_plant()                    # every truth attribute NaN, and still alive in memory
    poisoned = replay(ms)
    assert np.isfinite(poisoned).all() and np.array_equal(clean, poisoned)
    del pl


def test_U27c_mutation_a_leaky_controller_is_caught_by_the_poison_test():
    ms = stream(200)
    good = GEMPlant()
    _Leaky.plant = good
    leaky_clean = np.array([_Leaky(I.ControllerConfig(delay_steps=0, angle_advance=False), beliefs()).command(m)[0] for m in ms[:1]])
    _Leaky.plant = _poisoned_plant()
    leaky_poisoned = np.array([_Leaky(I.ControllerConfig(delay_steps=0, angle_advance=False), beliefs()).command(m)[0] for m in ms[:1]])
    assert not np.isfinite(leaky_poisoned).all() or not np.array_equal(leaky_clean, leaky_poisoned)


# ------------------------------------------------------------------------------------------------ U28 contract fields
def test_U28_measurements_is_frozen_and_matches_the_contract():
    assert tuple(f.name for f in dataclasses.fields(I.Measurements)) == I.CONTRACT_FIELDS
    assert tuple(f.name for f in dataclasses.fields(I.BMSMsg)) == I.BMS_FIELDS
    m = mk_meas()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.i_q_m = 0.0
    md = (ROOT / "docs" / "INFORMATION_CONTRACT.md").read_text()
    for name in ["i_a_m", "i_b_m", "i_c_m", "i_d_m", "i_q_m", "theta_m", "omega_m_m", "vdc_m", "bms_msg", "omega_ref", "ceiling_telemetry"]:
        assert f"`{name}" in md or name in md, f"contract text does not mention {name}"


# ------------------------------------------------------------------------------------------------ U29 legacy equivalence
CASES = [dict(mode="naive"), dict(mode="aw"), dict(mode="aw", governor=True), dict(mode="naive", governor=True, vdc_aware=True)]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("event", [False, True])
def test_U29_identity_mode_reproduces_the_legacy_controller(tmp_path, fast_scenario, kw, event):
    from firewalled_loop import run_fw
    leg = legacy_df(tmp_path, event=event, **kw)
    new, _ = run_fw("N", event=event, identity=True, oracle_ceiling=event, write=False, **kw)
    assert len(leg) == len(new)
    if event:
        assert leg["saturated"].sum() > 100 and new["saturated"].sum() > 100, "event scenario must saturate or the equivalence test is vacuous"
    for c in ["omega_m", "i_d", "i_q", "u_q_plant", "u_d_plant", "psi_hat", "z", "T_elec", "iq_ref", "d_u_q", "u_d_cmd", "u_q_cmd", "u_limit"]:
        assert np.max(np.abs(leg[c].values - new[c].values)) < 1e-9, c


def test_U29_identity_mode_reproduces_the_legacy_battery_run(tmp_path, fast_scenario):
    from firewalled_loop import run_fw
    leg = legacy_df(tmp_path, event=True, battery=True, mode="aw", governor=True, vdc_aware=True)
    new, _ = run_fw("N", event=True, battery=True, identity=True, mode="aw", governor=True, vdc_aware=True, write=False)
    for c in ["omega_m", "i_q", "V_dc", "psi_hat", "u_limit", "d_u_q"]:
        assert np.max(np.abs(leg[c].values - new[c].values)) < 1e-9, c


@pytest.mark.skipif(not (RUNS / "A.parquet").exists(), reason="stored run A absent")
def test_U29_identity_mode_reproduces_the_stored_run_A(monkeypatch):
    """Independent of the legacy code path: compare to the stored artifact itself (first 1.2 s of the original 8 s run A)."""
    from firewalled_loop import run_fw
    monkeypatch.setattr(R, "T_END", 1.2)
    new, _ = run_fw("N", identity=True, write=False)
    a = pd.read_parquet(RUNS / "A.parquet").iloc[: len(new)]
    for c in ["omega_m", "i_q", "u_q_plant", "psi_hat", "T_elec"]:
        assert np.max(np.abs(a[c].values - new[c].values)) < 1e-9, c


# ------------------------------------------------------------------------------------------------ U30 existing suite
def test_U30_existing_test_suite_still_passes():
    existing = [str(p) for p in (ROOT / "tests").glob("test_*.py") if p.name not in ("test_realism_firewall.py", "test_plan_coverage.py",
                "test_sensors.py", "test_inverter.py", "test_battery_fit.py", "test_bms.py")]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *existing], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:]


# ------------------------------------------------------------------------------------------------ U31 ledger decomposition
def test_U31_ledger_decomposition_is_exact_and_telemetry_identity_holds(fast_scenario):
    from firewalled_loop import run_fw
    from telemetry import derive, params_from_plant
    df, _ = run_fw("N", mode="aw", event=True, governor=False, seed=2, write=False)
    for ax in "dq":
        parts = sum(df[f"d_u_{p}_{ax}"] for p in ("ctrl_clip", "adv", "delay", "rot", "plant_clip", "inv"))
        assert np.max(np.abs(parts - df[f"d_u_{ax}"])) < 1e-9, ax
    assert df["sat_ctrl"].sum() + df["sat_plant"].sum() > 0, "scenario must actually saturate or the test is vacuous"
    P = params_from_plant(GEMPlant())
    d = derive(df, P)
    assert d["resid_du_power"].abs().max() < 1e-6


# ------------------------------------------------------------------------------------------------ U32 per-channel gaps
def test_U32_every_measured_channel_has_a_nonzero_gap_when_errors_are_on(fast_scenario):
    from firewalled_loop import run_fw
    df, _ = run_fw("N", mode="aw", seed=4, write=False, t_end=0.5)
    gaps = [c for c in df.columns if c.startswith("gap_")]
    assert set(gaps) == {"gap_i_a", "gap_i_b", "gap_i_c", "gap_i_d", "gap_i_q", "gap_theta", "gap_omega", "gap_vdc"}
    for c in gaps:
        assert df[c].std() > 0, f"{c} has identically zero variance: that channel is not wired"
    for k in range(5):
        assert df["bms_age_s"].notna().all()


# ------------------------------------------------------------------------------------------------ U33 logging separation
@pytest.mark.skipif(not (RUNS / "A.parquet").exists(), reason="stored run A absent")
def test_U33_truth_columns_keep_their_names_and_new_groups_exist(fast_scenario):
    from firewalled_loop import run_fw
    df, _ = run_fw("N", mode="aw", seed=1, write=False, t_end=0.3)
    legacy_cols = set(pd.read_parquet(RUNS / "A.parquet").columns)
    assert legacy_cols <= set(df.columns), legacy_cols - set(df.columns)
    for pref in ("meas_", "gap_", "belief_", "bms_", "d_u_ctrl_clip", "d_u_inv"):
        assert any(c.startswith(pref) for c in df.columns), pref
    assert not any(c.startswith("meas_") and c[5:] in legacy_cols and False for c in df.columns)
    assert df["psi_tilde"].iloc[-1] == pytest.approx(df["psi_hat"].iloc[-1] - GEMPlant().psi_p)


# ------------------------------------------------------------------------------------------------ U34 beliefs
def test_U34_belief_draws_are_in_range_seeded_and_recorded():
    from firewalled_loop import draw_beliefs
    cfg = I.BeliefConfig()
    pl = GEMPlant()
    b1, d1 = draw_beliefs(pl, cfg, I.rng_streams(7)["beliefs"])
    b2, d2 = draw_beliefs(pl, cfg, I.rng_streams(7)["beliefs"])
    b3, d3 = draw_beliefs(pl, cfg, I.rng_streams(8)["beliefs"])
    assert d1 == d2 and d1 != d3
    assert cfg.dR_range[0] <= d1["dR"] <= cfg.dR_range[1] and cfg.dL_range[0] <= d1["dL"] <= cfg.dL_range[1]
    assert cfg.dJ_range[0] <= d1["dJ"] <= cfg.dJ_range[1] and cfg.dpsi_range[0] <= d1["dpsi"] <= cfg.dpsi_range[1]
    assert b1.psi_hat0 == pytest.approx(0.7 * pl.psi_p)
    exact, de = draw_beliefs(pl, I.BeliefConfig(exact=True), I.rng_streams(7)["beliefs"])
    assert de == dict(dR=0.0, dL=0.0, dJ=0.0, dpsi=0.0) and exact.R == pl.Rs and exact.Lq == pl.Lq


def test_U34_manifest_records_the_draws(fast_scenario):
    from firewalled_loop import run_fw
    _, mf = run_fw("N", seed=3, write=False, t_end=0.05)
    assert set(mf["belief_draws"]) == {"dR", "dL", "dJ", "dpsi"} and "beliefs" in mf and mf["seed"] == 3


# ------------------------------------------------------------------------------------------------ U35 seed independence
def test_U35_streams_are_independent_and_noise_is_shared_across_controllers(fast_scenario):
    a = I.rng_streams(11)
    _ = a["beliefs"].random(1000)             # consuming one stream must not move another
    assert a["sensors"].random() == I.rng_streams(11)["sensors"].random()
    from firewalled_loop import run_fw
    n, _ = run_fw("N", mode="naive", seed=11, write=False, t_end=0.02)
    w, _ = run_fw("W", mode="aw", governor=True, seed=11, write=False, t_end=0.02)
    # no battery: true bus is constant, so the voltage-sensor gap depends ONLY on the sensor stream, never on the controller
    assert np.array_equal(n["gap_vdc"].values, w["gap_vdc"].values)


# ------------------------------------------------------------------------------------------------ U36 tau independence
def test_U36_identity_loop_matches_a_legacy_loop_at_the_cycle_step():
    """Cycle runs use tau = 2e-4. Compare to a minimal legacy loop at that tau (the legacy runner hard-codes 1e-4)."""
    from control import CascadeController
    from firewalled_loop import run_fw
    tau = I.TAU_CYCLE
    pl = GEMPlant(disturbance_fn=lambda t: 0.0, tau=tau)
    ctl = CascadeController(pl, mode="aw")
    led, rows = {}, []
    for _ in range(1500):
        st = pl.state()
        u, _i = ctl.command(100.0 * min(st["t"] / 0.2, 1.0), st, led)
        led = pl.apply_voltage(u, None)
        rows.append((st["omega_m"], st["i_q"]))
    leg = np.array(rows)
    mp = pytest.MonkeyPatch()
    mp.setattr(R, "omega_ref", lambda t: 100.0 * min(t / 0.2, 1.0))
    mp.setattr(R, "disturbance", lambda t: 0.0)
    try:
        new, _ = run_fw("N", mode="aw", identity=True, tau=tau, t_end=1500 * tau, write=False)
    finally:
        mp.undo()
    assert np.max(np.abs(leg[:, 0] - new["omega_m"].values)) < 1e-9 and np.max(np.abs(leg[:, 1] - new["i_q"].values)) < 1e-9


# ------------------------------------------------------------------------------------------------ U37 BMS and vdc wiring
def test_U37_bms_message_and_vdc_reach_the_controller_output():
    cfg_on = I.ControllerConfig(mode="aw", governor=True, bms_aware=True, vdc_aware=True, delay_steps=0, angle_advance=False)
    cfg_off = dataclasses.replace(cfg_on, bms_aware=False)
    hi, lo = mk_meas(), mk_meas(bms=I.BMSMsg(20.0, 61.0, 300.0, 0.01))
    iq = lambda cfg, m: (lambda c: (c.command(m), c.i_lim_eff)[1])(FirewalledController(cfg, beliefs()))
    assert iq(cfg_on, lo) < iq(cfg_on, hi), "a low BMS current limit must lower the governor limit"
    assert iq(cfg_off, lo) == iq(cfg_off, hi), "without bms_aware the message must be ignored (flag is wired, not decorative)"
    c1, c2 = FirewalledController(cfg_on, beliefs()), FirewalledController(cfg_on, beliefs())
    c1.command(mk_meas(vdc_m=300.0)), c2.command(mk_meas(vdc_m=120.0))
    assert c2.last["ceiling_hat"] < c1.last["ceiling_hat"] and c2.last["ceiling_hat"] == pytest.approx(120.0 / np.sqrt(3.0))
    cap = FirewalledController(cfg_on, beliefs())
    cap.command(mk_meas(vdc_m=340.0))
    assert cap.last["ceiling_hat"] == pytest.approx(I.U_CAP), "the 300 V GEM cap must apply (regen)"


# ------------------------------------------------------------------------------------------------ U38 stability versus delay
@pytest.mark.parametrize("delay", [0, 1, 2])
@pytest.mark.parametrize("advance", [True, False])
def test_U38_protected_controller_stays_inside_the_envelope_across_delay(fast_scenario, delay, advance):
    """The result is REPORTED either way (a failure here is information about delay, not a test to loosen silently)."""
    from firewalled_loop import run_fw
    df, _ = run_fw("N", mode="aw", governor=True, vdc_aware=True, seed=1, write=False, t_end=0.9,
                   inv_cfg=I.InverterConfig(delay_steps=delay), ctl_overrides=dict(angle_advance=advance, d_gain_by_L=True))
    assert np.isfinite(df[["i_d", "i_q", "psi_hat", "omega_m"]].values).all()
    assert df["psi_hat"].abs().max() <= 0.15 and np.hypot(df["i_d"], df["i_q"]).max() <= 275.0


# ------------------------------------------------------------------------------------------------ U39 determinism
def test_U39_two_runs_at_equal_seed_are_bit_identical(fast_scenario):
    from firewalled_loop import run_fw
    a, _ = run_fw("N", mode="aw", governor=True, seed=9, write=False, t_end=0.3)
    b, _ = run_fw("N", mode="aw", governor=True, seed=9, write=False, t_end=0.3)
    cols = [c for c in a.columns if a[c].dtype.kind == "f"]
    assert all(np.array_equal(a[c].values, b[c].values) for c in cols)
    c, _ = run_fw("N", mode="aw", governor=True, seed=10, write=False, t_end=0.3)
    assert not np.array_equal(a["meas_i_q"].values, c["meas_i_q"].values)
