"""Coverage gap tests: static and behavioural firewall of the controller, ledger closure and determinism of short firewalled runs."""
import ast
import dataclasses
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import interfaces as I
from control_fw import Beliefs, FirewalledController
from firewalled_loop import draw_beliefs, run_fw
from plant import GEMPlant

SRC = (ROOT / "control_fw.py").read_text()
TREE = ast.parse(SRC)
FIT = ROOT / "data" / "tum_battery_fit.json"
PIECES = ("ctrl_clip", "adv", "delay", "rot", "plant_clip", "inv")
WREF = lambda t: 200.0 * min(t / 0.1, 1.0)


# ----------------------------------------------------------------------------------------------- static firewall
def test_control_fw_imports_only_numpy_dataclasses_and_four_interface_names():
    mods, names = set(), set()
    for n in ast.walk(TREE):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module)
            names |= {a.name for a in n.names if n.module == "interfaces"}
    assert mods == {"__future__", "dataclasses", "numpy", "interfaces"}
    assert names == {"ControllerConfig", "Measurements", "U_CAP", "rot"}


def test_control_fw_has_no_dynamic_escape_hatches():
    called = {n.func.id for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "getattr", "setattr", "globals", "locals", "__import__", "open", "vars", "compile"}
    attrs = {n.attr for n in ast.walk(TREE) if isinstance(n, ast.Attribute)}
    assert not attrs & {"modules", "__dict__", "__globals__", "f_back", "read_text", "read_bytes", "stack"}
    assert not any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(TREE))
    funcs_with_import = [f for f in ast.walk(TREE) if isinstance(f, ast.FunctionDef) and any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(f))]
    assert funcs_with_import == []                                            # no lazy import hidden inside a method


def _attrs_of(var):
    return {n.attr for n in ast.walk(TREE) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == var}


def test_every_attribute_read_from_m_b_cfg_is_a_declared_contract_field():
    meas = {f.name for f in dataclasses.fields(I.Measurements)}
    assert _attrs_of("m") <= meas and _attrs_of("m") >= {"vdc_m", "omega_m_m", "omega_ref", "i_d_m", "i_q_m", "ceiling_telemetry", "bms"}
    bms_attrs = {n.attr for n in ast.walk(TREE) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "bms"}
    assert bms_attrs <= set(I.BMS_FIELDS) and bms_attrs == {"i_dis_lim"}
    assert _attrs_of("b") <= {f.name for f in dataclasses.fields(Beliefs)}
    assert _attrs_of("cfg") <= {f.name for f in dataclasses.fields(I.ControllerConfig)}


def test_controller_reads_no_measurement_field_that_is_truth_in_disguise():
    # the contract lists exactly 12 fields; the controller must not rely on anything else even via a subclass of Measurements
    assert tuple(f.name for f in dataclasses.fields(I.Measurements)) == I.CONTRACT_FIELDS
    assert not {"omega_m", "omega_el", "epsilon_el", "V_dc", "i_d", "i_q"} & _attrs_of("m")


def test_loop_hands_the_controller_only_the_measurements_object():
    loop = ast.parse((ROOT / "firewalled_loop.py").read_text())
    ctl_attr = [n for n in ast.walk(loop) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "ctl"]
    assert {n.attr for n in ctl_attr} == {"command"}
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "command" and getattr(n.func.value, "id", "") == "ctl"]
    assert len(calls) == 1 and [getattr(a, "id", None) for a in calls[0].args] == ["m"] and not calls[0].keywords


def test_beliefs_are_built_outside_the_controller_from_the_plant_and_a_dedicated_stream():
    pl = GEMPlant()
    b, dr = draw_beliefs(pl, I.BeliefConfig(), I.rng_streams(3)["beliefs"])
    assert b.R == pytest.approx(pl.Rs * (1 + dr["dR"])) and b.Ld == pytest.approx(pl.Ld * (1 + dr["dL"])) and b.Lq == pytest.approx(pl.Lq * (1 + dr["dL"]))
    assert b.Ld / b.Lq == pytest.approx(pl.Ld / pl.Lq)                       # one shared inductance error keeps the saliency ratio
    assert b.psi_hat0 == pytest.approx(0.7 * pl.psi_p) and b.J == pytest.approx(pl.J * (1 + dr["dJ"]))
    assert -0.15 <= dr["dR"] <= 0.40 and 0.0 <= dr["dL"] <= 0.25 and -0.2 <= dr["dJ"] <= 0.2 and -0.1 <= dr["dpsi"] <= 0.1


def test_exact_beliefs_equal_truth_but_still_consume_the_same_draws():
    pl = GEMPlant()
    r1, r2 = I.rng_streams(5)["beliefs"], I.rng_streams(5)["beliefs"]
    b, dr = draw_beliefs(pl, I.BeliefConfig(exact=True), r1)
    draw_beliefs(pl, I.BeliefConfig(exact=False), r2)
    assert (b.R, b.Ld, b.Lq, b.J, b.psi_name) == (pl.Rs, pl.Ld, pl.Lq, pl.J, pl.psi_p) and set(dr.values()) == {0.0}
    assert r1.uniform() == r2.uniform()                                       # exact mode does not shift the stream


# ----------------------------------------------------------------------------------------------- behavioural firewall
@pytest.fixture
def canned_sensors(monkeypatch):
    """Replace the sensor suite by a deterministic generator that does not look at the truth dict at all."""
    import sensors

    def fake(self, truth, omega_ref, bms, ceiling_telemetry=None):
        k = getattr(self, "_k", -1) + 1
        self._k = k
        self.last_delta = 0.0
        return I.Measurements(t=k * 1e-4, i_a_m=0.0, i_b_m=0.0, i_c_m=0.0, i_d_m=0.5 * np.sin(k / 50.0), i_q_m=20.0 + 5.0 * np.sin(k / 80.0),
                              theta_m=0.0, omega_m_m=100.0 + 0.01 * k, vdc_m=290.0 - 0.001 * k, bms=bms, omega_ref=omega_ref)
    monkeypatch.setattr(sensors.SensorSuite, "measure", fake)


CTRL_COLS = ["u_d_ctrl", "u_q_ctrl", "u_d_clip", "u_q_clip", "iq_ref", "psi_hat", "z", "e_q", "i_lim_eff", "ceiling_hat", "theta_adv", "i_ref_clip"]


def test_hidden_plant_state_cannot_change_controller_output_when_measurements_are_equal(canned_sensors):
    quiet, _ = run_fw("a", mode="aw", governor=True, seed=1, t_end=0.1, write=False, omega_ref_fn=WREF,
                      plant=GEMPlant(disturbance_fn=lambda t: 0.0))
    loaded, _ = run_fw("b", mode="aw", governor=True, seed=1, t_end=0.1, write=False, omega_ref_fn=WREF,
                       plant=GEMPlant(disturbance_fn=lambda t: 40.0, fault_fn=None))
    for c in CTRL_COLS:
        assert np.array_equal(quiet[c].to_numpy(), loaded[c].to_numpy()), c
    assert np.abs(quiet["i_q"] - loaded["i_q"]).max() > 0.1                   # the hidden state really differed (non-vacuous)
    assert np.abs(quiet["omega_m"] - loaded["omega_m"]).max() > 1e-3


def test_changing_a_measurement_does_change_controller_output(canned_sensors, monkeypatch):
    import sensors
    base, _ = run_fw("a", mode="naive", seed=1, t_end=0.02, write=False, omega_ref_fn=WREF)
    orig = sensors.SensorSuite.measure

    def shifted(self, *a, **k):
        m = orig(self, *a, **k)
        return dataclasses.replace(m, i_q_m=m.i_q_m + 3.0)

    monkeypatch.setattr(sensors.SensorSuite, "measure", shifted)
    alt, _ = run_fw("a", mode="naive", seed=1, t_end=0.02, write=False, omega_ref_fn=WREF)
    assert np.allclose(alt["e_q"] - base["e_q"], 3.0, atol=1e-12) and not np.array_equal(base["u_q_ctrl"], alt["u_q_ctrl"])


# ----------------------------------------------------------------------------------------------- ledger closure
@pytest.mark.parametrize("kw", [
    dict(mode="naive"), dict(mode="robust", governor=True), dict(mode="aw", governor=True, vdc_aware=True),
    dict(mode="aw", identity=True), dict(mode="aw", ctl_overrides={"d_gain_by_L": True, "clamp_psi": True}),
])
def test_ledger_six_pieces_sum_to_d_u_on_a_short_run(kw):
    df, _ = run_fw("L", seed=3, t_end=0.2, write=False, omega_ref_fn=lambda t: 220.0 * min(t / 0.05, 1.0), **kw)
    for ax in "dq":
        parts = sum(df[f"d_u_{p}_{ax}"] for p in PIECES)
        assert np.abs(parts - df[f"d_u_{ax}"]).max() < 1e-9, (kw, ax)
    assert df["d_u_d"].abs().max() > 0 or kw.get("identity")
    if kw.get("identity"):
        assert df[[f"d_u_{p}_{a}" for p in ("adv", "delay", "rot", "inv") for a in "dq"]].abs().max().max() == 0.0


def test_ledger_closes_on_a_battery_run_with_a_bms_derate_and_the_bus_identities_hold():
    from telemetry import derive, params_from_plant
    from battery import Battery
    import json
    fit = json.loads(FIT.read_text())
    plant = GEMPlant(battery=Battery(**fit["toy_battery_kwargs"]), fault_fn=lambda t: {"idis_scale": 0.02 if 0.1 <= t < 0.2 else 1.0}, tau=1e-4)
    df, man = run_fw("B", mode="aw", governor=True, bms_aware=True, vdc_aware=True, battery=True, fit_path=str(FIT), seed=2, t_end=0.3, write=False,
                     plant=plant, omega_ref_fn=WREF)
    for ax in "dq":
        assert np.abs(sum(df[f"d_u_{p}_{ax}"] for p in PIECES) - df[f"d_u_{ax}"]).max() < 1e-9
    d = derive(df, params_from_plant(plant))
    assert df["bms_limited"].max() == 1.0 and df["V_dc"].min() < df["V_dc"].iloc[0]
    assert d["resid_chain"].abs().max() < 1e-6 and d["resid_du_power"].abs().max() < 1e-6
    assert (df["bms_age_s"] >= 0).all() and df["bms_i_dis_lim"].min() >= 0


def test_hidden_plant_saturation_is_isolated_to_the_plant_clip_piece(monkeypatch):
    """The true ceiling drops to 20 V while the controller still believes ~173 V: the shortfall must show up in the plant-clip piece only."""
    import run_exp as R
    monkeypatch.setattr(R, "T_EVENT", (0.1, 0.15))
    monkeypatch.setattr(R, "U_EVENT", 20.0)
    df, _ = run_fw("S", mode="naive", event=True, seed=0, t_end=0.2, write=False, omega_ref_fn=lambda t: 200.0 * min(t / 0.05, 1.0))
    ev = df[(df.t >= 0.1) & (df.t < 0.15)]
    assert (ev["u_limit"] == 20.0).all() and (df[df.t < 0.1]["u_limit"] == I.U_CAP).all()        # true ceiling dropped, only for the window
    assert ev["sat_plant"].mean() > 0.3 and ev["sat_ctrl"].sum() == 0                              # the controller never saw it coming
    assert ev["d_u_plant_clip_q"].abs().max() > 2.0
    idle = df[df["sat_plant"] == 0]
    assert idle[["d_u_plant_clip_d", "d_u_plant_clip_q"]].abs().max().max() == 0.0               # piece 5 is nonzero only when the plant clipped
    assert (np.hypot(ev["u_d_app"] - ev["u_d_extra"], ev["u_q_app"] - ev["u_q_extra"]) <= 20.0 + 1e-9).all()


# ----------------------------------------------------------------------------------------------- determinism
def _num(df):
    return df.select_dtypes("number")


@pytest.mark.parametrize("kw", [dict(mode="naive"), dict(mode="robust", governor=True), dict(mode="aw", identity=True)])
def test_same_seed_gives_a_bit_identical_frame(kw):
    a, ma = run_fw("D", seed=11, t_end=0.1, write=False, omega_ref_fn=WREF, **kw)
    b, mb = run_fw("D", seed=11, t_end=0.1, write=False, omega_ref_fn=WREF, **kw)
    assert list(a.columns) == list(b.columns) and a.shape == b.shape
    assert _num(a).to_numpy().tobytes() == _num(b).to_numpy().tobytes()
    assert ma["belief_draws"] == mb["belief_draws"] and ma["beliefs"] == mb["beliefs"]


def test_different_seeds_differ_in_beliefs_sensor_errors_and_hence_commands():
    a, ma = run_fw("D", mode="aw", seed=1, t_end=0.1, write=False, omega_ref_fn=WREF)
    b, mb = run_fw("D", mode="aw", seed=2, t_end=0.1, write=False, omega_ref_fn=WREF)
    assert ma["belief_draws"] != mb["belief_draws"] and a["belief_R"].iloc[0] != b["belief_R"].iloc[0]
    for c in ("meas_i_a", "meas_theta", "meas_vdc", "u_q_cmd", "i_q"):
        assert not np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c


def test_seed_zero_run_does_not_depend_on_global_numpy_state():
    np.random.seed(0)
    a, _ = run_fw("D", mode="aw", seed=5, t_end=0.05, write=False, omega_ref_fn=WREF)
    np.random.seed(999)
    np.random.standard_normal(1000)
    b, _ = run_fw("D", mode="aw", seed=5, t_end=0.05, write=False, omega_ref_fn=WREF)
    assert _num(a).to_numpy().tobytes() == _num(b).to_numpy().tobytes()


def test_log_every_decimates_without_changing_the_trajectory():
    full, _ = run_fw("D", mode="aw", seed=6, t_end=0.1, write=False, omega_ref_fn=WREF, log_every=1)
    dec, _ = run_fw("D", mode="aw", seed=6, t_end=0.1, write=False, omega_ref_fn=WREF, log_every=5)
    assert len(dec) == len(full) // 5
    for c in ("i_q", "omega_m", "meas_i_q", "u_q_cmd", "psi_hat"):
        assert np.array_equal(full[c].to_numpy()[::5], dec[c].to_numpy()), c


def test_controller_choice_does_not_change_sensor_noise_realisation_at_step_zero():
    a, _ = run_fw("D", mode="naive", seed=8, t_end=0.01, write=False, omega_ref_fn=WREF)
    b, _ = run_fw("D", mode="aw", governor=True, seed=8, t_end=0.01, write=False, omega_ref_fn=WREF)
    assert a["meas_i_a"].iloc[0] == b["meas_i_a"].iloc[0] and a["meas_vdc"].iloc[0] == b["meas_vdc"].iloc[0]


def test_manifest_records_what_was_run():
    _, m = run_fw("M", mode="robust", governor=True, vdc_aware=True, seed=3, t_end=0.05, write=False, omega_ref_fn=WREF)
    assert (m["mode"], m["governor"], m["vdc_aware"], m["seed"], m["steps"]) == ("robust", True, True, 3, 500)
    assert m["controller_cfg"]["mode"] == "robust" and m["controller_cfg"]["delay_steps"] == m["inverter_cfg"]["delay_steps"] == 1
    assert m["psi_true"] == GEMPlant().psi_p and set(m["belief_draws"]) == {"dR", "dL", "dJ", "dpsi"}


def test_write_true_creates_parquet_and_json(tmp_path):
    import json
    import pandas as pd
    df, m = run_fw("W", mode="aw", seed=1, t_end=0.02, outdir=str(tmp_path / "out"), omega_ref_fn=WREF)
    pq, js = tmp_path / "out" / "W.parquet", tmp_path / "out" / "W.json"
    assert pq.exists() and js.exists()
    back = pd.read_parquet(pq)
    assert back.shape == df.shape and np.array_equal(back["i_q"].to_numpy(), df["i_q"].to_numpy())
    assert json.loads(js.read_text())["seed"] == 1


def test_controller_delay_override_beats_the_inverter_default_but_default_follows_the_inverter():
    _, m = run_fw("D", mode="aw", seed=1, t_end=0.01, write=False, omega_ref_fn=WREF, inv_cfg=I.InverterConfig(delay_steps=2), ctl_overrides={"d_gain_by_L": True})
    assert m["controller_cfg"]["delay_steps"] == 2
    _, m2 = run_fw("D", mode="aw", seed=1, t_end=0.01, write=False, omega_ref_fn=WREF, inv_cfg=I.InverterConfig(delay_steps=2), ctl_overrides={"delay_steps": 0, "d_gain_by_L": True})
    assert m2["controller_cfg"]["delay_steps"] == 0 and m2["inverter_cfg"]["delay_steps"] == 2


def test_unstable_d_loop_and_mismatched_time_steps_are_refused():
    import pytest
    with pytest.raises(ValueError, match="unstable"):
        run_fw("X", mode="aw", seed=1, t_end=0.01, write=False, omega_ref_fn=WREF, inv_cfg=I.InverterConfig(delay_steps=2))
    with pytest.raises(ValueError, match="time step mismatch"):
        run_fw("X", mode="aw", seed=1, t_end=0.01, write=False, omega_ref_fn=WREF, sensor_cfg=I.SensorConfig(tau=2e-4))
    assert I.d_loop_spectral_radius(3.0, 1e-4, 0.00037, 1) < 1 < I.d_loop_spectral_radius(3.0, 1e-4, 0.00037, 2)
    assert I.d_loop_spectral_radius(3.0, 2e-4, 0.00037, 1) > 1                  # the cycle step with the default gain
    assert I.d_loop_spectral_radius(3.0 * 0.00037 / 0.0012, 2e-4, 0.00037, 1) < 1   # ... and with d_gain_by_L
