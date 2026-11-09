"""Tests for bounds.py and the electrical reconstruction. Cheap: synthetic arrays plus the stored 8 s runs."""
import pathlib
import numpy as np
import pandas as pd
import pytest

from bounds import bounds_report
from battery import Battery
from plant import GEMPlant
from telemetry import derive, params_from_plant

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"
P0 = params_from_plant(GEMPlant(battery=Battery()))


def synth(n=2000, dt=1e-3, **over):
    t = np.arange(n) * dt
    d = pd.DataFrame(dict(t=t, u_d_app=np.zeros(n), u_q_app=np.full(n, 50.0), u_limit=np.full(n, 173.2), i_d=np.zeros(n), i_q=np.full(n, 10.0),
                          psi_tilde=np.zeros(n), e_q=np.zeros(n), omega_ref=np.full(n, 100.0), omega_m=np.full(n, 100.0), V=np.ones(n), saturated=np.zeros(n)))
    for k, v in over.items():
        d[k] = v
    return d


def test_input_bound_detects_a_violation():
    """If delivered voltage ever exceeds the ceiling the report must show input_bound > 1 (the by-construction claim is checkable)."""
    d = synth()
    d.loc[100, "u_q_app"] = 200.0
    assert bounds_report(d)["input_bound"] > 1.0
    assert bounds_report(synth())["input_bound"] < 1.0


def test_input_bound_uses_the_instantaneous_ceiling():
    """The ceiling moves with the bus; 60 V delivered against a 50 V ceiling is a violation even though 60 < 173."""
    d = synth(u_limit=np.full(2000, 50.0))
    d["u_q_app"] = 60.0
    assert bounds_report(d)["input_bound"] == pytest.approx(1.2)


def test_ultimate_bound_is_the_final_window_maximum():
    """psi error is large early and 0.01 at the end: the ultimate bound must be 0.01, the peak 0.5."""
    n = 2000
    psi = np.r_[np.full(n - 300, 0.5), np.full(300, 0.01)]
    r = bounds_report(synth(psi_tilde=psi), window_s=0.2)
    assert r["psi_err_peak"] == pytest.approx(0.5) and r["psi_err_ultimate"] == pytest.approx(0.01)


def test_settle_time_when_error_decays_and_stays():
    """After the error falls below its ultimate bound for good, t_settle is the first such sample (not the last sample)."""
    t = np.arange(4000) * 1e-3
    psi = np.where(t < 1.0, 0.4, 0.001)
    r = bounds_report(synth(n=4000, psi_tilde=psi), window_s=0.5)
    assert r["t_settle_psi_err"] == pytest.approx(1.0, abs=0.01)


def test_decay_rate_is_recovered_from_an_exponential():
    """V(t) = e^{-4 t} after a peak: the descriptive fit must return lambda close to 4 (a known answer, not a run)."""
    t = np.arange(4000) * 1e-3
    V = np.exp(-4.0 * t) * 100 + 1e-9
    r = bounds_report(synth(n=4000, V=V), window_s=0.5)
    assert r["lambda_hat"] == pytest.approx(4.0, rel=0.15)


def test_amplification_and_floor():
    V = np.r_[np.ones(500), np.full(500, 50.0), np.full(1000, 2.0)]
    r = bounds_report(synth(V=V), window_s=0.3)
    assert r["V_amplification"] == pytest.approx(50.0) and r["V_floor"] == pytest.approx(2.0)


def test_vdot_sign_split_by_saturation():
    """V rising only while saturated: the unsaturated fraction of positive V-dot must be ~0 and the saturated one ~1."""
    n = 2000
    sat = np.r_[np.zeros(1000), np.ones(1000)]
    V = np.r_[np.full(1000, 1.0), np.linspace(1, 100, 1000)]
    r = bounds_report(synth(V=V, saturated=sat))
    assert r["vdot_pos_frac_unsat"] < 0.01 and r["vdot_pos_frac_sat"] > 0.99


@pytest.mark.skipif(not (RUNS / "B.parquet").exists(), reason="stored runs absent")
@pytest.mark.parametrize("name", ["A", "B", "C", "D", "Dg", "Eb", "Eg"])
def test_stored_runs_have_bounded_input(name):
    """Bounded input is the one boundedness claim that is guaranteed by construction. It must hold in every stored run."""
    d = derive(pd.read_parquet(RUNS / f"{name}.parquet"), P0)
    assert bounds_report(d)["input_bound"] <= 1.0 + 1e-9


@pytest.mark.skipif(not (RUNS / "B.parquet").exists(), reason="stored runs absent")
def test_naive_run_is_not_ultimately_bounded_in_a_small_ball_but_protected_is():
    """The whole point of the protection, as a measured regression: B ends far from the true parameter, D ends on it."""
    b = bounds_report(derive(pd.read_parquet(RUNS / "B.parquet"), P0), window_s=1.0)
    dd = bounds_report(derive(pd.read_parquet(RUNS / "D.parquet"), P0), window_s=1.0)
    assert b["psi_err_ultimate"] > 0.5 and dd["psi_err_ultimate"] < 0.01
    assert b["V_amplification"] > 100 * dd["V_amplification"]


# ------------------------------------------------------------------------------------------- electrical reconstruction
@pytest.mark.skipif(not (RUNS / "B.parquet").exists(), reason="stored runs absent")
@pytest.mark.parametrize("name", ["A", "B", "D", "Dg", "Eb", "Eg"])
def test_electrical_gap_equals_shaft_plus_copper_plus_magnetic(name):
    """dP_elec_ref = dP_mech + dP_cu + P_mag up to the physics residual (P_balance_resid), which is a few tens of W against kW."""
    d = derive(pd.read_parquet(RUNS / f"{name}.parquet"), P0)
    lhs = d["dP_elec_ref"]
    rhs = d["dP_mech"] + d["dP_cu"] + d["P_mag"]
    assert np.sqrt(((lhs - rhs) ** 2).mean()) < 0.01 * np.sqrt((d["P_mech"] ** 2).mean())
    assert np.allclose(lhs - rhs, d["P_balance_resid"], atol=1e-6)


@pytest.mark.skipif(not (RUNS / "B.parquet").exists(), reason="stored runs absent")
@pytest.mark.parametrize("name", ["A", "B", "D", "Eb"])
def test_actuator_power_identity_is_exact(name):
    """dP_elec == 1.5 (du_d i_d + du_q i_q): pure algebra, so it must hold to rounding."""
    d = derive(pd.read_parquet(RUNS / f"{name}.parquet"), P0)
    assert d["resid_du_power"].abs().max() < 1e-6


@pytest.mark.skipif(not (RUNS / "D.parquet").exists(), reason="stored runs absent")
def test_feedback_power_is_the_gain_times_error_product():
    """P_fb is DEFINED as a residual, so check it equals the closed form -1.5 k (i_d^2 + e_q i_q) with k_i = 3 V/A."""
    d = derive(pd.read_parquet(RUNS / "D.parquet"), P0)
    assert np.abs(d["P_fb"] - (-1.5 * 3.0 * (d["i_d"] ** 2 + d["e_q"] * d["i_q"]))).max() < 1e-6


@pytest.mark.skipif(not (RUNS / "D.parquet").exists(), reason="stored runs absent")
def test_voltage_command_gap_splits_into_parameter_error_storage_and_feedback():
    """dP_elec = -1.5 w psi_tilde i_q + P_mag - P_fb, the place where the adaptive error enters the electrical gap."""
    d = derive(pd.read_parquet(RUNS / "D.parquet"), P0)
    assert np.sqrt((d["resid_req_decomp"] ** 2).mean()) < 0.01 * np.sqrt((d["P_mech"] ** 2).mean())


@pytest.mark.skipif(not (RUNS / "A.parquet").exists(), reason="stored runs absent")
def test_disturbance_vector_parts_sum_to_friction_torque():
    d = derive(pd.read_parquet(RUNS / "A.parquet"), P0)
    assert np.allclose(d["T_roll"] + d["T_visc"] + d["T_aero"], d["T_fric"], atol=1e-9)
