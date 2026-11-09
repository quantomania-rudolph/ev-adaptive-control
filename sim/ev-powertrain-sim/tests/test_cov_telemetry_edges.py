"""Coverage gap tests: telemetry.derive on edge inputs with known answers, catalog hygiene, and the two-state identities."""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telemetry import CATALOG, catalog_frame, derive, params_from_plant

R_, LD, LQ, PSI, J_ = 0.018, 0.00037, 0.0012, 0.066, 0.54
P = dict(p=3, Ld=LD, Lq=LQ, R=R_, psi=PSI, J=J_, tau=1e-4, id_ref=0.0, a_load=1.0, b_load=0.02, c_load=0.003)


def raw(n=6, dt=1e-4, **over):
    """Constant, hand-checkable operating point (no motion in the derivatives)."""
    c = lambda v: np.full(n, float(v))
    d = pd.DataFrame(dict(
        t=np.arange(n) * dt, omega_m=c(100), omega_el=c(300), i_d=c(0), i_q=c(10), u_d_cmd=c(5), u_q_cmd=c(50), u_d_app=c(4), u_q_app=c(48),
        d_u_d=c(-1), d_u_q=c(-2), u_limit=c(173.2), T_elec=c(1.5 * 3 * PSI * 10), T_fric=c(1.0), T_dist=c(2.0), omega_ref=c(110), iq_ref=c(12),
        T_cmd=c(1.5 * 3 * PSI * 12), psi_hat=c(0.05), psi_tilde=c(-0.016)))
    for k, v in over.items():
        d[k] = v
    return d


def test_catalog_names_are_unique_and_every_row_is_complete():
    names = [c[0] for c in CATALOG]
    assert len(names) == len(set(names))
    assert all(len(c) == 5 and all(isinstance(x, str) and x for x in c) for c in CATALOG)
    f = catalog_frame()
    assert list(f.columns) == ["name", "unit", "group", "role", "description"] and len(f) == len(CATALOG)


def test_params_from_plant_matches_nameplate():
    from battery import Battery
    from plant import GEMPlant
    pl = GEMPlant(battery=Battery())
    p = params_from_plant(pl)
    assert (p["p"], p["Ld"], p["Lq"], p["R"], p["psi"], p["J"]) == (pl.p, pl.Ld, pl.Lq, pl.Rs, pl.psi_p, pl.J)
    assert p["C_dc"] == pl.battery.C and p["R_bus"] == pl.battery.Rc and p["V_nom_ceiling"] == pl.u_max_hw
    assert "C_dc" not in params_from_plant(GEMPlant())


def test_derive_constant_operating_point_known_answers():
    d = derive(raw(), P)
    assert d["lam_q"].iloc[0] == pytest.approx(LQ * 10) and d["lam_d"].iloc[0] == pytest.approx(PSI)
    assert d["bemf"].iloc[0] == pytest.approx(300 * PSI)
    assert d["v_demand_mag"].iloc[0] == pytest.approx(np.hypot(5, 50)) and d["v_headroom"].iloc[0] == pytest.approx(173.2 - np.hypot(5, 50))
    assert d["f_iq"].iloc[0] == pytest.approx((-R_ * 10 - 300 * PSI) / LQ) and d["f_id"].iloc[0] == pytest.approx(300 * LQ * 10 / LD)
    assert d["d_w"].iloc[0] == pytest.approx(-2.0 / J_) and d["d_w_fric"].iloc[0] == pytest.approx(-1.0 / J_)
    assert d["d_iq"].iloc[0] == pytest.approx(-2.0 / LQ) and d["d_id"].iloc[0] == pytest.approx(-1.0 / LD)
    assert d["e_speed"].iloc[0] == 10.0
    assert d["P_req_elec"].iloc[0] == pytest.approx(750.0) and d["P_del_elec"].iloc[0] == pytest.approx(720.0) and d["dP_elec"].iloc[0] == pytest.approx(-30.0)
    assert d["P_cu"].iloc[0] == pytest.approx(1.5 * R_ * 100)
    assert d["resid_du_power"].abs().max() < 1e-9                         # dP = 1.5 (du_d i_d + du_q i_q) exactly
    assert d["P_mag"].abs().max() == 0.0 and d["P_kin"].abs().max() == 0.0     # constant currents/speed: no derivatives
    assert d["P_mech"].iloc[0] == pytest.approx(1.5 * 3 * PSI * 10 * 100)
    assert d["ts_z1"].iloc[0] == -10.0 and d["ts_z2"].iloc[0] == -2.0
    assert d["E_del"].iloc[-1] == pytest.approx(6 * 720.0 * 1e-4) and d["E_unserved"].iloc[-1] == pytest.approx(6 * 30.0 * 1e-4)
    assert d["T_visc"].iloc[0] == pytest.approx(0.02 * 100) and d["T_aero"].iloc[0] == pytest.approx(0.003 * 100 * 100)
    assert d["T_roll"].iloc[0] == pytest.approx(1.0 - 2.0 - 30.0)          # friction minus viscous minus aero (a synthetic, not physical, split)


def test_derive_does_not_mutate_its_input_and_adds_no_battery_columns_without_battery_data():
    r = raw()
    before = r.copy()
    d = derive(r, P)
    pd.testing.assert_frame_equal(r, before)
    assert not {"bus_sag", "ceiling_loss", "d_i_dc", "resid_chain"} & set(d.columns)
    assert "data_infeasible" not in d.columns
    no_load = derive(raw(), {k: v for k, v in P.items() if k not in ("a_load", "b_load", "c_load")})
    assert "T_aero" not in no_load.columns


def test_derive_needs_at_least_two_rows_one_row_raises_valueerror():
    with pytest.raises(ValueError):
        derive(raw(n=1), P)
    d2 = derive(raw(n=2), P)                                              # two rows is the minimum that works
    assert len(d2) == 2 and d2["E_del"].iloc[-1] == pytest.approx(2 * 720.0 * 1e-4)


def test_derive_empty_frame_raises_instead_of_returning_garbage():
    with pytest.raises((ValueError, KeyError, IndexError)):
        derive(raw(n=0), P)


def test_missing_iq_ref_column_gives_nan_ts_z2_instead_of_a_crash():
    d = derive(raw().drop(columns=["iq_ref"]), P)
    assert d["ts_z2"].isna().all()


@pytest.mark.parametrize("missing", ["psi_hat", "T_cmd", "u_limit", "d_u_q"])
def test_derive_missing_required_columns_raises_keyerror_naming_the_column(missing):
    with pytest.raises(KeyError) as e:
        derive(raw().drop(columns=[missing]), P)
    assert missing in str(e.value)


def test_row_spacing_wins_over_P_tau_and_warns():
    r = raw(dt=5e-3)
    with pytest.warns(UserWarning, match="row spacing"):
        d = derive(r, P)
    assert d.attrs["tau_used"] == pytest.approx(5e-3)
    assert d.attrs["derivative_channels_valid"] is False                  # 5 ms rows cannot resolve the current dynamics
    assert d["E_del"].iloc[-1] == pytest.approx(6 * 720.0 * 5e-3)         # the integral used the row spacing, not P['tau']


def test_matching_tau_is_silent_and_marks_derivatives_valid():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = derive(raw(dt=1.005e-4), P)                                   # within 1 %
    assert d.attrs["derivative_channels_valid"] is True and d.attrs["tau_used"] == P["tau"]


def test_valid_flag_threshold_is_four_tenths_of_a_millisecond():
    assert derive(raw(dt=4e-4), dict(P, tau=4e-4)).attrs["derivative_channels_valid"] is True
    assert derive(raw(dt=4.2e-4), dict(P, tau=4.2e-4)).attrs["derivative_channels_valid"] is False


def test_nonuniform_time_axis_uses_the_median_spacing():
    r = raw(n=7)
    t = r["t"].to_numpy().copy()
    t[4:] += 1e-3                                                         # one 1 ms hole in an otherwise 0.1 ms grid
    r["t"] = t
    d = derive(r, P)                                                      # median spacing is still 1e-4, so no warning and tau = P['tau']
    assert d.attrs["tau_used"] == pytest.approx(1e-4)
    assert d["E_del"].iloc[-1] == pytest.approx(7 * 720 * 1e-4)           # the hole is NOT accounted for: documents the limitation


def test_battery_columns_branch_known_answers():
    n = 6
    from interfaces import U_CAP
    Pb = dict(P, C_dc=2e-3, R_bus=0.01, V_nom_ceiling=U_CAP)
    r = raw(n, i_batt=np.full(n, 40.0), V_dc=np.full(n, 240.0), V_oc=np.full(n, 250.0), i_dc=np.full(n, 35.0), V_rc=np.full(n, 3.0),
            R0_eff=np.full(n, 0.05), P_bms_clip=np.full(n, 0.0))
    d = derive(r, Pb)
    assert (d["bus_sag"] == -10.0).all()
    assert d["ceiling_loss"].iloc[0] == pytest.approx(240.0 / np.sqrt(3) - U_CAP)
    assert (d["d_i_dc"] == 5.0).all()
    assert d["E_bus_unserved"].iloc[-1] == pytest.approx(240.0 * (35 - 40) * 6 * 1e-4)
    assert d["P_chem"].iloc[0] == 250.0 * 40
    assert d["P_batt_loss"].iloc[0] == pytest.approx(3.0 * 40 + (0.05 + 0.01) * 1600)
    assert d["P_bus_store"].iloc[0] == 0.0                                # diff().fillna(0) on the first row, constant V afterwards
    assert d["resid_chain"].iloc[0] == pytest.approx(250 * 40 - d["P_batt_loss"].iloc[0] - 240 * 40)
    assert d["resid_bus"].iloc[0] == pytest.approx(240 * 5)
    full = derive(r.assign(V_dc=300.0), Pb)
    assert (full["ceiling_loss"] == 0.0).all()                            # healthy bus: no headroom lost (min(173.2, V_nom) - V_nom)


def test_data_infeasible_flag_is_strictly_above_one():
    d = derive(raw(data_ratio=np.array([0.5, 1.0, 1.0000001, 2.0, np.nan, 0.0])), P)
    assert d["data_infeasible"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]        # NaN compares False: not flagged


def test_req_sign_conflict_needs_both_powers_above_10_w_and_opposite_signs():
    # P_req_elec = 1.5*(u_q_cmd*i_q); P_req_ref = 1.5 R iq*^2 + T_cmd w_m
    r = raw(u_q_cmd=np.array([-50, -50, -0.1, 50, -50, -50.0]), i_q=10.0, iq_ref=np.array([12, 12, 12, 12, 0, 12.0]),
            T_cmd=np.array([1.5 * 3 * PSI * 12] * 4 + [0.0, 1.5 * 3 * PSI * 12]))
    d = derive(r, P)
    assert d["P_req_elec"].iloc[0] < 0 < d["P_req_ref"].iloc[0]
    assert d["req_sign_conflict"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]        # small |P| (row 2), same sign (row 3), ref ~ 0 (row 4)


# ----------------------------------------------------------------------------------------------- two-state identities on a short run
@pytest.fixture(scope="module")
def short_fw():
    from firewalled_loop import run_fw
    df, man = run_fw("cov", mode="aw", governor=True, seed=4, t_end=0.3, write=False,
                     omega_ref_fn=lambda t: 200.0 * min(t / 0.1, 1.0))
    from battery import Battery
    from plant import GEMPlant
    return df, derive(df, dict(params_from_plant(GEMPlant()), tau=man["tau"]))


def test_ts_regrouping_columns_equal_their_definitions_row_by_row(short_fw):
    df, d = short_fw
    assert np.allclose(d["ts_d1"], d["d_w"] + d["ts_d1_red"], atol=1e-12)
    assert np.allclose(d["ts_d2"], d["d_u_q"] / LQ_P(df) + d["ts_d2_red"], atol=1e-9)
    assert np.allclose(d["ts_z1"], d["omega_m"] - d["omega_ref"]) and np.allclose(d["ts_z2"], d["i_q"] - d["iq_ref"])
    assert np.allclose(d["ts_z2"] - d["e_q"], -d["gap_i_q"], atol=1e-9)    # e_q is built from the MEASURED i_q, ts_z2 from the true one
    assert (d["ts_g2"] == 1.0 / LQ_P(df)).all()


def test_ts_z2_differs_from_e_q_by_exactly_the_current_measurement_gap(short_fw):
    _, d = short_fw
    assert np.allclose(d["e_q"] - d["ts_z2"], d["gap_i_q"], atol=1e-9)        # e_q uses the measured current, ts_z2 the true one


def LQ_P(df):
    from plant import GEMPlant
    return GEMPlant().Lq


def test_ts_residuals_equal_the_three_state_residuals(short_fw):
    _, d = short_fw
    assert np.abs(d["ts_res_iq"] - d["resid_iq"]).max() < 1e-6
    assert np.abs(d["ts_res_w"] - d["resid_w"]).max() < 1e-9


def test_ts_f1_plus_d1_plus_g1_iq_reproduces_net_acceleration(short_fw):
    from plant import GEMPlant
    _, d = short_fw
    lhs = d["ts_f1"] + d["ts_g1"] * d["i_q"] + d["ts_d1"]
    assert np.allclose(lhs, (d["T_elec"] - d["T_fric"] - d["T_dist"]) / GEMPlant().J, rtol=1e-9, atol=1e-9)
    assert d["i_d"].abs().max() > 0 and d["ts_d1_red"].abs().max() > 0       # non-vacuous: reluctance term is exercised


def test_two_state_reduction_vanishes_when_i_d_is_zero():
    d = derive(raw(i_d=0.0), P)
    assert (d["ts_d1_red"] == 0.0).all() and (d["ts_d2_red"] == 0.0).all()
    assert np.allclose(d["ts_d1"], d["d_w"]) and np.allclose(d["ts_d2"], d["d_u_q"] / LQ)


def test_two_state_reduction_terms_have_the_documented_closed_form():
    d = derive(raw(i_d=-20.0), P)
    assert d["ts_d1_red"].iloc[0] == pytest.approx(1.5 * 3 * (LD - LQ) * (-20.0) * 10.0 / J_)
    assert d["ts_d2_red"].iloc[0] == pytest.approx(-300 * LD * (-20.0) / LQ)
    assert d["ts_f2"].iloc[0] == pytest.approx((-R_ * 10 - 300 * PSI) / LQ) and d["ts_f1"].iloc[0] == pytest.approx(-1.0 / J_)
    assert d["ts_g1"].iloc[0] == pytest.approx(1.5 * 3 * PSI / J_)
