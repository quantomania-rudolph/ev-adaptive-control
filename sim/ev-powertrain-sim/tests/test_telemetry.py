"""Gate for the data product: catalog and derive() must agree, and the physics checks must hold on a short run."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from plant import GEMPlant
from control import CascadeController
from telemetry import derive, params_from_plant, CATALOG


def _short_run():
    from battery import Battery
    pl = GEMPlant(disturbance_fn=lambda t: 5.0 if t > 0.1 else 0.0, battery=Battery(), fault_fn=lambda t: {"idis_scale": 0.004 if 0.3 < t < 0.4 else 1.0})
    ctl = CascadeController(pl, mode="aw")
    rows, led = [], {}
    for _ in range(6000):
        st = pl.state()
        u, it = ctl.command(min(st["t"] * 50, 40.0), st, led)
        led = pl.apply_voltage(u)
        T = 1.5 * pl.p * pl.psi_p * it["iq_ref"]
        rows.append({**st, **it, **led, "T_cmd": T, "d_T": st["T_elec"] - T,
                     "dist_i_limit": 1 - np.hypot(st["i_d"], st["i_q"]) / 400, "V": 0.0})
    return derive(pd.DataFrame(rows), params_from_plant(pl))


FW_ONLY = {c[0] for c in CATALOG if c[3] in ("measured", "error", "belief", "internal", "ledger")} | {"sat_ctrl", "sat_plant", "v_ref_mps"}


def test_every_catalog_channel_exists_and_is_finite_in_a_firewalled_run():
    from firewalled_loop import run_fw
    df, _ = run_fw("t", mode="aw", battery=True, fit_path="data/tum_battery_fit.json", t_end=0.3, write=False, seed=2)
    d = derive(df, params_from_plant(__import__("plant").GEMPlant(battery=__import__("battery").Battery())))
    cycle_only = {"T_req_data", "T_avail", "data_ratio", "data_infeasible", "v_ref_mps"}
    for name, *_ in CATALOG:
        if name in cycle_only:
            continue
        assert name in d.columns, name
        assert np.isfinite(d[name]).all(), name


def test_every_catalog_channel_exists_and_is_finite():
    d = _short_run()
    cycle_only = {"T_req_data", "T_avail", "data_ratio", "data_infeasible"}          # logged by run_cycle.py, absent in a bare plant run
    for name, *_ in CATALOG:
        if name in cycle_only or name in FW_ONLY or name in {"V_e", "V_theta", "V_aw"}:      # V_* are added by the run loops
            continue
        assert name in d.columns, name
        assert np.isfinite(d[name]).all(), name


def test_physics_checks_hold():
    d = _short_run().iloc[50:-50]                      # skip finite-difference edges
    assert np.sqrt((d.resid_w ** 2).mean()) / np.sqrt((d.f_w ** 2).mean()) < 0.02
    # ~17 W of discretisation error is constant; on this low-power 0.6 s run that is 3 %, on the 8 s runs 0.3 %.
    assert np.sqrt((d.P_balance_resid ** 2).mean()) / np.sqrt((d.P_mech ** 2).mean()) < 0.05
    assert d.d_u_plant_d.abs().max() < 1e-6 and d.d_u_plant_q.abs().max() < 1e-6


def test_battery_books_close_and_derate_collapses_the_bus():
    d = _short_run().iloc[50:-50]
    assert d.resid_chain.abs().max() < 1e-6                       # pack energy chain is exact by construction
    assert np.sqrt((d.resid_bus ** 2).mean()) < 1e-3 * np.sqrt((d.P_dc ** 2).mean())   # current balance on the bus
    ev = d[(d.t > 0.3) & (d.t < 0.4)]
    assert ev.bms_limited.max() == 1.0 and ev.V_dc.min() < 0.8 * d.V_dc.iloc[0]


def test_two_state_regrouping_is_exact_and_residual_is_discretization_sized():
    """x2 = [omega, i_q]: f1 + g1 i_q + d1 and f2 + g2 u_q + d2_red must reproduce the measured forward differences to within
    discretization error (a few percent of the signal), and must equal the three-state residuals row by row."""
    import json, pathlib
    import numpy as np, pandas as pd
    root = pathlib.Path(__file__).resolve().parents[1]
    stem = root / "data/audit/fr_battery_sag_awg"
    if not stem.with_suffix(".parquet").exists():
        import pytest; pytest.skip("full-rate audit file not present")
    from plant import GEMPlant
    from battery import Battery
    from telemetry import derive, params_from_plant
    man = json.loads(stem.with_suffix(".json").read_text())
    d = derive(pd.read_parquet(stem.with_suffix(".parquet")).iloc[:20000], dict(params_from_plant(GEMPlant(battery=Battery())), tau=man["tau"]))
    assert np.abs(d.ts_res_iq - d.resid_iq).max() < 1e-6 and np.abs(d.ts_res_w - d.resid_w).max() < 1e-9
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x)[:-1] ** 2)))
    fwd = lambda c: np.r_[np.diff(d[c].to_numpy()) / man["tau"], 0.0]
    assert rms(d.ts_res_iq) < 0.03 * rms(fwd("i_q"))
    assert rms(d.ts_res_w) < 0.03 * rms(fwd("omega_m"))
    assert np.allclose(d.ts_z1, d.omega_m - d.omega_ref) and np.allclose(d.ts_z2, d.i_q - d.iq_ref)
