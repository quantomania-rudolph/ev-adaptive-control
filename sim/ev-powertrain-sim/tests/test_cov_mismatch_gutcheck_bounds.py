"""Coverage gap tests: mismatch_map.py, data_gutcheck.py and bounds.py edge cases on tiny synthetic frames with known answers."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_gutcheck as G
import mismatch_map as MM
from bounds import _first_stays_below, bounds_report

R_, LD, LQ, PSI, J_ = 0.018, 0.00037, 0.0012, 0.066, 0.54
P0 = dict(p=3, Ld=LD, Lq=LQ, R=R_, psi=PSI, J=J_, tau=0.01, id_ref=0.0, a_load=1.0, b_load=0.02, c_load=0.003)


def raw(n=1001, dt=0.01, **over):
    c = lambda v: np.full(n, float(v))
    d = pd.DataFrame(dict(
        t=np.arange(n) * dt, omega_m=c(100), omega_el=c(300), omega_ref=c(110), i_d=c(0), i_q=c(10), iq_ref=c(12), u_d_cmd=c(5), u_q_cmd=c(50),
        u_d_app=c(4), u_q_app=c(48), d_u_d=c(-1), d_u_q=c(-2), u_limit=c(173.2), T_elec=c(1.5 * 3 * PSI * 10), T_fric=c(1.0), T_dist=c(2.0),
        T_cmd=c(1.5 * 3 * PSI * 12), psi_hat=c(0.05), psi_tilde=c(-0.016), d_u_plant_d=c(0), d_u_plant_q=c(0), i_ref_clip=c(-3), d_T=c(0.5)))
    for k, v in over.items():
        d[k] = v
    return d


def rows_of(rows):
    return {r["mismatch"]: r for r in rows}


# ----------------------------------------------------------------------------------------------- mismatch_map
def test_scales_table_is_complete_and_positive():
    assert set(MM.SCALES) == {"M1", "M2", "M3", "M4", "M5", "M6b", "M6c", "M6d", "M9", "M10"}
    assert all(len(v) == 3 and v[2] > 0 for v in MM.SCALES.values())
    assert MM.T_CAP == pytest.approx(1.5 * 3 * 0.066 * 250.0) and MM.SCALES["M2"][2] == pytest.approx(173.2051)


@pytest.mark.parametrize("man,label", [({"mode": "robust"}, "robust"), ({"mode": "aw"}, "anti-windup"), ({"mode": "aw", "governor": True}, "anti-windup + governor"),
                                       ({"mode": "naive"}, "naive"), ({}, "naive"), ({"mode": "weird"}, "naive")])
def test_controller_label(man, label):
    assert MM.controller_label(man) == label


def test_series_for_known_values_without_battery_columns():
    from telemetry import derive
    s = MM.series_for(derive(raw(n=20), P0))
    assert set(s) == {"M1", "M2", "M3", "M4", "M5", "M6b", "M6c", "M6d"}
    assert (s["M1"] == 10.0).all() and np.allclose(s["M2"], np.sqrt(5)) and (s["M3"] == 0).all() and (s["M4"] == -3).all() and (s["M5"] == 0.5).all()
    assert np.allclose(s["M6b"], 100 * 4.5 * PSI * (10 - 12)) and np.allclose(s["M6c"], s["M6b"] + 1.5 * R_ * (100 - 144)) and np.allclose(s["M6d"], -30.0)


def test_summarize_run_row_statistics_and_index():
    rows, summ = MM.summarize_run("syn", raw(), {"tau_log": 0.01, "mode": "aw"}, P0)
    r = rows_of(rows)
    assert len(rows) == 8 and set(r) == {"M1", "M2", "M3", "M4", "M5", "M6b", "M6c", "M6d"}
    assert r["M1"]["rms"] == pytest.approx(10.0) and r["M1"]["rms_pct"] == pytest.approx(100 * 10 / 190) and r["M1"]["peak"] == 10.0 and r["M1"]["mean"] == 10.0
    assert r["M1"]["active_pct"] == 100.0 and r["M3"]["active_pct"] == 0.0 and r["M3"]["rms"] == 0.0
    assert r["M4"]["rms"] == pytest.approx(3.0) and r["M4"]["mean"] == -3.0 and r["M4"]["peak"] == 3.0 and r["M4"]["rms_pct"] == pytest.approx(100 * 3 / 250)
    assert r["M2"]["rms"] == pytest.approx(np.sqrt(5)) and r["M5"]["rms_pct"] == pytest.approx(100 * 0.5 / MM.T_CAP)
    assert r["M6c"]["mean"] == pytest.approx(-60.588) and r["M6d"]["mean"] == pytest.approx(-30.0)
    core = [r[k]["rms_pct"] for k in ("M1", "M2", "M4", "M5", "M6c")]
    assert summ["index"] == pytest.approx(np.mean(core))
    assert all(x["controller"] == "anti-windup" and x["dataset_class"] == "other" and x["event"] is False for x in rows)


def test_summarize_run_energy_owed_fractions_and_flags():
    _, s = MM.summarize_run("syn", raw(), {"tau_log": 0.01}, P0)
    req = 4.5 * PSI * 12 * 100                                                     # |T_cmd * omega| per row
    assert s["e_shaft_owed_frac"] == pytest.approx(59.4 / req) and s["e_elec_qs_owed_frac"] == pytest.approx(60.588 / req)
    assert s["duration"] == pytest.approx(10.0) and s["event"] is None and s["decimated"] is True
    assert s["data_ratio_gt1_pct"] is None and s["data_ratio_q99"] is None and s["req_sign_conflict_pct"] == 0.0
    _, fine = MM.summarize_run("syn", raw(n=100, dt=1e-4), {"tau_log": 1e-4}, dict(P0, tau=1e-4))
    assert fine["decimated"] is False


def test_attribution_without_events_is_all_remainder():
    _, s = MM.summarize_run("syn", raw(), {"tau_log": 0.01}, P0)
    assert s["m1_share"] == {"data": 0.0, "event": 0.0, "remainder": 1.0}


def _err_frame(lo, hi, **over):
    t = np.arange(1001) * 0.01
    return raw(omega_ref=np.where((t >= lo) & (t < hi), 120.0, 100.0), **over)


def test_attribution_event_window_is_event_plus_three_seconds():
    for man, lo in (({"event": True}, 4.6), ({"derate": [2.0, 1.0]}, 2.2)):
        _, s = MM.summarize_run("syn", _err_frame(lo, lo + 0.3), dict(man, tau_log=0.01), P0)
        assert s["m1_share"]["event"] == pytest.approx(1.0) and s["m1_share"]["data"] == 0.0 and s["m1_share"]["remainder"] == pytest.approx(0.0, abs=1e-12)
    assert s["event"] == [2.0, 3.0]
    _, outside = MM.summarize_run("syn", _err_frame(9.0, 9.5), {"event": True, "tau_log": 0.01}, P0)      # after event + 3 s = 8.5 s
    assert outside["m1_share"] == {"data": 0.0, "event": 0.0, "remainder": 1.0}


def test_attribution_data_window_is_two_seconds_after_an_infeasible_instant_and_beats_event():
    ratio = np.zeros(1001)
    ratio[500] = 2.0                                                              # infeasible at t = 5.0 s
    inside = _err_frame(5.2, 5.5, data_ratio=ratio)                               # error inside [5, 7]
    _, s = MM.summarize_run("syn", inside, {"event": True, "tau_log": 0.01}, P0)
    assert s["m1_share"]["data"] == pytest.approx(1.0) and s["m1_share"]["event"] == 0.0      # data takes precedence over the event window
    _, before = MM.summarize_run("syn", _err_frame(4.6, 4.9, data_ratio=ratio), {"event": True, "tau_log": 0.01}, P0)
    assert before["m1_share"]["data"] == 0.0 and before["m1_share"]["event"] == pytest.approx(1.0)   # before the instant: not data
    _, late = MM.summarize_run("syn", _err_frame(7.5, 7.8, data_ratio=ratio), {"tau_log": 0.01}, P0)
    assert late["m1_share"] == {"data": 0.0, "event": 0.0, "remainder": 1.0}
    assert s["data_ratio_gt1_pct"] == pytest.approx(100 / 1001) and s["data_ratio_q99"] == pytest.approx(0.0)


def test_shares_always_sum_to_one_and_zero_error_run_does_not_divide_by_zero():
    ratio = np.zeros(1001)
    ratio[100] = 3.0
    _, s = MM.summarize_run("syn", raw(omega_ref=100.0, data_ratio=ratio), {"event": True, "tau_log": 0.01}, P0)
    assert sum(s["m1_share"].values()) == pytest.approx(1.0) and all(np.isfinite(list(s["m1_share"].values())))
    _, s2 = MM.summarize_run("syn", _err_frame(1.0, 3.0, data_ratio=ratio), {"event": True, "tau_log": 0.01}, P0)
    assert sum(s2["m1_share"].values()) == pytest.approx(1.0)


def test_battery_columns_add_m9_and_m10():
    n = 1001
    b = raw(i_batt=np.full(n, 40.0), V_dc=np.full(n, 240.0), V_oc=np.full(n, 250.0), i_dc=np.full(n, 35.0), V_rc=np.zeros(n), R0_eff=np.full(n, 0.05), P_bms_clip=np.zeros(n))
    rows, _ = MM.summarize_run("syn", b, {"tau_log": 0.01}, dict(P0, C_dc=2e-3, R_bus=0.01, V_nom_ceiling=173.2051))
    r = rows_of(rows)
    assert r["M9"]["mean"] == -10.0 and r["M10"]["mean"] == 5.0 and r["M9"]["rms_pct"] == pytest.approx(100 * 10 / 300)


def test_legacy_names_get_their_class_and_controller_but_manifest_class_wins_for_unknown_names():
    r, s = MM.summarize_run("A", raw(), {"tau_log": 0.01}, P0)
    assert s["dataset_class"] == "8 s ramp, plain motor" and s["controller"] == "naive, no event"
    r, s = MM.summarize_run("T0p", raw(), {"tau_log": 0.01}, P0)
    assert s["dataset_class"] == "real mixed (Tesla track 0), full 283 s" and s["controller"] == "anti-windup + governor"
    r, s = MM.summarize_run("zzz", raw(), {"tau_log": 0.01, "dataset_class": "custom", "mode": "robust"}, P0)
    assert s["dataset_class"] == "custom" and s["controller"] == "robust"
    r, s = MM.summarize_run("A", raw(), {"tau_log": 0.01, "dataset_class": "override"}, P0)
    assert s["dataset_class"] == "override"                                       # manifest beats the legacy table


def test_tau_falls_back_to_manifest_tau_then_to_P0():
    _, a = MM.summarize_run("syn", raw(), {"tau": 0.01}, dict(P0, tau=0.5))
    _, b = MM.summarize_run("syn", raw(), {"tau_log": 0.01, "tau": 0.5}, dict(P0, tau=0.5))
    assert a["e_shaft_owed_frac"] == pytest.approx(b["e_shaft_owed_frac"])
    _, c = MM.summarize_run("syn", raw(), {}, P0)
    assert c["e_shaft_owed_frac"] == pytest.approx(a["e_shaft_owed_frac"])


def test_build_writes_csv_and_json_skips_measured_and_tolerates_missing_manifest(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    raw().to_parquet(runs / "A.parquet")
    (runs / "A.json").write_text(json.dumps({"tau_log": 0.01, "mode": "aw", "governor": True, "event": True}))
    raw().to_parquet(runs / "B.parquet")                                          # no manifest: tau falls back to P0 (1e-4) and derive warns
    raw().to_parquet(runs / "A_measured.parquet")
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="row spacing"):
        rows, summ = MM.build(str(runs), str(out))
    assert [s["run"] for s in summ] == ["A", "B"] and len(rows) == 16
    csv = pd.read_csv(out / "mismatch_map.csv")
    assert len(csv) == 16 and set(csv["run"]) == {"A", "B"} and set(csv["mismatch"]) == {"M1", "M2", "M3", "M4", "M5", "M6b", "M6c", "M6d"}
    js = json.loads((out / "mismatch_map.json").read_text())
    assert [r["run"] for r in js["runs"]] == ["A", "B"] and js["scales"]["M1"] == ["speed shortfall", "rad/s", 190.0]
    assert next(r for r in js["runs"] if r["run"] == "A")["event"] == [4.5, 5.5]


def test_build_on_an_empty_directory_writes_empty_outputs(tmp_path):
    (tmp_path / "runs").mkdir()
    rows, summ = MM.build(str(tmp_path / "runs"), str(tmp_path / "out"))
    assert rows == [] and summ == []
    assert json.loads((tmp_path / "out" / "mismatch_map.json").read_text())["runs"] == []


# ----------------------------------------------------------------------------------------------- data_gutcheck.runs_of_equal
@pytest.mark.parametrize("x,expect", [
    ([1, 1, 1, 2, 2], (0, 3)), ([1, 2, 3], (0, 1)), ([5.0], (0, 1)), ([], (0, 1)), ([2, 1, 1, 1, 1, 3, 3], (1, 4)),
    ([1, 1, np.nan, np.nan, 1, 1, 1], (4, 3)),            # NaN is never equal to NaN: it breaks a run rather than extending it
    ([np.nan] * 4, (0, 1)), ([0, 0, 0, 0], (0, 4)), ([1, 1, 2, 2], (0, 2)),   # tie: the first longest run wins
])
def test_runs_of_equal_known_answers(x, expect):
    assert G.runs_of_equal(x) == expect


def test_runs_of_equal_compares_bit_identical_not_approximate():
    assert G.runs_of_equal([1.0, 1.0 + 1e-15, 1.0 + 1e-15, 1.0 + 1e-15]) == (1, 3)


# ----------------------------------------------------------------------------------------------- data_gutcheck.scan_sim
def sim_frame(n=400, dt=0.01):
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    return pd.DataFrame(dict(
        t=t, omega_m=np.full(n, 50.0) + rng.standard_normal(n),                       # moving, noisy: ok
        const=np.full(n, 3.0), sat_flag=(rng.random(n) > 0.5).astype(float), saturated=np.zeros(n) + (np.arange(n) % 90 == 0),
        i_q_stuck0=np.where(t < 2.0, 0.0, rng.standard_normal(n)),                   # held at 0 for 2 s while moving
        i_q_stuck7=np.where(t < 2.0, 7.0, rng.standard_normal(n)),                   # held at 7 for 2 s while moving
        d_u_q=np.where(t < 2.0, 0.0, rng.standard_normal(n)),                        # ledger-like: zero when inactive
        i_lim_eff=np.where(t < 2.0, 250.0, rng.standard_normal(n)),                  # allowed plateau
        allnan=np.full(n, np.nan), label=["x"] * n))


def verdicts(df, **kw):
    s = G.scan_sim(df, "r", **kw)
    return dict(zip(s["col"], s["verdict"])), s


def test_scan_sim_verdicts_for_each_class_of_channel():
    v, s = verdicts(sim_frame())
    assert v["omega_m"] == "ok" and v["const"] == "constant-channel" and v["sat_flag"] == "flag" and v["saturated"] == "flag"
    assert v["i_q_stuck0"] == "ZERO-WHILE-MOVING" and v["i_q_stuck7"] == "SUSPECT"
    assert v["d_u_q"] == "zero-when-inactive" and v["i_lim_eff"] == "limit-plateau" and v["allnan"] == "ALL-NaN"
    assert "t" not in v and "label" not in v                                       # time and non-numeric columns are skipped


def test_scan_sim_reports_hold_length_and_value():
    _, s = verdicts(sim_frame())
    r = s[s["col"] == "i_q_stuck7"].iloc[0]
    assert r["held_value_moving"] == 7.0 and r["longest_hold_moving_s"] == pytest.approx(2.0, abs=0.02) and r["hold_t0_moving"] == 0.0
    assert r["n_unique"] > 100


def test_scan_sim_only_counts_holds_while_moving():
    df = sim_frame()
    df["omega_m"] = np.where(df["t"] < 2.0, 0.0, 50.0)                              # standing still exactly while the channel is held
    v, s = verdicts(df)
    assert v["i_q_stuck7"] == "ok" and v["i_q_stuck0"] == "ok"
    assert s[s["col"] == "i_q_stuck7"].iloc[0]["longest_hold_s"] == pytest.approx(2.0, abs=0.02)      # the plain hold is still recorded


def test_scan_sim_min_hold_threshold():
    df = pd.DataFrame(dict(t=np.arange(100) * 0.01, omega_m=np.full(100, 9.0), x=np.r_[np.full(10, 4.0), np.arange(90) + 100.0]))
    assert verdicts(df, min_hold_s=0.5)[0]["x"] == "ok"                             # 0.10 s hold below 0.5 s
    assert verdicts(df, min_hold_s=0.05)[0]["x"] == "SUSPECT"


def test_scan_sim_without_omega_m_treats_every_row_as_moving():
    df = sim_frame().drop(columns="omega_m")
    assert verdicts(df)[0]["i_q_stuck7"] == "SUSPECT"


def test_scan_sim_integer_columns_are_scanned_too():
    df = pd.DataFrame(dict(t=np.arange(50) * 0.01, omega_m=np.full(50, 5.0), k=np.arange(50)))
    assert verdicts(df)[0]["k"] == "ok"


# ----------------------------------------------------------------------------------------------- data_gutcheck.scan_tum
def test_scan_tum_known_update_rates_and_holds(tmp_path):
    n = 100
    p = tmp_path / "trk.csv"
    pd.DataFrame(dict(time=np.arange(n) * 0.1, fast=np.arange(n, dtype=float), slow=np.repeat(np.arange(n // 5, dtype=float), 5), empty=np.full(n, np.nan),
                      half_nan=np.where(np.arange(n) < 50, np.nan, np.arange(n, dtype=float)), txt=["a"] * n)).to_csv(p, index=False)
    out = G.scan_tum(p, "trk").set_index("col")
    assert set(out.index) == {"fast", "slow", "half_nan"}                           # time, text and all-NaN are skipped
    assert out.loc["fast", "frac_equal_prev"] == 0.0 and out.loc["fast", "eff_update_hz"] == 10.0 and out.loc["fast", "longest_hold_s"] == pytest.approx(0.1)
    assert out.loc["slow", "frac_equal_prev"] == pytest.approx(80 / 99, abs=1e-3) and out.loc["slow", "longest_hold_s"] == pytest.approx(0.5)
    assert out.loc["slow", "eff_update_hz"] == pytest.approx(10 * (1 - 80 / 99), abs=0.01) and out.loc["slow", "n_unique"] == 20
    assert out.loc["half_nan", "nan_frac"] == 0.5 and (out["track"] == "trk").all()


# ----------------------------------------------------------------------------------------------- bounds.py edges
def synth(n=2000, dt=1e-3, **over):
    t = np.arange(n) * dt
    d = pd.DataFrame(dict(t=t, u_d_app=np.zeros(n), u_q_app=np.full(n, 50.0), u_limit=np.full(n, 173.2), i_d=np.zeros(n), i_q=np.full(n, 10.0),
                          psi_tilde=np.zeros(n), e_q=np.zeros(n), omega_ref=np.full(n, 100.0), omega_m=np.full(n, 100.0), V=np.ones(n), saturated=np.zeros(n)))
    for k, v in over.items():
        d[k] = v
    return d


def test_first_stays_below_known_answers():
    t = np.arange(6.0)
    assert _first_stays_below(t, np.array([0, 0, 0, 0, 0, 0.0]), 1.0) == 0.0               # never over: settled from the start
    assert _first_stays_below(t, np.array([5, 0, 5, 0, 0, 0.0]), 1.0) == 3.0               # last excursion at index 2
    assert _first_stays_below(t, np.array([0, 0, 0, 0, 0, 5.0]), 1.0) is None              # still over at the final sample
    assert _first_stays_below(t, np.array([1.0] * 6), 1.0) == 0.0                           # equal to the level is not "over"


def test_event_keys_and_post_event_peak():
    n = 2000
    psi = np.zeros(n)
    psi[300:400] = 0.2                                                          # during the event window [0.3, 0.4]
    psi[900] = 0.05                                                             # after it
    r = bounds_report(synth(psi_tilde=psi), event=(0.3, 0.4))
    assert r["t_event"] == [0.3, 0.4] and r["psi_err_after_event_peak"] == pytest.approx(0.05) and r["psi_err_peak"] == pytest.approx(0.2)
    assert "t_event" not in bounds_report(synth())
    assert bounds_report(synth(), event=(5.0, 6.0))["psi_err_after_event_peak"] is None     # event after the end of the log


def test_default_window_is_ten_percent_of_the_run_or_five_samples():
    assert bounds_report(synth(n=2000))["window_s"] == pytest.approx(0.1 * 1.999)
    r = bounds_report(synth(n=20))
    assert r["window_s"] == pytest.approx(5e-3)                                  # 10 % of 19 ms is below five samples
    assert bounds_report(synth(), window_s=0.25)["window_s"] == 0.25


def test_flat_run_has_no_decay_rate_and_unit_amplification():
    r = bounds_report(synth())
    assert r["lambda_hat"] is None and r["V_amplification"] == 1.0 and r["V_floor"] == 1.0 and r["t_settle_psi_err"] == 0.0
    assert r["vdot_pos_frac_sat"] is None and r["vdot_pos_frac_unsat"] == 0.0


def test_zero_lyapunov_does_not_divide_by_zero():
    r = bounds_report(synth(V=np.zeros(2000)))
    assert r["V_amplification"] == 0.0 and r["lambda_hat"] is None and np.isfinite(r["V_floor"])


def test_all_saturated_run_reports_only_the_saturated_vdot_split():
    r = bounds_report(synth(saturated=np.ones(2000)))
    assert r["vdot_pos_frac_unsat"] is None and r["vdot_pos_frac_sat"] == 0.0


def test_growing_lyapunov_function_is_flagged_by_the_positive_vdot_fraction():
    t = np.arange(2000) * 1e-3
    r = bounds_report(synth(V=np.exp(2 * t)))
    assert r["vdot_pos_frac_unsat"] == pytest.approx(1.0) and r["lambda_hat"] is None and r["V_amplification"] == pytest.approx(np.exp(2 * t[-1]))


def test_current_and_speed_peaks_use_the_magnitude_and_absolute_error():
    r = bounds_report(synth(i_d=np.full(2000, -30.0), i_q=np.full(2000, 40.0), omega_m=np.full(2000, 130.0), e_q=np.full(2000, -2.0)), i_lim=100.0)
    assert r["i_peak"] == pytest.approx(50.0) and r["i_over_limit"] == pytest.approx(0.5)
    assert r["e_speed_peak"] == pytest.approx(30.0) and r["e_q_ultimate"] == pytest.approx(2.0)


def test_ceiling_range_and_input_peak():
    r = bounds_report(synth(u_limit=np.linspace(100.0, 173.2, 2000), u_q_app=np.full(2000, 60.0), u_d_app=np.full(2000, 80.0)))
    assert r["input_peak_v"] == pytest.approx(100.0) and r["ceiling_min_v"] == pytest.approx(100.0) and r["ceiling_max_v"] == pytest.approx(173.2)
    assert r["input_bound"] == pytest.approx(1.0)                                    # |u| = 100 against a 100 V ceiling at t = 0


def test_one_row_frame_raises_valueerror():
    with pytest.raises(ValueError):
        bounds_report(synth(n=1))


def test_two_row_frame_is_the_minimum_that_reports():
    r = bounds_report(synth(n=2))
    assert r["input_bound"] == pytest.approx(50 / 173.2) and r["window_s"] == pytest.approx(5e-3)


def test_missing_required_column_raises_keyerror():
    with pytest.raises(KeyError):
        bounds_report(synth().drop(columns=["psi_tilde"]))
