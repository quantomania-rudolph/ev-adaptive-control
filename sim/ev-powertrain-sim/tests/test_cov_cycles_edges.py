"""Coverage gap tests: cycles.py loader error paths, smoothing, grade, vehicle mapping and the TUM adapter on malformed or tiny inputs."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cycles import (G, KMH, MPH, Cycle, CycleError, VehicleEq, data_demand, fit_vehicle_to_motor, load_cycle, load_tum_tesla, read_table,
                    synthetic_cycle)


def tr(n=200, dt=1.0, vmax=12.0):
    t = np.arange(n) * dt
    return t, vmax * np.sin(np.pi * t / t[-1]) ** 2


def err(fn, *a, **k):
    with pytest.raises(CycleError) as e:
        fn(*a, **k)
    return e.value.code


# ----------------------------------------------------------------------------------------------- malformed input
def test_unknown_time_unit_is_refused_with_its_own_code():
    t, v = tr()
    assert err(load_cycle, (t, v), time_unit="hours") == "bad_time_unit"
    assert err(load_cycle, (t, v), speed_unit="knots") == "bad_speed_unit"


def test_utf16_with_byte_order_mark_is_decoded_and_reported():
    t, v = tr(60)
    body = "time,speed\n" + "\n".join(f"{a:.3f},{b:.4f}" for a, b in zip(t, v))
    c = load_cycle(body.encode("utf-16"), min_duration_s=10)
    assert "encoding_fallback" in c.codes() and c.report[0]["msg"] == "decoded as utf-16"
    assert c.v.max() == pytest.approx(v.max(), rel=0.02)


def test_text_with_no_numeric_rows_is_refused():
    assert err(load_cycle, b"alpha,beta\ngamma,delta\nepsilon,zeta") == "no_numeric_rows"
    assert err(load_cycle, b"time,speed\n1,x\n2,y") == "no_numeric_rows"             # only one numeric-looking column is not enough


def test_single_data_row_and_one_valid_timestamp_are_refused():
    assert err(load_cycle, b"time,speed\n0,5") == "too_few_samples"
    df = pd.DataFrame({"time": [0.0, np.nan, np.nan], "speed": [1.0, 2.0, 3.0]})
    assert err(load_cycle, df) == "too_few_samples"


def test_dataframe_with_no_rows_is_refused():
    assert err(load_cycle, pd.DataFrame({"time": [], "speed": []})) == "file_empty"


def test_rows_without_timestamp_are_dropped_and_counted():
    t, v = tr(100)
    df = pd.DataFrame({"time": t, "speed": v})
    df.loc[[10, 50, 51], "time"] = np.nan
    c = load_cycle(df)
    rec = next(r for r in c.report if r["code"] == "rows_without_time")
    assert rec["n"] == 3 and rec["level"] == "fix" and c.t[0] == 0.0


def test_named_columns_that_are_the_same_column_are_refused():
    df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)})
    assert err(load_cycle, df, time_col="a", speed_col="a") == "column_ambiguous"
    assert err(load_cycle, df, time_col="nope", speed_col="b") == "column_missing"


def test_two_columns_with_unrecognisable_names_use_the_positional_fallback():
    t, v = tr(100)
    c = load_cycle(pd.DataFrame({"foo": t, "bar": v}))
    assert "two_column_fallback" in c.codes() and c.report[0]["level"] == "info" and len(c.t) > 90
    assert err(load_cycle, pd.DataFrame({"foo": t, "bar": v, "baz": v})) == "column_ambiguous"


def test_bytes_and_text_file_inputs_give_identical_cycles(tmp_path):
    t, v = tr(80)
    body = ("time,speed\n" + "\n".join(f"{a:.2f},{b:.3f}" for a, b in zip(t, v))).encode()
    p = tmp_path / "c.csv"
    p.write_bytes(body)
    a, b = load_cycle(body), load_cycle(p)
    assert np.array_equal(a.v, b.v) and np.array_equal(a.t, b.t) and b.source == str(p)[:120]


def test_read_table_alone_reports_dropped_ragged_rows():
    rep = []
    df = read_table(b"time,speed\n1,2\n2,3\n3\n4,5\n", rep)
    assert list(df.columns) == ["time", "speed"] and len(df) == 3 and rep[0]["code"] == "ragged_rows_dropped" and rep[0]["n"] == 1


def test_whitespace_delimited_file_without_header_gets_generic_column_names():
    rep = []
    df = read_table(b"0 1.5\n1 2.5\n2 3.5\n", rep)
    assert list(df.columns) == ["col0", "col1"] and df.shape == (3, 2)


def test_sparse_sampling_warns_and_strict_turns_it_into_an_error():
    t = np.arange(25) * 3.0
    v = np.linspace(0, 20, 25)
    c = load_cycle((t, v), min_duration_s=10)
    assert "sparse_sampling" in c.codes() and next(r for r in c.report if r["code"] == "sparse_sampling")["level"] == "warn"
    assert err(load_cycle, (t, v), min_duration_s=10, strict=True) == "strict_warning"
    assert c.dt == 0.1 and c.v[0] == 0.0 and c.v[-1] == pytest.approx(20.0)


def test_empty_arrays_are_refused():
    assert err(load_cycle, (np.array([]), np.array([]))) == "file_empty"


def test_mismatched_array_lengths_raise_instead_of_misaligning():
    with pytest.raises(ValueError):
        load_cycle((np.arange(10.0), np.arange(9.0)))


# ----------------------------------------------------------------------------------------------- smoothing
def test_smoothing_is_a_moving_average_that_keeps_length_and_a_constant():
    t = np.arange(0, 100, 0.1)
    v = np.where(t < 50, 0.0, 10.0)
    raw = load_cycle((t, v), min_duration_s=10)
    sm = load_cycle((t, v), min_duration_s=10, smooth_s=2.0)
    assert len(sm.v) == len(raw.v) and "smoothed" in sm.codes() and "smoothed" not in raw.codes()
    assert sm.v[0] == 0.0 and sm.v[-1] == pytest.approx(10.0)
    mid = sm.v[(sm.t > 48.5) & (sm.t < 51.5)]
    assert np.all(np.diff(mid) >= -1e-12) and 0.0 < mid[len(mid) // 2] < 10.0           # the step became a ramp
    assert sm.v.max() <= raw.v.max() + 1e-12 and sm.v.min() >= -1e-12
    assert np.abs(sm.v - raw.v).max() <= 10.0 / 2                                         # a 2 s window cannot move a 10 m/s step by more than half
    assert abs(sm.v.sum() - raw.v.sum()) <= 10.0 * 10                                      # area preserved to within a window's worth of edge samples


def test_smoothing_window_of_one_sample_is_the_identity():
    t, v = tr(100)
    a = load_cycle((t, v), dt=1.0)
    b = load_cycle((t, v), dt=1.0, smooth_s=0.4)                                          # rounds to a 1-sample window
    assert np.allclose(a.v, b.v)


# ----------------------------------------------------------------------------------------------- Cycle helpers
def test_cycle_accel_note_and_report_frame():
    c = Cycle(t=np.arange(5.0), v=np.array([0, 1, 4, 9, 16.0]), dt=1.0)
    assert c.accel().tolist() == [1.0, 2.0, 4.0, 6.0, 7.0] and c.duration == 4.0
    c.note("x", "warn", "hello", n=np.int64(3))
    f = c.report_frame()
    assert list(f.columns) == ["code", "level", "msg", "n"] and f.iloc[0].tolist() == ["x", "warn", "hello", 3] and c.codes() == ["x"]
    assert list(Cycle(t=np.arange(2.0), v=np.zeros(2), dt=1.0).report_frame().columns) == ["code", "level", "msg", "n"]


# ----------------------------------------------------------------------------------------------- grade
def _elev_cycle(slope, v=10.0, n=3000, dt=0.1, **kw):
    t = np.arange(n) * dt
    df = pd.DataFrame({"time": t, "speed": np.full(n, v) + 0.5 * np.sin(t / 7.0), "elev": slope * np.cumsum(np.full(n, v)) * dt})
    return load_cycle(df, keep={"elevation": "elev"}, grade_cols=["elev"], **kw)


def test_grade_from_elevation_recovers_a_known_slope():
    c = _elev_cycle(0.04)
    assert c.grade.shape == c.t.shape
    mid = c.grade[len(c.grade) // 4: 3 * len(c.grade) // 4]
    assert np.allclose(mid, np.arctan(0.04), atol=0.003)
    rec = next(r for r in c.report if r["code"] == "grade_from_elevation")
    assert rec["level"] == "info" and rec["n"] == 0


def test_grade_is_capped_at_fifteen_percent_and_flagged():
    c = _elev_cycle(0.4)
    assert c.grade.max() <= np.arctan(0.15) + 1e-12 and c.grade.max() == pytest.approx(np.arctan(0.15))
    rec = next(r for r in c.report if r["code"] == "grade_from_elevation")
    assert rec["level"] == "warn" and rec["n"] > 0.02 * len(c.t)


def test_downhill_grade_is_negative_and_flat_road_is_zero():
    assert _elev_cycle(-0.03).grade[1000:2000].mean() == pytest.approx(np.arctan(-0.03), abs=0.003)
    assert np.abs(_elev_cycle(0.0).grade).max() < 1e-9


def test_short_cycle_with_elevation_does_not_crash_the_grade_step():
    c = _elev_cycle(0.04, n=250)                                    # 25 s at dt = 0.1: shorter than the 30 s slope window
    assert c.grade is None or len(c.grade) == len(c.t)


def test_grade_requested_without_elevation_warns_and_leaves_grade_none():
    t, v = tr(100)
    c = load_cycle(pd.DataFrame({"time": t, "speed": v}), grade_cols=["elev"])
    assert "grade_unavailable" in c.codes() and c.grade is None


def test_grade_with_standstill_does_not_blow_up():
    n = 3000
    t = np.arange(n) * 0.1
    v = np.where(t < 150, 0.0, 10.0)
    c = load_cycle(pd.DataFrame({"time": t, "speed": v, "elev": np.linspace(0, 30, n)}), keep={"elevation": "elev"}, grade_cols=["elev"])
    assert np.isfinite(c.grade).all() and np.abs(c.grade[:1000]).max() == 0.0               # no distance covered: slope forced to zero


# ----------------------------------------------------------------------------------------------- vehicle equations
def test_vehicle_plant_kwargs_and_grade_torque_formulas():
    v = VehicleEq(k=19.0, mass=1500.0, crr=0.01, cda=0.6, rho=1.2, drivetrain_b=0.02)
    pk = v.plant_kwargs()
    assert pk["j_load"] == pytest.approx(1500 / 19.0 ** 2) and pk["a"] == pytest.approx(1500 * G * 0.01 / 19.0)
    assert pk["b"] == 0.02 and pk["c"] == pytest.approx(0.5 * 1.2 * 0.6 / 19.0 ** 3)
    assert v.grade_torque(0.0) == 0.0 and v.grade_torque(np.pi / 6) == pytest.approx(1500 * G * 0.5 / 19.0)
    assert v.grade_torque(-0.1) == pytest.approx(-v.grade_torque(0.1))


def test_required_torque_at_constant_speed_is_road_load_plus_grade():
    n = 50
    c = Cycle(t=np.arange(n) * 0.1, v=np.full(n, 10.0), dt=0.1)
    v = VehicleEq(k=10.0, mass=1000.0)
    w, pk = 100.0, v.plant_kwargs()
    base = pk["a"] + pk["b"] * w + pk["c"] * w * w
    assert np.allclose(v.required_torque(c), base)
    c.grade = np.full(n, 0.05)
    assert np.allclose(v.required_torque(c), base + 1000 * G * np.sin(0.05) / 10.0)
    c.v, c.grade = np.zeros(n), None
    assert np.allclose(v.required_torque(c), 0.0)                                           # rolling term switches off at standstill


def test_data_demand_includes_grade_torque_in_the_segment_demand():
    t = np.arange(0, 20, 0.1)
    flat = Cycle(t=t, v=np.full(len(t), 10.0), dt=0.1)
    hill = Cycle(t=t, v=np.full(len(t), 10.0), dt=0.1, grade=np.full(len(t), 0.05))
    v = VehicleEq(k=10.0, mass=1000.0)
    a, b = data_demand(flat, v), data_demand(hill, v)
    assert np.allclose(b["T_req_seg"] - a["T_req_seg"], 1000 * G * np.sin(0.05) / 10.0)
    assert np.allclose(a["T_avail"], b["T_avail"]) and (b["ratio_seg"] > a["ratio_seg"]).all()


def test_road_scale_branch_shrinks_road_load_for_an_overpowered_cycle_and_records_it():
    cyc = synthetic_cycle("ramp", duration=60.0, v_peak=30.0)
    T_cap = 12.0
    with pytest.raises(CycleError) as e:
        fit_vehicle_to_motor(cyc, T_cap=T_cap)
    assert e.value.code == "cycle_infeasible"
    veq, info = fit_vehicle_to_motor(cyc, T_cap=T_cap, allow_road_scale=True)
    assert 0.02 < info["road_scale"] < 1.0 and veq.cda == pytest.approx(0.62 * info["road_scale"]) and veq.crr == pytest.approx(0.010 * info["road_scale"])
    assert np.quantile(np.abs(veq.required_torque(cyc)), 0.99) <= 0.85 * T_cap * 1.02
    assert info["omega_peak"] == pytest.approx(190.0)


def test_road_scale_gives_up_when_even_two_percent_road_load_is_too_much():
    cyc = synthetic_cycle("ramp", duration=60.0, v_peak=30.0)
    assert err(fit_vehicle_to_motor, cyc, T_cap=0.01, allow_road_scale=True) == "cycle_infeasible"


def test_fit_info_is_self_consistent_and_never_exceeds_mass_bounds():
    cyc = synthetic_cycle("sine", duration=60.0, v_peak=10.0)
    veq, info = fit_vehicle_to_motor(cyc, T_cap=74.0, mass_max=800.0)
    assert 20.0 <= veq.mass <= 800.0 and info["mass_eq"] == veq.mass and info["road_scale"] == 1.0
    assert info["T_q"] <= info["T_peak"] and 0.0 <= info["frac_over_cap"] <= 1.0
    assert veq.mass == pytest.approx(800.0) or info["T_q"] == pytest.approx(0.85 * 74.0, rel=0.02)


# ----------------------------------------------------------------------------------------------- TUM adapter
def _tum(tmp_path, n=300, **cols):
    d = {"time": 1.7e12 + np.arange(n) * 100.0}
    d.update(cols)
    p = tmp_path / "track.csv"
    pd.DataFrame(d).to_csv(p, index=False)
    return p


def test_tum_adapter_prefers_gps_speed_and_falls_back_to_vehicle_speed(tmp_path):
    n = 300
    gps = 10 + 5 * np.sin(np.arange(n) / 30.0)
    a = load_tum_tesla(_tum(tmp_path, n, speed=gps))
    assert a.v.max() == pytest.approx(gps.max(), rel=0.02) and a.source.startswith("TUM Tesla Model 3")
    kmh = np.round(gps * 3.6)
    b = load_tum_tesla(_tum(tmp_path, n, vehicle_speed=kmh, speed=np.full(n, np.nan)))
    assert b.v.max() == pytest.approx(kmh.max() * KMH, rel=0.02) and "speed_nan_interpolated" not in b.codes()


def test_tum_adapter_refuses_files_without_usable_speed_or_time(tmp_path):
    n = 300
    assert err(load_tum_tesla, _tum(tmp_path, n, speed=np.where(np.arange(n) < 100, 5.0, np.nan))) == "speed_mostly_missing"     # 33 % coverage
    assert err(load_tum_tesla, _tum(tmp_path, n, other=np.zeros(n))) == "speed_mostly_missing"
    p = tmp_path / "bad.csv"
    pd.DataFrame({"t": [1, 2], "speed": [1, 2]}).to_csv(p, index=False)
    assert err(load_tum_tesla, p) == "column_missing"


def test_tum_adapter_keeps_only_the_optional_channels_that_exist(tmp_path):
    n = 300
    n = 1000
    c = load_tum_tesla(_tum(tmp_path, n, speed=10 + np.sin(np.arange(n) / 20.0), hv_soc=np.linspace(80, 79, n), elevation=np.linspace(0, 3, n)))
    assert set(c.measured) == {"hv_soc", "elevation"} and len(c.measured["hv_soc"]) == len(c.t)
    assert c.measured["hv_soc"][0] == pytest.approx(80.0) and c.measured["hv_soc"][-1] == pytest.approx(79.0, abs=0.05)
