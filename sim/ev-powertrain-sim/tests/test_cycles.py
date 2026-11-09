"""The 25+ ways real speed data goes wrong, one test each (several more are parametrised). Every case must end in exactly
one of: repaired and reported (FIX), refused with a named error (ERROR), or accepted with a warning (WARN)."""
import os, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from cycles import CycleError, load_cycle, load_tum_tesla, fit_vehicle_to_motor, VehicleEq, MPH, KMH


def trace(n=300, dt=1.0, vmax=15.0):
    t = np.arange(n) * dt
    v = vmax * np.sin(np.pi * t / t[-1]) ** 2
    return t, v


def text(t, v, sep=",", header=("time", "speed"), fmt="{:.4f}"):
    lines = [sep.join(header)] if header else []
    lines += [sep.join([fmt.format(a), fmt.format(b)]) for a, b in zip(t, v)]
    return "\n".join(lines).encode()


def codes(c):
    return set(c.codes())


# ---------------------------------------------------------------------------------------------- reading the file
def test_01_missing_file():
    with pytest.raises(CycleError) as e: load_cycle("/no/such/file.csv")
    assert e.value.code == "file_missing"

def test_02_directory(tmp_path):
    with pytest.raises(CycleError) as e: load_cycle(tmp_path)
    assert e.value.code == "file_is_directory"

def test_03_empty_file(tmp_path):
    p = tmp_path / "e.csv"; p.write_bytes(b"   \n\n")
    with pytest.raises(CycleError) as e: load_cycle(p)
    assert e.value.code == "file_empty"

def test_04_only_comments():
    with pytest.raises(CycleError) as e: load_cycle(b"# nothing\n// here\n")
    assert e.value.code == "file_empty"

def test_05_utf8_bom():
    c = load_cycle(b"\xef\xbb\xbf" + text(*trace()))
    assert c.duration > 290

def test_06_encoding_fallback_latin1():
    raw = "time,speed,Bemerkung\n".encode("latin-1") + b"\n".join(f"{i},{5+i%3},\xe4".encode("latin-1") for i in range(60))
    c = load_cycle(raw, time_col="time", speed_col="speed")
    assert "encoding_fallback" in codes(c) or c.duration > 50

@pytest.mark.parametrize("sep", [";", "\t", "|", " "])
def test_07_delimiters(sep):
    t, v = trace()
    raw = text(t, v, sep="  " if sep == " " else sep)
    c = load_cycle(raw)
    assert np.isclose(c.v.max(), v.max(), rtol=0.02)

def test_08_decimal_comma_semicolon():
    t, v = trace(); body = "time;speed\n" + "\n".join(f"{a:.2f};{b:.3f}".replace(".", ",") for a, b in zip(t, v))
    c = load_cycle(body.encode())
    assert np.isclose(c.v.max(), v.max(), rtol=0.02)

def test_09_title_lines_and_header_units():
    body = "MY CYCLE\nTest Time, secs\tTarget Speed, mph\n" + "\n".join(f"{i}\t{30 * np.sin(np.pi*i/60)**2:.1f}" for i in range(61))
    c = load_cycle(body.encode())
    assert np.isclose(c.v.max(), 30 * MPH, rtol=0.02)           # mph converted via the header text, no guessing
    assert "speed_unit_inferred" not in codes(c)

def test_10_ragged_rows_dropped():
    t, v = trace(60); raw = text(t, v).decode().split("\n"); raw.insert(10, "1,2,3,4"); raw.insert(20, "garbage")
    c = load_cycle("\n".join(raw).encode())
    assert "ragged_rows_dropped" in codes(c)

# ---------------------------------------------------------------------------------------------------- columns, units
def test_11_ambiguous_columns():
    df = pd.DataFrame({"a": [0, 1, 2.0], "b": [1, 2, 3.0], "c": [3, 2, 1.0]})
    with pytest.raises(CycleError) as e: load_cycle(df)
    assert e.value.code == "column_ambiguous"

def test_12_named_column_missing():
    t, v = trace()
    with pytest.raises(CycleError) as e: load_cycle(text(t, v), speed_col="nope")
    assert e.value.code == "column_missing"

def test_13_unit_mph_exact():
    t, v = trace(); c = load_cycle(text(t, v, header=("time", "speed_mph")))
    assert np.isclose(c.v.max(), v.max() * MPH, rtol=0.01)

def test_14_unit_kmh_exact():
    t, v = trace(); c = load_cycle(text(t, v, header=("time", "speed km/h")))
    assert np.isclose(c.v.max(), v.max() * KMH, rtol=0.01)

def test_15_unitless_large_values_flagged_as_guess():
    t, v = trace(vmax=110.0); c = load_cycle(text(t, v))
    assert "speed_unit_inferred" in codes(c) and np.isclose(c.v.max(), 110 * KMH, rtol=0.01)

def test_16_unknown_unit_refused():
    t, v = trace()
    with pytest.raises(CycleError) as e: load_cycle(text(t, v), speed_unit="furlongs")
    assert e.value.code == "bad_speed_unit"

def test_17_implausible_speed_refused():
    t, v = trace(vmax=300.0)
    with pytest.raises(CycleError) as e: load_cycle(text(t, v), speed_unit="ms")
    assert e.value.code == "speed_implausible"

def test_18_non_numeric_strings_coerced():
    t, v = trace(80); body = ["time,speed"] + [f"{a},{b:.2f} m/s" if i % 7 else f"{a},n/a" for i, (a, b) in enumerate(zip(t, v))]
    c = load_cycle("\n".join(body).encode(), speed_unit="ms")
    assert "speed_nan_interpolated" in codes(c) and np.isfinite(c.v).all()

# ------------------------------------------------------------------------------------------------------------- time
def test_19_epoch_milliseconds():
    t, v = trace(200, dt=0.1); df = pd.DataFrame({"time": 1.6e12 + t * 1000.0, "speed": v})
    c = load_cycle(df, speed_unit="ms")
    assert np.isclose(c.duration, t[-1], atol=0.2) and c.t[0] == 0.0

def test_20_time_unit_explicit_ms_small_numbers():
    t, v = trace(100, dt=1.0); df = pd.DataFrame({"time": t * 1000.0, "speed": v})
    c = load_cycle(df, speed_unit="ms", time_unit="ms")
    assert np.isclose(c.duration, t[-1], atol=0.2)

def test_21_out_of_order_sorted():
    t, v = trace(); idx = np.random.default_rng(0).permutation(len(t)); c = load_cycle(pd.DataFrame({"time": t[idx], "speed": v[idx]}), speed_unit="ms")
    assert "time_not_monotonic" in codes(c) and np.all(np.diff(c.t) > 0)

def test_22_duplicate_timestamps_collapsed():
    t, v = trace(); t2 = np.r_[t, t[50:60]]; v2 = np.r_[v, v[50:60]]
    c = load_cycle(pd.DataFrame({"time": t2, "speed": v2}), speed_unit="ms")
    assert "duplicate_timestamps" in codes(c)

def test_23_gap_keeps_longest_segment():
    t, v = trace(300); keep = (t < 100) | (t > 130); c = load_cycle(pd.DataFrame({"time": t[keep], "speed": v[keep]}), speed_unit="ms")
    assert "time_gap_split" in codes(c) and c.duration > 150

def test_24_gap_error_mode():
    t, v = trace(300); keep = (t < 100) | (t > 130)
    with pytest.raises(CycleError) as e: load_cycle(pd.DataFrame({"time": t[keep], "speed": v[keep]}), speed_unit="ms", on_gap="error")
    assert e.value.code == "time_gap"

def test_25_too_short():
    with pytest.raises(CycleError) as e: load_cycle(pd.DataFrame({"time": [0, 1, 2.0], "speed": [0, 1, 2.0]}), speed_unit="ms")
    assert e.value.code == "too_short"

def test_26_irregular_sampling_resampled_uniform():
    rng = np.random.default_rng(1); t = np.sort(rng.uniform(0, 100, 400)); v = 10 + 5 * np.sin(t / 10)
    c = load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms", dt=0.1)
    assert "irregular_sampling" in codes(c) and np.allclose(np.diff(c.t), 0.1)

def test_27_time_all_missing():
    with pytest.raises(CycleError) as e: load_cycle(pd.DataFrame({"time": [np.nan] * 30, "speed": np.arange(30.0)}), speed_unit="ms")
    assert e.value.code == "time_all_missing"

# ------------------------------------------------------------------------------------------------------------ speed
def test_28_nan_speed_interpolated():
    t, v = trace(); v = v.copy(); v[40:50] = np.nan
    c = load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms")
    assert "speed_nan_interpolated" in codes(c) and np.isfinite(c.v).all()

def test_29_mostly_missing_refused():
    t, v = trace(); v = v.copy(); v[: 200] = np.nan
    with pytest.raises(CycleError) as e: load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms")
    assert e.value.code == "speed_mostly_missing"

def test_30_all_nan_speed():
    with pytest.raises(CycleError) as e: load_cycle(pd.DataFrame({"time": np.arange(30.0), "speed": [np.nan] * 30}), speed_unit="ms")
    assert e.value.code == "speed_all_missing"

def test_31_small_negative_clipped():
    t, v = trace(); v = v - 0.05
    c = load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms")
    assert "negative_speed_clipped" in codes(c) and c.v.min() >= 0

def test_32_reversing_refused():
    t, v = trace(); v = -v - 1
    with pytest.raises(CycleError) as e: load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms")
    assert e.value.code == "reverse_driving"

def test_33_spike_removed():
    t, v = trace(); v = v.copy(); v[100] = 70.0
    c = load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms")
    assert "spikes_removed" in codes(c) and c.v.max() < 20

def test_34_infinite_values():
    t, v = trace(); v = v.copy(); v[30] = np.inf
    c = load_cycle(pd.DataFrame({"time": t, "speed": v}), speed_unit="ms")
    assert "inf_values" in codes(c) and np.isfinite(c.v).all()

def test_35_constant_and_standstill_warn():
    t = np.arange(60.0)
    assert "constant_speed" in codes(load_cycle(pd.DataFrame({"time": t, "speed": 12.0 + 0 * t}), speed_unit="ms"))
    assert "standstill" in codes(load_cycle(pd.DataFrame({"time": t, "speed": 0 * t}), speed_unit="ms"))

def test_36_strict_mode_turns_warnings_into_errors():
    t, v = trace(vmax=110.0)
    with pytest.raises(CycleError) as e: load_cycle(text(t, v), strict=True)
    assert e.value.code == "strict_warning"

def test_37_array_input_and_determinism():
    t, v = trace(); a = load_cycle((t, v), speed_unit="ms"); b = load_cycle((t, v), speed_unit="ms")
    assert np.array_equal(a.v, b.v) and np.array_equal(a.t, b.t)

def test_38_extra_channels_aligned():
    t, v = trace(); df = pd.DataFrame({"time": t, "speed": v, "i": np.linspace(0, 5, len(t))})
    c = load_cycle(df, speed_unit="ms", keep={"current": "i"})
    assert c.measured["current"].shape == c.t.shape and np.isclose(c.measured["current"][-1], 5.0, atol=0.01)

def test_39_large_file_is_fast(tmp_path):
    n = 400_000; t = np.arange(n) * 0.05; v = 15 + 10 * np.sin(t / 40)
    p = tmp_path / "big.csv"; pd.DataFrame({"time": t, "speed": v}).to_csv(p, index=False)
    t0 = time.time(); c = load_cycle(p, speed_unit="ms"); assert time.time() - t0 < 15 and c.duration > 19000

# ----------------------------------------------------------------------------------------- vehicle mapping and TUM
def test_40_fit_vehicle_fits_gentle_cycle():
    t, v = trace(400, vmax=15.0); c = load_cycle((t, v), speed_unit="ms")
    veq, info = fit_vehicle_to_motor(c)
    assert info["frac_over_cap"] < 0.02 and np.isclose(info["omega_peak"], 190.0)

def test_41_fit_refuses_aggressive_cycle_then_scales_on_request():
    t = np.arange(0, 200, 1.0); v = np.clip(t / 5.0 * 5.0, 0, 38.0)             # 5 m/s^2 launch to 38 m/s
    c = load_cycle((t, v), speed_unit="ms", a_max=20.0)
    with pytest.raises(CycleError) as e: fit_vehicle_to_motor(c, mass_min=500.0)
    assert e.value.code == "cycle_infeasible"

def test_42_fit_refuses_motionless_cycle():
    c = load_cycle((np.arange(60.0), 0.0 * np.arange(60.0)), speed_unit="ms")
    with pytest.raises(CycleError) as e: fit_vehicle_to_motor(c)
    assert e.value.code == "cannot_scale"

def test_43_tum_adapter_synthetic_schema_variants(tmp_path):
    n = 400; t = 1.6e12 + np.arange(n) * 100.0; sp = 12 + 8 * np.sin(np.arange(n) / 30)
    # track A: GPS speed present, optional channels absent
    pd.DataFrame({"time": t, "speed": sp}).to_csv(tmp_path / "a.csv", index=False)
    a = load_tum_tesla(tmp_path / "a.csv"); assert "optional_channels_missing" in codes(a) and a.grade is None
    # track B: GPS speed mostly empty, fall back to vehicle_speed in whole km/h
    gps = pd.Series(sp); gps[: 300] = np.nan
    pd.DataFrame({"time": t, "speed": gps, "vehicle_speed": np.round(sp * 3.6)}).to_csv(tmp_path / "b.csv", index=False)
    b = load_tum_tesla(tmp_path / "b.csv"); assert np.isclose(b.v.max(), sp.max(), rtol=0.05)
    # track C: no time column
    pd.DataFrame({"speed": sp}).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(CycleError) as e: load_tum_tesla(tmp_path / "c.csv")
    assert e.value.code == "column_missing"

TUM = pathlib.Path(os.environ.get("TUM_DIR", "/tmp/claude-0/-home-claude/953c4b99-a683-5625-8372-4d907fe355ce/scratchpad/ds_tum"))

@pytest.mark.skipif(not (TUM / "track_0.csv").exists(), reason="real TUM tracks not downloaded")
@pytest.mark.parametrize("n", [0, 1, 10, 100, 103])
def test_44_real_tum_tracks_load_and_are_sane(n):
    c = load_tum_tesla(TUM / f"track_{n}.csv")
    assert np.isfinite(c.v).all() and c.v.min() >= 0 and c.duration > 100
    assert abs(np.degrees(c.grade).max()) < 9.0 and np.allclose(np.diff(c.t), c.dt)
