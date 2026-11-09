"""Battery fit tests (plan U20 to U25) and mutation tests. Run: pytest -q tests/test_battery_fit.py  (about 15 s)

Honest notes on ill-posed plan items:
  * U21 asks to "test the logic with a synthetic case where the fit cannot win". The fitted model family NESTS the constant
    resistance baseline (R0 = R_c, R1 = 0 and a flat OCV), so no synthetic data set can force it to lose strictly; a
    tie is the best a synthetic case gives. The status LOGIC is therefore tested on the decision function with numbers
    where the fit loses/ties, and the stored/real result is checked for consistency with that same function.
  * U22 'a trace shifted up by 20 %' is implemented as every current sample times 1.2 (discharge and regen magnitudes).
"""
import sys, pathlib, json, hashlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from functools import lru_cache
import numpy as np
import pandas as pd
import pytest

import battery_fit as bf
from interfaces import validate_fit_dict, FIT_REQUIRED

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tum_subset"


@lru_cache(maxsize=1)
def _tracks():
    return bf.load_tracks(DATA)


@lru_cache(maxsize=1)
def _clean():
    return bf.filter_all(_tracks())


@lru_cache(maxsize=1)
def _fit():
    return bf.fit_pack(_tracks())


def _synthetic_dir(tmp_path, r_true=0.045, n=4000, seed=0, p_i_update=0.5, p_v_update=0.95, lag=0, tid=7, soc_pct=60.0):
    """Write a TUM-format CSV (discharge NEGATIVE in the file) with a known series resistance and real-like holds."""
    rng = np.random.default_rng(seed)
    tt = np.arange(n) * 0.1
    ph = rng.uniform(0, 6.28, 3)
    i_true = 220.0 + 150.0 * np.sin(2 * np.pi * tt / 17.0 + ph[0]) + 90.0 * np.sin(2 * np.pi * tt / 5.3 + ph[1]) \
        + 40.0 * np.sin(2 * np.pi * tt / 1.9 + ph[2]) + np.convolve(rng.normal(0, 8, n), np.ones(5) / 5, "same")
    for a in range(0, n, 1000):                                    # parked periods (rest points for the OCV fit)
        i_true[a:a + 150] = rng.normal(0, 1.0, min(150, n - a))
    i_true = np.convolve(i_true, np.ones(3) / 3, "same")
    v_true = 350.0 - r_true * np.roll(i_true, lag)

    def hold(x, p):
        out, cur = np.empty(n), x[0]
        upd = rng.random(n) < p
        for k in range(n):
            if upd[k]:
                cur = x[k]
            out[k] = cur
        return out
    i_obs = np.round(hold(i_true, p_i_update), 1)
    v_obs = np.round(hold(v_true, p_v_update) + rng.normal(0, 0.02, n), 2)
    soc = np.full(n, np.nan)
    soc[::3] = soc_pct
    df = pd.DataFrame({"time": 1611305532000 + 100 * np.arange(n), "hv_battery_voltage": v_obs,
                       "hv_battery_current": -i_obs, "hv_soc": soc, "hv_temp_min": 25.0, "speed": 0.0,
                       "vehicle_speed": np.nan, "ambient_air_temp": 10.0})
    d = tmp_path / f"syn_{seed}_{lag}"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"track_{tid}_battery.csv", index=False)
    return d


# ---------------------------------------------------------------------------------------------- U20
def test_U20_fit_deterministic_positive_monotone_and_sign_convention(tmp_path):
    fit1 = bf.fit_pack(_tracks())
    fit2 = bf.fit_pack(bf.load_tracks(DATA))
    assert fit1 == fit2                                              # deterministic, including the bootstrap
    assert validate_fit_dict(fit1) == []
    pr = fit1["pack_real"]
    assert pr["v_nom"] > 0 and pr["p_base_w"] > 0 and pr["i_base_a"] > 0
    assert all(v > 0 for v in pr["r_eff_by_window_ohm"].values())
    assert all(v > 0 for v in pr["r_eff_pu"].values()) and pr["r_eff_pu"]["lo"] <= pr["r_eff_pu"]["mid"] <= pr["r_eff_pu"]["hi"]
    assert pr["rc"]["r1_pu"] > 0 and bf.TAU_BOUNDS[0] <= pr["rc"]["tau1_s"] <= bf.TAU_BOUNDS[1]
    assert pr["rc_detail"]["r0_ohm"] > 0
    # OCV monotone non-decreasing in SOC over the SOC range of the data (and by the K1 >= 0, K2 <= 0 bounds)
    clean, _ = _clean()
    soc = np.concatenate([t["soc"][t["valid"]] for t in clean.values()])
    grid = np.linspace(soc.min(), soc.max(), 400)
    ocv = bf.ocv_value(pr["ocv_k"], grid)
    assert np.all(np.diff(ocv) >= -1e-9) and ocv[-1] > ocv[0]
    assert pr["ocv_k"][1] >= 0 and pr["ocv_k"][2] <= 0
    # sign convention: TUM-format file with discharge NEGATIVE and a known 45 mOhm gives a POSITIVE resistance
    d = _synthetic_dir(tmp_path, r_true=0.045)
    syn, _ = bf.filter_all(bf.load_tracks(d))
    r = bf.reff_by_window(syn, n_boot=5)["pooled"]["held_aware_ols"]
    assert all(v > 0 for v in r.values())
    assert abs(r["0.5"] - 0.045) < 0.003


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_MUT_sign_flipped_current_gives_negative_resistance(tmp_path):
    d = _synthetic_dir(tmp_path, r_true=0.045)
    syn, _ = bf.filter_all(bf.load_tracks(d, discharge_sign=+1.0))   # MUTANT loader: file sign kept
    r = bf.reff_by_window(syn, n_boot=5)["pooled"]["held_aware_ols"]
    assert all(v < 0 for v in r.values())                           # the U20 sign check (v > 0) would fail
    with pytest.raises(ValueError):                                  # and fit_pack refuses such data
        bf.fit_pack(bf.load_tracks(DATA, discharge_sign=+1.0))


def test_estimator_recovers_known_resistance_under_realistic_holds_and_naive_does_not(tmp_path):
    d = _synthetic_dir(tmp_path, r_true=0.045, n=6000, seed=3)
    syn, _ = bf.filter_all(bf.load_tracks(d))
    res = bf.reff_by_window(syn, n_boot=5)
    for w in ("0.3", "0.5", "1.0"):
        assert abs(res["pooled"]["held_aware_ols"][w] - 0.045) / 0.045 < 0.05, w
    # the naive 0.1 s regression is biased far low, as in the real data
    assert res["pooled"]["naive_raw"]["0.1"] < 0.6 * 0.045
    # and with a 0.2 s voltage lag the short window under-reads (the reported window dependence is not only held samples)
    d2 = _synthetic_dir(tmp_path, r_true=0.045, n=6000, seed=3, lag=2)
    syn2, _ = bf.filter_all(bf.load_tracks(d2))
    res2 = bf.reff_by_window(syn2, n_boot=5)["pooled"]["held_aware_ols"]
    assert res2["0.1"] < res2["1.0"] < 0.045 * 1.05


def test_real_window_dependence_is_reported_with_uncertainty():
    pr = _fit()["pack_real"]
    ohm = pr["r_eff_by_window_ohm"]
    assert set(ohm) == {"0.1", "0.3", "0.5", "1.0"}
    assert ohm["0.1"] <= ohm["0.3"] <= ohm["0.5"] <= ohm["1.0"]      # measured fact on these tracks (held-aware)
    m = pr["r_eff_methods"]
    for meth in ("held_aware_ols", "naive_raw", "eiv_deming", "lag_aligned"):
        for w in ohm:
            lo, hi = m["ci95_ohm"][meth][w]
            assert lo <= m["pooled_ohm"][meth][w] <= hi or meth == "eiv_deming"
    # the naive 0.1 s value is the biased-low one the plan warned about, the held-aware one is several times larger
    assert m["pooled_ohm"]["naive_raw"]["0.1"] < 0.5 * m["pooled_ohm"]["held_aware_ols"]["0.1"]
    assert 0.012 > m["pooled_ohm"]["naive_raw"]["0.1"] > 0.007


# ---------------------------------------------------------------------------------------------- U21
def test_U21_leave_one_track_out_vs_three_baselines_and_status_consistency(tmp_path):
    fit = _fit()
    v = fit["validation"]
    loo = v["loo_rmse_v"]
    assert set(loo) == {"fit", "const_v", "const_r", "generic"} and all(np.isfinite(x) and x > 0 for x in loo.values())
    for nm, folds in v["per_fold_rmse_v"].items():
        assert len(folds) == 5
        assert abs(np.sqrt(np.mean([x * x for x in folds.values()])) - loo[nm]) < 1e-6
    best = min(loo[k] for k in ("const_v", "const_r", "generic"))
    assert (fit["status"] == "ok") == (loo["fit"] < best)           # ok iff the fit beats the best baseline
    assert fit["status"] == bf.decide_status(loo)[0]
    assert isinstance(v["status_reason"], str) and len(v["status_reason"]) > 10
    assert v["fit_minus_best_baseline_v"]["n_folds"] == 5            # the spread across folds is reported, not only the mean
    _status_logic_with_cases_where_the_fit_cannot_win(tmp_path)      # decision function + synthetic tie case


def _status_logic_with_cases_where_the_fit_cannot_win(tmp_path):
    base = dict(const_v=6.0, const_r=3.8, generic=13.0)
    assert bf.decide_status(dict(fit=2.9, **base))[0] == "ok"
    st, why = bf.decide_status(dict(fit=3.9, **base))                # worse than const_r
    assert st == "not_better_than_baseline" and "const_r" in why
    assert bf.decide_status(dict(fit=3.8, **base))[0] == "not_better_than_baseline"          # a tie is not a win
    assert bf.decide_status(dict(fit=5.0, const_v=6.0, const_r=7.0, generic=4.0))[0] == "not_better_than_baseline"
    assert bf.decide_status(dict(fit=3.0, const_v=6.0, const_r=7.0, generic=4.0))[0] == "ok"
    # end to end on synthetic tracks that are EXACTLY the constant-resistance model: the fit family nests it, so the
    # best possible outcome is a tie with const_r (both ~0 V); it must never be reported as a clear win over const_r
    tracks = {}
    for j in range(4):
        d = _synthetic_dir(tmp_path, r_true=0.05, n=3000, seed=10 + j, tid=20 + j,
                           p_i_update=1.0, p_v_update=1.0, soc_pct=55.0 + 10 * j)
        tracks.update(bf.load_tracks(d))
    clean, _ = bf.filter_all(tracks)
    loo = bf.leave_one_track_out(clean)["loo_rmse_v"]
    assert loo["const_r"] < 0.2 and loo["fit"] < 0.2                # both explain the data; nothing for the fit to win


# ---------------------------------------------------------------------------------------------- U22
def _observed(clean):
    soc = np.concatenate([t["soc"][t["valid"]] for t in clean.values()])
    i = np.concatenate([t["i_dis"][t["valid"]] for t in clean.values()])
    return soc, i


def test_U22_envelope_covers_99_percent_and_shifted_trace_falls_outside():
    fit = _fit()
    env = fit["envelope_real"]
    clean, _ = _clean()
    soc, i = _observed(clean)
    cov = bf.envelope_coverage(env, soc, i)
    assert cov["dis"] >= 0.99 and cov["chg"] >= 0.99
    assert (i > 0).sum() > 10000 and (i < 0).sum() > 3000           # both directions are actually populated
    assert "observed" in env["note"] and "NOT a BMS limit" in env["note"]
    # negative control: every current sample 20 % larger must NOT be covered
    cov_up = bf.envelope_coverage(env, soc, 1.2 * i)
    assert cov_up["dis"] < 0.99
    assert cov_up["chg"] < 0.99 or cov_up["chg"] < cov["chg"]       # regen is pinned near a cap in the data; at least it gets worse
    # and the envelope is not vacuous: 99.9 % coverage at most as large as 1.5x the observed maximum
    assert max(env["i_dis_lim_a"]) < 1.5 * i.max()


def test_MUT_too_tight_envelope_fails_coverage_check():
    fit = _fit()
    env = dict(fit["envelope_real"])
    env["i_dis_lim_a"] = [0.8 * x for x in env["i_dis_lim_a"]]      # MUTANT: an envelope 20 % too low
    clean, _ = _clean()
    soc, i = _observed(clean)
    assert bf.envelope_coverage(env, soc, i)["dis"] < 0.99


# ---------------------------------------------------------------------------------------------- U23
def _sag_pu(e_v, r_ohm, p_base_w, p_pu):
    """Per-unit sag of a pack (EMF e_v, series R) delivering p_pu * p_base: solve V = E - R*P/V  (V^2 - E V + R P = 0)."""
    p = p_pu * p_base_w
    v = 0.5 * (e_v + np.sqrt(e_v * e_v - 4.0 * r_ohm * p))
    return (e_v - v) / e_v


def scaling_check(fit, scaling) -> float:
    """Worst relative difference of per-unit sag between the real pack and the toy pack under the same per-unit power
    steps. Uses ONLY the resulting R values (real: r_eff_by_window_ohm['0.5'], toy: r_total_toy_ohm['mid']) and the
    voltage / power bases; it never evaluates the scaling formula."""
    pr = fit["pack_real"]
    worst = 0.0
    for p_pu in (0.25, 0.5, 1.0):
        s_real = _sag_pu(pr["v_nom"], pr["r_eff_by_window_ohm"]["0.5"], pr["p_base_w"], p_pu)
        s_toy = _sag_pu(scaling["v_toy"], scaling["r_total_toy_ohm"]["mid"], scaling["p_base_toy_w"], p_pu)
        worst = max(worst, abs(s_toy - s_real) / s_real)
    return worst


def test_U23_independent_scaling_check_per_unit_sag():
    fit = _fit()
    assert scaling_check(fit, fit["scaling"]) < 0.02
    for pb in ("65000", "31100"):
        assert scaling_check(fit, fit["scaling_sensitivity"][pb]) < 0.02, pb
    # lo and hi must map to the lo/hi window values as well (checked via the real windows 0.3 and 1.0)
    pr = fit["pack_real"]
    for key, w in (("lo", "0.3"), ("hi", "1.0")):
        s_real = _sag_pu(pr["v_nom"], pr["r_eff_by_window_ohm"][w], pr["p_base_w"], 0.5)
        s_toy = _sag_pu(300.0, fit["scaling"]["r_total_toy_ohm"][key], 65000.0, 0.5)
        assert abs(s_toy - s_real) / s_real < 0.02
    # sanity of the other scale factors against first principles (stated here independently of the module)
    sc = fit["scaling"]
    assert abs(sc["i_base_toy"] - 65000.0 / 300.0) < 1e-6
    assert abs(sc["i_dis_lim_toy_a"][-1] / fit["envelope_real"]["i_dis_lim_a"][-1] - sc["i_base_toy"] / pr["i_base_a"]) < 1e-6
    # kwargs are accepted by the existing toy Battery and consistent with the toy motor
    from battery import Battery
    b = Battery(**fit["toy_battery_kwargs"])
    out = b.step(1e-4, 20000.0)
    assert np.isfinite(out["V_dc"])
    assert fit["toy_battery_kwargs"]["i_dis_max"] <= 250.0
    assert sc["motor_consistency"]["consistent_with_toy_motor"] is True


def test_MUT_wrong_scaling_formula_fails_the_independent_check():
    fit = _fit()
    pr = fit["pack_real"]
    bad = dict(fit["scaling"])
    zb_wrong = 300.0 / 65000.0                                       # MUTANT: V instead of V^2 in the base impedance
    bad["r_total_toy_ohm"] = {k: v * zb_wrong for k, v in pr["r_eff_pu"].items()}
    assert scaling_check(fit, bad) > 0.02
    # and the real function satisfies it for another base (independent of the default)
    sc = bf.scale_to_toy({"pack_real": pr, "envelope_real": fit["envelope_real"]}, 50000.0, 280.0)["scaling"]
    assert scaling_check(fit, sc) < 0.02


def test_scaling_reports_both_bases_and_ranges():
    fit = _fit()
    s65, s31 = fit["scaling_sensitivity"]["65000"], fit["scaling_sensitivity"]["31100"]
    assert s65 == fit["scaling"]
    for s in (s65, s31):
        r = s["r_total_toy_ohm"]
        assert r["lo"] < r["mid"] < r["hi"]
    assert abs(s31["r_total_toy_ohm"]["mid"] / s65["r_total_toy_ohm"]["mid"] - 65000.0 / 31100.0) < 1e-6    # R_toy ~ 1/P_base
    assert s31["motor_consistency"]["consistent_with_toy_motor"] is False      # 31.1 kW base throttles the 250 A motor
    g = fit["generic_toy_pack_per_unit"]
    assert g["65000"]["r0_plus_rbus_pu"] > g["31100"]["r0_plus_rbus_pu"]       # the generic pack looks 2x 'better' at the small base


# ---------------------------------------------------------------------------------------------- U24
def provenance_problems(fit, directory) -> list:
    bad = []
    for name, rec in fit["source"]["files"].items():
        p = pathlib.Path(directory) / name
        if not p.exists():
            bad.append(f"{name} missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != rec:
            bad.append(f"{name} sha256 mismatch")
    if bf.canonical_hash(fit) != fit["source"]["fit_hash"]:
        bad.append("fit_hash does not reproduce from the file content")
    return bad


def test_U24_provenance_checksums_and_fit_hash_reproduce():
    stored = json.loads((ROOT / "data" / "tum_battery_fit.json").read_text())
    assert validate_fit_dict(stored) == []
    assert len(stored["source"]["files"]) == 5 and stored["source"]["n_tracks"] == 5
    assert provenance_problems(stored, DATA) == []
    # the stored fit is exactly what the code produces from the data today (hash reproduces from the DATA, not the file)
    fresh = _fit()
    assert fresh["source"]["fit_hash"] == stored["source"]["fit_hash"]
    assert fresh == stored
    # every file of the subset is covered, and the originals' checksums are carried along for reference
    assert sorted(stored["source"]["files"]) == sorted(p.name for p in DATA.glob("track_*_battery.csv"))
    assert len(stored["source"]["original_full_csv_sha256"]) == 5


def test_MUT_tampered_input_is_detected_by_provenance_check(tmp_path):
    stored = json.loads((ROOT / "data" / "tum_battery_fit.json").read_text())
    d = tmp_path / "copy"
    shutil.copytree(DATA, d)
    assert provenance_problems(stored, d) == []
    p = d / "track_0_battery.csv"
    txt = p.read_text().replace("350.49", "350.50", 1)
    p.write_text(txt)
    assert any("sha256 mismatch" in x for x in provenance_problems(stored, d))
    stored2 = json.loads(json.dumps(stored))
    stored2["pack_real"]["v_nom"] += 1.0                             # content changed, hash not updated
    assert any("fit_hash" in x for x in provenance_problems(stored2, DATA))


# ---------------------------------------------------------------------------------------------- U25
KNOWN_SENTINEL = {"track_1": list(range(4711, 4721))}


def filter_problems(tracks) -> list:
    """Compare glitch_filter with an INDEPENDENT derivation of the expected removals; returns a list of problems."""
    bad = []
    for name, tr in tracks.items():
        clean, rep = bf.glitch_filter(tr)
        raw_t = tr["temp_c"]
        exp_sent = np.where(raw_t == -25.0)[0].tolist()
        if exp_sent != KNOWN_SENTINEL.get(name, []):
            bad.append(f"{name}: test expectation of sentinel rows inconsistent with raw data")
        got_inv = np.where(~clean["valid"])[0].tolist()
        if got_inv != exp_sent:
            bad.append(f"{name}: invalid rows {len(got_inv)} != sentinel rows {len(exp_sent)}")
        s = tr["soc_pct"]
        ok = np.isfinite(s)
        med = np.median(s[ok])
        exp_soc = np.where(ok & (np.abs(s - med) > 20.0))[0].tolist()     # whole-track median, 20 point band: independent of the filter
        got_soc = np.where(clean["soc_glitch"])[0].tolist()
        if exp_soc != got_soc:
            bad.append(f"{name}: soc glitch rows differ ({len(got_soc)} vs {len(exp_soc)})")
        if not (np.array_equal(clean["v"], tr["v"]) and np.array_equal(clean["i_dis"], tr["i_dis"])):
            bad.append(f"{name}: voltage/current values were altered")
        # unflagged SOC samples are untouched
        keep = ok & ~clean["soc_glitch"]
        if not np.array_equal(clean["soc_pct"][keep], s[keep]):
            bad.append(f"{name}: unflagged SOC samples changed")
    return bad


def test_U25_glitch_filter_removes_exactly_the_known_glitches():
    tracks = _tracks()
    assert filter_problems(tracks) == []
    clean, rep = _clean()
    assert rep["track_1"]["temp_sentinel_row_ranges"] == [[4711, 4720]] and rep["track_1"]["temp_sentinel_rows_invalidated"] == 10
    assert sum(r["temp_sentinel_rows_invalidated"] for r in rep.values()) == 10
    # the +51.2 jump of track 10 (30.2 -> 81.4 at the start) and its 79.5 -> 28.3 -> 79.5 excursion are removed
    raw10 = tracks["track_10"]["soc_pct"]
    assert abs(np.nanmax(np.abs(np.diff(raw10[np.isfinite(raw10)]))) - 51.2) < 0.15
    assert clean["track_10"]["soc_glitch"][:36].any() and np.all(np.isfinite(raw10[clean["track_10"]["soc_glitch"]]))
    assert rep["track_10"]["soc_glitch_row_ranges"] == [[0, 35], [7227, 7240]]
    # tracks without glitches lose nothing
    for k in ("track_0", "track_100", "track_103"):
        assert rep[k]["soc_glitch_samples_removed"] == 0 and rep[k]["temp_sentinel_rows_invalidated"] == 0
    # held voltage runs are REPORTED (values untouched); counts are derived independently here
    for k, tr in tracks.items():
        v = tr["v"]
        runs, n = [], 1
        for a in range(1, len(v)):
            if v[a] == v[a - 1]:
                n += 1
            else:
                runs.append(n); n = 1
        runs.append(n)
        assert rep[k]["held_voltage_runs_ge4_rows"] == sum(1 for r in runs if r >= bf.STALE_RUN)
        assert rep[k]["voltage_current_values_altered"] == 0


def test_MUT_overzealous_or_sloppy_filters_are_flagged(monkeypatch):
    tracks = _tracks()
    monkeypatch.setattr(bf, "SENTINEL_TEMP_C", 30.0)                 # MUTANT 1: treats every normal temperature as sentinel
    assert any("invalid rows" in p for p in filter_problems(tracks))
    monkeypatch.setattr(bf, "SENTINEL_TEMP_C", -20.0)
    monkeypatch.setattr(bf, "SOC_JUMP_PTS", 0.05)                    # MUTANT 2: removes ordinary SOC drift too
    assert any("soc glitch rows differ" in p for p in filter_problems(tracks))
    monkeypatch.setattr(bf, "SOC_JUMP_PTS", 500.0)                   # MUTANT 3: removes no SOC glitch at all
    assert any("soc glitch rows differ" in p for p in filter_problems(tracks))


def test_loader_facts_and_schema():
    tracks = _tracks()
    assert sorted(tracks) == ["track_0", "track_1", "track_10", "track_100", "track_103"]
    for tr in tracks.values():
        assert tr["i_dis"].max() > 400 and tr["i_dis"].min() < -50       # discharge positive after load, regen negative
    fit = _fit()
    assert set(FIT_REQUIRED) <= set(fit) and fit["validation"]["glitches_removed"]["track_1"]["temp_sentinel_rows_invalidated"] == 10
    assert "DISCHARGE POSITIVE" in fit["sign_convention"]
    assert "NOT FITTED" in fit["pack_real"]["temperature"]
