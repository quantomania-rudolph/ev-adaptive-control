"""Battery fit from real TUM Tesla Model 3 CAN traces, observed-use envelope, and per-unit scaling to the toy pack.

SIGN CONVENTION (stated once): the TUM files record battery DISCHARGE current as NEGATIVE. Inside this module (and in
bms.py) discharge current is POSITIVE (I_dis > 0). load_tracks() multiplies the file current by `discharge_sign` (-1).

What this module does and does not claim (honest scope):
  * five 10 Hz tracks, 20 to 34 C only (no cold data): NO temperature dependence is fitted; the toy keeps the generic
    Arrhenius R0(T) from battery.py, labelled an ASSUMPTION.
  * CAN current is repeated (held) on 33 to 66 % of rows, voltage on 4 to 10 %, so a naive 0.1 s dV/dI regression is
    biased low. "Held-sample-aware" here means: every repeated value is treated as MISSING and both signals are linearly
    interpolated between their update (value-change) rows before differencing. Sample-to-sample timing of the CAN frames
    is not recorded, so a residual timing error and a lag between current and voltage remain; they are reported
    (lag-aligned diagnostic), not removed.
  * the apparent resistance depends on the difference window. R_eff(w) mixes ohmic and polarization resistance.
  * the OCV is flat (about 3.5 V over SOC 0.58 to 0.985 at rest) and sits on only four SOC clusters; K1, K2 are weakly
    identified. Uncertainty is reported, not hidden.
  * the envelope is OBSERVED USE (a quantile of what the car did), not a BMS limit.

Run `python battery_fit.py` to (re)write data/tum_battery_fit.json deterministically.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import lsq_linear, minimize_scalar, nnls
from scipy.signal import lfilter

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "data" / "tum_subset"
DEFAULT_OUT = ROOT / "data" / "tum_battery_fit.json"
TRACK_IDS = (0, 1, 10, 100, 103)
DT_S = 0.1                                   # all five tracks are 10 Hz (checked in load_tracks)
WINDOWS_S = (0.1, 0.3, 0.5, 1.0)             # difference windows
SIGN_CONVENTION = ("fit and BMS use DISCHARGE POSITIVE (I_dis > 0); TUM files store discharge as NEGATIVE and are "
                   "multiplied by -1 on load; regen is negative inside the fit")
SCHEMA_VERSION = 1
BLOCK_ROWS = 600                             # 60 s blocks for the bootstrap
N_BOOT = 200
BOOT_SEED = 20240611
TAU_BOUNDS = (1.0, 30.0)
REST_A = 5.0                                 # |I| < 5 A counts as rest-like for the OCV fit
SENTINEL_TEMP_C = -20.0
SOC_JUMP_PTS = 5.0
MAX_HOLD_ROWS = 10                           # a repeat run longer than 1.0 s is a genuine plateau (rest, cruise), not a held frame
STALE_RUN = 4                                # identical voltage for >= 4 consecutive rows is reported as a "held run"


# ---------------------------------------------------------------------------------------------------- loading
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(directory) -> pathlib.Path:
    p = pathlib.Path(directory)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def load_tracks(directory="data/tum_subset", ids=None, discharge_sign: float = -1.0) -> dict:
    """Read the TUM subset CSVs. Returns {"track_<id>": dict}. Values are untouched except the current is multiplied by
    `discharge_sign` (-1: file discharge-negative -> fit discharge-positive). Nothing is filtered here."""
    d = _resolve(directory)
    out = {}
    files = sorted(d.glob("track_*_battery.csv"), key=lambda p: int(p.name.split("_")[1]))
    for p in files:
        tid = int(p.name.split("_")[1])
        if ids is not None and tid not in ids:
            continue
        df = pd.read_csv(p)
        t = df["time"].to_numpy(dtype=float) / 1000.0
        dts = np.unique(np.round(np.diff(t), 6))
        if not (len(dts) == 1 and abs(dts[0] - DT_S) < 1e-6):
            raise ValueError(f"{p.name}: expected a uniform 10 Hz time base, got steps {dts[:5]}")
        out[f"track_{tid}"] = {
            "id": tid, "file": p.name, "sha256": sha256_file(p), "n": len(df), "t_s": t - t[0],
            "v": df["hv_battery_voltage"].to_numpy(dtype=float),
            "i_dis": float(discharge_sign) * df["hv_battery_current"].to_numpy(dtype=float),
            "soc_pct": df["hv_soc"].to_numpy(dtype=float),
            "temp_c": df["hv_temp_min"].to_numpy(dtype=float),
        }
    if not out:
        raise FileNotFoundError(f"no track_*_battery.csv in {d}")
    return out


# ---------------------------------------------------------------------------------------------------- glitch filter
def _runs_of_equal(x: np.ndarray):
    """(start, length) of maximal runs of identical consecutive values."""
    ch = np.r_[True, np.diff(x) != 0]
    st = np.where(ch)[0]
    return st, np.diff(np.r_[st, len(x)])


def glitch_filter(tr: dict) -> tuple[dict, dict]:
    """Remove ONLY: (a) temperature sentinel rows (hv_temp_min < -20 C): the whole row is marked invalid;
    (b) SOC samples that sit more than 5 points from the rolling median of the track's own SOC samples (these are the
    +/-51.2 / ~50 point excursions); (c) REPORT held voltage runs (>= 4 identical consecutive rows). Voltage/current
    VALUES are never altered and no row is dropped except (a). Held samples are not deleted: they are treated as missing
    when the signals are interpolated (see `reconstruct`). Returns (clean_track, report)."""
    n = tr["n"]
    valid = np.ones(n, bool)
    temp = tr["temp_c"].copy()
    sent = np.isfinite(temp) & (temp < SENTINEL_TEMP_C)
    valid[sent] = False
    temp[sent] = np.nan

    soc = tr["soc_pct"].copy()
    idx = np.where(np.isfinite(soc))[0]
    ss = soc[idx]
    raw_jumps = int(np.sum(np.abs(np.diff(ss)) > SOC_JUMP_PTS)) if len(ss) > 1 else 0
    glitch = np.zeros(n, bool)
    if len(ss) > 3:
        size = min(1001, len(ss) if len(ss) % 2 else len(ss) - 1)
        ref = median_filter(ss, size=size, mode="reflect")
        bad = np.abs(ss - ref) > SOC_JUMP_PTS
        glitch[idx[bad]] = True
    soc[glitch] = np.nan

    st, ln = _runs_of_equal(tr["v"])
    held_runs = [(int(a), int(l)) for a, l in zip(st, ln) if l >= STALE_RUN]
    repeated_v = int(np.sum(np.diff(tr["v"]) == 0))
    repeated_i = int(np.sum(np.diff(tr["i_dis"]) == 0))
    clean = dict(tr)
    clean.update({"valid": valid, "temp_c": temp, "soc_pct": soc, "soc_glitch": glitch})
    report = {
        "n_rows": int(n),
        "temp_sentinel_rows_invalidated": int(sent.sum()),
        "temp_sentinel_row_ranges": _ranges(np.where(sent)[0]),
        "soc_samples": int(len(ss)),
        "soc_jumps_over_5_points_raw": raw_jumps,
        "soc_glitch_samples_removed": int(glitch.sum()),
        "soc_glitch_row_ranges": _ranges(np.where(glitch)[0]),
        "held_voltage_runs_ge4_rows": len(held_runs),
        "held_voltage_rows_in_runs_ge4": int(sum(l - 1 for _, l in held_runs)),
        "longest_held_voltage_run_rows": int(max([l for _, l in held_runs] or [0])),
        "repeated_voltage_rows_total": repeated_v,
        "repeated_current_rows_total": repeated_i,
        "repeated_voltage_fraction": round(repeated_v / max(n - 1, 1), 4),
        "repeated_current_fraction": round(repeated_i / max(n - 1, 1), 4),
        "voltage_current_values_altered": 0,
    }
    return clean, report


def _ranges(rows: np.ndarray):
    """Compress sorted row indices into [first, last] ranges (JSON friendly)."""
    if len(rows) == 0:
        return []
    cuts = np.where(np.diff(rows) > 1)[0]
    starts = np.r_[rows[0], rows[cuts + 1]]
    ends = np.r_[rows[cuts], rows[-1]]
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def filter_all(tracks: dict) -> tuple[dict, dict]:
    clean, rep = {}, {}
    for k, tr in tracks.items():
        clean[k], rep[k] = glitch_filter(tr)
    # attach reconstructed (hold-aware) signals
    for k, tr in clean.items():
        tr["v_f"], tr["i_f"] = reconstruct(tr)
        tr["soc"] = _soc_fraction(tr)
    return clean, rep


def _fresh_interp(x: np.ndarray, usable: np.ndarray, plateau_abs=None) -> np.ndarray:
    """Treat repeated values (and unusable rows) as missing; linearly interpolate between update rows. If `plateau_abs` is
    given, identical runs longer than MAX_HOLD_ROWS with |value| < plateau_abs are kept as genuine plateaus (parked car:
    the current really is ~constant). Above that level a long identical run is a STALE frame: during such runs the
    voltage moves by 2 to 20 V (see long_hold_evidence), so the current cannot have been constant."""
    ch = np.r_[True, np.diff(x) != 0]
    if plateau_abs is not None:
        st, ln = _runs_of_equal(x)
        for a, l in zip(st, ln):
            if l > MAX_HOLD_ROWS and abs(x[a]) < plateau_abs:
                ch[a:a + l] = True
    upd = np.where(ch & usable)[0]
    if len(upd) == 0:
        return x.copy()
    return np.interp(np.arange(len(x)), upd, x[upd])


def long_hold_evidence(clean: dict) -> dict:
    """Why long repeated-current runs at |I| > 20 A are treated as stale frames: voltage keeps moving during them."""
    out = {}
    for k, t in clean.items():
        i, v = t["i_dis"], t["v"]
        st, ln = _runs_of_equal(i)
        rng_long = [float(v[a:a + l].max() - v[a:a + l].min()) for a, l in zip(st, ln) if l > MAX_HOLD_ROWS and abs(i[a]) > 20.0]
        rng_ref = []
        for a in range(0, len(v) - 11, 11):
            w = i[a:a + 11]
            if np.abs(w).min() > 20.0 and len(np.unique(w)) >= 6:
                rng_ref.append(float(v[a:a + 11].max() - v[a:a + 11].min()))
        out[k] = {"n_long_runs": len(rng_long),
                  "mean_voltage_range_in_long_runs_v": float(np.mean(rng_long)) if rng_long else None,
                  "mean_voltage_range_in_updating_windows_v": float(np.mean(rng_ref)) if rng_ref else None}
    return out


def reconstruct(tr: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hold-aware voltage and discharge current: repeated samples are missing, interpolate between updates."""
    u = tr["valid"]
    return _fresh_interp(tr["v"], u), _fresh_interp(tr["i_dis"], u, plateau_abs=REST_A)


def _soc_fraction(tr: dict) -> np.ndarray:
    s = tr["soc_pct"]
    idx = np.where(np.isfinite(s))[0]
    if len(idx) == 0:
        return np.full(tr["n"], np.nan)
    return np.interp(np.arange(tr["n"]), idx, s[idx]) / 100.0


# ---------------------------------------------------------------------------------------------------- resistance
def _pairs(v, i, valid, w):
    k = np.arange(0, len(v) - w)
    ok = valid[k] & valid[k + w]
    k = k[ok]
    return k, v[k + w] - v[k], i[k + w] - i[k]


def reff_by_window(clean: dict, windows=WINDOWS_S, seed=BOOT_SEED, n_boot=N_BOOT) -> dict:
    """Resistance (ohm, discharge positive: dV = -R dI) by difference window. Methods, per window:
       held_aware_ols : OLS through origin on hold-aware (interpolated) differences, pooled over tracks (PRIMARY)
       naive_raw      : same on the raw, un-reconstructed series (shows the held-sample bias)
       eiv_deming     : Deming regression through origin on the hold-aware differences, error-variance ratio from the
                        rest-noise of the raw signals (state: it is ~ OLS because measurement noise is tiny against
                        the dI of driving; the dominant error is TIMING, which noise-based EIV cannot see)
       lag_aligned    : per-track integer lag of the current relative to the voltage (chosen on the 0.3 s window by
                        maximum |corr|), then OLS: an upper bound on the short-window values if the lag is timing."""
    names = list(clean)
    # rest noise: std of first differences of the raw signals where the car is parked (|I| < 5 A, both fresh)
    dvn, din = [], []
    for tr in clean.values():
        v, i, ok = tr["v"], tr["i_dis"], tr["valid"]
        m = ok[1:] & ok[:-1] & (np.abs(i[1:]) < REST_A) & (np.abs(i[:-1]) < REST_A)
        dvn.append(np.diff(v)[m]); din.append(np.diff(i)[m])
    sv = float(np.std(np.concatenate(dvn)) / np.sqrt(2.0))
    si = float(np.std(np.concatenate(din)) / np.sqrt(2.0))
    lam = (sv / si) ** 2 if si > 0 else 0.0

    # per-track best lag (samples) at the 0.3 s window; positive lag = look later in current
    lags = {}
    for name in names:
        tr = clean[name]
        best = (0, -1.0)
        for s in range(-6, 3):
            k = np.arange(max(0, -s), len(tr["v"]) - 3 - max(0, s))
            ok = tr["valid"][k] & tr["valid"][k + 3]
            dv = tr["v_f"][k + 3] - tr["v_f"][k]
            di = tr["i_f"][k + 3 + s] - tr["i_f"][k + s]
            r = abs(np.corrcoef(dv[ok], di[ok])[0, 1])
            if r > best[1]:
                best = (s, r)
        lags[name] = best

    res = {"windows_s": list(windows), "per_track": {}, "pooled": {}, "ci95": {}, "method_notes": {}}
    rng = np.random.default_rng(seed)
    meth_keys = ("held_aware_ols", "naive_raw", "eiv_deming", "lag_aligned")
    blocks = {m: {} for m in meth_keys}   # method -> window -> list over tracks of per-block sums arrays
    for wi, wsec in enumerate(windows):
        w = int(round(wsec / DT_S))
        for name in names:
            tr = clean[name]
            for m in meth_keys:
                if m == "naive_raw":
                    k, dv, di = _pairs(tr["v"], tr["i_dis"], tr["valid"], w)
                elif m == "lag_aligned":
                    s = lags[name][0]
                    kk = np.arange(max(0, -s), len(tr["v"]) - w - max(0, s))
                    ok = tr["valid"][kk] & tr["valid"][kk + w]
                    k = kk[ok]
                    dv = tr["v_f"][k + w] - tr["v_f"][k]
                    di = tr["i_f"][k + w + s] - tr["i_f"][k + s]
                else:
                    k, dv, di = _pairs(tr["v_f"], tr["i_f"], tr["valid"], w)
                nb = int(len(tr["v"]) // BLOCK_ROWS + 1)
                b = k // BLOCK_ROWS
                S = np.zeros((nb, 3))
                np.add.at(S[:, 0], b, di * di)
                np.add.at(S[:, 1], b, di * dv)
                np.add.at(S[:, 2], b, dv * dv)
                blocks[m].setdefault(wsec, {})[name] = S

    def solve(m, S):
        sxx, sxy, syy = S[0], -S[1], S[2]           # y = -dV, x = dI ; R = y/x
        if m == "eiv_deming":
            if sxy <= 0:
                return float("nan")
            a = syy - lam * sxx
            return (a + np.sqrt(a * a + 4.0 * lam * sxy * sxy)) / (2.0 * sxy)
        return sxy / sxx if sxx > 0 else float("nan")

    for m in meth_keys:
        res["pooled"][m], res["per_track"][m], res["ci95"][m] = {}, {}, {}
        for wsec in windows:
            key = f"{wsec:g}" if wsec != 1.0 else "1.0"
            per = {n: float(solve(m, blocks[m][wsec][n].sum(0))) for n in names}
            tot = sum(blocks[m][wsec][n].sum(0) for n in names)
            res["per_track"][m][key] = per
            res["pooled"][m][key] = float(solve(m, tot))
            boots = []
            for _ in range(n_boot):
                acc = np.zeros(3)
                for n in names:
                    B = blocks[m][wsec][n]
                    pick = rng.integers(0, len(B), len(B))
                    acc += B[pick].sum(0)
                boots.append(solve(m, acc))
            res["ci95"][m][key] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    res["rest_noise"] = {"sigma_v": sv, "sigma_i": si, "deming_lambda_ohm2": lam}
    res["best_lag_samples_by_track"] = {n: {"lag": int(lags[n][0]), "abs_corr": float(lags[n][1])} for n in names}
    res["method_notes"] = {
        "primary": "held_aware_ols (interpolate between update rows, OLS through origin, pooled by sum of squares)",
        "ci": f"block bootstrap, {BLOCK_ROWS} rows (60 s) blocks within tracks, {n_boot} resamples, seed {seed}",
        "lag_note": "voltage responds 0.1 to 0.3 s after the recorded current on tracks 0/1/10: this is physics (fast polarization) "
                    "and/or CAN timing; the data cannot separate them",
    }
    return res


def hold_rule_sensitivity(tracks: dict) -> dict:
    """Pooled held-aware R_eff (ohm) if repeated current runs longer than G rows were treated as plateaus at ANY level
    (G = 4, 6, 10, 20 rows) instead of stale frames. The short windows depend strongly on this choice; 1.0 s does not."""
    global MAX_HOLD_ROWS
    keep = MAX_HOLD_ROWS
    out = {}
    try:
        for g in (4, 6, 10, 20):
            MAX_HOLD_ROWS = g
            clean = {}
            for k, tr in tracks.items():
                c, _ = glitch_filter(tr)
                c["v_f"] = _fresh_interp(c["v"], c["valid"])
                c["i_f"] = _fresh_interp(c["i_dis"], c["valid"], plateau_abs=1.0e9)
                clean[k] = c
            r = reff_by_window(clean, n_boot=1)["pooled"]["held_aware_ols"]
            out[f"plateau_if_run_gt_{g}_rows_any_level"] = r
    finally:
        MAX_HOLD_ROWS = keep
    return out


def _window_key(w):
    return "1.0" if w == 1.0 else f"{w:g}"


# ---------------------------------------------------------------------------------------------------- OCV
def ocv_value(k, soc):
    s = np.clip(np.asarray(soc, float), 1e-3, 1.0 - 1e-3)
    return k[0] + k[1] * np.log(s) + k[2] * np.log(1.0 - s)


def _rest_points(clean: dict):
    rows = []
    for name, tr in clean.items():
        m = tr["valid"] & (np.abs(tr["i_dis"]) < REST_A) & (np.abs(tr["i_f"]) < REST_A) & np.isfinite(tr["soc"])
        rows.append((name, np.where(m)[0]))
    return rows


def fit_ocv(clean: dict, boot_seed=BOOT_SEED, n_boot=N_BOOT, with_uncertainty=True) -> dict:
    """V = K0 + K1 ln(s) + K2 ln(1-s) on rest-like points (|I| < 5 A), each track weighted equally. Bounds K1 >= 0,
    K2 <= 0 make OCV monotone non-decreasing in SOC on (0,1). Linear least squares, so exact."""
    rest = _rest_points(clean)
    if sum(len(r) for _, r in rest) < 20:
        raise ValueError("not enough rest-like points for the OCV fit")

    def design(sel):
        A, y, wts = [], [], []
        for (name, rows), pick in zip(rest, sel):
            if len(pick) == 0:
                continue
            tr = clean[name]
            s = np.clip(tr["soc"][pick], 1e-3, 1 - 1e-3)
            A.append(np.c_[np.ones(len(pick)), np.log(s), np.log(1 - s)])
            y.append(tr["v_f"][pick])
            wts.append(np.full(len(pick), 1.0 / np.sqrt(len(pick))))
        A, y, w = np.vstack(A), np.concatenate(y), np.concatenate(wts)
        return A, y, w

    def solve(sel):
        A, y, w = design(sel)
        r = lsq_linear(A * w[:, None], y * w, bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, 0.0]), method="bvls")
        return r.x, A, w

    sel0 = [r for _, r in rest]
    k, A, w = solve(sel0)
    soc_all = np.concatenate([clean[n]["soc"][r] for n, r in rest])
    out = {"k": [float(x) for x in k],
           "soc_range_of_rest_points": [float(soc_all.min()), float(soc_all.max())],
           "n_rest_points": int(len(soc_all)),
           "bounds_active": {"K1_at_0": bool(k[1] <= 1e-9), "K2_at_0": bool(k[2] >= -1e-9)}}
    clusters = {}
    for n, r in rest:
        if len(r):
            clusters[n] = {"soc_mean": float(clean[n]["soc"][r].mean()), "v_mean": float(clean[n]["v_f"][r].mean()),
                           "v_sd": float(clean[n]["v_f"][r].std()), "n": int(len(r))}
    out["rest_clusters"] = clusters
    sv = np.linalg.svd(A * w[:, None], compute_uv=False)
    out["design_condition_number"] = float(sv[0] / max(sv[-1], 1e-300))
    if with_uncertainty:
        rng = np.random.default_rng(boot_seed)
        grid = np.array([0.5, 0.6, 0.8, 0.98])
        ks, vals = [], []
        for _ in range(n_boot):
            sel = []
            for n, r in rest:
                if len(r) == 0:
                    sel.append(r); continue
                blk = r // BLOCK_ROWS
                ub = np.unique(blk)
                pick_b = rng.choice(ub, len(ub))
                sel.append(np.concatenate([r[blk == b] for b in pick_b]))
            kb, _, _ = solve(sel)
            ks.append(kb); vals.append(ocv_value(kb, grid))
        ks, vals = np.array(ks), np.array(vals)
        out["k_ci95"] = [[float(np.percentile(ks[:, j], 2.5)), float(np.percentile(ks[:, j], 97.5))] for j in range(3)]
        out["ocv_at_soc"] = {f"{g:g}": {"mid": float(ocv_value(k, g)),
                                         "ci95": [float(np.percentile(vals[:, a], 2.5)), float(np.percentile(vals[:, a], 97.5))]}
                             for a, g in enumerate(grid)}
        # leave-one-track-out spread of the SOC-slope between 0.58 and 0.98 (shows how much one track decides)
        loo = {}
        for drop in range(len(rest)):
            sel = [r if j != drop else r[:0] for j, (_, r) in enumerate(rest)]
            if sum(len(s) for s in sel) < 20:
                continue
            kl, _, _ = solve(sel)
            loo[rest[drop][0]] = float(ocv_value(kl, 0.98) - ocv_value(kl, 0.58))
        out["delta_ocv_0p58_to_0p98_when_track_dropped_V"] = loo
    out["note"] = ("FLAT and WEAKLY IDENTIFIED: rest points sit on four SOC clusters (about 0.53, 0.58-0.63, 0.80, 0.985); "
                   "rest voltage still contains unrelaxed polarization; K1 and K2 are strongly correlated and the lower "
                   "SOC range (< 0.5) is NOT observed at all. Use OCV(s) only inside the observed SOC range.")
    return out


# ---------------------------------------------------------------------------------------------------- RC branch
def _filt(x, tau):
    a = float(np.exp(-DT_S / tau))
    return lfilter([1.0 - a], [1.0, -a], x, zi=[a * x[0]])[0]


def _demeaned_design(clean, ocv_k, tau, trend=True):
    """Per-track nuisance (offset, and a linear drift if `trend`) is profiled out by projection. It absorbs the OCV
    level error and slow SOC/relaxation drift that would otherwise be mistaken for a slow RC branch. Each track gets
    equal total weight (rows scaled by 1/sqrt(n)) so the 18 000-row track 1 does not decide alone."""
    X, y, w = [], [], []
    for tr in clean.values():
        m = tr["valid"]
        f = _filt(tr["i_f"], tau)
        yy = (tr["v_f"] - ocv_value(ocv_k, tr["soc"]))[m]
        cols = np.c_[tr["i_f"], f][m]
        t = tr["t_s"][m]
        B = np.c_[np.ones(m.sum()), (t - t.mean()) / (np.ptp(t) + 1e-9)] if trend else np.ones((m.sum(), 1))
        Q, _ = np.linalg.qr(B)
        X.append(cols - Q @ (Q.T @ cols))
        y.append(yy - Q @ (Q.T @ yy))
        w.append(np.full(m.sum(), 1.0 / np.sqrt(m.sum())))
    return np.vstack(X), np.concatenate(y), np.concatenate(w)


def _rc_solve(clean, ocv_k, tau):
    X, y, w = _demeaned_design(clean, ocv_k, tau)
    coef, rn = nnls(X * w[:, None], -y * w)        # V - OCV = -(R0 I + R1 F)  => nnls with R >= 0
    rmse = float(np.sqrt(np.mean((y + X @ coef) ** 2)))
    return coef, float(rn), rmse


def _rc_search(clean, ocv_k, lo_tau, hi_tau, n_grid=25):
    grid = np.exp(np.linspace(np.log(lo_tau), np.log(hi_tau), n_grid))
    rn = [_rc_solve(clean, ocv_k, t)[1] for t in grid]
    j = int(np.argmin(rn))
    lo, hi = np.log(grid[max(j - 1, 0)]), np.log(grid[min(j + 1, len(grid) - 1)])
    r = minimize_scalar(lambda lt: _rc_solve(clean, ocv_k, float(np.exp(lt)))[1], bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-4})
    tau = float(np.clip(np.exp(r.x), lo_tau, hi_tau))
    coef, _, rmse = _rc_solve(clean, ocv_k, tau)
    return tau, coef, rmse


def fit_rc(clean: dict, ocv_k, tau_bounds=TAU_BOUNDS, with_jackknife=True) -> dict:
    """V = OCV(s) + nuisance_track - R0*I - R1*F_tau(I), F = first-order low-pass of I. For fixed tau this is linear in
    (R0, R1) (non-negative LS); tau is found by grid + bounded 1-D search inside `tau_bounds`. With >= 3 tracks a
    leave-one-track-out refit (including the OCV) shows whether the branch is identified."""
    tau, coef, rmse = _rc_search(clean, ocv_k, *tau_bounds)
    out = {"r0_ohm": float(coef[0]), "r1_ohm": float(coef[1]), "tau1_s": tau, "rmse_in_sample_v": rmse,
           "tau_at_bound": bool(min(abs(tau - tau_bounds[0]), abs(tau - tau_bounds[1])) < 0.05)}
    if with_jackknife and len(clean) >= 3:
        jk = {}
        for drop in clean:
            sub = {k: v for k, v in clean.items() if k != drop}
            ok = fit_ocv(sub, with_uncertainty=False)
            t2, c2, _ = _rc_search(sub, ok["k"], *tau_bounds)
            jk[drop] = {"r0_ohm": float(c2[0]), "r1_ohm": float(c2[1]), "tau1_s": t2}
        taus = [v["tau1_s"] for v in jk.values()]
        out["drop_one_track"] = jk
        out["identified"] = bool(not out["tau_at_bound"] and max(taus) / min(taus) < 3.0)
        out["identified_note"] = ("tau and R1 are taken as IDENTIFIED only if the pooled tau is off both bounds and the "
                                  "drop-one-track refits agree within 3x; otherwise the slow/fast split is not supported by the data")
    return out


def rc_free_tau_diagnostic(clean, ocv_k) -> dict:
    """Same fit with tau allowed below the 1 s lower bound: what does the data prefer? (diagnostic only)"""
    grid = np.exp(np.linspace(np.log(0.05), np.log(30.0), 40))
    rows = []
    for t in grid:
        coef, rn, rmse = _rc_solve(clean, ocv_k, float(t))
        rows.append((float(t), float(coef[0]), float(coef[1]), rmse))
    best = min(rows, key=lambda r: r[3])
    return {"best_tau_s": best[0], "r0_ohm": best[1], "r1_ohm": best[2], "rmse_in_sample_v": best[3],
            "rmse_at_tau": {f"{t:.3g}": round(r, 4) for t, _, _, r in rows[::4]}}


# ---------------------------------------------------------------------------------------------------- bases
def real_bases(clean: dict) -> dict:
    """v_nom: median over tracks of the per-track median bus voltage. p_base: the largest discharge power seen in any
    track (a USE maximum, not a pack rating). i_base = p_base / v_nom."""
    med = [float(np.median(tr["v"][tr["valid"]])) for tr in clean.values()]
    pk = []
    for tr in clean.values():
        m = tr["valid"]
        pk.append(float(np.max(tr["v"][m] * tr["i_dis"][m])))
    v_nom, p_base = float(np.median(med)), float(max(pk))
    return {"v_nom": v_nom, "p_base_w": p_base, "i_base_a": p_base / v_nom,
            "median_bus_v_by_track": {k: m for k, m in zip(clean, med)},
            "peak_discharge_power_w_by_track": {k: p for k, p in zip(clean, pk)}}


# ---------------------------------------------------------------------------------------------------- predictors / validation
def _generic_params():
    """Constants of the existing generic toy pack, read from battery.Battery (not duplicated)."""
    from battery import Battery
    b = Battery()
    return {"ocv": b.ocv, "r0": b.R0_25, "rc": b.Rc, "r1": b.R1, "tau1": b.tau1, "v_ref": 300.0, "p_ref": 65000.0}


def predict_fit(model: dict, tr: dict) -> np.ndarray:
    return (ocv_value(model["ocv_k"], tr["soc"]) - model["r0_ohm"] * tr["i_f"]
            - model["r1_ohm"] * _filt(tr["i_f"], model["tau1_s"]))


def _train_models(train: dict) -> dict:
    ocv = fit_ocv(train, with_uncertainty=False)
    rc = fit_rc(train, ocv["k"], with_jackknife=False)
    bases = real_bases(train)
    # constant resistance: pooled hold-aware R_eff(0.5 s) of the training tracks, offset from their mean
    rr = reff_by_window(train, windows=(0.5,), n_boot=2)
    r_c = rr["pooled"]["held_aware_ols"]["0.5"]
    a_c = float(np.mean(np.concatenate([tr["v_f"][tr["valid"]] + r_c * tr["i_f"][tr["valid"]] for tr in train.values()])))
    v_c = float(np.mean(np.concatenate([tr["v_f"][tr["valid"]] for tr in train.values()])))
    g = _generic_params()
    # generic pack scaled to the real pack: one voltage gain from the TRAINING mean voltage, per-unit resistances
    ocv_g = np.concatenate([[g["ocv"](s) for s in tr["soc"][tr["valid"]]] for tr in train.values()])
    k_v = v_c / float(np.mean(ocv_g))
    zb_real = bases["v_nom"] ** 2 / bases["p_base_w"]
    zb_toy = g["v_ref"] ** 2 / g["p_ref"]
    to_real = zb_real / zb_toy
    generic = {"k_v": k_v, "r_ohm": (g["r0"] + g["rc"]) * to_real, "r1_ohm": g["r1"] * to_real, "tau1_s": g["tau1"], "ocv": g["ocv"]}
    return {"fit": {"ocv_k": ocv["k"], "r0_ohm": rc["r0_ohm"], "r1_ohm": rc["r1_ohm"], "tau1_s": rc["tau1_s"]},
            "const_v": {"v": v_c}, "const_r": {"a": a_c, "r_ohm": r_c}, "generic": generic}


def _predict(name: str, m: dict, tr: dict) -> np.ndarray:
    if name == "fit":
        return predict_fit(m, tr)
    if name == "const_v":
        return np.full(tr["n"], m["v"])
    if name == "const_r":
        return m["a"] - m["r_ohm"] * tr["i_f"]
    if name == "generic":
        ocv = np.array([m["ocv"](s) for s in tr["soc"]])
        return m["k_v"] * ocv - m["r_ohm"] * tr["i_f"] - m["r1_ohm"] * _filt(tr["i_f"], m["tau1_s"])
    raise KeyError(name)


MODEL_NAMES = ("fit", "const_v", "const_r", "generic")


def leave_one_track_out(clean: dict) -> dict:
    """Train on 4 tracks, predict the 5th with each model; RMSE (V) per fold on valid rows. Overall = sqrt(mean of fold MSE)."""
    names = list(clean)
    folds = {m: {} for m in MODEL_NAMES}
    for held in names:
        train = {k: v for k, v in clean.items() if k != held}
        models = _train_models(train)
        tr = clean[held]
        m = tr["valid"]
        for nm in MODEL_NAMES:
            err = _predict(nm, models[nm], tr)[m] - tr["v_f"][m]
            folds[nm][held] = float(np.sqrt(np.mean(err ** 2)))
    overall = {nm: float(np.sqrt(np.mean([r ** 2 for r in folds[nm].values()]))) for nm in MODEL_NAMES}
    best_name = min(("const_v", "const_r", "generic"), key=lambda n: overall[n])
    diff = np.array([folds["fit"][h] - folds[best_name][h] for h in names])
    return {"loo_rmse_v": overall, "per_fold_rmse_v": folds, "best_baseline": best_name,
            "fit_minus_best_baseline_v": {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                                          "folds_fit_wins": int((diff < 0).sum()), "n_folds": int(len(diff))}}


def decide_status(loo_rmse_v: dict) -> tuple[str, str]:
    """'ok' iff the fit has strictly lower leave-one-track-out RMSE than EVERY baseline."""
    base = {k: loo_rmse_v[k] for k in ("const_v", "const_r", "generic")}
    best = min(base, key=base.get)
    if loo_rmse_v["fit"] < base[best]:
        return "ok", (f"fit RMSE {loo_rmse_v['fit']:.3f} V beats the best baseline ({best}, {base[best]:.3f} V)")
    return "not_better_than_baseline", (f"fit RMSE {loo_rmse_v['fit']:.3f} V does not beat the best baseline "
                                        f"({best}, {base[best]:.3f} V); the toy should keep the generic pack until a better fit exists")


# ---------------------------------------------------------------------------------------------------- envelope
ENV_GRID = tuple(round(0.5 + 0.05 * j, 2) for j in range(11))      # SOC 0.50 .. 1.00
ENV_HALF_WIDTH = 0.025
ENV_Q = 0.995
ENV_MIN_N = 1500


def envelope_eval(env: dict, soc, which: str):
    return np.interp(soc, env["soc_grid"], env["i_dis_lim_a" if which == "dis" else "i_chg_lim_a"])


def envelope_coverage(env: dict, soc, i_dis) -> dict:
    """Fraction of discharge samples (I>0) below i_dis_lim(soc) and of regen samples (I<0) with |I| below i_chg_lim(soc)."""
    soc, i = np.asarray(soc), np.asarray(i_dis)
    d, c = i > 0, i < 0
    return {"dis": float(np.mean(i[d] <= envelope_eval(env, soc[d], "dis"))),
            "chg": float(np.mean(-i[c] <= envelope_eval(env, soc[c], "chg")))}


def _pooled_samples(clean):
    soc = np.concatenate([tr["soc"][tr["valid"]] for tr in clean.values()])
    i = np.concatenate([tr["i_dis"][tr["valid"]] for tr in clean.values()])
    return soc, i


def observed_envelope(clean: dict, grid=ENV_GRID, q=ENV_Q) -> dict:
    """OBSERVED-USE envelope of current vs SOC: per SOC bin, the q-quantile of discharge current and of regen
    magnitude. Bins with too few samples use the pooled quantile (flagged). Inflated by the smallest 0.5 % steps so
    that >= 99 % of the observed discharge AND regen samples lie inside after linear interpolation (true by
    construction; it is not a BMS limit)."""
    soc, i = _pooled_samples(clean)
    dis, chg = i > 0, i < 0
    pooled = {"dis": float(np.quantile(i[dis], q)), "chg": float(np.quantile(-i[chg], q))}
    vals = {"dis": [], "chg": []}
    counts, fallback = [], []
    for g in grid:
        m = np.abs(soc - g) <= ENV_HALF_WIDTH
        nd, nc = int(np.sum(m & dis)), int(np.sum(m & chg))
        counts.append([nd, nc])
        vals["dis"].append(float(np.quantile(i[m & dis], q)) if nd >= ENV_MIN_N else pooled["dis"])
        vals["chg"].append(float(np.quantile(-i[m & chg], q)) if nc >= ENV_MIN_N // 3 else pooled["chg"])
        fallback.append([nd < ENV_MIN_N, nc < ENV_MIN_N // 3])
    env = {"soc_grid": [float(g) for g in grid], "i_dis_lim_a": vals["dis"], "i_chg_lim_a": vals["chg"]}
    f = 1.0
    for _ in range(200):
        e2 = dict(env, i_dis_lim_a=[x * f for x in vals["dis"]], i_chg_lim_a=[x * f for x in vals["chg"]])
        cov = envelope_coverage(e2, soc, i)
        if min(cov.values()) >= 0.995:
            break
        f += 0.005
    env = {"soc_grid": env["soc_grid"], "i_dis_lim_a": [round(x * f, 3) for x in vals["dis"]],
           "i_chg_lim_a": [round(x * f, 3) for x in vals["chg"]]}
    env["coverage"] = {k: round(v, 5) for k, v in envelope_coverage(env, soc, i).items()}
    env["quantile"] = q
    env["inflation_factor"] = round(f, 4)
    env["samples_per_bin_dis_chg"] = counts
    env["bin_uses_pooled_fallback_dis_chg"] = fallback
    env["pooled_quantile_a"] = pooled
    env["observed_soc_range"] = [float(np.nanmin(soc)), float(np.nanmax(soc))]
    env["i_dis_lim_flat_a"] = [round(pooled["dis"] * f, 3)] * len(grid)
    env["i_chg_lim_flat_a"] = [round(pooled["chg"] * f, 3)] * len(grid)
    env["soc_dependence_identified"] = False
    env["soc_dependence_note"] = ("each SOC cluster is ONE trip, so SOC dependence is confounded with driving style/route/ambient; "
                                  "the per-bin values are what was observed, not a SOC law. *_flat_a is the pooled alternative.")
    env["note"] = ("observed-use envelope: the q-quantile of what the cars drew, NOT a BMS limit and NOT a pack rating. "
                   "Covers >= 99 % of the observed samples by construction. Only SOC 0.52-0.60, 0.62, 0.80 and 0.985 were "
                   "observed; bins marked as fallback use the pooled quantile and outside [0.50, 1.00] the value is held flat.")
    return env


# ---------------------------------------------------------------------------------------------------- scaling
def scale_to_toy(real: dict, p_base_toy_w: float = 65000.0, v_toy: float = 300.0, motor_i_lim_a: float = 250.0) -> dict:
    """Per-unit similarity scaling (R_pu = R / Z_base, Z_base = V^2/P).
        R_toy = R_pu * v_toy**2 / p_base_toy_w      I_base_toy = p_base_toy_w / v_toy
        OCV scaled by v_toy / v_nom_real            current limits scaled by I_base_toy / I_base_real
    `real` is a (partial) fit dict with `pack_real` and `envelope_real`. Returns {"scaling": {...}, "toy_battery_kwargs": {...}}.
    NOT preserved: thermal mass, absolute energy (q_ah kept at the toy value), and cell time constants are kept in
    seconds, not scaled. Temperature dependence is the generic Arrhenius R0(T) of battery.py (an ASSUMPTION)."""
    pr, en = real["pack_real"], real["envelope_real"]
    i_base_toy = p_base_toy_w / v_toy
    cur = i_base_toy / pr["i_base_a"]
    r_pu = pr["r_eff_pu"]
    zb_toy = v_toy ** 2 / p_base_toy_w
    r_tot = {k: float(r_pu[k] * zb_toy) for k in ("lo", "mid", "hi")}
    dis = [float(x * cur) for x in en["i_dis_lim_a"]]
    chg = [float(x * cur) for x in en["i_chg_lim_a"]]
    # split the mid total into ohmic + RC for Battery(...): RC branch from the time-domain fit, per unit
    rc = pr["rc"]
    if rc.get("identified", False):
        r1_toy, tau, rc_source = float(rc["r1_pu"] * zb_toy), float(rc["tau1_s"]), "fitted (identified)"
    else:                                       # data cannot fix the slow branch: keep the generic toy RC, say so
        g = _generic_params()
        r1_toy, tau, rc_source = float(g["r1"]), float(g["tau1"]), "generic toy RC kept (fitted R1/tau NOT identified)"
    w_mid = 0.5
    ns, r_bus = 80, 5.0e-3                      # n_series and busbar kept at the generic values (not identified by the data)
    r0_total = r_tot["mid"] - r1_toy * (1.0 - np.exp(-w_mid / tau))     # series (R0 + R_bus) so that sag at 0.5 s = mid
    r0_cell = max((r0_total - r_bus) / ns, 1.0e-5)
    kwargs = {"n_series": ns, "q_ah": 60.0, "r0_cell": float(r0_cell), "r1_cell": float(r1_toy / ns), "tau1": tau,
              "r_bus": r_bus, "i_dis_max": float(max(dis)), "i_chg_max": float(max(chg))}
    sc = {"p_base_toy_w": float(p_base_toy_w), "v_toy": float(v_toy), "i_base_toy": float(i_base_toy),
          "default_r_pu": float(r_pu["mid"]), "r_total_toy_ohm": r_tot, "i_dis_lim_toy_a": dis, "i_chg_lim_toy_a": chg,
          "rc_source": rc_source, "ocv_scale_factor": float(v_toy / pr["v_nom"]), "current_scale_factor": float(cur),
          "ocv_toy_v_at_soc": {k: float(v * v_toy / pr["v_nom"]) for k, v in pr.get("ocv_real_v_at_soc", {}).items()},
          "ocv_toy_note": "Battery() has a fixed generic OCV table (about 302 V at SOC 0.5 to 332 V at 0.9); the fitted real OCV is much FLATTER "
                          "(about 1.2 % over SOC 0.6 to 0.98). It cannot be passed through Battery(...) kwargs.",
          "motor_consistency": {
              "toy_motor_i_lim_a": motor_i_lim_a, "max_scaled_i_dis_a": float(max(dis)),
              "headroom_fraction_of_motor_limit": float(max(dis) / motor_i_lim_a),
              "consistent_with_toy_motor": bool(0.5 * motor_i_lim_a <= max(dis) <= motor_i_lim_a),
              "consistency_rule": "scaled max discharge limit must be <= the motor's 250 A and >= 125 A (else it either exceeds the motor "
                                  "or throttles it far below its rating)",
              "max_scaled_i_chg_a": float(max(chg)),
              "regen_over_discharge_current": float(max(chg) / max(dis))},
          "not_preserved": ["thermal mass", "absolute energy (q_ah)", "cell time constants are kept in seconds",
                            "temperature dependence (generic Arrhenius R0(T), assumption)"]}
    return {"scaling": sc, "toy_battery_kwargs": kwargs}


# ---------------------------------------------------------------------------------------------------- assembly
def _round(obj, sig=9):
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return float(f"{x:.{sig}g}")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _round(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_round(v, sig) for v in obj]
    return obj


def canonical_hash(fit: dict) -> str:
    """sha256 of the canonical JSON of the fit with source.fit_hash blanked."""
    d = json.loads(json.dumps(fit))
    d["source"]["fit_hash"] = ""
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_pack(tracks: dict) -> dict:
    """Fit the real pack from loaded (unfiltered) tracks and assemble the dict that satisfies interfaces.validate_fit_dict.
    Raises ValueError if the pooled resistance is not positive (sign convention violated)."""
    clean, reports = filter_all(tracks)
    reff = reff_by_window(clean)
    prim = reff["pooled"]["held_aware_ols"]
    if not all(prim[_window_key(w)] > 0 for w in WINDOWS_S):
        raise ValueError("pooled resistance <= 0: current sign convention is wrong (discharge must be positive)")
    bases = real_bases(clean)
    ocv = fit_ocv(clean)
    rc = fit_rc(clean, ocv["k"])
    rc_free = rc_free_tau_diagnostic(clean, ocv["k"])
    loo = leave_one_track_out(clean)
    status, reason = decide_status(loo["loo_rmse_v"])
    env = observed_envelope(clean)

    zb = bases["v_nom"] ** 2 / bases["p_base_w"]
    r_by_w = {_window_key(w): prim[_window_key(w)] for w in WINDOWS_S}
    r_pu_by_w = {k: v / zb for k, v in r_by_w.items()}
    r_pu_ci = {k: [x / zb for x in reff["ci95"]["held_aware_ols"][k]] for k in r_by_w}
    long_w = [r_pu_by_w[k] for k in ("0.3", "0.5", "1.0")]
    pack_real = {
        "v_nom": bases["v_nom"], "p_base_w": bases["p_base_w"], "i_base_a": bases["i_base_a"],
        "base_note": "p_base_w is the largest discharge power SEEN in the five tracks (use maximum, not a pack rating)",
        "r_eff_by_window_ohm": r_by_w,
        "r_eff_by_window_pu": r_pu_by_w,
        "r_eff_by_window_pu_ci95": r_pu_ci,
        "r_eff_pu": {"lo": min(long_w), "mid": r_pu_by_w["0.5"], "hi": max(long_w)},
        "r_eff_pu_ci95_outer": {"lo": min(v[0] for k, v in r_pu_ci.items() if k != "0.1"),
                                "hi": max(v[1] for k, v in r_pu_ci.items() if k != "0.1")},
        "r_eff_pu_definition": "lo/hi = min/max of the held-aware pooled R_eff over the 0.3, 0.5, 1.0 s windows, mid = 0.5 s window",
        "ocv_k": ocv["k"],
        "rc": {"r1_pu": rc["r1_ohm"] / zb, "tau1_s": rc["tau1_s"], "identified": bool(rc.get("identified", False))},
        "rc_detail": {**rc, "r0_pu": rc["r0_ohm"] / zb, "tau_bounds_s": list(TAU_BOUNDS),
                      "free_tau_diagnostic": rc_free,
                      "r0_plus_r1_ohm": rc["r0_ohm"] + rc["r1_ohm"]},
        "ocv_detail": ocv,
        "ocv_real_v_at_soc": {"0.6": float(ocv_value(ocv["k"], 0.6)), "0.8": float(ocv_value(ocv["k"], 0.8)),
                              "0.98": float(ocv_value(ocv["k"], 0.98))},
        "r_eff_methods": {
            "pooled_ohm": reff["pooled"], "per_track_ohm": reff["per_track"], "ci95_ohm": reff["ci95"],
            "rest_noise": reff["rest_noise"], "long_hold_evidence": long_hold_evidence(clean),
            "hold_rule_sensitivity_ohm": hold_rule_sensitivity(tracks), "best_lag_samples_by_track": reff["best_lag_samples_by_track"],
            "notes": reff["method_notes"],
            "eiv_statement": "EIV estimate = Deming regression through the origin with the noise variance ratio taken from rest-time "
                             "sample noise; it is ~ OLS because sample noise is tiny against the driving dI. The dominant "
                             "errors-in-variables effect is TIMING (held frames, current-voltage lag), reported as the lag-aligned "
                             "diagnostic. No claim is made that either removes the timing error.",
        },
        "temperature": "NOT FITTED: tracks span 20-34 C, one band per track, no cold data. Toy keeps the generic Arrhenius R0(T) (assumption).",
        "track_bases": {"median_bus_v_by_track": bases["median_bus_v_by_track"],
                        "peak_discharge_power_w_by_track": bases["peak_discharge_power_w_by_track"]},
    }
    partial = {"pack_real": pack_real, "envelope_real": env}
    s65 = scale_to_toy(partial, 65000.0, 300.0)
    s31 = scale_to_toy(partial, 31100.0, 300.0)
    gen = _generic_params()
    g_pu = {}
    for pb in (65000.0, 31100.0):
        zbt = 300.0 ** 2 / pb
        g_pu[f"{pb:g}"] = {"r0_plus_rbus_pu": (gen["r0"] + gen["rc"]) / zbt, "r1_pu": gen["r1"] / zbt,
                           "total_pu": (gen["r0"] + gen["rc"] + gen["r1"]) / zbt}
    files = {tr["file"]: tr.get("sha256", "n/a") for tr in tracks.values()}
    nglitch = {k: v for k, v in reports.items()}
    fit = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "sign_convention": SIGN_CONVENTION,
        "source": {"files": files, "n_tracks": len(tracks), "fit_hash": "",
                   "filters": {"temp_sentinel_c": SENTINEL_TEMP_C, "soc_jump_points": SOC_JUMP_PTS, "held_voltage_run_rows": STALE_RUN,
                               "rest_current_a": REST_A, "block_rows": BLOCK_ROWS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                               "tau_bounds_s": list(TAU_BOUNDS)},
                   "original_full_csv_sha256": _source_original_hashes()},
        "pack_real": pack_real,
        "envelope_real": env,
        "scaling": s65["scaling"],
        "scaling_sensitivity": {"65000": s65["scaling"], "31100": s31["scaling"]},
        "toy_battery_kwargs": s65["toy_battery_kwargs"],
        "toy_battery_kwargs_alt_p_base_31100": s31["toy_battery_kwargs"],
        "generic_toy_pack_per_unit": g_pu,
        "validation": {"loo_rmse_v": loo["loo_rmse_v"], "status_reason": reason, "glitches_removed": nglitch,
                       "per_fold_rmse_v": loo["per_fold_rmse_v"], "best_baseline": loo["best_baseline"],
                       "fit_minus_best_baseline_v": loo["fit_minus_best_baseline_v"],
                       "protocol": "leave-one-track-out on 5 tracks; RMSE of predicted pack voltage from SOC and current on valid rows; "
                                   "overall = sqrt(mean fold MSE); all models see the same hold-aware current; baselines: const_v = training "
                                   "mean voltage, const_r = a - R_eff(0.5 s)*I, generic = battery.Battery OCV/R0/Rc/R1/tau scaled to the real "
                                   "pack (one voltage gain from the training mean, per-unit resistances at 65 kW toy base)"},
    }
    fit = _round(fit)
    fit["source"]["fit_hash"] = canonical_hash(fit)
    return fit


def _source_original_hashes():
    p = DEFAULT_DIR / "SOURCE.json"
    try:
        return json.loads(p.read_text()).get("files_sha256_of_full_original_csv", {})
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    sys.path.insert(0, str(ROOT))
    from interfaces import validate_fit_dict
    tracks = load_tracks(DEFAULT_DIR)
    fit = fit_pack(tracks)
    bad = validate_fit_dict(fit)
    if bad:
        print("schema problems:", bad)
        return 1
    DEFAULT_OUT.write_text(json.dumps(fit, indent=1, sort_keys=True) + "\n")
    print(f"wrote {DEFAULT_OUT}  status={fit['status']}  hash={fit['source']['fit_hash'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
