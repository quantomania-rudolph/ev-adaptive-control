"""Held-value and plausibility scan over sim runs and TUM data (read-only). Run: python data_gutcheck.py

A 'hold' is a run of consecutive bit-identical values. Most holds are legitimate (zero at standstill, flags, clip plateaus, ledger
rows that are zero when nothing saturates). The point of the scan is the REST: a continuous physical channel that sits still while
the vehicle is moving. Verdict rules are explicit so they can be argued with.
"""
import sys, pathlib, json
import numpy as np, pandas as pd

FLAGS = {"saturated", "sat_ctrl", "sat_plant", "bms_limited", "regen_active", "data_infeasible", "req_sign_conflict"}
INACTIVE_ZERO = ("du_", "gap_", "d_u_", "V_aw", "z", "i_ref_clip", "P_bms_clip", "d_u_norm", "u_d_extra", "u_q_extra", "hidden", "ceiling_loss")
PLATEAU = {"i_lim_eff", "u_limit", "i_batt_max", "T_avail", "T_roll", "psi_hat", "ceiling_hat", "omega_ref", "T_cmd", "iq_ref", "id_ref", "i_q_ref", "i_d_ref", "T_dist", "belief_psi_hat0"}     # values that legitimately sit at a limit for a long time


def runs_of_equal(x):
    """(start, length) of the longest run of bit-identical non-NaN values."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    eq = (x[1:] == x[:-1]) & ok[1:] & ok[:-1]
    if not eq.any():
        return 0, 1
    d = np.diff(np.r_[0, eq.view(np.int8), 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    i = int(np.argmax(e - s))
    return int(s[i]), int(e[i] - s[i]) + 1


def scan_sim(df, label, min_hold_s=0.05):
    t = df["t"].values
    dt = float(np.median(np.diff(t)))
    moving = np.abs(df["omega_m"].values) > 1.0 if "omega_m" in df else np.ones(len(df), bool)
    out = []
    for c in df.columns:
        if df[c].dtype.kind not in "fi" or c in ("t",):
            continue
        x = df[c].values
        if np.isnan(x).all():
            out.append(dict(run=label, col=c, verdict="ALL-NaN")); continue
        nun = len(np.unique(x[~np.isnan(x)]))
        s, L = runs_of_equal(x)
        # longest hold restricted to moving samples
        xm = np.where(moving, x, np.nan)
        sm, Lm = runs_of_equal(xm)
        val = float(x[s]) if L > 1 else np.nan
        valm = float(xm[sm]) if Lm > 1 else np.nan
        if nun == 1:
            v = "constant-channel"
        elif c in FLAGS or df[c].dropna().isin([0.0, 1.0]).all():
            v = "flag"
        elif Lm * dt < min_hold_s:
            v = "ok"
        elif valm == 0.0 and any(c.startswith(p) or c == p for p in INACTIVE_ZERO):
            v = "zero-when-inactive"
        elif c in PLATEAU:
            v = "limit-plateau"
        elif valm == 0.0:
            v = "ZERO-WHILE-MOVING"
        else:
            v = "SUSPECT"
        out.append(dict(run=label, col=c, n_unique=nun, longest_hold_s=L * dt, longest_hold_moving_s=Lm * dt, held_value_moving=valm,
                        hold_t0_moving=float(t[sm]) if Lm > 1 else np.nan, verdict=v))
    return pd.DataFrame(out)


def scan_tum(path, label):
    d = pd.read_csv(path)
    out = []
    for c in d.columns:
        if d[c].dtype.kind not in "fi" or c == "time":
            continue
        x = d[c].values.astype(float)
        nn = float(np.isnan(x).mean())
        if nn == 1.0:
            continue
        eq = (x[1:] == x[:-1]) & ~np.isnan(x[1:]) & ~np.isnan(x[:-1])
        s, L = runs_of_equal(x)
        out.append(dict(track=label, col=c, nan_frac=round(nn, 3), frac_equal_prev=round(float(eq.mean()), 3), eff_update_hz=round(10 * (1 - float(eq.mean())), 2),
                        longest_hold_s=L * 0.1, held_value=float(x[s]) if L > 1 else np.nan, n_unique=len(np.unique(x[~np.isnan(x)]))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    import glob
    root = pathlib.Path(__file__).parent
    sims, tum = [], []
    files = sorted(glob.glob(str(root / "runs" / "*.parquet")) + glob.glob(str(root / "data" / "matrix" / "*.parquet")))
    for f in files:
        p = pathlib.Path(f)
        if p.name.endswith("_measured.parquet"):
            continue
        try:
            df = pd.read_parquet(f)
            if "omega_m" not in df or "t" not in df: continue
            sims.append(scan_sim(df, p.stem))
        except Exception as e:
            print("skip", p.name, e)
    for f in sorted(glob.glob(str(root / "data" / "tum_subset" / "track_*_battery.csv"))):
        tum.append(scan_tum(f, pathlib.Path(f).stem))
    S = pd.concat(sims, ignore_index=True); T = pd.concat(tum, ignore_index=True)
    out = root / "data" / "gutcheck"; out.mkdir(exist_ok=True)
    S.to_csv(out / "sim_holds.csv", index=False); T.to_csv(out / "tum_holds.csv", index=False)
    print(S.verdict.value_counts().to_string())
    bad = S[S.verdict.isin(["SUSPECT", "ZERO-WHILE-MOVING"])]
    print(bad.groupby("col").agg(runs=("run", "count"), maxhold=("longest_hold_moving_s", "max"), ex=("run", "first"), val=("held_value_moving", "first")).sort_values("maxhold", ascending=False).to_string())
