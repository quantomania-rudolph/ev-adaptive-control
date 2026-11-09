"""Summaries and pre-registered comparisons (plan A1, A4, A5, A6) for data/matrix/plain_metrics.csv."""
import sys, numpy as np, pandas as pd
from scipy import stats

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def summarize(df):
    rows = []
    for (s, c), g in df.groupby(["scenario", "controller"]):
        k = int(g.envelope_ok.sum()); lo, hi = wilson(k, len(g))
        rows.append(dict(scenario=s, controller=c, n=len(g), ult_psi_median=g.ultimate_psi_err.median(), ult_psi_p90=g.ultimate_psi_err.quantile(.9),
                         envelope_ok=f"{k}/{len(g)}", env_wilson=f"[{lo:.2f},{hi:.2f}]", speed_rms_post_med=g.speed_rms_post.median(),
                         E_unserved_kj_med=g.E_mech_unserved_kj.median(), hidden_sat_rms_med=g.hidden_sat_rms.median(), du_known_rms_med=g.du_known_rms.median(),
                         u_bound_max=g.u_bound_true.max()))
    return pd.DataFrame(rows)

def a5(df, scen):
    out = []
    n = df[(df.scenario == scen) & (df.controller == "naive")].ultimate_psi_err
    for c in ("robust", "aw", "awg"):
        p = df[(df.scenario == scen) & (df.controller == c)].ultimate_psi_err
        if len(p) < 3 or len(n) < 3: continue
        u = stats.mannwhitneyu(p, n, alternative="less")
        out.append(dict(scenario=scen, protected=c, median_ratio=p.median() / n.median(), mannwhitney_p=u.pvalue, passes_A5=bool(p.median() < 0.25 * n.median() and u.pvalue < 0.01)))
    return out

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/matrix/plain_metrics.csv"
    df = pd.read_csv(path)
    scens = ("sagbase", "sag") if "battery" in path else ("oracle", "hidden", "base")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summarize(df).round(4).to_string(index=False))
    print("\nA1 bounded input (true ceiling): max |u|/ceiling over all runs =", df.u_bound_true.max(), "(<= 1+1e-9:", df.u_bound_true.max() <= 1 + 1e-9, ")")
    print("\nA5 protected vs naive:")
    for s in scens:
        for r in a5(df, s): print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
