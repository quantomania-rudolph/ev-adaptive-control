"""A7 verdict per (parameter, level). With only 3 seeds the exact test cannot reach p<0.01, so the rule is complete separation:
the WORST protected seed must be below 0.25x the BEST (smallest) naive seed. Baseline (1x) = battery matrix seeds 0-2."""
import pandas as pd, numpy as np
d = pd.read_csv("data/sensitivity/metrics.csv")
b = pd.read_csv("data/matrix/battery_metrics.csv"); b = b[(b.scenario == "sag") & (b.seed < 3) & b.controller.isin(["naive", "aw"])].assign(param="baseline", level="1x")
d = pd.concat([d, b], ignore_index=True)
rows = []
for (p, l), g in d.groupby(["param", "level"], sort=False):
    n, a = g[g.controller == "naive"].ultimate_psi_err, g[g.controller == "aw"].ultimate_psi_err
    rows.append(dict(param=p, level=l, n_naive=len(n), n_aw=len(a), naive_min=n.min(), aw_max=a.max(), ratio=a.max() / n.min(),
                     holds_A5=bool(a.max() < 0.25 * n.min()), aw_env_ok=f"{int(g[g.controller=='aw'].envelope_ok.sum())}/{len(a)}",
                     naive_env_ok=f"{int(g[g.controller=='naive'].envelope_ok.sum())}/{len(n)}", u_bound_max=g.u_bound_true.max()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
print("\nconclusion-deciding (A5 fails):", R[~R.holds_A5][["param", "level"]].values.tolist() or "none")
print("A1 over all sweep runs:", d.u_bound_true.max() <= 1 + 1e-9)
