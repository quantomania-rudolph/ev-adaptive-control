"""Figure 2: two distinct failure mechanisms. Estimator windup (fixed by anti-windup) vs infeasible command (fixed by governor)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"B": "#eb6834", "Bg": "#6250d6", "D": "#1baf7a", "Dg": "#2a78d6"}       # fixed colour per run
LAB = {"B": "B  naive", "Bg": "Bg naive + governor", "D": "D  anti-windup", "Dg": "Dg anti-windup + governor"}
INK, GRID = "#0b0b0b", "#e6e5e0"
R = {n: pd.read_parquet(f"runs/{n}.parquet").iloc[::20] for n in COL}
fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
def panel(ax, f, title, yl, scale=None, hline=None):
    for n, d in R.items(): ax.plot(d.t, f(d), color=COL[n], lw=1.4, label=LAB[n])
    ax.axvspan(4.5, 5.5, color="#000", alpha=0.06, lw=0)
    if hline is not None: ax.axhline(hline, color=INK, lw=0.8, ls=(0, (4, 3)))
    if scale: ax.set_yscale("symlog", linthresh=scale)
    ax.set_title(title, loc="left", fontsize=10.5); ax.set_ylabel(yl, fontsize=9); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
panel(axs[0,0], lambda d: d.psi_hat, "Estimator windup: governor does not fix it", "ψ̂ (Wb, symlog)", scale=0.1, hline=0.066)
panel(axs[0,1], lambda d: d.omega_ref - d.omega_m, "Speed error after the ceiling is restored", "rad/s")
panel(axs[1,0], lambda d: d.i_d, "d-axis current: the voltage-limited trap (i_d should be 0)", "A")
panel(axs[1,1], lambda d: d.d_T, "M5  torque mismatch ΔT", "N·m")
for ax in axs[1]: ax.set_xlabel("time (s); shaded = unmodeled 25 V ceiling", fontsize=9)
h, l = axs[0,0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig("figures/exp_governor.png", dpi=130, facecolor="white"); print("saved")
