"""Figure for Experiments A-D. Small multiples, one measure per axis (no dual axes), fixed colour per experiment."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}      # validated categorical slots 1-4
LAB = {"A": "A  no saturation", "B": "B  naive adaptive + saturation", "C": "C  sigma-mod + projection", "D": "D  anti-windup"}
INK, GRID = "#0b0b0b", "#e6e5e0"
runs = {n: pd.read_parquet(f"runs/{n}.parquet").iloc[::20] for n in "ABCD"}   # 500 Hz is plenty for plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
def panel(ax, col, title, ylabel, f=lambda d, c: d[c], scale=None, hline=None):
    for n, d in runs.items():
        ax.plot(d.t, f(d, col), color=COL[n], lw=1.4, label=LAB[n], zorder=3 if n in "BD" else 2)
    ax.axvspan(4.5, 5.5, color="#000", alpha=0.06, lw=0)
    if hline is not None: ax.axhline(hline, color=INK, lw=0.8, ls=(0, (4, 3)))
    if scale: ax.set_yscale(*scale[:1], **scale[1])
    ax.set_title(title, loc="left", fontsize=10.5, color=INK); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
panel(axs[0,0], "omega_m", "Speed tracks reference until the voltage ceiling drops", "rad/s")
axs[0,0].plot(runs["A"].t, runs["A"].omega_ref, color=INK, lw=0.8, ls=(0,(4,3)))
panel(axs[0,1], None, "Current magnitude vs the 400 A limit (dashed)", "A",
      f=lambda d, c: np.hypot(d.i_d, d.i_q), hline=400)
panel(axs[1,0], None, "M2  voltage mismatch |Δu| = |u − u_c|", "V (symlog)",
      f=lambda d, c: np.hypot(d.d_u_d, d.d_u_q), scale=("symlog", dict(linthresh=1)))
panel(axs[1,1], None, "M5  torque mismatch ΔT = T_elec − T_cmd", "N·m", f=lambda d, c: d.d_T)
panel(axs[2,0], "psi_hat", "Estimate ψ̂ (true 0.066 Wb)", "Wb (symlog)", scale=("symlog", dict(linthresh=0.1)), hline=0.066)
panel(axs[2,1], "V", "Lyapunov candidate V = V_e + V_θ + V_aw", "(log)", scale=("log", {}))
axs[2,1].set_ylim(1e-6, None)
for ax in axs[2]: ax.set_xlabel("time (s); shaded = unmodeled voltage ceiling 25 V", fontsize=9)
h, l = axs[0,0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/exp_ABCD.png", dpi=130, facecolor="white")
print("saved")
