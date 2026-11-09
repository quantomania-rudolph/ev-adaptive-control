"""Figures for the explainer PDF. Run from the repo root: python docs/make_figures.py"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#14181f", "#6b727c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9aa0a8", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

# ---------------- Figure 1: architecture
fig, ax = plt.subplots(figsize=(10.2, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, title, sub, fc="#eef2f7", ec="#9aa0a8", bold=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=9, fontweight="bold", color=bold)
    ax.text(x + w / 2, y + h - 5.2, sub, ha="center", va="top", fontsize=7.4, color=INK, linespacing=1.25)
def arrow(x1, y1, x2, y2, label="", c=INK, dy=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.1, color=c))
    if label: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=6.8, color=c)
box(1, 28, 17, 15, "Real drive data", "TUM Tesla Model 3\nEPA UDDS\n(speed vs time)")
box(21, 28, 17, 15, "cycles.py", "clean + check\nunits, gaps, spikes\nreport of every fix")
box(41, 28, 17, 15, "VehicleEq", "speed -> motor speed\nmass, drag, grade\nreflected to shaft")
box(61, 28, 17, 15, "control.py", "speed PI -> current\nadaptive q-loop\nanti-windup, governor")
box(81, 28, 18, 15, "plant.py", "voltage request\nTHEN our saturation\nGEM motor + inverter", fc="#fdf0ea", bold=ORANGE)
box(81, 4, 18, 17, "battery.py", "pack OCV, R0, RC\nBMS current limit\nDC-link capacitor")
box(51, 4, 20, 17, "telemetry.py", "98 channels\nf, g, d split\nmismatch + energy", fc="#e8f7f1", bold=AQUA)
box(7, 4, 40, 17, "Outputs the data team uses", "runs/*.parquet  (every logged step)\nviewer/data_viewer.html  (interactive)\nruns/*_cycle_report.csv  (what was repaired)", fc="#f6f7f9")
arrow(18, 35.5, 21, 35.5); arrow(38, 35.5, 41, 35.5); arrow(58, 35.5, 61, 35.5, "w_ref", dy=1.0); arrow(78, 35.5, 81, 35.5)
ax.text(79.5, 40.6, "u", fontsize=7, ha="center")
arrow(86, 28, 86, 21, "", c=ORANGE); ax.text(84.6, 24.3, "P_el", fontsize=6.8, color=ORANGE, ha="right")
arrow(94, 21, 94, 28, "", c=BLUE); ax.text(95.4, 24.3, "V_dc sets\nthe ceiling", fontsize=6.8, color=BLUE, ha="left")
arrow(81, 12.5, 71, 12.5, "", c=AQUA); arrow(65, 28, 65, 21, "", c=AQUA); arrow(51, 12.5, 47, 12.5, "", c=AQUA)
ax.text(50, 44.8, "Each arrow is a hand-off where requested and delivered can differ; each difference is logged.", ha="center", fontsize=8, color=MUTED)
fig.savefig("docs/figures/fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: real drive results
n, p = json.load(open("runs/T0n.json")), json.load(open("runs/T0p.json"))
dn, dp = pd.read_parquet("runs/T0n.parquet"), pd.read_parquet("runs/T0p.parquet")
t0, t1 = n["derate"][0], n["derate"][0] + n["derate"][1]
fig, axs = plt.subplots(3, 1, figsize=(10.2, 7.2), gridspec_kw=dict(height_ratios=[1.1, 0.9, 1.1], hspace=0.42))
a = axs[0]; a.plot(dn.t, dn.omega_ref / 7.983, color=MUTED, lw=1.2, ls="--", label="real speed request (Tesla GPS speed)")
a.plot(dp.t, dp.omega_m / 7.983, color=BLUE, lw=1.1, label="simulated vehicle speed, protected"); a.axvspan(t0, t1, color=ORANGE, alpha=.18, lw=0)
a.set_ylabel("vehicle speed (m/s)"); a.set_title("A. A full real 283 s drive through the simulated powertrain", loc="left", fontsize=9.5, fontweight="bold"); a.legend(frameon=False, fontsize=7.5, loc="lower right"); a.set_xlim(0, dn.t.max())
a = axs[1]; a.plot(dn.t, dn.V_dc, color=ORANGE, lw=1.1, label="naive controller"); a.plot(dp.t, dp.V_dc, color=BLUE, lw=1.1, label="protected controller"); a.axvspan(t0, t1, color=ORANGE, alpha=.18, lw=0)
a.set_ylabel("DC-link voltage (V)"); a.set_title("B. The BMS current cut collapses the bus (shaded), and each controller reacts differently", loc="left", fontsize=9.5, fontweight="bold"); a.legend(frameon=False, fontsize=7.5, loc="lower left"); a.set_xlim(0, dn.t.max())
a = axs[2]; m = (dn.t > t0 - 8) & (dn.t < t1 + 14)
a.plot(dn.t[m], dn.omega_ref[m], color=MUTED, lw=1.3, ls="--", label="request"); a.plot(dn.t[m], dn.omega_m[m], color=ORANGE, lw=1.4, label="naive"); a.plot(dp.t[m], dp.omega_m[m], color=BLUE, lw=1.4, label="protected")
a.axvspan(t0, t1, color=ORANGE, alpha=.18, lw=0); a.set_ylabel("motor speed (rad/s)"); a.set_xlabel("time (s)")
a.set_title("C. Zoom on the event: the unprotected controller falls 32 rad/s behind the request, the protected one 22", loc="left", fontsize=9.5, fontweight="bold"); a.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.savefig("docs/figures/fig_realdrive.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 3: simulated vs real battery power
mm = pd.read_parquet("runs/T0p_measured.parquet"); t = mm.t.to_numpy()
Pm = -(mm.hv_battery_current * mm.hv_battery_voltage).to_numpy(); Ps = np.interp(t, dp.t, dp.V_dc * dp.i_batt)
sm = lambda x, w: pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
ok = np.isfinite(Pm) & (t > 8); r = np.corrcoef(sm(Pm, 50)[ok], sm(Ps, 50)[ok])[0, 1]
kap = p["vehicle"]["mass_eq"] / 1900.0
fig, ax = plt.subplots(figsize=(10.2, 3.2))
ax.plot(t, sm(Pm, 50) * kap / 1e3, color=MUTED, lw=1.3, label="real Tesla battery power x mass ratio 0.069")
ax.plot(t, sm(Ps, 50) / 1e3, color=BLUE, lw=1.3, label="simulated battery power"); ax.axhline(0, color="#c3c8cf", lw=.8)
ax.set_xlabel("time (s)"); ax.set_ylabel("kW, 5 s average"); ax.legend(frameon=False, fontsize=7.5)
ax.set_title(f"Same drive, real battery vs simulated battery. Correlation after 5 s smoothing: {r:.2f}", loc="left", fontsize=9.5, fontweight="bold")
fig.savefig("docs/figures/fig_realvsim.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok; corr", round(r, 3))
