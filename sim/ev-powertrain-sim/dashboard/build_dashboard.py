"""Builds dashboard/dashboard.html: one self-contained page for the whole powertrain story.

    python dashboard/build_dashboard.py

Learning note: like the viewer, this page only plots columns that telemetry.derive() produced from the run files, so
every number on screen is a number a data teammate can get from runs/*.parquet. Two things are computed here, on the FULL
rate data, and shipped next to the thinned traces: the per-run peak statistics (a thinned plot can miss a spike) and the
run-away time. The traces themselves are thinned to ~800-1000 points per channel so the page stays small.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plant import GEMPlant
from battery import Battery
from telemetry import derive, params_from_plant, CATALOG
from bounds import bounds_report

PSI_TRUE = 0.066
D_TESTS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# ---- which runs, and how they are presented --------------------------------------------------------------------------
RUNS = {
    "A":   dict(name="A  baseline, no event",              mode="naive",  scen=1, color="base", pts=800),
    "B":   dict(name="B  naive adaptive",                  mode="naive",  scen=1, color="naive", pts=800),
    "C":   dict(name="C  robust (leak + clamp)",           mode="robust", scen=1, color="robust", pts=800),
    "D":   dict(name="D  anti-windup",                     mode="aw",     scen=1, color="aw", pts=800),
    "Dg":  dict(name="Dg anti-windup + governor",          mode="aw+gov", scen=1, color="awgov", pts=800),
    "Eb":  dict(name="Eb naive, BMS cuts current",         mode="naive",  scen=2, color="naive", pts=800),
    "Eg":  dict(name="Eg protected, sees bus voltage",     mode="aw+gov", scen=2, color="awgov", pts=800),
    "T0n": dict(name="T0n real Tesla drive, naive",        mode="naive",  scen=3, color="naive", pts=1000),
    "T0p": dict(name="T0p real Tesla drive, protected",    mode="aw+gov", scen=3, color="awgov", pts=1000),
}

# ---- which card each catalog channel belongs to ----------------------------------------------------------------------
NODES = [
    dict(id="req", title="Driver request", sub="the speed we want",
         blurb="Where the story starts: a speed-versus-time request. In the 8 s experiments it is a ramp to 200 rad/s. In the real-drive runs it is a recorded Tesla Model 3 speed trace, cleaned by cycles.py and mapped onto the motor shaft. The controller never sees the vehicle, only this request.",
         watch="Mismatch M1 (speed requested minus speed actually reached) is the first number a driver would feel.",
         live=[["omega_ref", "request", "rad/s"], ["omega_m", "actual", "rad/s"], ["data_ratio", "data demand / capability", ""]], gap=["e_speed", "M1 speed not delivered", "rad/s"],
         ch=["t", "omega_ref", "T_req_data", "T_avail", "data_ratio", "data_infeasible"]),
    dict(id="ctl", title="Controller", sub="decides current and voltage",
         blurb="A cascade like a real drive. A slow speed loop turns the speed gap into a current request. A fast adaptive loop turns the current gap into a voltage request while it learns the magnet strength (psi) online. The governor trims the current request to what the voltage budget can sustain. Everything the controller believes or learns is logged here.",
         watch="psi_hat is the learned magnet strength. If it leaves the true value 0.066 Wb by a lot, the controller has been fooled. V is the energy-like Lyapunov candidate the theory team wants to bound.",
         live=[["psi_hat", "psi estimate", "Wb"], ["iq_ref", "current request", "A"], ["V", "V (Lyapunov candidate)", ""]], gap=["i_ref_clip", "M4 current request clipped", "A"],
         ch=["iq_ref", "i_ref_clip", "e_speed", "psi_hat", "psi_tilde", "e_q", "e_qa", "z", "i_lim_eff", "V", "T_cmd", "u_d_cmd", "u_q_cmd",
             "v_demand_mag", "dist_i_limit", "P_that", "P_psi_err", "P_fb", "req_sign_conflict"]),
    dict(id="sat", title="Voltage limiter", sub="where saturation happens",
         blurb="The inverter cannot make more than V_dc/√3 volts (173 V at a 300 V bus). We apply that limit ourselves, as a radial projection onto a disc, so we can measure it. Requested voltage in, delivered voltage out, and the difference (delta u) is the actuator disturbance from the research notes. The check d_u_plant must stay 0, which proves the simulator never clipped behind our back.",
         watch="Delta u is the key disturbance. When 'saturated' flips to 1 the controller is asking for more voltage than exists.",
         live=[["v_demand_mag", "asked", "V"], ["u_limit", "ceiling", "V"], ["d_u_norm", "short by", "\u00d7 ceiling"]], gap=["d_u_plant_q", "M3 hidden clipping (must be 0)", "V"],
         ch=["u_d_app", "u_q_app", "d_u_d", "d_u_q", "d_u_norm", "u_limit", "saturated", "d_u_plant_d", "d_u_plant_q", "v_headroom", "duty_peak"]),
    dict(id="mot", title="Inverter + motor", sub="turns voltage into torque",
         blurb="The Gym-Electric-Motor permanent-magnet motor in the dq frame. Currents i_d and i_q, phase currents and voltages, torque, flux and back-EMF live here, along with the electrical and shaft power ledgers and the cumulative energy that was asked for but not delivered.",
         watch="M5 is the torque we got minus the torque the current request implied. E_mech_unserved is the shaft energy owed so far.",
         live=[["i_q", "q current", "A"], ["T_elec", "torque", "N·m"], ["P_req_ref", "electrical request", "W"]], gap=["dP_qs", "M6c electrical power gap (shaft + copper)", "W"],
         ch=["omega_el", "epsilon_el", "i_d", "i_q", "i_a", "i_b", "i_c", "u_a", "u_b", "u_c", "T_elec", "d_T", "lam_d", "lam_q", "bemf", "P_el", "P_cu", "P_mag",
             "P_mech", "P_req_elec", "P_del_elec", "dP_elec", "P_mech_req", "dP_mech", "E_req", "E_del", "E_unserved", "E_mech_unserved", "E_cu",
             "P_req_ref", "P_cu_ref", "dP_cu", "dP_elec_ref", "E_elec_ref_unserved", "E_mag_store"]),
    dict(id="load", title="Road and vehicle", sub="what pushes back",
         blurb="Rolling resistance, drag and grade are reflected onto the motor shaft as friction and an external disturbance torque. The speed of the shaft is the state the driver finally feels. In real-drive runs this is the Tesla's road load, scaled to what the small Tier 1 motor can move.",
         watch="T_dist is the outside disturbance d(t) in the notes. It is the one disturbance the controller can never remove.",
         live=[["omega_m", "speed", "rad/s"], ["T_dist", "road torque", "N·m"], ["T_fric", "friction + drag", "N·m"]], gap=None,
         ch=["omega_m", "T_fric", "T_dist", "P_load", "P_kin", "d_w", "d_w_fric", "T_roll", "T_visc", "T_aero"]),
    dict(id="bat", title="Battery pack", sub="80 cells, current limits",
         blurb="Open-circuit voltage from state of charge, a resistance that rises when cold, a diffusion branch, and BMS current limits. When the BMS cuts the allowed current, the pack delivers less than the inverter asks for. That shortfall is what makes the bus voltage fall. Pack numbers are generic assumptions.",
         watch="i_batt_max is the BMS limit. bms_limited = 1 means the pack is being held back. Compare i_batt (delivered) with i_dc on the DC link card (requested).",
         live=[["i_batt", "pack current", "A"], ["i_batt_max", "BMS limit", "A"], ["soc", "state of charge", ""]], gap=["d_i_dc", "M10 pack minus inverter current", "A"],
         ch=["V_oc", "V_rc", "V_term", "soc", "T_batt", "R0_eff", "i_batt", "i_batt_max", "bms_limited", "P_chem", "P_batt_loss", "P_bms_clip", "resid_chain"]),
    dict(id="bus", title="DC link", sub="bus voltage sets the ceiling",
         blurb="A 2 mF capacitor between battery and inverter. Its voltage V_dc is a real state. The inverter ceiling is V_dc/√3, so a sagging bus lowers the voltage limiter in the next step. This is how a battery event becomes a voltage saturation event.",
         watch="bus_sag (M9) is how far the bus fell below the pack's open-circuit voltage. ceiling_loss is the inverter headroom that cost.",
         live=[["V_dc", "bus voltage", "V"], ["i_dc", "inverter draw", "A"], ["ceiling_loss", "ceiling lost", "V"]], gap=["bus_sag", "M9 bus sag", "V"],
         ch=["V_dc", "i_dc", "P_dc", "P_inv_loss", "P_bus_store", "bus_sag", "ceiling_loss", "d_i_dc", "E_bus_unserved", "resid_bus"]),
    dict(id="math", title="Math layer", sub="f, g, d split and checks",
         blurb="The plant written as x' = f(x) + g(x)u + d. f is what the motor does on its own, g is how strongly the voltage pushes, and d is what is wrong: the actuator mismatch (g times delta u) and the road load. The residual rows check that these terms really add up to the measured derivatives. The viewer marks them SKIP on thinned real-drive logs.",
         watch="d_iq and d_id are the actuator disturbance in current units. resid_* and the power balance should sit near zero in full-rate logs.",
         live=[["f_iq", "f for i_q", "A/s"], ["d_iq", "actuator d on i_q", "A/s"], ["d_w", "road d on speed", "rad/s²"]], gap=None,
         ch=["f_id", "f_iq", "f_w", "g_id", "g_iq", "d_id", "d_iq", "resid_id", "resid_iq", "resid_w", "P_balance_resid", "resid_elec_decomp", "resid_req_decomp", "resid_du_power"]),
]
EXPORT = sorted({c for n in NODES for c in n["ch"]} | {"t", "omega_ref", "omega_m", "V_e", "V_theta", "V_aw", "dP_mech"})

# every catalog channel must belong to exactly one card
assigned = [c for n in NODES for c in n["ch"]]
cat_names = [c[0] for c in CATALOG]
missing = [c for c in cat_names if c not in assigned]
dups = sorted({c for c in assigned if assigned.count(c) > 1})
assert not missing, f"catalog channels without a card: {missing}"
assert not dups, f"channels on two cards: {dups}"
assert not [c for c in assigned if c not in cat_names], "a card lists a channel that is not in the catalog"

sig = lambda v: None if not np.isfinite(v) else float(f"{v:.4g}")
P0 = params_from_plant(GEMPlant(battery=Battery()))

out_runs, stats = {}, {}
for key, meta in RUNS.items():
    pq = ROOT / "runs" / f"{key}.parquet"
    man = json.loads((ROOT / "runs" / f"{key}.json").read_text())
    tau = man.get("tau_log", man.get("tau", P0["tau"]))
    d = derive(pd.read_parquet(pq), dict(P0, tau=tau))
    n = len(d)
    if man.get("derate"):
        ev = [man["derate"][0], man["derate"][0] + man["derate"][1]]
    elif man.get("event"):
        ev = [4.5, 5.5]
    else:
        ev = None
    step = max(1, n // meta["pts"])
    has_batt = "i_batt" in d.columns
    cols = {}
    for c in EXPORT:
        if c in d.columns and not (c in ("i_batt", "soc") and not has_batt):
            cols[c] = [sig(v) for v in d[c].to_numpy()[::step]]
    # derived magnitudes used by the charts
    cols["u_cmd_n"] = [sig(v) for v in np.hypot(d.u_d_cmd, d.u_q_cmd).to_numpy()[::step]]
    cols["u_app_n"] = [sig(v) for v in np.hypot(d.u_d_app, d.u_q_app).to_numpy()[::step]]
    cols["i_mag"] = [sig(v) for v in np.hypot(d.i_d, d.i_q).to_numpy()[::step]]
    cols["du_abs"] = [sig(v) for v in np.hypot(d.d_u_d, d.d_u_q).to_numpy()[::step]]
    cols["psi_err"] = [sig(v) for v in d.psi_tilde.abs().to_numpy()[::step]]
    cols["Eun_kj"] = [sig(v) for v in (d.E_mech_unserved / 1000).to_numpy()[::step]]
    cols["Eel_kj"] = [sig(v) for v in (d.E_elec_ref_unserved / 1000).to_numpy()[::step]]
    cols["dP_qs"] = [sig(v) for v in (d.dP_mech + d.dP_cu).to_numpy()[::step]]          # quasi-steady electrical gap: valid at any logging rate
    cols["negP_fb"] = [sig(v) for v in (-d.P_fb).to_numpy()[::step]]
    # peaks on the FULL data
    after = d[d.t >= ev[0]] if ev else d
    e_speed = (d.omega_ref - d.omega_m)
    sp = e_speed[(d.t >= ev[0]) & (d.t <= ev[1] + 2.0)] if ev else e_speed
    big = np.nonzero(d.psi_tilde.abs().to_numpy() > 0.05)[0]
    stats[key] = dict(
        rows=n, step=step, tau=tau, event=ev, duration=float(d.t.iloc[-1]),
        psi_peak=sig(d.psi_hat.abs().max()), psi_end=sig(d.psi_hat.iloc[-1]), psi_err_peak=sig(d.psi_tilde.abs().max()),
        i_peak=sig(np.hypot(d.i_d, d.i_q).max()), V0=sig(d.V.iloc[0]), V_peak=sig(d.V.max()), V_end=sig(d.V.iloc[-1]),
        du_peak=sig(np.hypot(d.d_u_d, d.d_u_q).max()), sat_pct=sig(100 * d.saturated.mean()),
        speed_short_peak=sig(sp.max()), speed_rms_after=sig(float(np.sqrt((after.omega_ref - after.omega_m).pow(2).mean()))),
        eun_kj=sig(d.E_mech_unserved.iloc[-1] / 1000), runaway_t=sig(float(d.t.iloc[big[0]])) if len(big) else None,
        vdc_min=sig(d.V_dc.min()) if has_batt else None, bms_pct=sig(100 * d.bms_limited.mean()) if has_batt else None,
        all_finite=bool(np.isfinite(d[[c for c in EXPORT if c in d.columns]].to_numpy(dtype=float)).all()),
        m3_max=float(f"{max(d.d_u_plant_d.abs().max(), d.d_u_plant_q.abs().max()):.2g}"),
        vdc_aware=bool(man.get("vdc_aware")), governor=bool(man.get("governor")), battery=has_batt,
    )
    decimated = bool(tau > 1.5e-4 * 3)
    rms = lambda c: float(np.sqrt((d[c] ** 2).mean()))
    stats[key]["decimated"] = decimated
    stats[key]["bounds"] = {k_: (sig(v) if isinstance(v, float) else v) for k_, v in bounds_report(d, event=ev, window_s=1.0 if n < 100000 else None).items()}
    stats[key]["elec"] = dict(
        rms_dP_mech=sig(rms("dP_mech")), rms_dP_cu=sig(rms("dP_cu")), rms_dP_qs=sig(float(np.sqrt(((d.dP_mech + d.dP_cu) ** 2).mean()))),
        e_shaft_kj=sig(float(d.dP_mech.sum() * tau / 1000)), e_cu_kj=sig(float(d.dP_cu.sum() * tau / 1000)),
        sign_conflict_pct=sig(float(100 * d.req_sign_conflict.mean())), resid_du_rms=float(f"{rms('resid_du_power'):.2g}"),
        data_ratio_gt1_pct=sig(float(100 * (d.data_ratio > 1).mean())) if "data_ratio" in d else None,
        **({} if decimated else dict(rms_dP_elec_ref=sig(rms("dP_elec_ref")), rms_P_mag=sig(rms("P_mag")), rms_resid=sig(rms("resid_elec_decomp")),
            e_mag_kj=sig(float(d.P_mag.sum() * tau / 1000)), e_elec_kj=sig(float(d.dP_elec_ref.sum() * tau / 1000)))))
    out_runs[key] = dict(meta, event=ev, cols=cols, has_batt=has_batt)
    print(key, n, "rows ->", len(cols["t"]), "pts; psi_peak", stats[key]["psi_peak"], "V_peak", stats[key]["V_peak"], "finite", stats[key]["all_finite"])

cat = {n: dict(unit=u, group=g, role=r, desc=ds) for n, u, g, r, ds in CATALOG}
mm = json.loads((ROOT / "data" / "mismatch_map.json").read_text()) if (ROOT / "data" / "mismatch_map.json").exists() else None
payload = dict(runs=out_runs, stats=stats, nodes=NODES, catalog=cat, psi_true=PSI_TRUE, tests=D_TESTS, mmap=mm)
tpl = (ROOT / "dashboard" / "dashboard_template.html").read_text()
html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
(ROOT / "dashboard" / "dashboard.html").write_text(html)
print("wrote dashboard.html", f"{len(html)/1e6:.2f} MB")
