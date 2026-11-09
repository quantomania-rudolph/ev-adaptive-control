"""Monte Carlo matrix for the realism build (plan section 4). Resumable, writes one metrics row per run, keeps parquet only for seed 0.

    python run_matrix_fw.py plain|battery [n_seeds] [workers]

Scenario names: base (no event), oracle (25 V event, controller told the ceiling: legacy reproduction, NON-physical),
hidden (same event, controller NOT told: the realistic unobservable-limit case).
Controllers: naive, robust, aw (no governor), awg (aw + governor + vdc_aware).
"""
import sys, os, json, pathlib, itertools, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT = pathlib.Path("data/matrix"); OUT.mkdir(parents=True, exist_ok=True)
CTRL = {"naive": dict(mode="naive"), "robust": dict(mode="robust"), "aw": dict(mode="aw"), "awg": dict(mode="aw", governor=True, vdc_aware=True)}
SCEN = {"base": dict(event=False), "oracle": dict(event=True, oracle_ceiling=True), "hidden": dict(event=True, oracle_ceiling=False)}
FIT = "data/tum_battery_fit.json"
BAT_SCEN = {"sagbase": dict(battery=True, event=False, fit_path=FIT), "sag": dict(battery=True, event=True, fit_path=FIT)}   # observable sag: the bus collapses through the TUM-fitted pack
ENV_PSI, ENV_I = 0.15, 275.0
STORE_STRIDE = 17        # prime and coprime to the controller's 10-step speed-loop cadence (a stride of 20 sampled only update instants and doubled RMS mismatch)


def metrics(df, mf):
    from bounds import bounds_report
    from telemetry import derive, params_from_plant
    from plant import GEMPlant
    from battery import Battery
    P = params_from_plant(GEMPlant(battery=Battery()))       # battery-aware parameter set also works for plain (constant bus) runs
    t = df.t.values
    last = df[df.t >= df.t.iloc[-1] - 1.0]
    ev = (4.5, 5.5)
    d = derive(df, P)
    f_mag = np.hypot(df.u_d_app - df.d_u_inv_d, df.u_q_app - df.d_u_inv_q)       # plant-clipped voltage, before the inverter error
    post = df[df.t >= ev[1]]
    br = bounds_report(d, event=ev, window_s=1.0)
    return dict(
        ultimate_psi_err=float(last.psi_tilde.abs().max()), psi_hat_max=float(df.psi_hat.abs().max()), psi_tilde_end=float(df.psi_tilde.iloc[-1]),
        i_max=float(np.hypot(df.i_d, df.i_q).max()), envelope_ok=float((df.psi_hat.abs().max() <= ENV_PSI) and (np.hypot(df.i_d, df.i_q).max() <= ENV_I)),
        speed_rms=float(np.sqrt(((df.omega_m - df.omega_ref) ** 2).mean())), speed_rms_post=float(np.sqrt(((post.omega_m - post.omega_ref) ** 2).mean())),
        u_bound_true=float((f_mag / df.u_limit).max()), E_mech_unserved_kj=float(d["E_mech_unserved"].iloc[-1] / 1e3),
        V_peak=float(df.V.max()), V_amp=float(br.get("V_amplification", np.nan)),
        hidden_sat_rms=float(np.sqrt((df.d_u_plant_clip_q ** 2 + df.d_u_plant_clip_d ** 2).mean())),
        du_known_rms=float(np.sqrt((df.d_u_ctrl_clip_q ** 2 + df.d_u_ctrl_clip_d ** 2).mean())),
        inv_err_rms=float(np.sqrt((df.d_u_inv_q ** 2 + df.d_u_inv_d ** 2).mean())),
        sat_ctrl_frac=float(df.sat_ctrl.mean()), sat_plant_frac=float(df.sat_plant.mean()),
        gap_iq_rms=float(np.sqrt((df.gap_i_q ** 2).mean())), gap_omega_rms=float(np.sqrt((df.gap_omega ** 2).mean())),
        torque_gap_rms=float(np.sqrt(((df.T_elec - df.T_cmd) ** 2).mean())), finite=float(np.isfinite(df[["i_d", "i_q", "psi_hat", "omega_m"]].values).all()))


def one(args):
    ctl, scen, seed = args
    from firewalled_loop import run_fw
    kw = {**CTRL[ctl], **(BAT_SCEN[scen] if scen in BAT_SCEN else SCEN[scen])}
    if scen in BAT_SCEN and ctl == "awg":
        kw["bms_aware"] = True
    t0 = time.time()
    df, mf = run_fw(f"{scen}_{ctl}_s{seed}", seed=seed, write=False, **kw)
    m = metrics(df, mf)
    if seed == 0:
        (OUT / "runs").mkdir(exist_ok=True)
        df.iloc[::STORE_STRIDE].to_parquet(OUT / "runs" / f"{scen}_{ctl}_s0.parquet")       # stride 17 (1.7 ms): NOT a multiple of the controller's 10-step cadence, so instantaneous stats are unbiased
    return dict(scenario=scen, controller=ctl, seed=seed, wall_s=round(time.time() - t0, 1), **m,
                beliefs=json.dumps(mf["belief_draws"]))


if __name__ == "__main__":
    kind = sys.argv[1]
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    assert kind in ("plain", "battery")
    scen_set = SCEN if kind == "plain" else BAT_SCEN
    path = OUT / f"{kind}_metrics.csv"
    done = set()
    if path.exists():
        e = pd.read_csv(path); done = set(zip(e.controller, e.scenario, e.seed))
    todo = [(c, s, k) for k in range(nseeds) for s in scen_set for c in CTRL if (c, s, k) not in done]
    print("todo", len(todo), flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for fu in as_completed([ex.submit(one, a) for a in todo]):
            r = fu.result(); pd.DataFrame([r]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(r["scenario"], r["controller"], r["seed"], r["wall_s"], flush=True)
