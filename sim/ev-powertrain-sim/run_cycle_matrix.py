"""Drive-cycle runs through the firewall: naive vs awg (aw + governor + vdc/bms aware), N seeds. Resumable.
    python run_cycle_matrix.py udds|tesla [n_seeds] [workers]
"""
import sys, os, json, time, pathlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

SC = pathlib.Path(__file__).resolve().parent / "data" / "inputs"      # inputs live in the repo (data/inputs/README.md)
CYC = {"udds": (f"udds:{SC}/udds.txt", 120.0), "tesla": (f"tum:{SC}/ds_tum/track_0.csv", 150.0)}
CTRL = {"naive": dict(mode="naive"), "awg": dict(mode="aw", governor=True, vdc_aware=True, bms_aware=True)}
OUT = pathlib.Path("data/cycles"); OUT.mkdir(parents=True, exist_ok=True)


def one(a):
    kind, ctl, seed = a
    import run_cycle as RC
    from firewalled_cycle import run_fw_cycle
    src, tmax = CYC[kind]
    t0 = time.time()
    df, mf = (
        run_fw_cycle(RC._load(src), f"{kind}_{ctl}_s{seed}", seed=seed, t_max=tmax, log_every=23, outdir=str(OUT / "runs") if seed == 0 else "/tmp/cyc_scratch", ctl_extra={"d_gain_by_L": True}, **CTRL[ctl]))
    f_mag = np.hypot(df.u_d_app - df.d_u_inv_d, df.u_q_app - df.d_u_inv_q)
    ev = df.bms_i_dis_lim.values < 0.5 * df.bms_i_dis_lim.max()          # BMS derate window as the controller was told
    err = df.omega_m - df.omega_ref
    return dict(cycle=kind, controller=ctl, seed=seed, wall_s=round(time.time() - t0, 1),
                speed_rms=float(np.sqrt((err ** 2).mean())), speed_rms_derate=float(np.sqrt((err[ev] ** 2).mean())) if ev.any() else np.nan,
                speed_max_abs=float(err.abs().max()), psi_hat_max=float(df.psi_hat.abs().max()), psi_tilde_final=float(df.psi_tilde.iloc[-1]),
                psi_tilde_absmax=float(df.psi_tilde.abs().max()), i_max=float(np.hypot(df.i_d, df.i_q).max()),
                u_bound_true=float((f_mag / df.u_limit).max()), sat_plant_frac=float(df.sat_plant.mean()), sat_ctrl_frac=float(df.sat_ctrl.mean()),
                V_dc_min=float(df.V_dc.min()), soc_end=float(df.soc.iloc[-1]) if "soc" in df else np.nan,
                gap_omega_rms=float(np.sqrt((df.gap_omega ** 2).mean())), gap_iq_rms=float(np.sqrt((df.gap_i_q ** 2).mean())),
                inv_err_rms=float(np.sqrt((df.d_u_inv_q ** 2 + df.d_u_inv_d ** 2).mean())),
                finite=float(np.isfinite(df[["i_d", "i_q", "psi_hat", "omega_m"]].values).all()), derate_rows=int(ev.sum()))


if __name__ == "__main__":
    kind = sys.argv[1]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 5; w = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    path = OUT / f"{kind}_metrics.csv"
    done = set(zip(*[pd.read_csv(path)[c] for c in ("controller", "seed")])) if path.exists() else set()
    todo = [(kind, c, s) for s in range(ns) for c in CTRL if (c, s) not in done]
    print("todo", len(todo), flush=True)
    with ProcessPoolExecutor(w) as ex:
        for fu in as_completed([ex.submit(one, a) for a in todo]):
            r = fu.result(); pd.DataFrame([r]).to_csv(path, mode="a", header=not path.exists(), index=False); print(r["controller"], r["seed"], r["wall_s"], r["speed_rms"], flush=True)
