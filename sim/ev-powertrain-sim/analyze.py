"""Summarise experiments A-D from the logged ledger. Everything here is computed from the Parquet files."""
import pandas as pd, numpy as np, json, sys
rows=[]
for n in ["A","B","C","D","Ag","Bg","Cg","Dg"]:
    df=pd.read_parquet(f"runs/{n}.parquet"); m=json.load(open(f"runs/{n}.json"))
    ev=(df.t>=4.5)&(df.t<5.5); post=df.t>=5.5
    rec = df[df.t>=6.5]   # settled window after the event
    rows.append(dict(exp=n, mode=m["mode"], event=m["event"],
        sat_frac_event=round(df.loc[ev,"saturated"].mean(),3),
        max_du_V=round(np.hypot(df.d_u_d,df.d_u_q).max(),1),
        psi_hat_min=round(df.psi_hat.min(),4), psi_hat_max=round(df.psi_hat.max(),4),
        psi_hat_end=round(df.psi_hat.iloc[-1],4), psi_true=m["psi_true"],
        max_abs_psi_err=round(df.psi_tilde.abs().max(),4),
        peak_i_A=round(np.hypot(df.i_d,df.i_q).max(),1),
        min_dist_i_limit=round(df.dist_i_limit.min(),3),
        omega_min_after3s=round(df.loc[df.t>4.5,"omega_m"].min(),1),
        omega_err_rms_5to65=round(float(np.sqrt(((rec.omega_ref-rec.omega_m)**2).mean())),2),
        max_M3_V=float(df[["d_u_plant_d","d_u_plant_q"]].abs().max().max()),
        max_duty=round(df.duty_peak.max(),3)))
r=pd.DataFrame(rows); pd.set_option("display.width",250); print(r.T.to_string())
