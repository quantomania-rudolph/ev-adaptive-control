import json,re,sys,numpy as np,pandas as pd
sys.path.insert(0,'.')
html=open('dashboard/dashboard.html').read()
m=re.search(r'const D = (\{.*?\});\n',html,re.S); D=json.loads(m.group(1))
bad=0
def chk(name,ok,info=""):
    global bad
    print(("PASS " if ok else "FAIL ")+name,info); bad+=0 if ok else 1
for k,r in D['runs'].items():
    d=pd.read_parquet(f'runs/{k}.parquet'); st=D['stats'][k]; c=r['cols']; step=st['step']
    chk(f"{k} psi_peak",abs(d.psi_hat.abs().max()-st['psi_peak'])<=1e-3*max(1,st['psi_peak']),st['psi_peak'])
    chk(f"{k} i_peak",abs(np.hypot(d.i_d,d.i_q).max()-st['i_peak'])<0.01*st['i_peak'])
    chk(f"{k} V_peak",abs(d.V.max()-st['V_peak'])<=1e-3*st['V_peak'])
    # trace spot-check vs parquet at stride
    idx=np.arange(0,len(d),step)
    for ch,col in [('omega_m','omega_m'),('psi_hat','psi_hat'),('i_q','i_q'),('V','V')]:
        a=np.array(c[ch],float); b=d[col].to_numpy()[idx]
        chk(f"{k} trace {ch}",np.allclose(a,b,rtol=2e-3,atol=1e-6))
    chk(f"{k} t monotonic",bool(np.all(np.diff(c['t'])>0)))
    chk(f"{k} no nulls in key channels",all(None not in c[x] for x in ['t','omega_m','psi_hat','V','i_mag']))
    chk(f"{k} lengths equal",len({len(v) for v in c.values()})==1)
# claims on the page
B,D_,Dg=D['stats']['B'],D['stats']['D'],D['stats']['Dg']
chk("fact 42x",round(B['psi_peak']/0.066)==42); chk("fact 1.01x",abs(D_['psi_peak']/0.066-1.01)<0.005)
chk("fact 589 A",round(B['i_peak'])==589); chk("Dg 224 A",round(Dg['i_peak'])==224)
chk("M3 hidden clipping tiny in all",all(s['m3_max']<1e-6 for s in D['stats'].values()),[s['m3_max'] for s in D['stats'].values()])
chk("all channels in catalog assigned",set(D['catalog'])==set(c for n in D['nodes'] for c in n['ch']))
print("FAILS",bad)
