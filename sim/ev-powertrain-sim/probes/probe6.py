import numpy as np, gym_electric_motor as gem
env = gem.make("Cont-CC-PMSM-v0", constraints=())
ps = env.unwrapped.physical_system
names=list(ps.state_names); lim=np.array(ps.limits); ix={n:i for i,n in enumerate(names)}
U=150.0
def trial(mag, ang, inject):
    obs,_=env.reset(seed=0); s=obs[0]*lim
    eps=s[ix['epsilon']]
    cmd = mag*np.array([np.cos(ang), np.sin(ang)])
    d = ps.dq_to_abc_space(cmd, eps)/U
    if inject: d = d - (d.max()+d.min())/2          # min-max zero-sequence injection (SVPWM-equivalent)
    obs,_,_,_,_ = env.step(np.clip(d,-1,1)); s2=obs[0]*lim
    ap = np.array([s2[ix['u_sd']], s2[ix['u_sq']]])
    return cmd, ap, np.abs(d).max()
print("mag  angle  inject | commanded           applied              |err|   max|duty|")
for mag in [140, 160, 170, 180, 200]:
  for ang in [0.0, np.pi/6, np.pi/3, np.pi/2]:
    for inj in [False, True]:
        c,a,m = trial(mag, ang, inj)
        print(f"{mag:4d} {np.degrees(ang):5.0f}  {str(inj):5s} | {np.round(c,1)}  {np.round(a,1)}  {np.linalg.norm(a-c):6.1f}  {m:5.2f}")
