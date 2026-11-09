import numpy as np, gym_electric_motor as gem
env = gem.make("Cont-CC-PMSM-v0", constraints=())
ps = env.unwrapped.physical_system
names=list(ps.state_names); lim=np.array(ps.limits); ix={n:i for i,n in enumerate(names)}
obs,_=env.reset(seed=0)
U=150.0
def phys(o): return o[0]*lim
cmd = np.array([10.0, 40.0])
print("k  eps_before  cmd_dq     applied_dq(reported)   err")
for k in range(8):
    s = phys(obs); eps = s[ix['epsilon']]            # electrical angle [rad], limit pi
    duty = ps.dq_to_abc_space(cmd, eps) / U
    obs,_,t,tr,_ = env.step(np.clip(duty,-1,1))
    s2 = phys(obs)
    ap = np.array([s2[ix['u_sd']], s2[ix['u_sq']]])
    print(k, round(eps,3), cmd, np.round(ap,2), np.round(ap-cmd,2))
# max reachable dq voltage magnitude with sinusoidal duties (no zero-seq injection)
print("reachable |u_dq| with |duty|<=1 per leg, sinusoidal:", U, "; with zero-seq injection up to", round(2*U/np.sqrt(3),1))
