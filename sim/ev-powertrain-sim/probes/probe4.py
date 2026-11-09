import numpy as np, gym_electric_motor as gem
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
env = gem.make("Cont-CC-PMSM-v0")
ps = env.unwrapped.physical_system
print("PS methods:", [m for m in dir(ps) if ('dq' in m or 'abc' in m or 'alpha' in m)])
names = list(ps.state_names); lim = np.array(ps.limits)
idx = {n:i for i,n in enumerate(names)}
def run(env, action_fn, n, reset_kwargs=None):
    obs,_ = env.reset(seed=0)
    log=[]
    for k in range(n):
        a = action_fn(k, obs)
        obs, r, term, trunc, _ = env.step(a)
        s = obs[0]*lim
        log.append((k, s.copy(), term, trunc))
        if term or trunc: break
    return log
# S2/S3: step response of voltages to a duty command, same-step vs delayed
log = run(env, lambda k,o: np.array([0.5,0.5,0.5]) if k>=2 else np.zeros(3), 8)
print("--- converter response to duty step (0.5 at k=2): u_a [V], u_sd, u_sq")
for k,s,t,tr in log: print(k, round(s[idx['u_a']],1), round(s[idx['u_sd']],1), round(s[idx['u_sq']],1), t, tr)
# S2b: out-of-range action
log = run(env, lambda k,o: np.array([3.0,3.0,3.0]), 3)
print("--- action 3.0 (out of range): u_a =", [round(s[idx['u_a']],1) for _,s,_,_ in log], "(150 = clipped to 1.0)")
# S4: limit violation -> terminate?
log = run(env, lambda k,o: np.array([1.0,-0.5,-0.5]), 5000)
print("--- full duty on one leg until limit: steps run", len(log), "terminated:", log[-1][2], "truncated:", log[-1][3])
s = log[-1][1]; print("final i_sd,i_sq,omega,torque:", [round(s[idx[n]],1) for n in ['i_sd','i_sq','omega','torque']])
print("limits i:", lim[idx['i_sd']])
