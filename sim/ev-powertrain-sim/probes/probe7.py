import numpy as np, gym_electric_motor as gem
from scipy.integrate import solve_ivp
env = gem.make("Cont-CC-PMSM-v0", constraints=())
ps = env.unwrapped.physical_system
names=list(ps.state_names); lim=np.array(ps.limits); ix={n:i for i,n in enumerate(names)}
mp = ps.electrical_motor.motor_parameter
p,Ld,Lq,Rs,psi = mp['p'],mp['l_d'],mp['l_q'],mp['r_s'],mp['psi_p']
U=150.0; tau=1e-4
obs,_=env.reset(seed=0)
# fixed mechanical speed (ConstantSpeedLoad omega_fixed default=100 rad/s reported in state)
s=obs[0]*lim; om = s[ix['omega']]; print("omega (rad/s):", om, " eps0:", s[ix['epsilon']])
cmd_seq=[]; gem_id=[]; gem_iq=[]; eps_log=[]
for k in range(3000):
    s=obs[0]*lim; eps=s[ix['epsilon']]
    cmd = np.array([30.0, 60.0]) if k<1500 else np.array([-20.0, 30.0])
    d = ps.dq_to_abc_space(cmd, eps)/U; d = d-(d.max()+d.min())/2
    obs,_,_,_,_=env.step(np.clip(d,-1,1)); s=obs[0]*lim
    cmd_seq.append(cmd); gem_id.append(s[ix['i_sd']]); gem_iq.append(s[ix['i_sq']])
# our dq model, same voltage, same speed
wel = p*om
def f(t,x,u):
    i_d,i_q=x
    return [ (u[0]-Rs*i_d + wel*Lq*i_q)/Ld, (u[1]-Rs*i_q - wel*(Ld*i_d+psi))/Lq ]
x=np.zeros(2); mi=[]
for k in range(3000):
    u=cmd_seq[k]
    sol=solve_ivp(f,[0,tau],x,args=(u,),method='RK45',rtol=1e-8,atol=1e-10); x=sol.y[:,-1]; mi.append(x.copy())
mi=np.array(mi); gi=np.array([gem_id,gem_iq]).T
err=np.abs(mi-gi)
print("max |i_d err| A:", err[:,0].max(), " max |i_q err| A:", err[:,1].max(), " final gem:", gi[-1].round(2), " final ours:", mi[-1].round(2))
print("signal range i_q (A):", gi[:,1].min().round(1), gi[:,1].max().round(1))
# torque check
T_gem = None
print("torque formula 1.5 p (psi+(Ld-Lq) id) iq at end:", (1.5*p*(psi+(Ld-Lq)*gi[-1,0])*gi[-1,1]).round(2), " GEM torque state:", (obs[0]*lim)[ix['torque']].round(2))
