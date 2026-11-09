import time, numpy as np, gym_electric_motor as gem, inspect
env = gem.make("Cont-CC-PMSM-v0")
ps = env.unwrapped.physical_system
names = list(ps.state_names)
obs, info = env.reset(seed=0)
state, ref = obs
print("reset state (normalised?):", dict(zip(names, np.round(state,4))))
print("ref:", ref)
# motor params
em = ps.electrical_motor
print("motor params:", {k:v for k,v in em.motor_parameter.items()})
print("motor limits:", em.limits if hasattr(em,'limits') else None)
print("mech load:", type(ps.mechanical_load).__name__, getattr(ps.mechanical_load,'load_parameter',None) or getattr(ps.mechanical_load,'_load_parameter',None))
print("inertia j_total:", getattr(ps.mechanical_load,'j_total',None))
print("env.max_steps:", getattr(env.unwrapped,'_max_steps',None), "reward fn:", type(env.unwrapped.reward_function).__name__)
print("constraints:", [type(c).__name__ for c in getattr(env.unwrapped.constraint_monitor,'_constraints',[])])
print("ode solver:", type(ps.ode_solver).__name__ if hasattr(ps,'ode_solver') else None)
# step-time benchmark
env.reset(seed=1)
N=20000; t=time.perf_counter(); n=0
a = np.zeros(3)
for _ in range(N):
    o,r,term,trunc,i = env.step(a); n+=1
    if term or trunc: env.reset()
dt=time.perf_counter()-t
print(f"steps/s: {N/dt:.0f}  -> real time per sim second (tau=1e-4): {1/(1e4/(N/dt)) :.3f} (wall s per sim s)")
