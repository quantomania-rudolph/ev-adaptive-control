import time, numpy as np, gym_electric_motor as gem
env = gem.make("Cont-CC-PMSM-v0")
obs, info = env.reset(seed=0) if True else None
print("obs type", type(obs), [getattr(o,'shape',None) for o in obs] if isinstance(obs,tuple) else None)
ps = env.unwrapped.physical_system
print("state_names:", list(ps.state_names))
print("limits:", dict(zip(ps.state_names, ps.limits)))
print("nominal:", dict(zip(ps.state_names, ps.nominal_state)))
print("action space:", env.action_space)
print("tau:", env.unwrapped.physical_system.tau)
print("converter:", type(ps.converter).__name__, "supply:", type(ps.electrical_motor).__name__)
