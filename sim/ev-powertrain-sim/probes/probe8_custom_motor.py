"""Can GEM be re-parameterised to an EV-traction-scale motor and a 350 V supply?
ILLUSTRATIVE numbers only (not a real vehicle). Result: accepted, steps fine."""
import numpy as np, gym_electric_motor as gem
from gym_electric_motor.physical_systems.voltage_supplies import IdealVoltageSupply
motor = dict(
    motor_parameter=dict(p=4, l_d=0.00020, l_q=0.00055, j_rotor=0.05, r_s=0.008, psi_p=0.085),
    limit_values=dict(i=700.0, omega=900.0, u=350.0),
    nominal_values=dict(i=500.0, omega=600.0, u=350.0),
)
env = gem.make("Cont-CC-PMSM-v0", motor=motor, supply=IdealVoltageSupply(350.0), constraints=())
ps = env.unwrapped.physical_system
print("limits:", {k: float(v) for k, v in zip(ps.state_names, ps.limits) if k in ("omega", "torque", "i_sd", "u_sd", "u_sup")})
env.reset(seed=0); [env.step(np.zeros(3)) for _ in range(5)]; print("stepping ok")
