import numpy as np, gym_electric_motor as gem, inspect
from gym_electric_motor.physical_systems import mechanical_loads as ml
print("loads:", [n for n in dir(ml) if n.endswith('Load')])
for n in ['PolynomialStaticLoad','ExternalSpeedLoad','ConstantSpeedLoad']:
    c = getattr(ml,n,None)
    if c: print(n, "params:", inspect.signature(c.__init__))
print(inspect.getsource(ml.PolynomialStaticLoad)[:2500])
