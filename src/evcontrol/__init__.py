"""EV adaptive control under input/state constraints.

Three segments, each owned by a different person/subteam:

    evcontrol.datagen     -> produces references and disturbances (inputs to a simulation)
    evcontrol.actuator    -> saturation + actuator model (u = sat(u_c), delta_u = u - u_c)
    evcontrol.evaluation  -> metrics that score a finished run

evcontrol.pipeline glues them together for the end-to-end integration test.
"""

__version__ = "0.1.0"
