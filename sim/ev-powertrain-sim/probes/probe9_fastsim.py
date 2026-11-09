"""FASTSim 3.1.0: which cycles ship, what the Bolt EV model exposes, and does its battery have a voltage?"""
import fastsim as fsim, yaml
print("cycles bundled:", [str(p) for p in fsim.Cycle.list_resources()])
print("vehicles:", [str(p) for p in fsim.Vehicle.list_resources()])
veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
cyc = fsim.Cycle.from_resource("udds.csv")
sd = fsim.SimDrive(veh, cyc); sd.walk()
df = sd.to_dataframe(); print("columns:", df.shape[1], "rows:", df.shape[0])
d = yaml.safe_load(veh.to_yaml())
def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k != "history": yield from flat(v, f"{p}.{k}" if p else k)
    else: yield p, x
items = list(flat(d))
for k, v in items:
    if any(s in k.lower() for s in ["drag_coef", "frontal", "wheel_radius", "rr_coef", "pwr_out_max", "energy_capacity"]): print(k, "=", v)
print("any voltage field in the battery (RES)?", any("volt" in k.lower() for k, _ in items))
