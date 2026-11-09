"""Writes data/catalog.csv (data dictionary) and data/params.json (every physical and configuration number, with units and source)."""
import sys, json, pathlib, dataclasses
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import dataset, telemetry, interfaces as I
from plant import GEMPlant
from battery import Battery

cat = dataset.catalog()
idx = dataset.index()
stored = set()
import pyarrow.parquet as pq
for p in idx.path[:1]: stored |= set(pq.ParquetFile(ROOT / p).schema_arrow.names)
cat["in_raw_stored_files"] = cat.name.isin(stored) | cat.name.str.startswith("ts_").eq(False) & False
cat.drop(columns="in_raw_stored_files").to_csv(ROOT / "data/catalog.csv", index=False)

pl = GEMPlant(battery=Battery()); P = telemetry.params_from_plant(pl)
U = {"p": "", "Ld": "H", "Lq": "H", "R": "ohm", "psi": "Wb", "J": "kg m^2", "tau": "s", "a_load": "N m", "b_load": "N m s/rad", "c_load": "N m s^2/rad^2"}
out = {
  "_about": "True plant values (what the physics uses) and default settings of everything else. Source = file that defines it.",
  "plant_true": {k: {"value": float(P[k]), "unit": U.get(k, ""), "source": "plant.py / Gym-Electric-Motor 3.0.3 PMSM defaults"} for k in ("p","Ld","Lq","R","psi","J","a_load","b_load","c_load","tau")},
  "plant_limits": {"V_dc_nominal_V": pl.V_dc, "voltage_ceiling_V": pl.u_max_hw, "current_limit_A": pl.limits["i"], "speed_limit_rad_s": pl.limits["omega"],
                   "note": "ceiling = V_dc/sqrt(3) (inscribed disc of the hexagon); radial projection onto the disc"},
  "cycle_runs_note": "Cycle runs replace J and the load terms with a fitted vehicle (cycles.fit_vehicle_to_motor); see dataset.params_for().",
  "controller_default": dataclasses.asdict(I.ControllerConfig()),
  "belief_ranges": dataclasses.asdict(I.BeliefConfig()),
  "sensors": dataclasses.asdict(I.SensorConfig()),
  "inverter": dataclasses.asdict(I.InverterConfig()),
  "bms": dataclasses.asdict(I.BMSConfig()),
  "battery_default": {k: float(v) for k, v in vars(Battery()).items() if isinstance(v, (int, float))},
  "battery_fit_file": "data/tum_battery_fit.json",
  "units_convention": "SI: V, A, Wb, H, ohm, rad/s (mechanical), s, N m, W, J. Electrical speed = p * mechanical speed.",
}
(ROOT / "data/params.json").write_text(json.dumps(out, indent=1, default=str))
print(len(cat), "catalog rows;", len(out), "param sections")
