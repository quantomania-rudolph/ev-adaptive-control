"""Generates docs/ASSUMPTIONS.md from the code, so the list cannot drift from the defaults.   python tools/make_assumptions.py
Each setting's value comes from the dataclass default; its rationale and provenance tag ([T1] measured from TUM, [T2] typical value, unverified,
[T3] design choice) are harvested from the comment on the same source line."""
import sys, re, inspect, dataclasses, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import interfaces as I, dataset
from battery import Battery

def table(cls, title, note=""):
    src = inspect.getsource(cls).splitlines(); com = {}
    for l in src:
        m = re.match(r"\s{4}(\w+)\s*:\s*[^=]+=\s*[^#]+?(?:#\s*(.*))?$", l)
        if m: com[m.group(1)] = (m.group(2) or "").strip()
    out = [f"### {title}", note, "", "| Setting | Default | Source comment |", "|---|---|---|"]
    for f in dataclasses.fields(cls):
        out.append(f"| `{f.name}` | `{getattr(cls(), f.name)}` | {com.get(f.name, '').replace('|', '/')} |")
    return "\n".join(out) + "\n"

P = dataset.params()
txt = f"""# Assumptions register (generated; do not edit by hand: `python tools/make_assumptions.py`)

Provenance tags: **[T1]** measured from the TUM traces by a script in this repo; **[T2]** typical class value, UNVERIFIED; **[T3]** design choice (swept in the sensitivity study where it matters). Anything without a tag is a setting of the simulation itself.
Independent audit of these numbers: `docs/AUDIT_PARAMETERS.md` (6 wrong, 14 suspicious items found 2026-11-09, listed there).

## 1. The toy scale
The simulator is a TOY: {P['plant_true']['p']['value']:.0f} pole pairs, {P['plant_limits']['V_dc_nominal_V']:.0f} V bus (ceiling {P['plant_limits']['voltage_ceiling_V']:.1f} V), motor from Gym-Electric-Motor 3.0.3 defaults. "Realistic" means the controller receives the same KINDS of imperfect information a production drive does, at toy scale; it is not a digital twin of the Tesla.

## 2. True plant (units in `data/params.json`)
| Symbol | Value | Unit | Source |
|---|---|---|---|
""" + "\n".join(f"| {k} | {v['value']:.6g} | {v['unit']} | {v['source']} |" for k, v in P["plant_true"].items()) + f"""

Limits: current {P['plant_limits']['current_limit_A']:.0f} A (plant termination), speed {P['plant_limits']['speed_limit_rad_s']:.1f} rad/s. Voltage limit is a disc of radius V_dc/sqrt(3) (our own radial projection). Cycle runs replace J and the load terms with a fitted vehicle (`cycles.fit_vehicle_to_motor`).

## 3. Settings of the information path and controller
""" + table(I.SensorConfig, "Sensors (`SensorConfig`)") + "\n" + table(I.InverterConfig, "Inverter (`InverterConfig`)") + "\n" + \
    table(I.BMSConfig, "BMS (`BMSConfig`)") + "\n" + table(I.BeliefConfig, "Controller beliefs about the plant (`BeliefConfig`): drawn per seed, relative to truth") + "\n" + \
    table(I.ControllerConfig, "Controller (`ControllerConfig`)") + f"""
## 4. Battery
Pack numbers come from `data/tum_battery_fit.json` ([T1]: resistance scale, capacity, current limits; `docs/BATTERY_FIT.md`). The open-circuit-voltage curve, the temperature law of R0, the RC branch and the thermal mass are generic NMC assumptions [T2]. `Battery()` with no fit file uses defaults that differ from the fitted pack (audit item).

## 5. Run design choices
* Event: ceiling forced to 25 V for t = 4.5 to 5.5 s (`hidden`, `oracle`); BMS derate for the battery scenarios; cycle runs derate to 0.02 for 3 s at peak power minus 0.5 s.
* Integration step tau = 1e-4 s (plain and battery runs), 2e-4 s (cycle runs). The speed loop runs every 10 steps.
* Seeds: 20 per scenario and controller (matrix), 5 (cycles), 3 (sensitivity). Beliefs, sensor offsets and noise are drawn per seed from separate streams.
* Stored seed-0 rows: every 17th step (matrix), every 23rd (cycles); full rate only in `data/audit/fr_*`.
* Tesla trace: scaled to the toy motor (vehicle fitted to the motor); asks for up to 1.5x its torque by design.
* Envelope test `envelope_ok`: current below 275 A and flux estimate below 0.15 Wb.
* Statistics: Wilson 95 % intervals; Mann-Whitney for the pass/fail rule A5; n = 3 sweep uses complete separation.

## 6. Known limits that are assumptions, not bugs
Single-run evidence, not proofs. Sensor, inverter and BMS values are assumed. TUM request holding/quantization (`ref_hold_s`, `ref_quant`) is available but off in the runs. Current-limit numbers (240/250/275/400 A) have different roles (pack, controller, envelope, plant) and are not one parameter.
"""
(ROOT / "docs/ASSUMPTIONS.md").write_text(txt)
print(len(txt.splitlines()), "lines")
