"""Plant adapter: wraps Gym-Electric-Motor (GEM) PMSM drive so that saturation is MEASURED, not inferred.

Why this file exists (lessons from probing GEM 3.0.3, see RED_TEAM.md):
  * GEM's action is three phase duty cycles, not the (u_d, u_q) our theory uses -> we convert.
  * GEM silently clips each duty to [-1, 1]. That clip is direction dependent in the dq frame, and
    it reports nothing. So we do the saturation ourselves, with a clean, isotropic definition
    (a disc in the dq plane), BEFORE GEM sees the command. Then GEM's own clip never fires and
    the ledger row "M3" (plant-reported minus ours) must stay ~0; if it does not, something hidden
    is happening and we find out.
  * GEM terminates an episode when a current limit is exceeded. We switch that off (constraints=())
    and monitor distance-to-limit ourselves, so a constraint violation is DATA, not a crash.
  * GEM's default load is a constant-speed load (speed is imposed, not a state). We subclass the
    polynomial load so speed is a real state and a time-varying road load d(t) can be injected.
"""
from __future__ import annotations

import numpy as np
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad

SQRT3 = np.sqrt(3.0)


class RoadLoad(PolynomialStaticLoad):
    """Polynomial friction/drag load plus an external disturbance torque d(t).

    The notes' plant is x_dot = f(x) + g(x)u + d(x,t). GEM's mechanical_ode already receives the
    simulation time t, so injecting d(t) is a one-line override (verified in the GEM source).
    Sign convention: positive disturbance opposes motor torque (like a hill or headwind).
    """

    def __init__(self, load_parameter, limits, disturbance_fn=None):
        super().__init__(load_parameter=load_parameter, limits=limits)
        self._d = disturbance_fn or (lambda t: 0.0)

    def mechanical_ode(self, t, mechanical_state, torque):
        omega = mechanical_state[self.OMEGA_IDX]
        total = torque - self._static_load(omega) - self._d(t)
        return np.array([total / self._j_total])


class GEMPlant:
    """Plant adapter. Controllers see ONLY this class, never GEM, so Tier 2 can swap the plant."""

    # --- names are OUR schema (see plan, "State-vector contract"); GEM names mapped here once ---
    GEM_TO_OURS = {
        "omega": "omega_m", "torque": "T_elec", "i_sd": "i_d", "i_sq": "i_q",
        "u_sd": "u_d_plant", "u_sq": "u_q_plant", "epsilon": "epsilon_el", "u_sup": "V_dc",
        # phase-domain signals: what the inverter legs actually do (new: a real EV data pipeline sees these)
        "i_a": "i_a", "i_b": "i_b", "i_c": "i_c", "u_a": "u_a", "u_b": "u_b", "u_c": "u_c",
    }

    def __init__(self, j_load=0.5, b=0.02, c=1.0e-4, disturbance_fn=None, seed=0, battery=None, fault_fn=None,
                 a=0.0, tau=1.0e-4):
        load = RoadLoad(
            load_parameter=dict(a=a, b=b, c=c, j_load=j_load),   # a: constant (rolling) torque, b: viscous, c: quadratic (drag)
            limits=dict(omega=418.879),  # rad/s, same as GEM's default PMSM speed limit
            disturbance_fn=disturbance_fn,
        )
        self.load = load
        self.env = gem.make("Cont-CC-PMSM-v0", load=load, constraints=(), tau=tau)
        self.ps = self.env.unwrapped.physical_system
        self.tau = float(self.ps.tau)
        names = list(self.ps.state_names)
        self._lim = np.array(self.ps.limits, dtype=float)
        self._ix = {n: i for i, n in enumerate(names)}
        mp = self.ps.electrical_motor.motor_parameter
        self.p, self.Ld, self.Lq = mp["p"], mp["l_d"], mp["l_q"]
        self.Rs, self.psi_p, self.J = mp["r_s"], mp["psi_p"], load._j_total
        self.V_dc = float(self._lim[self._ix["u_sup"]])          # 300 V
        self.u_max_hw = self.V_dc / SQRT3                         # 173.2 V: inscribed disc of the hexagon
        self.limits = {"i": 400.0, "omega": float(self._lim[self._ix["omega"]]), "u": self.u_max_hw}
        self.t = 0.0
        # Optional battery + DC link (battery.py). With battery=None the supply is the constant 300 V of Tier 1 v0.
        # fault_fn(t) -> {"r0_mult": x, "idis_scale": y} injects physical battery faults (cold pack, BMS derate).
        self.battery, self.fault_fn = battery, fault_fn
        self._vdc = battery.v_dc if battery is not None else self.V_dc
        self._obs, _ = self.env.reset(seed=seed)

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        s = self._obs[0] * self._lim                              # GEM returns normalised states
        out = {self.GEM_TO_OURS[k]: float(s[self._ix[k]]) for k in self.GEM_TO_OURS}
        out["omega_el"] = self.p * out["omega_m"]
        out["V_dc"] = self._vdc          # TRUE bus voltage (GEM's own u_sup is only a fixed numerical scale, see apply_voltage)
        out["t"] = self.t
        # Load-side truth: split the mechanical torque balance so analysts see each term, not just the sum.
        # J dw/dt = T_elec - T_friction(w) - T_dist(t)   (T_dist > 0 opposes motion: hill, headwind)
        out["T_fric"] = float(self.load._static_load(out["omega_m"]))
        out["T_dist"] = float(self.load._d(self.t))
        return out

    # ------------------------------------------------------------------ actuator (the important bit)
    def true_ceiling(self, u_max_imposed: float | None = None) -> float:
        """The physical voltage ceiling NOW (single definition, used by apply_voltage and by the realism loop's oracle channel)."""
        u_ceiling = min(self._vdc / SQRT3, self.u_max_hw) if self.battery is not None else self.u_max_hw
        return u_ceiling if u_max_imposed is None else min(u_ceiling, float(u_max_imposed))

    def apply_voltage(self, u_cmd: np.ndarray, u_max_imposed: float | None = None, u_extra: np.ndarray | None = None) -> dict:
        """Apply a dq voltage command through OUR saturation, step GEM, return the mismatch record.

        u_max_imposed lets an experiment impose a sudden, unmodeled limit (e.g. a DC-link sag)
        that the controller does not know about. Tier 2 will produce this physically from the
        battery model; Tier 1 imposes it here so the failure is reproducible.
        """
        u_cmd = np.asarray(u_cmd, dtype=float)
        # Physical ceiling NOW. Capped at the 300 V-scale hexagon GEM can represent: when regeneration pushes the real
        # bus above 300 V the true ceiling would be higher, but GEM's duty would exceed 1 and clip silently (measured:
        # up to 4 V of hidden clipping at V_dc = 307 V). Capping keeps M3 = 0; the cost is up to ~4 V of unused headroom
        # during regen, a known limitation that Tier 2 removes by building the converter on the real bus voltage.
        u_lim = self.true_ceiling(u_max_imposed)
        mag = float(np.hypot(*u_cmd))
        # Radial projection onto a disc: sat(u) = u * min(1, u_lim/|u|). Isotropic, Lipschitz, and
        # keeps the direction of u_c, which is the saturation model the proof can use.
        u_app = u_cmd if mag <= u_lim else u_cmd * (u_lim / mag)
        # u_extra (realism layer): an additive voltage error AFTER the clip (inverter dead-time and device drops are real
        # voltage errors that no controller clip can see). None (default) = legacy behaviour, bit for bit.
        u_gem = u_app if u_extra is None else u_app + np.asarray(u_extra, dtype=float)

        s = self._obs[0] * self._lim
        eps = float(s[self._ix["epsilon"]])
        duty = self.ps.dq_to_abc_space(u_gem, eps) / (self.V_dc / 2.0)
        # Min-max zero-sequence injection: shifts all three legs equally, does not change the dq
        # voltage, and lets the disc radius reach V_dc/sqrt(3) instead of V_dc/2 (verified).
        duty = duty - 0.5 * (duty.max() + duty.min())
        raw_duty_peak = float(np.abs(duty).max())                  # >1 would mean GEM clips internally
        self._obs, _, _, _, _ = self.env.step(np.clip(duty, -1.0, 1.0))
        self.t += self.tau

        st = self.state()
        u_plant = np.array([st["u_d_plant"], st["u_q_plant"]])
        batt = {}
        if self.battery is not None:
            # Electrical power the motor takes from the inverter output, from the DELIVERED voltage and the new currents.
            # NOTE the GEM converter is still scaled to a fixed 300 V (duty = u_app/(300/2)); because we pass the
            # voltage we want, GEM applies exactly u_app whatever the real bus is. The real bus only sets the ceiling.
            p_el = 1.5 * (u_gem[0] * st["i_d"] + u_gem[1] * st["i_q"])
            f = self.fault_fn(self.t) if self.fault_fn else {}
            batt = self.battery.step(self.tau, p_el, r0_mult=f.get("r0_mult", 1.0), idis_scale=f.get("idis_scale", 1.0))
            batt["P_el"] = p_el
            self._vdc = batt["V_dc"]
        return {**batt,
            "u_d_cmd": u_cmd[0], "u_q_cmd": u_cmd[1],             # request  (u_c)
            "u_d_app": u_app[0], "u_q_app": u_app[1],             # delivered by our saturation (u)
            "d_u_d": u_app[0] - u_cmd[0], "d_u_q": u_app[1] - u_cmd[1],   # M2: delta u = u - u_c
            "d_u_norm": mag and (np.hypot(*(u_app - u_cmd)) / u_lim),
            "u_limit": u_lim,
            "saturated": float(mag > u_lim),
            "d_u_plant_d": u_plant[0] - u_gem[0],                 # M3: hidden clipping check, expect ~0
            "d_u_plant_q": u_plant[1] - u_gem[1],
            "u_d_extra": u_gem[0] - u_app[0], "u_q_extra": u_gem[1] - u_app[1],   # realism layer: error added after the clip
            "duty_peak": raw_duty_peak,
        }
