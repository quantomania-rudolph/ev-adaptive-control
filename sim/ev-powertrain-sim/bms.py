"""BMS message model (contract row M6): what the controller is told over CAN about the pack, and when.

Every `cfg.period_s` the model builds ONE message from the battery truth AS IT WAS `cfg.latency_s` earlier and holds it:

    i_dis_lim = quantize((1 + e) * min(envelope_toy_dis(soc_sampled), i_batt_max_sampled))
    i_chg_lim = quantize((1 + e) * envelope_toy_chg(soc_sampled))
    v_pack    = V_dc sampled                       (no extra noise: the SENSOR layer already models the divider)
    age_s     = t - t_sample   (grows between messages; the message itself is the same object until the next period)

with e ~ N(0, cfg.rel_err) drawn ONCE per message from the BMS rng stream (one draw covers both limits).

Properties that tests pin down (tests/test_bms.py):
  * CAUSAL: the message at t depends only on truth sampled at times <= t - latency (a history buffer is kept; before
    enough history exists the earliest sample is used).
  * a fault that lowers truth i_batt_max (idis_scale) is NOT visible until `latency` after it happens.
  * cadence: the message only changes on multiples of `period_s`.
  * cfg.enabled False: the message equals the current truth (error-free, unquantized, zero latency, age 0).

The envelopes come from the fit file (observed-use envelope, scaled to the toy by battery_fit.scale_to_toy); they are
NOT BMS limits of the real car (see docs/BATTERY_FIT.md).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from interfaces import BMSConfig, BMSMsg


class BMSModel:
    def __init__(self, cfg: BMSConfig, fit: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._grid = np.asarray(fit["envelope_real"]["soc_grid"], dtype=float)
        self._dis = np.asarray(fit["scaling"]["i_dis_lim_toy_a"], dtype=float)
        self._chg = np.asarray(fit["scaling"]["i_chg_lim_toy_a"], dtype=float)
        if not (len(self._grid) == len(self._dis) == len(self._chg) >= 2):
            raise ValueError("fit envelope arrays must have equal length >= 2")
        self.reset()

    # ------------------------------------------------------------------ helpers
    def env_dis(self, soc: float) -> float:
        return float(np.interp(soc, self._grid, self._dis))

    def env_chg(self, soc: float) -> float:
        return float(np.interp(soc, self._grid, self._chg))

    def _quant(self, x: float) -> float:
        q = self.cfg.quant_a
        x = max(float(x), 0.0)
        return x if q <= 0.0 else float(np.round(x / q) * q)

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        self._hist: deque = deque()          # (t, soc, V_dc, i_batt_max), strictly causal (only past/present truth)
        self._k_last = -1
        self._msg = None
        self._t_sample = 0.0
        self._t_prev = -np.inf

    def step(self, t: float, battery_truth: dict) -> BMSMsg:
        """Call once per simulation step with the TRUE battery state at time t (keys: interfaces.BATTERY_TRUTH_KEYS)."""
        t = float(t)
        if t < self._t_prev - 1e-12:
            raise ValueError("BMSModel.step: time went backwards (call reset())")
        self._t_prev = t
        soc, v_dc, i_max = float(battery_truth["soc"]), float(battery_truth["V_dc"]), float(battery_truth["i_batt_max"])
        if not self.cfg.enabled:
            return BMSMsg(i_dis_lim=min(self.env_dis(soc), i_max), i_chg_lim=self.env_chg(soc), v_pack=v_dc, age_s=0.0)

        self._hist.append((t, soc, v_dc, i_max))
        k = int(np.floor(t / self.cfg.period_s + 1e-9))
        if k > self._k_last:                                    # a period boundary has been crossed: build a new message
            self._k_last = k
            target = t - self.cfg.latency_s
            while len(self._hist) > 1 and self._hist[1][0] <= target + 1e-12:
                self._hist.popleft()                            # older samples can never be needed again (targets only grow)
            ts, s_soc, s_v, s_imax = self._hist[0]              # latest sample <= target, else the earliest available
            e = float(self.rng.normal(0.0, self.cfg.rel_err))
            self._msg = (self._quant((1.0 + e) * min(self.env_dis(s_soc), s_imax)),
                         self._quant((1.0 + e) * self.env_chg(s_soc)), s_v)
            self._t_sample = ts
        i_dis, i_chg, v_pack = self._msg
        return BMSMsg(i_dis_lim=i_dis, i_chg_lim=i_chg, v_pack=v_pack, age_s=t - self._t_sample)
