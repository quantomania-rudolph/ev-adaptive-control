"""Boundedness measures for a finished run: what we can MEASURE about BIBS / UUB, and what we cannot.

Learning note (read this before quoting any number from here):
  * Bounded INPUT is guaranteed by construction. Our saturation is a radial projection onto a disc, so the delivered
    voltage can never exceed the ceiling. `input_bound` checks that on the logged data; it should be exactly 1.0 or less.
  * Bounded STATE is only ever MEASURED here, on one simulated run. A run that stays bounded is evidence, not a proof:
    the theory team still has to show V-dot <= -lambda V + c for ALL admissible disturbances.
  * 'Ultimate bound' (UUB) is reported as the largest value a state takes in a final window, plus the time after which
    it never left that ball. It describes this run only.
  * lambda_hat and the floor are DESCRIPTIVE fits of how fast V fell back and where it settled. They are not a proof of
    the inequality, only the numbers to compare a proof's lambda and c/lambda against.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _first_stays_below(t, x, level):
    """First time after which x never exceeds level again (None if it never settles)."""
    over = np.nonzero(x > level)[0]
    if len(over) == 0:
        return float(t[0])
    k = over[-1] + 1
    return float(t[k]) if k < len(t) else None


def bounds_report(d: pd.DataFrame, *, event=None, window_s=None, u_nom=173.2051, i_lim=250.0) -> dict:
    """d must hold t, u_d_app, u_q_app, u_limit, i_d, i_q, psi_tilde, e_q, omega_ref, omega_m, V, saturated."""
    t = d["t"].to_numpy(float)
    T = float(t[-1] - t[0])
    window = window_s if window_s else max(0.1 * T, 5 * float(np.median(np.diff(t))))
    last = t >= t[-1] - window
    u = np.hypot(d["u_d_app"], d["u_q_app"]).to_numpy()
    ceiling = d["u_limit"].to_numpy()
    imag = np.hypot(d["i_d"], d["i_q"]).to_numpy()
    psi_err = d["psi_tilde"].abs().to_numpy()
    e_q = d["e_q"].abs().to_numpy()
    e_w = (d["omega_ref"] - d["omega_m"]).abs().to_numpy()
    V = d["V"].to_numpy()
    out = dict(
        # --- bounded input (by construction)
        input_bound=float(np.max(u / np.maximum(ceiling, 1e-9))),                 # <= 1 always
        input_peak_v=float(u.max()), ceiling_min_v=float(ceiling.min()), ceiling_max_v=float(ceiling.max()),
        # --- bounded state (measured)
        psi_err_peak=float(psi_err.max()), i_peak=float(imag.max()), i_over_limit=float(imag.max() / i_lim),
        e_speed_peak=float(e_w.max()), V0=float(V[0]), V_peak=float(V.max()), V_amplification=float(V.max() / max(V[0], 1e-12)),
        # --- ultimate bound over the final window
        window_s=float(window), psi_err_ultimate=float(psi_err[last].max()), e_q_ultimate=float(e_q[last].max()),
        e_speed_ultimate=float(e_w[last].max()), V_floor=float(np.median(V[last])),
    )
    # settle time: after this the estimate error never leaves the larger of the ultimate bound and 2 % of its own peak
    # (a bare 'ultimate bound' is the noise floor when the error is ~0, which would make every settle time the last sample)
    out["t_settle_psi_err"] = _first_stays_below(t, psi_err, max(out["psi_err_ultimate"], 0.02 * out["psi_err_peak"]))
    # --- descriptive decay fit: log-slope of the running-max envelope of V from its peak down to 10 % of the peak
    ip = int(np.argmax(V))
    env = pd.Series(V).rolling(max(3, int(0.2 / max(float(np.median(np.diff(t))), 1e-9))), min_periods=1).max().to_numpy()
    tail = np.arange(ip, len(t))
    below = tail[env[tail] < 0.1 * V[ip]]
    j = int(below[0]) if len(below) else len(t) - 1
    lam = None
    # fit only where the envelope has left its plateau (below 90 % of the peak) and is above 10 %
    seg_idx = np.arange(ip, j + 1)
    seg_idx = seg_idx[env[seg_idx] <= 0.9 * V[ip]]
    if len(seg_idx) >= 5 and V[ip] > 0:
        slope = np.polyfit(t[seg_idx], np.log(np.maximum(env[seg_idx], 1e-300)), 1)[0]
        lam = float(-slope) if slope < 0 else None
    out["lambda_hat"] = lam
    # --- sign structure of V-dot, split by saturation (descriptive: the exact claim needs the proof)
    vd = np.gradient(V, t)
    sat = d["saturated"].to_numpy() > 0.5
    tol = 1e-6 * max(float(V.max()), 1e-12) / max(T, 1e-12)          # dead-band: float rounding of the time axis must not count as growth
    out["vdot_pos_frac_unsat"] = float((vd[~sat] > tol).mean()) if (~sat).any() else None
    out["vdot_pos_frac_sat"] = float((vd[sat] > tol).mean()) if sat.any() else None
    if event is not None:
        out["t_event"] = list(event)
        after = t >= event[1]
        out["psi_err_after_event_peak"] = float(psi_err[after].max()) if after.any() else None
    return out
