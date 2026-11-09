# Closed decisions (2026-11-09, project lead)

| ID | Decision | Why |
|---|---|---|
| D3 | `d_gain_by_L` is NOT the default. The dataclass default stays `False` (d gain k_d = 3). Envelope compliance of the protected controller is the priority (with it on, `envelope_ok` falls from 3/3 to 0/3 in the sag sweep). | A weaker d gain lets i_d drift under saturation. |
| D3 note | Where the default gain is unstable (delay >= 2, or tau = 2e-4 as in the cycle runs) the loop refuses to run unless you pass `d_gain_by_L=True` or a smaller `k_i`. The cycle runner passes it for non-identity runs because the default cannot run there at all. Do not read cycle results as envelope-compliance evidence. | Guard in `firewalled_loop.run_fw` (`check_d_loop`). |
| D1 | BIBS statements are made on the unmodeled-voltage and road-load bounds in `docs/TWO_STATE_MODEL.md`, for the protected controllers (aw/awg) only. No bound is claimed for naive (i_d up to 731 A). | The bound is a property of the controller. |
| D2 | EKF is not the default estimator. | Keeps the controller's information realistic and simple. |

Still open (not decisions): live Postgres/Timescale load, per-seed disturbance envelopes, repo owner for any push.
