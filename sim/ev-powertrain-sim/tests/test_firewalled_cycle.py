"""Cycle path through the firewall. Slow (about 1 min per 30 s run); marked so a quick suite can skip them with -m 'not slow'."""
import numpy as np, pytest
import run_cycle as RC
import firewalled_cycle as FC
import run_cycle_matrix as M

pytestmark = pytest.mark.slow
SRC = M.CYC["udds"][0]
PARTS = ["ctrl_clip", "adv", "delay", "rot", "plant_clip", "inv"]


def _run(**kw):
    kw.setdefault("t_max", 30)
    return FC.run_fw_cycle(RC._load(SRC), "t", outdir="/tmp/eq", **kw)[0]


def test_cycle_identity_matches_legacy_bit_for_bit():
    leg, _ = RC.run_cycle(RC._load(SRC), "leg", mode="aw", governor=True, vdc_aware=True, battery=False, t_max=30, outdir="/tmp/eq")
    fw = _run(mode="aw", governor=True, vdc_aware=True, battery=False, derate=False, identity=True, fit_path=None)
    assert len(leg) == len(fw)
    for c in ("omega_m", "i_q", "i_d", "psi_hat", "u_q_app"):
        assert np.array_equal(leg[c].values, fw[c].values), c


@pytest.fixture(scope="module")
def real():
    a = _run(mode="aw", governor=True, vdc_aware=True, bms_aware=True, seed=3)
    b = _run(mode="aw", governor=True, vdc_aware=True, bms_aware=True, seed=3)
    c = _run(mode="aw", governor=True, vdc_aware=True, bms_aware=True, seed=4)
    return a, b, c


def test_cycle_realism_ledger_closes_and_bound_holds(real):
    df = real[0]
    for ax in "dq":
        assert np.abs(sum(df[f"d_u_{p}_{ax}"] for p in PARTS) - df[f"d_u_{ax}"]).max() < 1e-9
    f = np.hypot(df.u_d_app - df.d_u_inv_d, df.u_q_app - df.d_u_inv_q)
    assert (f / df.u_limit).max() <= 1 + 1e-9


def test_cycle_realism_deterministic_and_seed_dependent(real):
    a, b, c = real
    assert np.array_equal(a.meas_i_a.values, b.meas_i_a.values)
    assert not np.array_equal(a.meas_i_a.values, c.meas_i_a.values)


def test_cycle_realism_controller_does_not_see_truth(real):
    df = real[0]
    assert np.abs(df.gap_i_q).max() > 0 and np.abs(df.gap_omega).max() > 0         # measurements differ from truth
