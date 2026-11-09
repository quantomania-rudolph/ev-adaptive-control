"""Regression gate for the plant adapter and the ledger. Run: pytest -q tests/

Each test pins down a fact we measured about GEM 3.0.3 (see README). If a GEM upgrade changes any of
them, these fail BEFORE a stability claim silently changes.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from plant import GEMPlant


def test_unsaturated_voltage_is_delivered_exactly():
    # GEM reports the applied dq voltage; with no saturation it must equal our command (measured: ~1e-13 V).
    pl = GEMPlant()
    led = pl.apply_voltage(np.array([10.0, 40.0]))
    assert abs(led["d_u_d"]) < 1e-9 and abs(led["d_u_q"]) < 1e-9
    assert abs(led["d_u_plant_d"]) < 1e-6 and abs(led["d_u_plant_q"]) < 1e-6
    assert led["saturated"] == 0.0


def test_saturation_is_an_isotropic_disc_and_gem_never_clips_hidden():
    # Command far outside the disc at several directions: delivered magnitude = ceiling in every direction,
    # direction preserved, and GEM's own per-leg clip never fires (M3 stays ~0).
    for ang in np.linspace(0, 2 * np.pi, 13):
        pl = GEMPlant()
        cmd = 400.0 * np.array([np.cos(ang), np.sin(ang)])
        led = pl.apply_voltage(cmd)
        app = np.array([led["u_d_app"], led["u_q_app"]])
        assert abs(np.hypot(*app) - pl.u_max_hw) < 1e-6
        cross2d = cmd[0] * app[1] - cmd[1] * app[0]                      # np.cross on 2-vectors is gone in NumPy 2
        assert abs(cross2d) < 1e-6 * 400 * pl.u_max_hw                   # same direction
        assert abs(led["d_u_plant_d"]) < 1e-6 and abs(led["d_u_plant_q"]) < 1e-6
        assert led["saturated"] == 1.0


def test_imposed_ceiling_is_respected_and_logged():
    pl = GEMPlant()
    led = pl.apply_voltage(np.array([0.0, 100.0]), u_max_imposed=25.0)
    assert abs(led["u_q_app"] - 25.0) < 1e-9 and abs(led["d_u_q"] + 75.0) < 1e-9


def test_current_limit_is_data_not_a_crash():
    # GEM's default terminates the episode above 400 A; the adapter disables that so violations are logged.
    pl = GEMPlant()
    for _ in range(300):
        pl.apply_voltage(np.array([100.0, 0.0]))          # drives i_d far past 400 A (Rs is only 18 mOhm)
    st = pl.state()
    assert abs(st["i_d"]) > 400.0                          # we got there and the sim is still running


def test_rerun_is_bit_identical():
    def go():
        pl = GEMPlant(disturbance_fn=lambda t: 5.0)
        out = []
        for _ in range(200):
            pl.apply_voltage(np.array([5.0, 20.0]))
            out.append(pl.state()["omega_m"])
        return np.array(out)
    assert np.array_equal(go(), go())


def test_battery_regen_overshoot_never_causes_hidden_clipping():
    # Regression for a bug found by the viewer's own check: with the bus above 300 V (regen) the ceiling must stay
    # inside GEM's representable hexagon, otherwise GEM clips duty silently and M3 != 0.
    from battery import Battery
    b = Battery(soc0=0.5)
    pl = GEMPlant(battery=b)
    pl._vdc = b.v_dc = 320.0
    led = pl.apply_voltage(np.array([0.0, 300.0]))
    assert led["u_limit"] <= pl.u_max_hw + 1e-9
    assert abs(led["d_u_plant_d"]) < 1e-6 and abs(led["d_u_plant_q"]) < 1e-6
