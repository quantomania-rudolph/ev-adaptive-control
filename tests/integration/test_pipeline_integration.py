"""Integration tests: all three segments working together.

These are the tests that catch "my part works, but it broke yours" problems.
If one fails after you change an interface, tell the owner of the other segment.
"""

import numpy as np
import pytest

from evcontrol.evaluation import saturation_fraction, tracking_rmse, ultimate_bound
from evcontrol.pipeline import simulate

pytestmark = pytest.mark.integration


def test_all_output_arrays_have_same_length():
    out = simulate(n_steps=300)
    assert {len(v) for v in out.values()} == {300}


def test_actuator_limit_is_never_exceeded():
    out = simulate(u_limit=1.0)
    assert np.max(np.abs(out["u"])) <= 1.0


def test_delta_u_identity_holds_through_pipeline():
    out = simulate()
    assert np.allclose(out["u"], out["u_c"] + out["delta_u"])


def test_big_step_forces_saturation():
    # Reference 3.0 with limit 1.0: the controller must ask for more than the actuator gives.
    out = simulate(ref_amplitude=3.0, u_limit=1.0)
    assert saturation_fraction(out["delta_u"]) > 0.0


def test_state_stays_bounded_under_saturation_and_disturbance():
    """BIBS-flavoured check from the notes.

    For x_dot = -x + u + d with |u| <= u_max and |d| <= d_max, the state cannot settle
    above u_max + d_max (plus a little numerical slack for Euler integration).
    """
    u_limit, d_bound = 1.0, 0.1
    out = simulate(u_limit=u_limit, d_bound=d_bound, ref_amplitude=10.0)
    assert ultimate_bound(out["x"]) <= u_limit + d_bound + 1e-6


def test_same_seed_gives_identical_runs():
    a = simulate(seed=3)
    b = simulate(seed=3)
    assert np.array_equal(a["x"], b["x"])


def test_unsaturated_case_tracks_reasonably():
    # Small reference: actuator never limits, so tracking should be decent.
    out = simulate(ref_amplitude=0.1, u_limit=5.0, d_bound=0.0)
    assert saturation_fraction(out["delta_u"]) == 0.0
    assert tracking_rmse(out["r"], out["x"]) < 0.1
