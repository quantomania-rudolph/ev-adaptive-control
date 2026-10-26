"""Unit tests for the actuator segment."""

import numpy as np
import pytest

from evcontrol.actuator import Actuator, saturate, saturation_mismatch


def test_command_inside_limits_passes_through():
    assert saturate(0.5, -1.0, 1.0) == 0.5


def test_command_above_limit_is_clipped():
    # Mirrors the notes: controller asks 450 N*m, hardware gives 300 N*m.
    assert saturate(450.0, -300.0, 300.0) == 300.0


def test_mismatch_is_zero_when_feasible():
    assert saturation_mismatch(0.2, -1.0, 1.0) == 0.0


def test_mismatch_sign_and_size_when_clipped():
    # delta_u = u - u_c = 300 - 450 = -150
    assert saturation_mismatch(450.0, -300.0, 300.0) == -150.0


def test_saturate_works_on_arrays():
    out = saturate(np.array([-5.0, 0.0, 5.0]), -1.0, 1.0)
    assert np.array_equal(out, [-1.0, 0.0, 1.0])


def test_output_never_exceeds_limits():
    rng = np.random.default_rng(0)
    u = saturate(rng.normal(0, 10, 1000), -2.0, 3.0)
    assert u.min() >= -2.0 and u.max() <= 3.0


def test_invalid_limits_raise():
    with pytest.raises(ValueError):
        Actuator(1.0, -1.0)


def test_actuator_step_returns_u_and_delta_u():
    act = Actuator(-1.0, 1.0)
    u, delta_u = act.step(4.0)
    assert u == 1.0
    assert delta_u == -3.0
    assert u == pytest.approx(4.0 + delta_u)  # identity: u = u_c + delta_u


@pytest.mark.skip(
    reason="TODO(actuator owner): speed/SOC/temperature-dependent limits, rate limits"
)
def test_state_dependent_limits_placeholder():
    raise NotImplementedError
