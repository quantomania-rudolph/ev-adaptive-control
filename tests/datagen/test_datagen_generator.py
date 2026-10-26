"""Unit tests for the datagen segment.

HOW TO ADD A TEST: copy any function below, rename it ``test_<what_it_checks>``,
and assert something you know must be true. pytest finds it automatically.
"""

import numpy as np
import pytest

from evcontrol.datagen import generate_disturbance, generate_reference, time_vector


def test_time_vector_length_and_spacing():
    t = time_vector(5, 0.1)
    assert len(t) == 5
    assert np.allclose(np.diff(t), 0.1)


@pytest.mark.parametrize("kind", ["step", "sine", "ramp"])
def test_reference_has_requested_length(kind):
    assert generate_reference(100, 0.01, kind=kind).shape == (100,)


def test_step_reference_is_constant():
    r = generate_reference(10, 0.01, kind="step", amplitude=2.5)
    assert np.all(r == 2.5)


def test_unknown_reference_kind_raises():
    with pytest.raises(ValueError):
        generate_reference(10, 0.01, kind="banana")


def test_disturbance_respects_bound():
    # The BIBS/UUB theorem assumes a bounded disturbance, so this MUST hold.
    d = generate_disturbance(10_000, bound=0.3, seed=1)
    assert np.max(np.abs(d)) <= 0.3


def test_disturbance_is_reproducible_with_same_seed():
    a = generate_disturbance(50, 1.0, seed=7)
    b = generate_disturbance(50, 1.0, seed=7)
    assert np.array_equal(a, b)


def test_disturbance_differs_across_seeds():
    assert not np.array_equal(
        generate_disturbance(50, 1.0, seed=1), generate_disturbance(50, 1.0, seed=2)
    )


@pytest.mark.skip(
    reason="TODO(datagen owner): add tests once dataset loading / EV drive cycles exist"
)
def test_drive_cycle_loader_placeholder():
    raise NotImplementedError
