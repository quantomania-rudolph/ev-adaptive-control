"""Unit tests for the evaluation segment."""

import numpy as np
import pytest

from evcontrol.evaluation import (
    constraint_violations,
    max_abs_error,
    saturation_fraction,
    tracking_rmse,
    ultimate_bound,
)


def test_perfect_tracking_has_zero_error():
    r = np.ones(10)
    assert tracking_rmse(r, r) == 0.0
    assert max_abs_error(r, r) == 0.0


def test_rmse_known_value():
    # constant error of 2 everywhere -> RMSE is exactly 2
    assert tracking_rmse(np.full(5, 3.0), np.full(5, 1.0)) == pytest.approx(2.0)


def test_max_abs_error_picks_worst_sample():
    assert max_abs_error([0, 0, 0], [0, -4, 1]) == 4.0


def test_saturation_fraction():
    assert saturation_fraction([0.0, 0.0, -1.0, 2.0]) == 0.5


def test_no_saturation_gives_zero_fraction():
    assert saturation_fraction(np.zeros(20)) == 0.0


def test_constraint_violations_counts_outside_samples():
    x = np.array([-2.0, 0.0, 0.5, 1.5])
    assert constraint_violations(x, -1.0, 1.0) == 2


def test_constraint_boundary_is_allowed():
    assert constraint_violations([-1.0, 1.0], -1.0, 1.0) == 0


def test_ultimate_bound_ignores_transient():
    x = np.concatenate([np.full(50, 100.0), np.full(50, 0.5)])  # big transient, small tail
    assert ultimate_bound(x, settle_fraction=0.5) == 0.5


def test_ultimate_bound_rejects_bad_fraction():
    with pytest.raises(ValueError):
        ultimate_bound([1.0, 2.0], settle_fraction=1.0)


@pytest.mark.skip(
    reason="TODO(evaluation owner): parameter-drift, V(t), distance-to-constraint metrics"
)
def test_parameter_drift_metric_placeholder():
    raise NotImplementedError
