"""Tests for probcal.metrics._safe (mixture-LR grade e-test)."""

import inspect

import numpy as np
import pytest

from probcal.metrics import hl_e_test


def _calibrated_data(
    n_per_grade: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = np.concatenate([np.full(n_per_grade, 0.1), np.full(n_per_grade, 0.3)])
    y = (rng.random(2 * n_per_grade) < p).astype(np.float64)
    grades = np.array(["A"] * n_per_grade + ["B"] * n_per_grade)
    return y, p, grades


def test_product_identity() -> None:
    y, p, grades = _calibrated_data()
    res = hl_e_test(y, p, grades)
    np.testing.assert_allclose(res.e_value, np.prod(res.e_grade), rtol=1e-9)
    assert res.p_value == min(1.0, 1.0 / res.e_value)
    assert res.construction == "mixture-lr"


def test_interpret_sentences() -> None:
    y, p, grades = _calibrated_data()
    res = hl_e_test(y, p, grades)
    interp = res.interpret()
    assert interp.method == "HlETest"
    assert interp.param_names == ("e_value", "p_value")
    assert interp.param_values == (res.e_value, res.p_value)
    # One message per grade plus a closing sentence.
    assert len(interp.messages) == len(res.grades) + 1
    assert "grade A: e =" in interp.messages[0]
    assert "grade B: e =" in interp.messages[1]
    closing = interp.messages[-1]
    assert "1/alpha" in closing
    assert "p = min(1, 1/e)" in closing


def test_miscalibrated_grade_drives_e_value_large() -> None:
    # Grade A is well calibrated; grade B's assigned PD is wildly understated
    # relative to its realized event rate -- an obvious miscalibration that
    # the e-value should flag with a large e_value.
    rng = np.random.default_rng(3)
    n = 500
    p_a = np.full(n, 0.1)
    y_a = (rng.random(n) < 0.1).astype(np.float64)
    p_b = np.full(n, 0.01)
    y_b = (rng.random(n) < 0.5).astype(np.float64)
    y = np.concatenate([y_a, y_b])
    p = np.concatenate([p_a, p_b])
    grades = np.array(["A"] * n + ["B"] * n)
    res = hl_e_test(y, p, grades)
    assert res.e_value > 1000.0
    assert res.p_value < 1e-3


def test_weights_are_scale_invariant_exponents() -> None:
    # Each row is one Bernoulli draw: weights become exponents min(1, w/mean(w))
    # per grade, so rescaling all weights changes nothing and constant weights
    # reproduce the unweighted e-value.
    grades = np.array(["A", "A", "B", "B"])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    p = np.array([0.2, 0.2, 0.05, 0.05])
    w = np.array([3.0, 1.0, 2.0, 4.0])

    weighted = hl_e_test(y, p, grades, sample_weight=w)
    scaled = hl_e_test(y, p, grades, sample_weight=10.0 * w)
    np.testing.assert_allclose(weighted.e_value, scaled.e_value, rtol=1e-12)

    unweighted = hl_e_test(y, p, grades)
    constant = hl_e_test(y, p, grades, sample_weight=np.full(4, 7.0))
    assert constant.e_value == unweighted.e_value


def test_default_mixture_grid_matches_monitor_default() -> None:
    # CalibrationMonitor's default mixture_grid; hl_e_test documents sharing
    # the same default and symmetrization convention.
    from probcal.monitor import CalibrationMonitor

    default = inspect.signature(hl_e_test).parameters["mixture_grid"].default
    assert CalibrationMonitor().mixture_grid == default == (0.1, 0.25, 0.5, 1.0)


def test_bad_arguments_raise() -> None:
    with pytest.raises(ValueError):  # grades length
        hl_e_test(np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.2, 0.3]), np.array(["A", "B"]))
    y, p, grades = _calibrated_data()
    with pytest.raises(ValueError):
        hl_e_test(y, p, grades, mixture_grid=())


def test_hl_e_test_weighted_null_stays_valid() -> None:
    rng = np.random.default_rng(4)
    n, reps = 400, 300
    rejections = 0
    for _ in range(reps):
        p = rng.uniform(0.02, 0.3, n)
        y = (rng.random(n) < p).astype(float)
        w = rng.lognormal(0.0, 1.0, n)
        grades = np.where(p < 0.1, "A", "B")
        rejections += hl_e_test(y, p, grades, sample_weight=w).e_value >= 20.0
    assert rejections / reps <= 0.05
