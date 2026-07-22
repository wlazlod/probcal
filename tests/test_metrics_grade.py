"""Tests for probcal.metrics.grade."""

import numpy as np
import pytest

from probcal.metrics.grade import binomial_grade_test, jeffreys_grade_test

RNG = np.random.default_rng(71)


def test_binomial_exact_hand_case() -> None:
    # One grade: n=2, k=1, PD=0.5 -> P(X >= 1) = 0.75.
    y = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    grades = np.array(["A", "A"])
    res = binomial_grade_test(y, p, grades)
    assert res.grades == ("A",)
    np.testing.assert_allclose(res.p_exact, [0.75], atol=1e-12)


def test_jeffreys_symmetric_hand_case() -> None:
    # k=1, n=2, PD=0.5: posterior Beta(1.5, 1.5) is symmetric -> P(theta <= 0.5) = 0.5.
    y = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    grades = np.array(["A", "A"])
    res = jeffreys_grade_test(y, p, grades)
    np.testing.assert_allclose(res.p_value, [0.5], atol=1e-12)


def test_traffic_lights() -> None:
    # Grade with far too many defaults for its PD -> red.
    n = 200
    y = np.concatenate([np.ones(30), np.zeros(n - 30)])
    p = np.full(n, 0.02)
    grades = np.array(["B"] * n)
    res = jeffreys_grade_test(y, p, grades)
    assert res.light == ("red",)
    # Consistent grade -> green.
    y2 = np.concatenate([np.ones(4), np.zeros(n - 4)])
    res2 = jeffreys_grade_test(y2, p, grades)
    assert res2.light == ("green",)


def test_multiple_grades_ordered_output() -> None:
    y = (RNG.random(300) < 0.05).astype(float)
    p = np.full(300, 0.05)
    grades = np.array(["g1"] * 100 + ["g2"] * 100 + ["g3"] * 100)
    res = binomial_grade_test(y, p, grades)
    assert res.grades == ("g1", "g2", "g3")
    assert len(res.p_exact) == 3


def test_nonuniform_weights_warn() -> None:
    y = np.array([1.0, 0.0, 0.0, 1.0])
    p = np.full(4, 0.3)
    grades = np.array(["A"] * 4)
    with pytest.warns(UserWarning, match="weights"):
        binomial_grade_test(y, p, grades, sample_weight=np.array([1.0, 2.0, 1.0, 1.0]))


@pytest.mark.reference
def test_binomial_exact_vs_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    n, k, pd = 100, 7, 0.03
    y = np.concatenate([np.ones(k), np.zeros(n - k)])
    p = np.full(n, pd)
    grades = np.array(["A"] * n)
    res = binomial_grade_test(y, p, grades)
    expected = float(stats.binom.sf(k - 1, n, pd))
    np.testing.assert_allclose(res.p_exact, [expected], atol=1e-10)
