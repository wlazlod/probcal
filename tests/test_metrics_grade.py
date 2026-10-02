"""Tests for probcal.metrics.grade."""

import numpy as np
import pytest

from probcal._math import betainc, bisect
from probcal.metrics.grade import binomial_grade_test, jeffreys_grade_test

RNG = np.random.default_rng(71)


def test_grade_cis_contain_observed_rate() -> None:
    y = (RNG.random(300) < 0.05).astype(float)
    p = np.full(300, 0.05)
    grades = np.array(["g1"] * 100 + ["g2"] * 100 + ["g3"] * 100)
    for res in (binomial_grade_test(y, p, grades), jeffreys_grade_test(y, p, grades)):
        rate = res.k / res.n
        assert np.all(res.ci_low <= rate + 1e-12)
        assert np.all(rate <= res.ci_high + 1e-12)


def test_jeffreys_ci_hand_anchor() -> None:
    # k=1, n=100: central 90% interval of Beta(1.5, 99.5), independently by bisection.
    y = np.concatenate([np.ones(1), np.zeros(99)])
    p = np.full(100, 0.02)
    res = jeffreys_grade_test(y, p, np.array(["A"] * 100))
    lo = bisect(lambda x: float(betainc(1.5, 99.5, x)) - 0.05, 0.0, 1.0, tol=1e-12)
    hi = bisect(lambda x: float(betainc(1.5, 99.5, x)) - 0.95, 0.0, 1.0, tol=1e-12)
    np.testing.assert_allclose(res.ci_low, [lo], atol=1e-9)
    np.testing.assert_allclose(res.ci_high, [hi], atol=1e-9)


def test_clopper_pearson_edge_cases() -> None:
    # Two grades in one call so y has both classes: "hi" is all events (k=n),
    # "lo" is event-free (k=0).
    y = np.concatenate([np.ones(30), np.zeros(70)])
    p = np.full(100, 0.05)
    grades = np.array(["hi"] * 30 + ["lo"] * 70)
    res = binomial_grade_test(y, p, grades)
    i_hi, i_lo = res.grades.index("hi"), res.grades.index("lo")
    assert res.ci_high[i_hi] == 1.0  # k = n
    assert res.ci_low[i_lo] == 0.0  # k = 0
    assert res.ci_low[i_hi] > 0.0 and res.ci_high[i_lo] < 1.0


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


# ------------------------------------------------------------------ 0.4.0 fixes


def test_grades_length_validated() -> None:
    # Regression (MET-11): a short label array silently broadcast or crashed.
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.full(4, 0.3)
    for fn in (binomial_grade_test, jeffreys_grade_test):
        with pytest.raises(ValueError, match="one label per observation"):
            fn(y, p, np.array(["A", "B"]))


def test_weights_length_validated() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.full(4, 0.3)
    grades = np.array(["A"] * 4)
    with pytest.raises(ValueError, match="sample_weight"):
        binomial_grade_test(y, p, grades, sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="sample_weight"):
        jeffreys_grade_test(y, p, grades, sample_weight=np.array([]))


def test_order_keyword_reorders_rows() -> None:
    y = (RNG.random(300) < 0.05).astype(float)
    y[:2] = 1.0
    p = np.full(300, 0.05)
    grades = np.array(["g1"] * 100 + ["g2"] * 100 + ["g10"] * 100)
    base = binomial_grade_test(y, p, grades)
    assert base.grades == ("g1", "g10", "g2")  # lexicographic
    res = binomial_grade_test(y, p, grades, order=("g1", "g2", "g10"))
    assert res.grades == ("g1", "g2", "g10")
    perm = [base.grades.index(g) for g in res.grades]
    np.testing.assert_array_equal(res.n, base.n[perm])
    np.testing.assert_array_equal(res.p_exact, base.p_exact[perm])
    jef = jeffreys_grade_test(y, p, grades, order=("g10", "g2", "g1"))
    assert jef.grades == ("g10", "g2", "g1")
    with pytest.raises(ValueError, match="order"):
        jeffreys_grade_test(y, p, grades, order=("g1", "g2"))


def test_binomial_p_value_alias() -> None:
    y = np.array([1.0, 0.0])
    res = binomial_grade_test(y, np.array([0.5, 0.5]), np.array(["A", "A"]))
    assert res.p_value is res.p_exact


def test_binomial_normal_p_tail_accurate() -> None:
    n, k = 1000, 120
    y = np.concatenate([np.ones(k), np.zeros(n - k)])
    res = binomial_grade_test(y, np.full(n, 0.02), np.array(["A"] * n))
    assert 0.0 < res.p_normal[0] < 1e-20
