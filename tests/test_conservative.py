"""Tests for probcal.metrics._conservative (Pluto-Tasche most-prudent PDs)."""

import numpy as np
import pytest

from probcal.metrics import PlutoTascheResult, pluto_tasche, pluto_tasche_from_arrays
from probcal.metrics.grade import pluto_tasche as pluto_tasche_via_grade


def test_reexport_identity() -> None:
    # Same function object via metrics/__init__.py and metrics/grade.py.
    assert pluto_tasche is pluto_tasche_via_grade


def test_zero_default_closed_form_hand_case() -> None:
    # Pluto & Tasche (2005) worked example: grades A/B/C, n=(100, 400, 300),
    # all zero-default. Pooled (best -> worst, worse grades pooled in):
    # n* = (800, 700, 300), d* = (0, 0, 0). For d*=0 the upper bound has the
    # exact closed form p = 1 - (1 - confidence)^(1/n*) (I_p(1, n) = 1-(1-p)^n).
    n = np.array([100.0, 400.0, 300.0])
    d = np.array([0.0, 0.0, 0.0])
    confidence = 0.9
    res = pluto_tasche(n, d, confidence=confidence, grades=("A", "B", "C"))
    n_pooled_expected = np.array([800.0, 700.0, 300.0])
    np.testing.assert_allclose(res.n_pooled, n_pooled_expected)
    np.testing.assert_allclose(res.d_pooled, [0.0, 0.0, 0.0])
    closed_form = 1.0 - (1.0 - confidence) ** (1.0 / n_pooled_expected)
    np.testing.assert_allclose(res.pd_upper, closed_form, atol=1e-12)
    assert res.grades == ("A", "B", "C")
    assert res.monotonized is False


@pytest.mark.reference
def test_zero_default_vs_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    n = np.array([100.0, 400.0, 300.0])
    d = np.array([0.0, 0.0, 0.0])
    confidence = 0.9
    res = pluto_tasche(n, d, confidence=confidence)
    n_pooled = np.array([800.0, 700.0, 300.0])
    expected = stats.beta.ppf(confidence, 1.0, n_pooled)
    np.testing.assert_allclose(res.pd_upper, expected, atol=1e-8)


@pytest.mark.reference
def test_nonzero_default_vs_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    n = np.array([500.0, 300.0, 200.0])
    d = np.array([1.0, 2.0, 3.0])
    confidence = 0.95
    res = pluto_tasche(n, d, confidence=confidence)
    n_pooled = np.cumsum(n[::-1])[::-1]
    d_pooled = np.cumsum(d[::-1])[::-1]
    expected = stats.beta.ppf(confidence, d_pooled + 1.0, n_pooled - d_pooled)
    np.testing.assert_allclose(res.pd_upper, expected, atol=1e-8)


def test_pd_upper_always_non_decreasing() -> None:
    # pd_upper is always non-decreasing best -> worst on return: pooling is
    # nested, so a real violation can only come from noisy per-grade default
    # rates, and the PAVA touch-up (applied unconditionally) resolves it.
    for trial in range(20):
        r = np.random.default_rng(trial)
        n = r.integers(50, 1000, size=6).astype(np.float64)
        d = np.floor(n * r.uniform(0.0, 0.05, size=6))
        res = pluto_tasche(n, d, confidence=0.9)
        assert np.all(np.diff(res.pd_upper) >= -1e-12)


def test_no_touchup_for_realistic_monotone_grade_structure() -> None:
    # With a genuinely monotone true PD by grade and enough obligors that
    # sampling noise cannot flip the pooled-rate ordering, the PAVA
    # touch-up is a no-op (monotonized is False) -- the expected case.
    rng = np.random.default_rng(3)
    n = np.array([5000.0, 4000.0, 3000.0, 2000.0])
    pd_true = np.array([0.002, 0.01, 0.03, 0.08])
    d = rng.binomial(n.astype(np.int64), pd_true).astype(np.float64)
    res = pluto_tasche(n, d, confidence=0.9)
    assert np.all(np.diff(res.pd_upper) >= 0.0)
    assert res.monotonized is False


def test_zero_default_grades_strictly_positive() -> None:
    n = np.array([100.0, 200.0])
    d = np.array([0.0, 0.0])
    res = pluto_tasche(n, d, confidence=0.9)
    assert np.all(res.pd_upper > 0.0)


def test_all_default_grade_gives_one() -> None:
    n = np.array([10.0, 5.0])
    d = np.array([0.0, 5.0])
    res = pluto_tasche(n, d, confidence=0.9)
    # Worst grade's pooled set is itself: d* == n* == 5 -> upper bound 1.0.
    assert res.pd_upper[-1] == 1.0


def test_zero_pooled_obligors_raises() -> None:
    n = np.array([100.0, 0.0])
    d = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="obligors"):
        pluto_tasche(n, d, confidence=0.9)


def test_default_exceeds_obligors_raises() -> None:
    with pytest.raises(ValueError):
        pluto_tasche(np.array([10.0]), np.array([11.0]))


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        pluto_tasche(np.array([10.0, 20.0]), np.array([1.0]))


def test_default_grade_labels() -> None:
    res = pluto_tasche(np.array([10.0, 20.0]), np.array([0.0, 0.0]))
    assert res.grades == ("1", "2")


def test_array_convenience_equals_count_form() -> None:
    rng = np.random.default_rng(7)
    grades = np.array(["A"] * 50 + ["B"] * 80 + ["C"] * 40)
    y = (rng.random(170) < 0.03).astype(float)
    count_res = pluto_tasche(
        np.array([50.0, 80.0, 40.0]),
        np.array(
            [
                float(y[:50].sum()),
                float(y[50:130].sum()),
                float(y[130:].sum()),
            ]
        ),
        confidence=0.9,
        grades=("A", "B", "C"),
    )
    array_res = pluto_tasche_from_arrays(grades, y, order=("A", "B", "C"), confidence=0.9)
    np.testing.assert_allclose(array_res.pd_upper, count_res.pd_upper)
    np.testing.assert_allclose(array_res.n, count_res.n)
    np.testing.assert_allclose(array_res.d, count_res.d)
    assert array_res.grades == count_res.grades


def test_array_convenience_weighted() -> None:
    grades = np.array(["A", "A", "B", "B"])
    y = np.array([0.0, 0.0, 0.0, 1.0])
    w = np.array([2.0, 3.0, 1.5, 2.5])
    res = pluto_tasche_from_arrays(grades, y, order=("A", "B"), confidence=0.9, sample_weight=w)
    np.testing.assert_allclose(res.n, [5.0, 4.0])
    np.testing.assert_allclose(res.d, [0.0, 2.5])


def test_order_mismatch_raises() -> None:
    grades = np.array(["A", "B", "C"])
    y = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="order"):
        pluto_tasche_from_arrays(grades, y, order=("A", "B"), confidence=0.9)


def test_allows_all_zero_y() -> None:
    # Pluto-Tasche targets low/zero-default portfolios; unlike
    # probcal._validation.validate_binary_y this must not require both
    # classes present.
    grades = np.array(["A"] * 10 + ["B"] * 10)
    y = np.zeros(20)
    res = pluto_tasche_from_arrays(grades, y, order=("A", "B"), confidence=0.9)
    assert np.all(res.pd_upper > 0.0)


def test_interpret_contents() -> None:
    n = np.array([100.0, 400.0, 300.0])
    d = np.array([0.0, 0.0, 0.0])
    res = pluto_tasche(n, d, confidence=0.9, grades=("A", "B", "C"))
    interp = res.interpret()
    assert interp.method == "PlutoTasche"
    assert interp.param_names == ("pd_upper.A", "pd_upper.B", "pd_upper.C")
    np.testing.assert_allclose(interp.param_values, res.pd_upper)
    assert len(interp.messages) == 3
    msg_a = interp.messages[0]
    assert "grade A" in msg_a
    assert "0" in msg_a and "100" in msg_a
    assert "n*=800" in msg_a
    assert "d*=0" in msg_a
    assert "90%" in msg_a
    assert "0.29%" in msg_a


def test_result_is_dataclass_instance() -> None:
    res = pluto_tasche(np.array([10.0]), np.array([0.0]))
    assert isinstance(res, PlutoTascheResult)
    d = res.as_dict()
    assert set(d.keys()) == {
        "grades",
        "n",
        "d",
        "n_pooled",
        "d_pooled",
        "pd_upper",
        "confidence",
        "monotonized",
    }
