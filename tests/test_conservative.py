"""Tests for probcal.metrics._conservative (Pluto-Tasche most-prudent PDs)."""

import warnings

import numpy as np
import pytest

from probcal import IsotonicCalibrator
from probcal._math import beta_ppf, pava
from probcal.metrics import (
    jeffreys_upper_bands,
    pluto_tasche,
    pluto_tasche_from_arrays,
)
from probcal.thresholds import calibrated_bands_to_raw


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


def _raw_pd_upper(n: np.ndarray, d: np.ndarray, confidence: float) -> np.ndarray:
    # Reference re-implementation of pluto_tasche's pre-hull bound, so the
    # monotonization test can compare the hull against the un-touched-up
    # values independently of the function under test.
    n_pooled = np.cumsum(n[::-1])[::-1]
    d_pooled = np.cumsum(d[::-1])[::-1]
    out = np.empty(len(n), dtype=np.float64)
    for i in range(len(n)):
        ns, ds = n_pooled[i], d_pooled[i]
        out[i] = 1.0 if ds == ns else beta_ppf(confidence, ds + 1.0, ns - ds)
    return out


def test_pd_upper_always_non_decreasing() -> None:
    # pd_upper is always non-decreasing best -> worst on return: pooling is
    # nested, so a real violation can only come from noisy per-grade default
    # rates, and the running-maximum touch-up (applied unconditionally)
    # resolves it -- as the cumulative maximum of the raw bounds, exactly,
    # and never below the raw bound at any grade (a most-prudent estimator
    # may only raise a bound, never lower one).
    for trial in range(20):
        r = np.random.default_rng(trial)
        n = r.integers(50, 1000, size=6).astype(np.float64)
        d = np.floor(n * r.uniform(0.0, 0.05, size=6))
        res = pluto_tasche(n, d, confidence=0.9)
        raw = _raw_pd_upper(n, d, 0.9)
        assert np.all(np.diff(res.pd_upper) >= -1e-12)
        assert np.all(res.pd_upper >= raw - 1e-12)
        np.testing.assert_allclose(res.pd_upper, np.maximum.accumulate(raw))


def test_all_default_grade_gives_one() -> None:
    n = np.array([10.0, 5.0])
    d = np.array([0.0, 5.0])
    res = pluto_tasche(n, d, confidence=0.9)
    # Worst grade's pooled set is itself: d* == n* == 5 -> upper bound 1.0.
    assert res.pd_upper[-1] == 1.0


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


def test_array_convenience_integer_weight_matches_row_duplication() -> None:
    # Integer sample_weight on a small dataset must equal the unweighted
    # call on the row-duplicated expansion: same n, d, pd_upper to 1e-12.
    grades = np.array(["A", "B", "B"])
    y = np.array([1.0, 0.0, 1.0])
    w = np.array([3, 1, 2])

    weighted = pluto_tasche_from_arrays(
        grades, y, order=("A", "B"), confidence=0.9, sample_weight=w
    )

    grades_dup = np.repeat(grades, w)
    y_dup = np.repeat(y, w)
    unweighted = pluto_tasche_from_arrays(grades_dup, y_dup, order=("A", "B"), confidence=0.9)

    np.testing.assert_allclose(weighted.n, unweighted.n, atol=1e-12)
    np.testing.assert_allclose(weighted.d, unweighted.d, atol=1e-12)
    np.testing.assert_allclose(weighted.pd_upper, unweighted.pd_upper, atol=1e-12)


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


# ---------------------------------------------------------------- jeffreys_upper_bands


def _band_dataset():
    # Large per-grade counts and well-separated true PDs, so sampling noise
    # cannot flip the pooled-rate ordering (matches the scale used for the
    # analogous pluto_tasche "realistic monotone" fixture).
    rng = np.random.default_rng(3)
    n_per_grade = np.array([5000, 4000, 3000])
    pd_true = np.array([0.002, 0.01, 0.05])
    grades = np.array(["A"] * n_per_grade[0] + ["B"] * n_per_grade[1] + ["C"] * n_per_grade[2])
    p = np.repeat(pd_true, n_per_grade)
    y = np.concatenate(
        [rng.binomial(1, pt, n).astype(float) for pt, n in zip(pd_true, n_per_grade, strict=True)]
    )
    return grades, y, p


def test_all_grades_zero_default_still_positive_and_monotone() -> None:
    # The module's own motivating case (see the module docstring): every
    # grade has zero observed defaults. Unlike the standard
    # probcal._validation.validate_binary_y path, jeffreys_upper_bands must
    # still accept this (matching pluto_tasche_from_arrays), and every
    # grade's hi must come out strictly positive from the Jeffreys prior
    # alone.
    grades = np.array(["A"] * 300 + ["B"] * 200 + ["C"] * 100)
    y = np.zeros(600)
    p = np.array([0.005] * 300 + [0.02] * 200 + [0.06] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        bands = jeffreys_upper_bands(y, p, grades, confidence=0.9)
    order = ["A", "B", "C"]
    prev_hi = 0.0
    for g in order:
        lo, hi = bands[g]
        assert hi > 0.0
        assert lo == pytest.approx(prev_hi)
        assert hi >= lo
        prev_hi = hi


def test_default_order_is_by_mean_p_ascending() -> None:
    # Grade labels deliberately out of alphabetical order relative to risk.
    grades = np.array(["Z"] * 100 + ["A"] * 100)
    y = np.array([0.0] * 100 + [1.0] * 5 + [0.0] * 95)
    p = np.array([0.01] * 100 + [0.05] * 100)  # Z is lower-risk than A here
    bands = jeffreys_upper_bands(y, p, grades, confidence=0.9)
    assert bands["Z"][0] == 0.0
    assert bands["Z"][1] == pytest.approx(bands["A"][0])


def test_explicit_order_is_honored() -> None:
    grades, y, p = _band_dataset()
    # Reversed order: every later bound is below C's, so the running maximum
    # would leave B and A zero-width bands; they merge into C's band.
    with pytest.warns(UserWarning, match="running maximum"):
        bands = jeffreys_upper_bands(y, p, grades, confidence=0.9, order=("C", "B", "A"))
    ordered = jeffreys_upper_bands(y, p, grades, confidence=0.9)
    assert list(bands) == ["C+B+A"]
    assert bands["C+B+A"] == (0.0, ordered["C"][1])


def test_hand_computed_two_grade_example() -> None:
    # Two grades, own-grade-only Jeffreys posterior, no cross-grade pooling.
    grades = np.array(["A"] * 50 + ["B"] * 50)
    y = np.array([0.0] * 50 + [1.0] * 2 + [0.0] * 48)
    p = np.array([0.01] * 50 + [0.05] * 50)
    bands = jeffreys_upper_bands(y, p, grades, confidence=0.9)
    expected_a = beta_ppf(0.9, 0.5, 50.5)
    expected_b = beta_ppf(0.9, 2.5, 48.5)
    np.testing.assert_allclose(bands["A"][1], expected_a, atol=1e-12)
    np.testing.assert_allclose(bands["B"][1], expected_b, atol=1e-12)


def test_monotone_hull_never_lowers_a_bound() -> None:
    # Regression (MET-2): 0.3.x pooled with size-weighted PAVA, which lowered
    # grade A's upper bound below its own Jeffreys bound and, with a large
    # grade B, produced a (near) zero-width band for B.
    grades = np.array(["A"] * 50 + ["B"] * 5000)
    y = np.array([1.0] * 5 + [0.0] * 45 + [1.0] * 50 + [0.0] * 4950)
    p = np.array([0.01] * 50 + [0.05] * 5000)
    with pytest.warns(UserWarning, match="running maximum"):
        bands = jeffreys_upper_bands(y, p, grades, confidence=0.9)
    own_a = beta_ppf(0.9, 5.5, 45.5)
    own_b = beta_ppf(0.9, 50.5, 4950.5)
    lo, hi = bands["A+B"]  # B cannot be ordered above A: merged, never zero-width
    assert hi == pytest.approx(own_a)  # never lowered
    assert hi >= own_b
    assert hi > lo
    # The PAVA fit the old code used would have lowered A's bound.
    old = pava(np.array([own_a, own_b]), np.array([50.0, 5000.0])).fitted
    assert old[0] < own_a


def test_level_keyword_deprecated() -> None:
    grades, y, p = _band_dataset()
    with pytest.warns(DeprecationWarning, match="confidence"):
        old = jeffreys_upper_bands(y, p, grades, level=0.8)
    assert old == jeffreys_upper_bands(y, p, grades, confidence=0.8)


def test_bad_arguments_raise() -> None:
    with pytest.raises(ValueError, match="obligors"):
        pluto_tasche(np.array([100.0, 0.0]), np.array([0.0, 0.0]), confidence=0.9)
    with pytest.raises(ValueError):  # more defaults than obligors
        pluto_tasche(np.array([10.0]), np.array([11.0]))
    with pytest.raises(ValueError):  # length mismatch
        pluto_tasche(np.array([10.0, 20.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="order"):
        pluto_tasche_from_arrays(np.array(["A", "B", "C"]), np.zeros(3), order=("A", "B"))
    grades, y, p = _band_dataset()
    with pytest.raises(ValueError, match="order"):
        jeffreys_upper_bands(y, p, grades, order=("A", "B"))
    with pytest.raises(ValueError, match="confidence"):
        jeffreys_upper_bands(y, p, grades, confidence=1.5)
    with pytest.raises(ValueError):  # grades length
        jeffreys_upper_bands(np.array([0.0, 1.0]), np.array([0.1, 0.2]), np.array(["A"]))


def test_jeffreys_bands_strictly_increasing() -> None:
    grades = np.array(["A"] * 100 + ["B"] * 1000)
    y = np.r_[np.zeros(97), np.ones(3), np.zeros(995), np.ones(5)]
    p = np.r_[np.full(100, 0.01), np.full(1000, 0.02)]
    with pytest.warns(UserWarning, match="'B'.*merged into 'A\\+B'"):
        bands = jeffreys_upper_bands(y, p, grades)
    assert list(bands) == ["A+B"]
    lo, hi = bands["A+B"]
    assert lo == 0.0 < hi
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 3000)
    cal = IsotonicCalibrator().fit(s, (rng.random(3000) < s * 0.2).astype(float))
    raw = calibrated_bands_to_raw(cal, bands)
    assert list(raw) == ["A+B"]


def test_jeffreys_bands_merge_only_violators() -> None:
    grades = np.array(["A"] * 200 + ["B"] * 50 + ["C"] * 2000 + ["D"] * 300)
    y = np.r_[np.zeros(200), np.ones(1), np.zeros(49), np.ones(4), np.zeros(1996), np.ones(30)]
    y = np.r_[y, np.zeros(300 - 30)]
    p = np.r_[np.full(200, 0.001), np.full(50, 0.01), np.full(2000, 0.02), np.full(300, 0.1)]
    with pytest.warns(UserWarning, match="merged"):
        bands = jeffreys_upper_bands(y, p, grades)
    assert list(bands) == ["A", "B+C", "D"]
    his = [hi for _, hi in bands.values()]
    los = [lo for lo, _ in bands.values()]
    assert los[0] == 0.0 and los[1:] == his[:-1]
    assert all(lo < hi for lo, hi in bands.values())
