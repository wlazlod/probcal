"""Tests for probcal.metrics._conservative (Pluto-Tasche most-prudent PDs)."""

import warnings

import numpy as np
import pytest

from probcal import BetaCalibrator, make_pd_portfolio
from probcal._math import beta_ppf, pava
from probcal.metrics import (
    PlutoTascheResult,
    jeffreys_upper_bands,
    pluto_tasche,
    pluto_tasche_from_arrays,
)
from probcal.metrics.grade import jeffreys_upper_bands as jeffreys_upper_bands_via_grade
from probcal.metrics.grade import pluto_tasche as pluto_tasche_via_grade
from probcal.thresholds import calibrated_bands_to_raw


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


def test_no_touchup_for_realistic_monotone_grade_structure() -> None:
    # With a genuinely monotone true PD by grade and enough obligors that
    # sampling noise cannot flip the pooled-rate ordering, the
    # running-maximum touch-up is a no-op (monotonized is False) -- the
    # expected case.
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


# ---------------------------------------------------------------- jeffreys_upper_bands


def test_jeffreys_upper_bands_reexport_identity() -> None:
    assert jeffreys_upper_bands is jeffreys_upper_bands_via_grade


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


def test_bands_are_contiguous_and_non_decreasing() -> None:
    grades, y, p = _band_dataset()
    bands = jeffreys_upper_bands(y, p, grades, level=0.9)
    order = sorted(bands, key=lambda g: bands[g][1])
    # Default order is best-to-worst by mean p, matching increasing hi here.
    assert order == ["A", "B", "C"]
    prev_hi = 0.0
    for g in order:
        lo, hi = bands[g]
        assert lo == pytest.approx(prev_hi)
        assert hi >= lo
        prev_hi = hi


def test_bands_feed_into_calibrated_bands_to_raw() -> None:
    d = make_pd_portfolio(n=1000, random_state=5)
    cal = BetaCalibrator().fit(d.scores, d.y)
    grades, y, p = _band_dataset()
    bands = jeffreys_upper_bands(y, p, grades, level=0.9)
    raw_bands = calibrated_bands_to_raw(cal, bands)
    assert set(raw_bands) == set(bands)
    for raw_lo, raw_hi in raw_bands.values():
        assert raw_lo <= raw_hi


def test_zero_default_grade_still_gets_positive_hi() -> None:
    grades = np.array(["A"] * 100 + ["B"] * 100)
    y = np.array([0.0] * 100 + [1.0] * 3 + [0.0] * 97)
    p = np.array([0.01] * 100 + [0.05] * 100)
    bands = jeffreys_upper_bands(y, p, grades, level=0.9)
    assert bands["A"][1] > 0.0


def test_default_order_is_by_mean_p_ascending() -> None:
    # Grade labels deliberately out of alphabetical order relative to risk.
    grades = np.array(["Z"] * 100 + ["A"] * 100)
    y = np.array([0.0] * 100 + [1.0] * 5 + [0.0] * 95)
    p = np.array([0.01] * 100 + [0.05] * 100)  # Z is lower-risk than A here
    bands = jeffreys_upper_bands(y, p, grades, level=0.9)
    assert bands["Z"][0] == 0.0
    assert bands["Z"][1] == pytest.approx(bands["A"][0])


def test_explicit_order_is_honored() -> None:
    grades, y, p = _band_dataset()
    # Reversed order forces PAVA to pool everything into one flat band.
    with pytest.warns(UserWarning, match="PAVA"):
        bands = jeffreys_upper_bands(y, p, grades, level=0.9, order=("C", "B", "A"))
    assert bands["C"][0] == 0.0
    assert bands["C"][1] == pytest.approx(bands["B"][0])
    assert bands["B"][1] == pytest.approx(bands["A"][0])


def test_jeffreys_upper_bands_order_mismatch_raises() -> None:
    grades, y, p = _band_dataset()
    with pytest.raises(ValueError, match="order"):
        jeffreys_upper_bands(y, p, grades, order=("A", "B"))


def test_invalid_level_raises() -> None:
    grades, y, p = _band_dataset()
    with pytest.raises(ValueError, match="level"):
        jeffreys_upper_bands(y, p, grades, level=1.5)


def test_mismatched_grades_length_raises() -> None:
    y = np.array([0.0, 1.0])
    p = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        jeffreys_upper_bands(y, p, np.array(["A"]))


def test_hand_computed_two_grade_example() -> None:
    # Two grades, own-grade-only Jeffreys posterior, no cross-grade pooling.
    grades = np.array(["A"] * 50 + ["B"] * 50)
    y = np.array([0.0] * 50 + [1.0] * 2 + [0.0] * 48)
    p = np.array([0.01] * 50 + [0.05] * 50)
    bands = jeffreys_upper_bands(y, p, grades, level=0.9)
    expected_a = beta_ppf(0.9, 0.5, 50.5)
    expected_b = beta_ppf(0.9, 2.5, 48.5)
    np.testing.assert_allclose(bands["A"][1], expected_a, atol=1e-12)
    np.testing.assert_allclose(bands["B"][1], expected_b, atol=1e-12)


def test_pava_monotonization_warns_only_when_it_changes_something() -> None:
    # Grade "A" (best, lower p) engineered to have more relative defaults than
    # grade "B" (worse), so its raw own-grade Jeffreys upper bound exceeds
    # grade B's -- PAVA must pool them, and a UserWarning must fire.
    grades = np.array(["A"] * 100 + ["B"] * 100)
    y = np.array([1.0] * 10 + [0.0] * 90 + [0.0] * 100)
    p = np.array([0.01] * 100 + [0.05] * 100)
    with pytest.warns(UserWarning, match="PAVA"):
        bands = jeffreys_upper_bands(y, p, grades, level=0.9)
    n = np.array([100.0, 100.0])
    hi_raw = np.array([beta_ppf(0.9, 10.5, 90.5), beta_ppf(0.9, 0.5, 100.5)])
    expected_hi = pava(hi_raw, n).fitted
    np.testing.assert_allclose([bands["A"][1], bands["B"][1]], expected_hi)
    assert bands["A"][1] <= bands["B"][1]


def test_no_warning_for_realistic_monotone_grade_structure() -> None:
    grades, y, p = _band_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        jeffreys_upper_bands(y, p, grades, level=0.9)
