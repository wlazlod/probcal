"""Tests for probcal.vennabers: IVAP and CVAP."""

import numpy as np
import pytest

from probcal._math import pava
from probcal.vennabers import CrossVennAbersCalibrator, VennAbersCalibrator

RNG = np.random.default_rng(29)
GRID = np.linspace(0.01, 0.99, 60)


def _sample(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.01, 0.99, n)
    y = (RNG.random(n) < s).astype(float)
    return s, y


def test_interval_orders_and_bounds() -> None:
    cal = VennAbersCalibrator().fit(*_sample())
    intervals = cal.predict_interval(GRID)
    p0, p1 = intervals[:, 0], intervals[:, 1]
    assert intervals.shape == (len(GRID), 2)
    assert np.all(p0 <= p1 + 1e-12)
    # Two-fit tail protection: never an infinitely confident 0 or 1.
    assert np.all(p1 > 0.0)
    assert np.all(p0 < 1.0)


def test_scalar_inside_interval_and_monotone() -> None:
    cal = VennAbersCalibrator().fit(*_sample())
    p = cal.predict_proba(GRID)
    intervals = cal.predict_interval(GRID)
    assert np.all(p >= intervals[:, 0] - 1e-12)
    assert np.all(p <= intervals[:, 1] + 1e-12)
    assert np.all(np.diff(p) >= -1e-10)


def test_ivap_agrees_with_direct_refit() -> None:
    s, y = _sample(150)
    cal = VennAbersCalibrator().fit(s, y)
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    for x in (0.15, 0.5, 0.85):
        idx = int(np.searchsorted(s_sorted, x, side="left"))
        expected = []
        for label in (0.0, 1.0):
            y_aug = np.insert(y_sorted, idx, label)
            w_aug = np.ones(len(y_aug))
            expected.append(pava(y_aug, w_aug).fitted[idx])
        got = cal.predict_interval(np.array([x]))[0]
        np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.slow
def test_interval_width_shrinks_with_n() -> None:
    small = VennAbersCalibrator().fit(*_sample(80))
    large = VennAbersCalibrator().fit(*_sample(3000))
    w_small = np.diff(small.predict_interval(GRID), axis=1).mean()
    w_large = np.diff(large.predict_interval(GRID), axis=1).mean()
    assert w_large < w_small


def test_ivap_interpret_reports_widths() -> None:
    cal = VennAbersCalibrator().fit(*_sample(200))
    interp = cal.interpret()
    assert "mean_width" in interp.param_names
    assert "max_width" in interp.param_names


def test_cvap_reproducible_and_monotone() -> None:
    s, y = _sample(500)
    p_a = CrossVennAbersCalibrator(cv=5, random_state=1).fit(s, y).predict_proba(GRID)
    p_b = CrossVennAbersCalibrator(cv=5, random_state=1).fit(s, y).predict_proba(GRID)
    np.testing.assert_allclose(p_a, p_b)
    assert np.all(np.diff(p_a) >= -1e-10)


def test_cvap_scalar_inside_envelope() -> None:
    s, y = _sample(500)
    cal = CrossVennAbersCalibrator(cv=5, random_state=2).fit(s, y)
    p = cal.predict_proba(GRID)
    env = cal.predict_interval(GRID)
    assert np.all(p >= env[:, 0] - 1e-9)
    assert np.all(p <= env[:, 1] + 1e-9)


@pytest.mark.slow
def test_cvap_close_to_ivap_on_ample_data() -> None:
    s, y = _sample(4000)
    p_ivap = VennAbersCalibrator().fit(s, y).predict_proba(GRID)
    p_cvap = CrossVennAbersCalibrator(cv=5, random_state=3).fit(s, y).predict_proba(GRID)
    assert np.max(np.abs(p_ivap - p_cvap)) < 0.1


def test_exports() -> None:
    import probcal

    for name in (
        "IsotonicCalibrator",
        "CenteredIsotonicCalibrator",
        "VennAbersCalibrator",
        "CrossVennAbersCalibrator",
    ):
        assert name in probcal.__all__


# ---------------------------------------------------------------- 0.3.4 regressions


def test_ivap_ties_pooled_and_order_invariant() -> None:
    """CAL-6: tied calibration scores are pooled; row order cannot move the output."""
    s, y = _sample(400)
    s = np.round(s, 1)
    perm = RNG.permutation(len(s))
    a = VennAbersCalibrator().fit(s, y)
    b = VennAbersCalibrator().fit(s[perm], y[perm])
    np.testing.assert_array_equal(a.predict_interval(GRID), b.predict_interval(GRID))
    np.testing.assert_array_equal(a.predict_interval(s[:20]), b.predict_interval(s[:20]))
    assert len(a.scores_) == len(np.unique(s))
    assert len(a.F0_) == len(a.scores_) + 1


def test_ivap_state_drops_unused_labels_and_weights() -> None:
    """CAL-23: only the distinct scores and the two tables are serialized."""
    cal = VennAbersCalibrator().fit(*_sample(200))
    state = cal.to_dict()["state"]
    assert set(state) == {"scores_", "F0_", "F1_", "ties_pooled_"}
    again = VennAbersCalibrator.from_json(cal.to_json())
    np.testing.assert_array_equal(again.predict_interval(GRID), cal.predict_interval(GRID))


def test_cvap_public_state_and_cv_validation() -> None:
    s, y = _sample(300)
    cal = CrossVennAbersCalibrator(cv=3).fit(s, y)
    assert len(cal.ivaps_) == 3
    assert set(cal.to_dict()["state"]) == {"ivaps_"}
    for bad in (1, 2.5, True):
        with pytest.raises(ValueError, match="cv"):
            CrossVennAbersCalibrator(cv=bad).fit(s, y)  # type: ignore[arg-type]
