"""Tests for probcal.vennabers: IVAP and CVAP."""

import numpy as np

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


def test_cvap_reproducible_and_monotone() -> None:
    s, y = _sample(500)
    p_a = CrossVennAbersCalibrator(cv=5, random_state=1).fit(s, y).predict_proba(GRID)
    p_b = CrossVennAbersCalibrator(cv=5, random_state=1).fit(s, y).predict_proba(GRID)
    np.testing.assert_allclose(p_a, p_b)
    assert np.all(np.diff(p_a) >= -1e-10)
    env = CrossVennAbersCalibrator(cv=5, random_state=1).fit(s, y).predict_interval(GRID)
    assert np.all(p_a >= env[:, 0] - 1e-9)
    assert np.all(p_a <= env[:, 1] + 1e-9)


# ---------------------------------------------------------------- 0.3.4 regressions


def test_ivap_state_drops_unused_labels_and_weights() -> None:
    """CAL-23: only the distinct scores and the two tables are serialized."""
    cal = VennAbersCalibrator().fit(*_sample(200))
    state = cal.to_dict()["state"]
    assert set(state) == {"scores_", "F0_", "F1_", "ties_pooled_"}
    again = VennAbersCalibrator.from_json(cal.to_json())
    np.testing.assert_array_equal(again.predict_interval(GRID), cal.predict_interval(GRID))


def test_cvap_public_state() -> None:
    cal = CrossVennAbersCalibrator(cv=3).fit(*_sample(300))
    assert len(cal.ivaps_) == 3
    assert set(cal.to_dict()["state"]) == {"ivaps_"}
