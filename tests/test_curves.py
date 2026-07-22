"""Tests for probcal.curves."""

import numpy as np

from probcal._math import expit, logit
from probcal._results import BeltResult, ReliabilityCurve
from probcal.curves import (
    calibration_belt,
    reliability_binned,
    reliability_loess,
    reliability_spline,
)

RNG = np.random.default_rng(83)


def _calibrated(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_reliability_binned_structure() -> None:
    y, p = _calibrated(2000)
    curve = reliability_binned(y, p, n_bins=10)
    assert isinstance(curve, ReliabilityCurve)
    assert np.all(np.diff(curve.pred_mean) > 0)
    np.testing.assert_allclose(curve.pred_mean_logit, logit(curve.pred_mean), atol=1e-12)
    assert int(curve.count.sum()) == 2000
    assert np.all(curve.ci_low <= curve.event_rate)
    assert np.all(curve.event_rate <= curve.ci_high)


def test_wilson_ci_hand_case() -> None:
    # One bin: n=100, k=10. Wilson at z=1.96.
    y = np.concatenate([np.ones(10), np.zeros(90)])
    p = np.full(100, 0.1)
    curve = reliability_binned(y, p, n_bins=1)
    z = 1.959963984540054
    n, rate = 100.0, 0.1
    denom = 1.0 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denom
    np.testing.assert_allclose(curve.ci_low, [center - half], atol=1e-12)
    np.testing.assert_allclose(curve.ci_high, [center + half], atol=1e-12)


def test_reliability_loess_near_diagonal() -> None:
    y, p = _calibrated()
    curve = reliability_loess(y, p)
    assert len(curve.grid_p) == 100
    assert np.max(np.abs(curve.event_rate - curve.grid_p)) < 0.06
    np.testing.assert_allclose(curve.grid_logit, logit(curve.grid_p), atol=1e-12)


def test_reliability_spline_near_diagonal() -> None:
    y, p = _calibrated()
    curve = reliability_spline(y, p)
    assert np.max(np.abs(curve.event_rate - curve.grid_p)) < 0.06


def test_belt_calibrated_data() -> None:
    y, p = _calibrated(8000)
    belt = calibration_belt(y, p)
    assert isinstance(belt, BeltResult)
    assert belt.p_value > 0.01
    assert 1 <= belt.degree <= 4
    inside = (belt.lower_95 <= belt.grid_p) & (belt.grid_p <= belt.upper_95)
    assert inside.mean() >= 0.9
    assert np.all(belt.lower_80 >= belt.lower_95 - 1e-12)
    assert np.all(belt.upper_80 <= belt.upper_95 + 1e-12)
    assert np.all(belt.lower_95 <= belt.upper_95)


def test_belt_rejects_distortion() -> None:
    y, p = _calibrated(8000)
    p_bad = expit(0.5 * logit(p) - 0.7)
    belt = calibration_belt(y, p_bad)
    assert belt.p_value < 1e-4
    outside = (belt.grid_p < belt.lower_95) | (belt.grid_p > belt.upper_95)
    assert outside.any()


def test_belt_grid_scales_consistent() -> None:
    y, p = _calibrated(3000)
    belt = calibration_belt(y, p)
    np.testing.assert_allclose(belt.grid_logit, logit(belt.grid_p), atol=1e-12)
