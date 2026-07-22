"""Tests for probcal.metrics.regression."""

import numpy as np

from probcal._math import expit, logit
from probcal.metrics.regression import (
    calibration_guardrails,
    calibration_intercept,
    calibration_slope,
    calibration_test,
)

RNG = np.random.default_rng(67)


def _from_distortion(a: float, b: float, n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-1.0, 1.4, n))
    y = (RNG.random(n) < expit(a * logit(p) + b)).astype(float)
    return y, p


def test_slope_recovers_distortion() -> None:
    y, p = _from_distortion(0.7, 0.0)
    assert abs(calibration_slope(y, p) - 0.7) < 0.06


def test_intercept_recovers_pure_shift() -> None:
    y, p = _from_distortion(1.0, -0.6)
    assert abs(calibration_intercept(y, p) - (-0.6)) < 0.08


def test_calibration_test_null_and_alternative() -> None:
    y_ok, p_ok = _from_distortion(1.0, 0.0)
    y_bad, p_bad = _from_distortion(0.6, -0.5)
    res_ok = calibration_test(y_ok, p_ok)
    res_bad = calibration_test(y_bad, p_bad)
    assert res_ok.p_value > 0.01
    assert res_bad.p_value < 1e-6
    assert abs(res_bad.beta - 0.6) < 0.06


def test_guardrails_pass_on_calibrated() -> None:
    y, p = _from_distortion(1.0, 0.0)
    g = calibration_guardrails(y, p)
    assert g.slope_ok and g.intercept_ok and g.spiegelhalter_ok
    assert g.all_ok


def test_guardrails_fail_on_distorted() -> None:
    y, p = _from_distortion(0.5, -0.8)
    g = calibration_guardrails(y, p)
    assert not g.all_ok
    assert not g.slope_ok
