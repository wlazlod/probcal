"""Tests for probcal.metrics.regression."""

import numpy as np
import pytest

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


# ------------------------------------------------------------------ 0.4.0 fixes


def _old_calibration_test_lr(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    from probcal._math import irls_logistic

    z = logit(p)
    X = np.column_stack([np.ones_like(z), z])
    fit = irls_logistic(X, y, w=w)

    def _ll(prob: np.ndarray) -> float:
        prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
        return float(np.sum(w * (y * np.log(prob) + (1.0 - y) * np.log1p(-prob))))

    return max(2.0 * (_ll(expit(X @ fit.beta)) - _ll(p)), 0.0)


def test_calibration_test_unit_weights_unchanged() -> None:
    y, p = _from_distortion(0.9, 0.05, n=3000)
    res = calibration_test(y, p)
    assert res.statistic == _old_calibration_test_lr(y, p, np.ones(len(y)))
    assert calibration_test(y, p, sample_weight=np.ones(len(y))) == res


def test_calibration_test_weights_are_relative() -> None:
    # Regression (MET-6): 0.3.x read weights as frequencies, so multiplying
    # every weight by 10 multiplied the LR statistic by 10.
    y, p = _from_distortion(0.9, 0.05, n=3000)
    w = np.random.default_rng(2).uniform(0.5, 2.0, len(y))
    a = calibration_test(y, p, sample_weight=w)
    b = calibration_test(y, p, sample_weight=10.0 * w)
    assert b.statistic == pytest.approx(a.statistic, rel=1e-8)
    n_eff = w.sum() ** 2 / np.dot(w, w)
    raw = _old_calibration_test_lr(y, p, w)
    assert a.statistic == pytest.approx(raw * n_eff / w.sum(), rel=1e-8)


def test_calibration_test_tail_and_aliases() -> None:
    y, p = _from_distortion(0.75, 0.0, n=20000)
    res = calibration_test(y, p)
    assert 80.0 < res.statistic < 1400.0
    # chi2(2) upper tail is exp(-x / 2); 1 - P(chi2) rounded it to 0 (MET-5).
    assert res.p_value == pytest.approx(np.exp(-res.statistic / 2.0), rel=1e-10)
    assert res.intercept == res.alpha and res.slope == res.beta


def test_guardrails_match_individual_metrics() -> None:
    from probcal.metrics import spiegelhalter_z

    y, p = _from_distortion(0.95, 0.02, n=4000)
    w = np.random.default_rng(4).uniform(0.5, 2.0, len(y))
    g = calibration_guardrails(y, p, sample_weight=w)
    assert g.slope == calibration_slope(y, p, sample_weight=w)
    assert g.intercept == calibration_intercept(y, p, sample_weight=w)
    assert g.spiegelhalter_p == spiegelhalter_z(y, p, sample_weight=w).p_value
