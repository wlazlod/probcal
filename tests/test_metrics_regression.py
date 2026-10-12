"""Tests for probcal.metrics.regression."""

import math
import warnings

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


def test_guardrails_flags() -> None:
    y, p = _from_distortion(1.0, 0.0)
    g = calibration_guardrails(y, p)
    assert g.slope_ok and g.intercept_ok and g.spiegelhalter_ok
    assert g.all_ok
    y, p = _from_distortion(0.5, -0.8)
    g = calibration_guardrails(y, p)
    assert not g.all_ok
    assert not g.slope_ok


# ------------------------------------------------------------------ 0.3.4 fixes


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
    assert b.df == pytest.approx(a.df, rel=1e-8)
    # Second-order Rao-Scott: lr / (lbar (1 + a2)) on chi2(2 / (1 + a2)),
    # lambda = eig(I^-1 J) at the null.
    raw = _old_calibration_test_lr(y, p, w)
    z = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(z), z])
    v = p * (1 - p)
    lam = np.linalg.eigvals(np.linalg.solve((X.T * (w * v)) @ X, (X.T * (w * w * v)) @ X)).real
    a2 = np.var(lam) / lam.mean() ** 2
    assert a.statistic == pytest.approx(raw / (lam.mean() * (1 + a2)), rel=1e-8)
    assert a.df == pytest.approx(2 / (1 + a2), rel=1e-10)


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


def test_calibration_test_rao_scott_type_one_error_and_units() -> None:
    # Heavy rows (weight 10) near p=0.15, the rest near p=0.5.
    rng = np.random.default_rng(1)
    hi = np.arange(4000) < 400
    p = np.where(hi, 0.15, 0.5) + rng.uniform(-0.01, 0.01, 4000)
    w = np.where(hi, 10.0, 1.0)
    rng = np.random.default_rng(5)
    rej = [
        calibration_test((rng.random(len(p)) < p).astype(float), p, sample_weight=w).p_value < 0.05
        for _ in range(400)
    ]
    assert 0.02 <= np.mean(rej) <= 0.09
    y, pp = _from_distortion(1.0, 0.0, n=3000)
    unit = calibration_test(y, pp)
    assert unit.df == 2.0
    const = calibration_test(y, pp, sample_weight=np.full(len(y), 4.0))
    assert const.statistic == pytest.approx(unit.statistic, rel=1e-10)
    assert const.df == 2.0
    weighted = calibration_test(y, pp, sample_weight=np.where(pp < 0.2, 5.0, 1.0))
    assert 0.0 < weighted.df < 2.0


def test_constant_p_slope_nan_and_intercept_only_test() -> None:
    c = 0.05
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0.0])
    p = np.full(10, c)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="not identified"):
            calibration_slope(y, p)  # the warning is not about separation
    with pytest.warns(UserWarning, match="not identified"):
        assert math.isnan(calibration_slope(y, p))
    with pytest.warns(UserWarning, match="not identified"):
        res = calibration_test(y, p)
    assert res.df == 1.0 and math.isnan(res.beta)
    # Intercept-only LR with slope fixed at 1: the fitted rate is the observed rate.
    rate = y.mean()
    lr = 2 * np.sum(y * np.log(rate / c) + (1 - y) * np.log((1 - rate) / (1 - c)))
    assert res.statistic == pytest.approx(max(lr, 0.0), abs=1e-9)
    assert res.alpha == pytest.approx(np.log(rate / (1 - rate)) - np.log(c / (1 - c)), abs=1e-7)
    with pytest.warns(UserWarning, match="not identified"):
        g = calibration_guardrails(y, p)
    assert math.isnan(g.slope) and not g.slope_ok
