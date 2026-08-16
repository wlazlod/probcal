"""Recalibration-regression framework: calibration intercept, slope, and joint test.

The Cox (1958) framework; lineage through Miller, Hui & Tierney (1991).
Theory: ``docs/concepts/metrics.md``.
"""

from dataclasses import dataclass

import numpy as np

from .._math import expit, gammainc_lower, irls_logistic, logit
from .scores import _prep
from .smooth import spiegelhalter_z


def calibration_intercept(y: object, p: object, *, sample_weight: object = None) -> float:
    """Calibration-in-the-large in log-odds.

    Logistic intercept with the slope fixed at 1 (offset regression on
    logit(p)).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    float
        Fitted intercept in log-odds units.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    z = logit(p_arr)
    res = irls_logistic(np.ones((len(z), 1)), y_arr, w=w, offset=z)
    return float(res.beta[0])


def calibration_slope(y: object, p: object, *, sample_weight: object = None) -> float:
    """Cox calibration slope.

    ``< 1`` means overfitting/overconfident spread, ``> 1`` underfitting.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    float
        Fitted slope on the logit scale.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    z = logit(p_arr)
    X = np.column_stack([np.ones_like(z), z])
    res = irls_logistic(X, y_arr, w=w)
    return float(res.beta[1])


@dataclass(frozen=True)
class CalibrationTestResult:
    """2-df likelihood-ratio test of (intercept, slope) = (0, 1) — the
    Cox-framed weak calibration test.

    Attributes
    ----------
    statistic : float
        Likelihood-ratio test statistic (chi-square, 2 df).
    p_value : float
        Upper-tail p-value of the statistic.
    alpha : float
        Fitted intercept.
    beta : float
        Fitted slope.
    """

    statistic: float
    p_value: float
    alpha: float
    beta: float


def calibration_test(
    y: object, p: object, *, sample_weight: object = None
) -> CalibrationTestResult:
    """Likelihood-ratio test of joint calibration (alpha, beta) = (0, 1).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    CalibrationTestResult
        Test statistic, p-value, and fitted intercept/slope.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    z = logit(p_arr)
    X = np.column_stack([np.ones_like(z), z])
    fit = irls_logistic(X, y_arr, w=w)
    alpha, beta = float(fit.beta[0]), float(fit.beta[1])

    def _ll(prob: np.ndarray) -> float:
        prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
        return float(np.sum(w * (y_arr * np.log(prob) + (1.0 - y_arr) * np.log1p(-prob))))

    ll_fit = _ll(expit(X @ fit.beta))
    ll_null = _ll(p_arr)
    lr = max(2.0 * (ll_fit - ll_null), 0.0)
    p_value = 1.0 - float(gammainc_lower(1.0, lr / 2.0))  # chi-square, df = 2
    return CalibrationTestResult(statistic=lr, p_value=p_value, alpha=alpha, beta=beta)


@dataclass(frozen=True)
class GuardrailReport:
    """Three-flag calibration health summary used across the package.

    Thresholds are conventions, not theorems: slope within [0.9, 1.1],
    intercept within +/-0.1 log-odds, Spiegelhalter p above 0.05.

    Attributes
    ----------
    slope : float
        Fitted Cox calibration slope.
    intercept : float
        Fitted calibration-in-the-large intercept (log-odds).
    spiegelhalter_p : float
        Spiegelhalter test p-value.
    slope_ok : bool
        Whether ``slope`` lies in ``[0.9, 1.1]``.
    intercept_ok : bool
        Whether ``abs(intercept) <= 0.1``.
    spiegelhalter_ok : bool
        Whether ``spiegelhalter_p > 0.05``.
    all_ok : bool
        Conjunction of the three flags above.
    """

    slope: float
    intercept: float
    spiegelhalter_p: float
    slope_ok: bool
    intercept_ok: bool
    spiegelhalter_ok: bool
    all_ok: bool


def calibration_guardrails(
    y: object, p: object, *, sample_weight: object = None
) -> GuardrailReport:
    """Evaluate the three guardrail flags.

    Printed in selection reports and offset audit reports.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    GuardrailReport
        Slope, intercept, and Spiegelhalter-p values with pass/fail flags.
    """
    slope = calibration_slope(y, p, sample_weight=sample_weight)
    intercept = calibration_intercept(y, p, sample_weight=sample_weight)
    sp = spiegelhalter_z(y, p, sample_weight=sample_weight)
    slope_ok = 0.9 <= slope <= 1.1
    intercept_ok = abs(intercept) <= 0.1
    sp_ok = sp.p_value > 0.05
    return GuardrailReport(
        slope=slope,
        intercept=intercept,
        spiegelhalter_p=sp.p_value,
        slope_ok=slope_ok,
        intercept_ok=intercept_ok,
        spiegelhalter_ok=sp_ok,
        all_ok=slope_ok and intercept_ok and sp_ok,
    )
