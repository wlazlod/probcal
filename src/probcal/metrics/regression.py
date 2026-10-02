"""Recalibration-regression framework: calibration intercept, slope, and joint test.

The Cox (1958) framework; lineage through Miller, Hui & Tierney (1991).
Theory: ``docs/concepts/metrics.md``. Sample weights follow the package
convention in :mod:`probcal.metrics._common`.
"""

from dataclasses import dataclass

import numpy as np

from .._math import chi2_sf, expit, irls_logistic, logit
from .._validation import EPS
from ._common import _prep, effective_weights
from .smooth import _spiegelhalter


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
        Optional positive weights, same length as ``y``.

    Returns
    -------
    float
        Fitted intercept in log-odds units.
    """
    return _intercept(*_prep(y, p, sample_weight))


def _intercept(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    z = logit(p)
    return float(irls_logistic(np.ones((len(z), 1)), y, w=w, offset=z).beta[0])


def _slope_fit(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix ``[1, logit(p)]`` and the fitted (intercept, slope)."""
    z = logit(p)
    X = np.column_stack([np.ones_like(z), z])
    return X, irls_logistic(X, y, w=w).beta


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
        Optional positive weights, same length as ``y``.

    Returns
    -------
    float
        Fitted slope on the logit scale.
    """
    return float(_slope_fit(*_prep(y, p, sample_weight))[1][1])


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
        Fitted intercept (also available as :attr:`intercept`).
    beta : float
        Fitted slope (also available as :attr:`slope`).
    """

    statistic: float
    p_value: float
    alpha: float
    beta: float

    @property
    def intercept(self) -> float:
        """Fitted intercept; the name used by the rest of the package for ``alpha``."""
        return self.alpha

    @property
    def slope(self) -> float:
        """Fitted slope; the name used by the rest of the package for ``beta``."""
        return self.beta


def calibration_test(
    y: object, p: object, *, sample_weight: object = None
) -> CalibrationTestResult:
    """Likelihood-ratio test of joint calibration (alpha, beta) = (0, 1).

    With weights, the fit uses ``sample_weight`` as given (the MLE is
    scale-invariant) and the LR statistic uses the Kish-rescaled weights, so
    it is invariant to rescaling the weights; unit weights give the plain
    test (see :mod:`probcal.metrics._common`).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    CalibrationTestResult
        Test statistic, p-value, and fitted intercept/slope.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    X, beta_hat = _slope_fit(y_arr, p_arr, w)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    # Likelihood ratio on the Kish-rescaled weights (package convention):
    # invariant to rescaling the weights, the plain LR for unit weights.
    we = effective_weights(w)

    def _ll(prob: np.ndarray) -> float:
        prob = np.clip(prob, EPS, 1.0 - EPS)
        return float(np.sum(we * (y_arr * np.log(prob) + (1.0 - y_arr) * np.log1p(-prob))))

    lr = max(2.0 * (_ll(expit(X @ beta_hat)) - _ll(p_arr)), 0.0)
    p_value = chi2_sf(lr, 2.0)
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
        Optional positive weights, same length as ``y``.

    Returns
    -------
    GuardrailReport
        Slope, intercept, and Spiegelhalter-p values with pass/fail flags.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    slope = float(_slope_fit(y_arr, p_arr, w)[1][1])
    intercept = _intercept(y_arr, p_arr, w)
    sp = _spiegelhalter(y_arr, p_arr, w)
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
