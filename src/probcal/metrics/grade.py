"""Per-grade binomial and Jeffreys backtests (credit-risk rating grades).

Supervisory context (BCBS WP14; ECB 2019 instructions): each rating grade's
realized default count is tested against its assigned PD. Theory:
``docs/concepts/metrics.md``.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .._math import beta_ppf, betainc, norm_cdf

# Re-exported so `probcal.metrics.grade.pluto_tasche` sits alongside the
# per-grade backtests it complements; defined in `_conservative.py`.
from ._conservative import (  # noqa: F401
    PlutoTascheResult,
    jeffreys_upper_bands,
    pluto_tasche,
    pluto_tasche_from_arrays,
)

# Re-exported so `probcal.metrics.grade.hl_e_test` sits alongside the other
# per-grade tests; defined in `_safe.py`.
from ._safe import HlEResult, hl_e_test  # noqa: F401
from .scores import _prep


def _traffic_light(p_value: float) -> str:
    if p_value <= 0.01:
        return "red"
    if p_value <= 0.05:
        return "amber"
    return "green"


def _per_grade(
    y: np.ndarray, p: np.ndarray, grades: np.ndarray
) -> tuple[tuple, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.unique(grades)
    n = np.empty(len(labels), dtype=np.int64)
    k = np.empty(len(labels), dtype=np.int64)
    pd = np.empty(len(labels))
    for i, g in enumerate(labels):
        mask = grades == g
        n[i] = int(np.sum(mask))
        k[i] = int(np.sum(y[mask]))
        pd[i] = float(np.mean(p[mask]))
    return tuple(str(g) for g in labels), n, k, pd


def _check_weights(sample_weight: object, n_obs: int) -> None:
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        if not np.allclose(w, w[0]):
            warnings.warn(
                "grade tests use raw integer counts; non-uniform sample weights are ignored",
                UserWarning,
                stacklevel=3,
            )


@dataclass(frozen=True)
class BinomialGradeResult:
    """Exact and approximate one-sided binomial backtest per rating grade.

    Attributes
    ----------
    grades : tuple of str
        Sorted grade labels.
    n : numpy.ndarray
        Observation count per grade.
    k : numpy.ndarray
        Default count per grade.
    pd : numpy.ndarray
        Assigned PD per grade (mean of ``p`` within the grade).
    p_exact : numpy.ndarray
        Exact binomial tail p-value per grade.
    p_normal : numpy.ndarray
        Normal-approximation p-value per grade.
    light : tuple of str
        Traffic light per grade (``"green"``, ``"amber"``, or ``"red"``),
        derived from ``p_exact``.
    ci_low, ci_high : numpy.ndarray
        90% Clopper-Pearson display interval for the observed rate.
    """

    grades: tuple
    n: np.ndarray
    k: np.ndarray
    pd: np.ndarray
    p_exact: np.ndarray
    p_normal: np.ndarray
    light: tuple
    ci_low: np.ndarray
    ci_high: np.ndarray


def binomial_grade_test(
    y: object, p: object, grades: object, *, sample_weight: object = None
) -> BinomialGradeResult:
    """Exact binomial tail test per grade: P(X >= k | n, PD).

    Small p-values flag grades with more defaults than the assigned PD
    supports. The exact tail uses the incomplete-beta identity
    ``P(X >= k) = I_PD(k, n - k + 1)``; the normal approximation is reported
    alongside. Traffic lights: green > 0.05, amber > 0.01, red <= 0.01.
    ``ci_low``/``ci_high`` are 90% Clopper-Pearson display intervals for the
    observed rate; the traffic light itself remains the one-sided exact test,
    unchanged.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities (assigned PDs) in ``[0, 1]``.
    grades : array_like
        Rating grade label per observation.
    sample_weight : array_like or None, keyword-only
        Not used: grade tests use raw integer counts. A ``UserWarning`` is
        emitted if the weights are non-uniform.

    Returns
    -------
    BinomialGradeResult
        Per-grade counts, p-values, traffic lights, and display intervals.
    """
    y_arr, p_arr, _ = _prep(y, p, None)
    _check_weights(sample_weight, len(y_arr))
    g_arr = np.asarray(grades)
    labels, n, k, pd = _per_grade(y_arr, p_arr, g_arr)
    p_exact = np.empty(len(labels))
    p_normal = np.empty(len(labels))
    for i in range(len(labels)):
        if k[i] == 0:
            p_exact[i] = 1.0
        else:
            p_exact[i] = float(betainc(float(k[i]), float(n[i] - k[i] + 1), pd[i]))
        se = np.sqrt(n[i] * pd[i] * (1.0 - pd[i]))
        z = (k[i] - n[i] * pd[i]) / se if se > 0 else 0.0
        p_normal[i] = float(1.0 - norm_cdf(np.array([z]))[0])
    light = tuple(_traffic_light(v) for v in p_exact)
    ci_low = np.empty(len(labels))
    ci_high = np.empty(len(labels))
    for i in range(len(labels)):
        ki, ni = int(k[i]), int(n[i])
        ci_low[i] = 0.0 if ki == 0 else beta_ppf(0.05, float(ki), float(ni - ki + 1))
        ci_high[i] = 1.0 if ki == ni else beta_ppf(0.95, float(ki + 1), float(ni - ki))
    return BinomialGradeResult(
        grades=labels,
        n=n,
        k=k,
        pd=pd,
        p_exact=p_exact,
        p_normal=p_normal,
        light=light,
        ci_low=ci_low,
        ci_high=ci_high,
    )


@dataclass(frozen=True)
class JeffreysGradeResult:
    """Jeffreys-posterior backtest per rating grade (ECB IRB practice).

    Attributes
    ----------
    grades : tuple of str
        Sorted grade labels.
    n : numpy.ndarray
        Observation count per grade.
    k : numpy.ndarray
        Default count per grade.
    pd : numpy.ndarray
        Assigned PD per grade (mean of ``p`` within the grade).
    p_value : numpy.ndarray
        Posterior ``P(theta <= PD | k, n)`` per grade.
    light : tuple of str
        Traffic light per grade (``"green"``, ``"amber"``, or ``"red"``),
        derived from ``p_value``.
    ci_low, ci_high : numpy.ndarray
        Central 90% Jeffreys posterior display interval.
    """

    grades: tuple
    n: np.ndarray
    k: np.ndarray
    pd: np.ndarray
    p_value: np.ndarray
    light: tuple
    ci_low: np.ndarray
    ci_high: np.ndarray


def jeffreys_grade_test(
    y: object, p: object, grades: object, *, sample_weight: object = None
) -> JeffreysGradeResult:
    """Jeffreys test per grade: posterior P(theta <= PD | k, n) under Beta(k+1/2, n-k+1/2).

    One-sided and conservative by design: a small value flags a grade whose
    PD is likely understated. Do not read it two-sided (a recurring
    validation error — see the metrics chapter). ``ci_low``/``ci_high`` are
    the central 90% Jeffreys posterior display intervals; the traffic light
    itself remains the one-sided posterior test, unchanged.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities (assigned PDs) in ``[0, 1]``.
    grades : array_like
        Rating grade label per observation.
    sample_weight : array_like or None, keyword-only
        Not used: grade tests use raw integer counts. A ``UserWarning`` is
        emitted if the weights are non-uniform.

    Returns
    -------
    JeffreysGradeResult
        Per-grade counts, p-values, traffic lights, and display intervals.
    """
    y_arr, p_arr, _ = _prep(y, p, None)
    _check_weights(sample_weight, len(y_arr))
    g_arr = np.asarray(grades)
    labels, n, k, pd = _per_grade(y_arr, p_arr, g_arr)
    p_value = np.empty(len(labels))
    for i in range(len(labels)):
        p_value[i] = float(betainc(k[i] + 0.5, n[i] - k[i] + 0.5, pd[i]))
    light = tuple(_traffic_light(v) for v in p_value)
    ci_low = np.empty(len(labels))
    ci_high = np.empty(len(labels))
    for i in range(len(labels)):
        a, b = k[i] + 0.5, n[i] - k[i] + 0.5
        ci_low[i] = beta_ppf(0.05, a, b)
        ci_high[i] = beta_ppf(0.95, a, b)
    return JeffreysGradeResult(
        grades=labels,
        n=n,
        k=k,
        pd=pd,
        p_value=p_value,
        light=light,
        ci_low=ci_low,
        ci_high=ci_high,
    )
