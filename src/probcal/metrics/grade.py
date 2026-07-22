"""Per-grade binomial and Jeffreys backtests (credit-risk rating grades).

Supervisory context (BCBS WP14; ECB 2019 instructions): each rating grade's
realized default count is tested against its assigned PD. Theory:
``docs/concepts/metrics.md``.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .._math import betainc, norm_cdf
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
    """Exact and approximate one-sided binomial backtest per rating grade."""

    grades: tuple
    n: np.ndarray
    k: np.ndarray
    pd: np.ndarray
    p_exact: np.ndarray
    p_normal: np.ndarray
    light: tuple


def binomial_grade_test(
    y: object, p: object, grades: object, *, sample_weight: object = None
) -> BinomialGradeResult:
    """Exact binomial tail test per grade: P(X >= k | n, PD).

    Small p-values flag grades with more defaults than the assigned PD
    supports. The exact tail uses the incomplete-beta identity
    ``P(X >= k) = I_PD(k, n - k + 1)``; the normal approximation is reported
    alongside. Traffic lights: green > 0.05, amber > 0.01, red <= 0.01.
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
    return BinomialGradeResult(
        grades=labels, n=n, k=k, pd=pd, p_exact=p_exact, p_normal=p_normal, light=light
    )


@dataclass(frozen=True)
class JeffreysGradeResult:
    """Jeffreys-posterior backtest per rating grade (ECB IRB practice)."""

    grades: tuple
    n: np.ndarray
    k: np.ndarray
    pd: np.ndarray
    p_value: np.ndarray
    light: tuple


def jeffreys_grade_test(
    y: object, p: object, grades: object, *, sample_weight: object = None
) -> JeffreysGradeResult:
    """Jeffreys test per grade: posterior P(theta <= PD | k, n) under Beta(k+1/2, n-k+1/2).

    One-sided and conservative by design: a small value flags a grade whose
    PD is likely understated. Do not read it two-sided (a recurring
    validation error — see the metrics chapter).
    """
    y_arr, p_arr, _ = _prep(y, p, None)
    _check_weights(sample_weight, len(y_arr))
    g_arr = np.asarray(grades)
    labels, n, k, pd = _per_grade(y_arr, p_arr, g_arr)
    p_value = np.empty(len(labels))
    for i in range(len(labels)):
        p_value[i] = float(betainc(k[i] + 0.5, n[i] - k[i] + 0.5, pd[i]))
    light = tuple(_traffic_light(v) for v in p_value)
    return JeffreysGradeResult(grades=labels, n=n, k=k, pd=pd, p_value=p_value, light=light)
