"""Calibration metrics and statistical tests (flat re-exports).

`evaluate` lives here because it aggregates across every submodule
(DECISIONS entry). Selection guidance — what may be optimized and what is
report-only — is the table in ``docs/concepts/metrics.md``.
"""

import numpy as np

from .._results import MetricReport
from .binned import (
    HosmerLemeshowResult,
    adaptive_ece,
    ece,
    ece_debiased,
    ece_sweep,
    hosmer_lemeshow,
)
from .grade import (
    BinomialGradeResult,
    JeffreysGradeResult,
    binomial_grade_test,
    jeffreys_grade_test,
)
from .regression import (
    CalibrationTestResult,
    GuardrailReport,
    calibration_guardrails,
    calibration_intercept,
    calibration_slope,
    calibration_test,
)
from .scores import (
    LogLossDecomposition,
    MurphyDecomposition,
    brier_score,
    brier_skill_score,
    log_loss,
    logloss_calibration_refinement,
    murphy_decomposition,
)
from .smooth import (
    EcceResult,
    SpiegelhalterResult,
    e50,
    e90,
    ecce,
    emax,
    ici,
    smooth_ece,
    spiegelhalter_z,
)

__all__ = [
    "BinomialGradeResult",
    "CalibrationTestResult",
    "EcceResult",
    "GuardrailReport",
    "HosmerLemeshowResult",
    "JeffreysGradeResult",
    "LogLossDecomposition",
    "MurphyDecomposition",
    "SpiegelhalterResult",
    "adaptive_ece",
    "binomial_grade_test",
    "brier_score",
    "brier_skill_score",
    "calibration_guardrails",
    "calibration_intercept",
    "calibration_slope",
    "calibration_test",
    "e50",
    "e90",
    "ecce",
    "ece",
    "ece_debiased",
    "ece_sweep",
    "emax",
    "evaluate",
    "hosmer_lemeshow",
    "ici",
    "jeffreys_grade_test",
    "log_loss",
    "logloss_calibration_refinement",
    "murphy_decomposition",
    "smooth_ece",
    "spiegelhalter_z",
]


def _point_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> dict[str, float]:
    from .._math import loess

    sp = spiegelhalter_z(y, p, sample_weight=w)
    ec = ecce(y, p, sample_weight=w)
    # One shared LOESS fit powers the whole ICI family (ici/e50/e90/emax use
    # the same distances; refitting four times would quadruple bootstrap cost).
    d = np.abs(loess(p, y, frac=0.75) - p)
    w_arr = np.ones(len(p)) if w is None else w
    return {
        "log_loss": log_loss(y, p, sample_weight=w),
        "brier": brier_score(y, p, sample_weight=w),
        "brier_skill": brier_skill_score(y, p, sample_weight=w),
        "ece": ece(y, p, sample_weight=w),
        "ece_debiased": ece_debiased(y, p, sample_weight=w),
        "mce": ece(y, p, norm="max", sample_weight=w),
        "ece_sweep": ece_sweep(y, p, sample_weight=w),
        "smooth_ece": smooth_ece(y, p, sample_weight=w),
        "ecce_max": ec.stat_max,
        "ecce_mean": ec.stat_mean,
        "ici": float(np.average(d, weights=w_arr)),
        "e50": float(np.quantile(d, 0.5)),
        "e90": float(np.quantile(d, 0.9)),
        "emax": float(np.max(d)),
        "spiegelhalter_z": sp.z,
        "spiegelhalter_p": sp.p_value,
        "intercept": calibration_intercept(y, p, sample_weight=w),
        "slope": calibration_slope(y, p, sample_weight=w),
    }


def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> MetricReport:
    """Full metric report with seeded bootstrap percentile confidence intervals.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None
        Observation weights (resampled together with the observations).
    n_boot : int
        Case-resampling bootstrap replicates (percentile CIs at 2.5/97.5).
    seed : int
        RNG seed; results are bit-reproducible given the seed.

    Returns
    -------
    MetricReport
        Point estimates and CI bounds for the full catalog. Note the caveat
        from the metrics chapter: a bootstrap CI around a *biased* estimator
        (plain ECE) quantifies its variance, not its bias.
    """
    from .scores import _prep

    y_arr, p_arr, w_arr = _prep(y, p, sample_weight)
    point = _point_metrics(y_arr, p_arr, w_arr)
    names = tuple(point)
    values = np.array([point[k] for k in names])

    rng = np.random.default_rng(seed)
    n = len(y_arr)
    boot = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb, wb = y_arr[idx], p_arr[idx], w_arr[idx]
        if yb.min() == yb.max():  # degenerate resample: keep the point estimate
            boot[b] = values
            continue
        pm = _point_metrics(yb, pb, wb)
        boot[b] = [pm[k] for k in names]
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    return MetricReport(names=names, values=values, ci_low=ci_low, ci_high=ci_high)
