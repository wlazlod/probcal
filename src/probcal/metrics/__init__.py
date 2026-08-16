"""Calibration metrics and statistical tests (flat re-exports).

`evaluate` lives here because it aggregates across every submodule
(DECISIONS entry). Selection guidance — what may be optimized and what is
report-only — is the table in ``docs/concepts/metrics.md``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

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
from .kernel import (
    SkceTestResult,
    skce,
    skce_test,
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
    "ReliabilitySummary",
    "SkceTestResult",
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
    "reliability_summary",
    "skce",
    "skce_test",
    "smooth_ece",
    "spiegelhalter_z",
]


_METRIC_CATALOG: tuple[str, ...] = (
    "log_loss",
    "brier",
    "brier_skill",
    "ece",
    "ece_debiased",
    "mce",
    "ece_sweep",
    "smooth_ece",
    "ecce_max",
    "ecce_mean",
    "ici",
    "e50",
    "e90",
    "emax",
    "spiegelhalter_z",
    "spiegelhalter_p",
    "intercept",
    "slope",
)


def _point_metrics(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray | None,
    names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    sel = set(_METRIC_CATALOG if names is None else names)
    dispatch: dict[str, Callable[[], float]] = {
        "log_loss": lambda: log_loss(y, p, sample_weight=w),
        "brier": lambda: brier_score(y, p, sample_weight=w),
        "brier_skill": lambda: brier_skill_score(y, p, sample_weight=w),
        "ece": lambda: ece(y, p, sample_weight=w),
        "ece_debiased": lambda: ece_debiased(y, p, sample_weight=w),
        "mce": lambda: ece(y, p, norm="max", sample_weight=w),
        "ece_sweep": lambda: ece_sweep(y, p, sample_weight=w),
        "smooth_ece": lambda: smooth_ece(y, p, sample_weight=w),
        "intercept": lambda: calibration_intercept(y, p, sample_weight=w),
        "slope": lambda: calibration_slope(y, p, sample_weight=w),
    }
    out: dict[str, float] = {k: fn() for k, fn in dispatch.items() if k in sel}

    if sel & {"ecce_max", "ecce_mean"}:
        ec = ecce(y, p, sample_weight=w)
        out["ecce_max"] = ec.stat_max
        out["ecce_mean"] = ec.stat_mean

    if sel & {"ici", "e50", "e90", "emax"}:
        from .._math import loess, weighted_quantile

        # One shared LOESS fit powers the whole ICI family (ici/e50/e90/emax
        # use the same distances; refitting four times would quadruple
        # bootstrap cost). Distances themselves stay unweighted; sample_weight,
        # when given and not uniform, weights only the e50/e90 quantile step.
        d = np.abs(loess(p, y, frac=0.75, grid_size=512) - p)
        w_arr = np.ones(len(p)) if w is None else w
        uniform_w = w is None or bool(np.all(w == w[0]))
        if "ici" in sel:
            out["ici"] = float(np.average(d, weights=w_arr))
        if "e50" in sel:
            out["e50"] = (
                float(np.quantile(d, 0.5)) if uniform_w else float(weighted_quantile(d, 0.5, w))
            )
        if "e90" in sel:
            out["e90"] = (
                float(np.quantile(d, 0.9)) if uniform_w else float(weighted_quantile(d, 0.9, w))
            )
        if "emax" in sel:
            out["emax"] = float(np.max(d))

    if sel & {"spiegelhalter_z", "spiegelhalter_p"}:
        sp = spiegelhalter_z(y, p, sample_weight=w)
        if "spiegelhalter_z" in sel:
            out["spiegelhalter_z"] = sp.z
        if "spiegelhalter_p" in sel:
            out["spiegelhalter_p"] = sp.p_value

    return {k: out[k] for k in _METRIC_CATALOG if k in sel}


def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    seed: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
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
    metrics : sequence of str or None
        Subset of catalog names to compute; ``None`` computes the full
        catalog. The report follows catalog order regardless of the order
        given here. Raises ``ValueError`` for unknown names.
    stratify : bool
        If ``True`` (default), each bootstrap replicate resamples the
        negative and positive classes separately (case resampling within
        strata), preserving the observed class counts exactly — the
        pROC-style default. This conditions the CI on the observed class
        balance: it removes the additional variance a plain i.i.d. bootstrap
        picks up from the replicate-to-replicate event *count* fluctuating,
        and it makes every replicate well-defined (never a single-class
        resample) on rare-event data, at the cost of not propagating
        sampling variance in the event rate itself. ``y`` must already
        contain both classes (checked unconditionally, independent of this
        flag). If ``False``, replicates draw i.i.d. from all ``n`` rows; a
        degenerate (single-class) draw is redrawn up to 100 times before
        raising ``RuntimeError``.

    Returns
    -------
    MetricReport
        Point estimates and CI bounds for the requested catalog. Note the
        caveat from the metrics chapter: a bootstrap CI around a *biased*
        estimator (plain ECE) quantifies its variance, not its bias.

    Notes
    -----
    Cost model per replicate: scores and regression metrics and ECCE are
    O(n); binned ECEs are O(n log n); the ICI family (ici/e50/e90/emax)
    shares one LOESS fit at O(grid_size * frac * n); smECE is
    O(n + 257 * bins) per bisection step. All of the above are paid
    ``n_boot`` times — for n > 1e6, reduce ``n_boot`` or pass a ``metrics=``
    subset.
    """
    from .scores import _prep

    if metrics is not None:
        unknown = sorted(set(metrics) - set(_METRIC_CATALOG))
        if unknown:
            raise ValueError(
                f"unknown metric names {unknown}; valid names: {list(_METRIC_CATALOG)}"
            )
        names = tuple(k for k in _METRIC_CATALOG if k in set(metrics))
    else:
        names = _METRIC_CATALOG

    y_arr, p_arr, w_arr = _prep(y, p, sample_weight)
    point = _point_metrics(y_arr, p_arr, w_arr, names)
    values = np.array([point[k] for k in names])

    rng = np.random.default_rng(seed)
    n = len(y_arr)
    # _prep -> validate_binary_y already rejects single-class y unconditionally
    # (both idx0 and idx1 are therefore guaranteed non-empty here).
    idx0 = np.flatnonzero(y_arr == 0)
    idx1 = np.flatnonzero(y_arr == 1)

    boot = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        if stratify:
            idx = np.concatenate(
                [
                    idx0[rng.integers(0, len(idx0), len(idx0))],
                    idx1[rng.integers(0, len(idx1), len(idx1))],
                ]
            )
        else:
            for _attempt in range(100):
                idx = rng.integers(0, n, n)
                if y_arr[idx].min() != y_arr[idx].max():
                    break
            else:
                raise RuntimeError(
                    "100 consecutive degenerate (single-class) bootstrap draws; "
                    "pass stratify=True or supply more data"
                )
        yb, pb, wb = y_arr[idx], p_arr[idx], w_arr[idx]
        pm = _point_metrics(yb, pb, wb, names)
        boot[b] = [pm[k] for k in names]
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    return MetricReport(names=names, values=values, ci_low=ci_low, ci_high=ci_high)


@dataclass(frozen=True)
class ReliabilitySummary:
    """Stats-box aggregate for the annotated reliability diagram."""

    n: int
    events: int
    intercept: float
    slope: float
    ici: float
    e90: float
    spiegelhalter_p: float


def reliability_summary(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> ReliabilitySummary:
    """Assemble the annotated-reliability stats box from existing metrics.

    No new math: intercept and slope from the recalibration regression, ICI
    and E90 from the LOESS distance family, and Spiegelhalter's p-value.
    Lives here because, like `evaluate`, it aggregates across submodules;
    ``probcal.plots`` only formats the result. ``grid_size=None`` recovers
    0.1.2 values exactly.
    """
    from .scores import _prep

    y_arr, p_arr, w = _prep(y, p, sample_weight)
    wq = None if sample_weight is None else w
    return ReliabilitySummary(
        n=len(y_arr),
        events=int(y_arr.sum()),
        intercept=calibration_intercept(y_arr, p_arr, sample_weight=wq),
        slope=calibration_slope(y_arr, p_arr, sample_weight=wq),
        ici=ici(y_arr, p_arr, sample_weight=wq, grid_size=grid_size),
        e90=e90(y_arr, p_arr, sample_weight=wq, grid_size=grid_size),
        spiegelhalter_p=spiegelhalter_z(y_arr, p_arr, sample_weight=wq).p_value,
    )
