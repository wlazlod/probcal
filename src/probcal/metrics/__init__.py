"""Calibration metrics and statistical tests (flat re-exports).

`evaluate` lives here because it aggregates across every submodule
(DECISIONS entry). Selection guidance — what may be optimized and what is
report-only — is the table in ``docs/concepts/metrics.md``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast, overload

import numpy as np

from .._results import GroupedMetricReport, MetricReport
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
    HlEResult,
    JeffreysGradeResult,
    PlutoTascheResult,
    binomial_grade_test,
    hl_e_test,
    jeffreys_grade_test,
    jeffreys_upper_bands,
    pluto_tasche,
    pluto_tasche_from_arrays,
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
    MurphyCurve,
    MurphyDecomposition,
    brier_score,
    brier_skill_score,
    log_loss,
    logloss_calibration_refinement,
    murphy_curve,
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
    "GroupedMetricReport",
    "GuardrailReport",
    "HlEResult",
    "HosmerLemeshowResult",
    "JeffreysGradeResult",
    "LogLossDecomposition",
    "MurphyCurve",
    "MurphyDecomposition",
    "PlutoTascheResult",
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
    "hl_e_test",
    "hosmer_lemeshow",
    "ici",
    "jeffreys_grade_test",
    "jeffreys_upper_bands",
    "log_loss",
    "logloss_calibration_refinement",
    "murphy_curve",
    "murphy_decomposition",
    "pluto_tasche",
    "pluto_tasche_from_arrays",
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


def _binned_presorted(
    y: np.ndarray, p: np.ndarray, w: np.ndarray | None, sel: set[str]
) -> dict[str, float]:
    """The binned family over ``p``-ascending arrays, sharing one binning pass.

    ``ece``/``ece_debiased``/``mce`` all bin at ``n_bins=15, strategy="mass"``,
    so one :func:`~probcal.metrics.binned._bin_gaps` result serves all three
    instead of three identical quantile-and-bincount passes; ``ece_sweep``'s
    ~99-candidate monotonicity scan runs on cut positions rather than a rebuilt
    length-n bin index, and still takes its returned value from the unchanged
    ``ece`` call at the chosen bin count.
    """
    from .binned import (
        _bin_gaps,
        _ece_debiased_from_gaps,
        _ece_from_gaps,
        _ece_sweep_presorted,
    )

    out: dict[str, float] = {}
    if not sel & {"ece", "ece_debiased", "mce", "ece_sweep"}:
        return out
    w_arr = np.ones(len(p)) if w is None else w
    if sel & {"ece", "ece_debiased", "mce"}:
        shares, gaps, rates, counts = _bin_gaps(y, p, w_arr, 15, "mass")
        if "ece" in sel:
            out["ece"] = _ece_from_gaps(shares, gaps, "l1")
        if "ece_debiased" in sel:
            out["ece_debiased"] = _ece_debiased_from_gaps(shares, gaps, rates, counts)
        if "mce" in sel:
            out["mce"] = _ece_from_gaps(shares, gaps, "max")
    if "ece_sweep" in sel:
        out["ece_sweep"] = _ece_sweep_presorted(y, p, w_arr)
    return out


def _point_metrics(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray | None,
    names: tuple[str, ...] | None = None,
    *,
    presorted: bool = False,
) -> dict[str, float]:
    """Point estimates for ``names``; ``presorted`` is the bootstrap fast path.

    ``presorted=True`` declares that ``p`` is already sorted ascending, which
    lets every consumer that would sort internally skip doing so and lets the
    binned family share one binning pass. It is only ever passed by
    :func:`evaluate`'s replicate loop, which sorts each replicate once; the
    reported *point* estimates always come off the default path, so they stay
    bit-for-bit what earlier releases produced (``tests/test_metrics_evaluate.py``
    pins this against a verbatim copy of the pre-0.3.0 body). Bootstrap values
    may differ in the last bits from the reordered summation, which moves the
    percentile CI bounds at the ~1e-15 level.
    """
    sel = set(_METRIC_CATALOG if names is None else names)
    dispatch: dict[str, Callable[[], float]] = {
        "log_loss": lambda: log_loss(y, p, sample_weight=w),
        "brier": lambda: brier_score(y, p, sample_weight=w),
        "brier_skill": lambda: brier_skill_score(y, p, sample_weight=w),
        "smooth_ece": lambda: smooth_ece(y, p, sample_weight=w),
        "intercept": lambda: calibration_intercept(y, p, sample_weight=w),
        "slope": lambda: calibration_slope(y, p, sample_weight=w),
    }
    if not presorted:
        dispatch |= {
            "ece": lambda: ece(y, p, sample_weight=w),
            "ece_debiased": lambda: ece_debiased(y, p, sample_weight=w),
            "mce": lambda: ece(y, p, norm="max", sample_weight=w),
            "ece_sweep": lambda: ece_sweep(y, p, sample_weight=w),
        }
    out: dict[str, float] = {k: fn() for k, fn in dispatch.items() if k in sel}
    if presorted:
        out |= _binned_presorted(y, p, w, sel)

    if sel & {"ecce_max", "ecce_mean"}:
        ec = ecce(y, p, sample_weight=w, presorted=presorted)
        out["ecce_max"] = ec.stat_max
        out["ecce_mean"] = ec.stat_mean

    if sel & {"ici", "e50", "e90", "emax"}:
        from .._math import loess, weighted_quantile

        # One shared LOESS fit powers the whole ICI family (ici/e50/e90/emax
        # use the same distances; refitting four times would quadruple
        # bootstrap cost). Distances themselves stay unweighted; sample_weight,
        # when given and not uniform, weights only the e50/e90 quantile step.
        d = np.abs(loess(p, y, frac=0.75, grid_size=512, presorted=presorted) - p)
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


@overload
def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    seed: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
    by: None = None,
) -> MetricReport: ...


@overload
def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    seed: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
    by: object,
) -> GroupedMetricReport: ...


def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    seed: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
    by: object = None,
) -> MetricReport | GroupedMetricReport:
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
        given here.
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
    by : array_like or None, keyword-only
        Optional group labels, one per observation (same length as ``y``).
        ``None`` (default) is the plain report above, unchanged. Otherwise
        each label is stringified and a separate report is computed per
        sorted label — group ``i`` (in sorted-label order) is evaluated
        with ``seed + 1000 * i``, a fixed offset so results are
        reproducible independent of the label values or how many groups
        exist — plus a pooled report on the full data using ``seed``
        unchanged. Returns a :class:`~probcal._results.GroupedMetricReport`
        instead of a plain report. Group-conditional statistical *testing*
        (formal multiplicity-adjusted comparisons across groups) is out of
        scope here; see ``docs/guide/groups.md``.

    Returns
    -------
    MetricReport or GroupedMetricReport
        Point estimates and CI bounds for the requested catalog
        (``by=None``, the default), or a pooled report plus one report per
        group (``by`` given). Note the caveat from the metrics chapter: a
        bootstrap CI around a *biased* estimator (plain ECE) quantifies its
        variance, not its bias.

    Raises
    ------
    ValueError
        If ``metrics`` contains names outside the metric catalog; if
        ``by`` is given with a length that does not match ``y``; or if a
        group has only one outcome class (the underlying
        ``"y must contain both classes"`` error, re-raised naming the
        group).
    RuntimeError
        If ``stratify=False`` and 100 consecutive bootstrap draws are all
        single-class.

    Notes
    -----
    Cost model per replicate: scores and regression metrics and ECCE are
    O(n); binned ECEs are O(n log n); the ICI family (ici/e50/e90/emax)
    shares one LOESS fit at O(grid_size * frac * n); smECE is
    O(n + 257 * bins) per bisection step. All of the above are paid
    ``n_boot`` times — for n > 1e6, reduce ``n_boot`` or pass a ``metrics=``
    subset. With ``by`` given, the whole cost model above is paid once per
    group plus once for the pooled report.

    Each replicate is sorted by prediction once and that order is shared:
    the LOESS fit and ECCE skip their own sorts, ``ece``/``ece_debiased``/
    ``mce`` share one 15-bin equal-mass binning pass, ``ece_sweep``'s
    ~99-candidate scan reads per-bin sums off prefix-sum differences at
    ``searchsorted`` cut positions, and the LOESS anchor fits are solved in
    vectorized blocks rather than one Python iteration per anchor. The
    reported *point* estimates are computed on the unsorted, scalar path and
    are bit-for-bit what 0.2.x produced; only the replicates take the fast
    path, whose reordered sums move percentile CI bounds in their last bits
    (measured <= 4e-11 relative) and whose tricube weight cubes by
    multiplication rather than ``** 3`` (<= 2.3e-16 relative on a
    well-conditioned window; on a rank-deficient one the
    ``abs(det) < _FPMIN`` guard in the local-linear solve can select a
    different branch than the scalar loop, where the ``swy / sw`` branch is
    the well-defined answer — see ``_math._loess_fit_sorted_vec``. Anchors
    are data quantiles, so this has not been observed to reach a reported
    value). On the dev
    host at n=1e4 a full-catalog replicate costs 0.089s — 58% of it the ICI
    family's LOESS fit, 27% the ``ece_sweep`` scan, 10% intercept/slope,
    0.5% the whole binned ECE family — and the full run
    (``n_boot=1000``) takes 87s against 304s in 0.2.x. Excluding the ICI
    family via ``metrics=`` remains the single largest lever on cost. See
    ``docs/concepts/metrics.md`` for the measured table.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import evaluate
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p).astype(float)
    >>> segment = np.where(p < 0.2, "low", "high")
    >>> grouped = evaluate(y, p, n_boot=50, metrics=("brier",), by=segment)
    >>> grouped.groups
    ('high', 'low')
    >>> len(grouped.reports) == len(grouped.groups)
    True
    """
    if by is not None:
        return _evaluate_grouped(
            y,
            p,
            by,
            sample_weight=sample_weight,
            n_boot=n_boot,
            seed=seed,
            metrics=metrics,
            stratify=stratify,
        )

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
        # One stable sort per replicate, shared by every metric that would
        # otherwise sort (or re-bin) on its own; see ``_point_metrics``.
        idx = idx[np.argsort(p_arr[idx], kind="stable")]
        yb, pb, wb = y_arr[idx], p_arr[idx], w_arr[idx]
        pm = _point_metrics(yb, pb, wb, names, presorted=True)
        boot[b] = [pm[k] for k in names]
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    return MetricReport(names=names, values=values, ci_low=ci_low, ci_high=ci_high)


def _evaluate_grouped(
    y: object,
    p: object,
    by: object,
    *,
    sample_weight: object,
    n_boot: int,
    seed: int,
    metrics: Sequence[str] | None,
    stratify: bool,
) -> GroupedMetricReport:
    """``evaluate(..., by=...)``: a pooled report plus one report per sorted group."""
    y_len = len(np.asarray(y))
    by_arr = np.asarray(by)
    if len(by_arr) != y_len:
        raise ValueError(f"by must have the same length as y ({y_len}), got {len(by_arr)}")
    labels = np.array([str(g) for g in by_arr])
    groups = tuple(sorted(set(labels.tolist())))

    pooled = cast(
        MetricReport,
        evaluate(
            y,
            p,
            sample_weight=sample_weight,
            n_boot=n_boot,
            seed=seed,
            metrics=metrics,
            stratify=stratify,
        ),
    )

    p_arr = np.asarray(p)
    y_arr = np.asarray(y)
    w_full = None if sample_weight is None else np.asarray(sample_weight)
    reports = []
    counts = []
    for i, g in enumerate(groups):
        mask = labels == g
        counts.append(int(mask.sum()))
        sw = None if w_full is None else w_full[mask]
        try:
            rep = cast(
                MetricReport,
                evaluate(
                    y_arr[mask],
                    p_arr[mask],
                    sample_weight=sw,
                    n_boot=n_boot,
                    seed=seed + 1000 * i,
                    metrics=metrics,
                    stratify=stratify,
                ),
            )
        except ValueError as exc:
            raise ValueError(f"group {g!r}: {exc}") from exc
        reports.append(rep)

    return GroupedMetricReport(
        pooled=pooled,
        groups=groups,
        reports=tuple(reports),
        counts=np.array(counts),
    )


@dataclass(frozen=True)
class ReliabilitySummary:
    """Stats-box aggregate for the annotated reliability diagram.

    Attributes
    ----------
    n : int
        Observation count.
    events : int
        Event count (``sum(y)``).
    intercept : float
        Calibration-in-the-large intercept (log-odds).
    slope : float
        Cox calibration slope.
    ici : float
        Integrated calibration index.
    e90 : float
        90th percentile of the LOESS distances.
    spiegelhalter_p : float
        Spiegelhalter test p-value.
    """

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

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size for the ICI/E90 terms; ``None`` recovers
        0.1.2 values exactly.

    Returns
    -------
    ReliabilitySummary
        Stats-box fields for the annotated reliability diagram.
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
