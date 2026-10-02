"""``evaluate``: the metric catalog with seeded bootstrap confidence intervals."""

from collections.abc import Callable, Sequence
from typing import overload

import numpy as np

from .._results import GroupedMetricReport, MetricReport
from ._common import _prep
from ._deprecation import UNSET, renamed_kwarg
from .binned import _ece_family, ece_sweep
from .regression import calibration_intercept, calibration_slope
from .scores import brier_score, brier_skill_score, log_loss
from .smooth import _ecce_stats, _ecce_walk, _ici_family, smooth_ece, spiegelhalter_z

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
    *,
    presorted: bool = False,
) -> dict[str, float]:
    """Point estimates for ``names``; ``presorted`` is the bootstrap fast path.

    ``presorted=True`` declares that ``p`` is already sorted ascending, which
    lets the two sort-heavy consumers -- the ICI family's LOESS fit and the
    ECCE walk -- skip their own sorts. It is only ever passed by
    :func:`evaluate`'s replicate loop, which sorts each replicate once; the
    reported *point* estimates always come off the default path. ``ece``,
    ``ece_debiased`` and ``mce`` share one 15-bin binning pass on both paths
    (bit-identical to the public calls); every other metric runs its public
    code path.
    """
    sel = set(_METRIC_CATALOG if names is None else names)
    dispatch: dict[str, Callable[[], float]] = {
        "log_loss": lambda: log_loss(y, p, sample_weight=w),
        "brier": lambda: brier_score(y, p, sample_weight=w),
        "brier_skill": lambda: brier_skill_score(y, p, sample_weight=w),
        "ece_sweep": lambda: ece_sweep(y, p, sample_weight=w),
        "smooth_ece": lambda: smooth_ece(y, p, sample_weight=w),
        "intercept": lambda: calibration_intercept(y, p, sample_weight=w),
        "slope": lambda: calibration_slope(y, p, sample_weight=w),
    }
    out: dict[str, float] = {k: fn() for k, fn in dispatch.items() if k in sel}
    if sel & {"ece", "ece_debiased", "mce"}:  # one shared 15-bin pass
        yv, pv, wv = _prep(y, p, w)
        out |= _ece_family(yv, pv, wv, sel)

    if sel & {"ecce_max", "ecce_mean"}:
        w_arr = np.ones(len(p)) if w is None else w
        ec = _ecce_stats(_ecce_walk(y, p, w_arr, presorted=presorted))
        out["ecce_max"] = ec.stat_max
        out["ecce_mean"] = ec.stat_mean

    ici_names = sel & {"ici", "e50", "e90", "emax"}
    if ici_names:  # one shared LOESS fit powers the whole ICI family
        out |= _ici_family(y, p, w, ici_names, presorted=presorted)

    if sel & {"spiegelhalter_z", "spiegelhalter_p"}:
        sp = spiegelhalter_z(y, p, sample_weight=w)
        out["spiegelhalter_z"] = sp.z
        out["spiegelhalter_p"] = sp.p_value

    return {k: out[k] for k in _METRIC_CATALOG if k in sel}


def _validate_n_boot(n_boot: object) -> int:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or int(n_boot) < 0:
        raise ValueError(f"n_boot must be a non-negative integer, got {n_boot!r}")
    return int(n_boot)


@overload
def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    random_state: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
    by: None = None,
    seed: object = UNSET,
) -> MetricReport: ...


@overload
def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    random_state: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
    by: object,
    seed: object = UNSET,
) -> GroupedMetricReport: ...


def evaluate(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    n_boot: int = 1000,
    random_state: int = 42,
    metrics: Sequence[str] | None = None,
    stratify: bool = True,
    by: object = None,
    seed: object = UNSET,
) -> MetricReport | GroupedMetricReport:
    """Full metric report with seeded bootstrap percentile confidence intervals.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None
        Positive observation weights (resampled together with the
        observations); see the weight convention in
        :mod:`probcal.metrics._common`.
    n_boot : int
        Case-resampling bootstrap replicates (percentile CIs at 2.5/97.5).
        ``0`` skips the bootstrap: point estimates only, CI bounds ``nan``.
    random_state : int
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
        rows are grouped by their *raw* label values (``1`` and ``"1"`` are
        different groups) and a separate report is computed per group, in
        the order of the groups' display names ``str(label)`` — group ``i``
        in that order is evaluated with ``random_state + 1000 * i``, a fixed
        offset so results are reproducible independent of the label values
        or how many groups exist — plus a pooled report on the full data
        using ``random_state`` unchanged. Returns a
        :class:`~probcal._results.GroupedMetricReport` instead of a plain
        report. Group-conditional statistical *testing* (formal
        multiplicity-adjusted comparisons across groups) is out of scope
        here; see ``docs/guide/groups.md``.
    seed : int, keyword-only
        Deprecated spelling of ``random_state`` (removed in 0.5.0).

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
        ``n_boot`` is negative; if ``by`` is given with a length that does
        not match ``y``, or two distinct ``by`` values share a display name;
        or if a group has only one outcome class (the underlying
        ``"y must contain both classes"`` error, re-raised naming the
        group).
    RuntimeError
        If ``stratify=False`` and 100 consecutive bootstrap draws are all
        single-class.

    Notes
    -----
    Every metric is recomputed ``n_boot`` times; for n > 1e6 reduce
    ``n_boot`` or pass a ``metrics=`` subset (the ICI family's LOESS fit is
    the largest single cost). With ``by`` given, the cost is paid once per
    group plus once for the pooled report. Each replicate is sorted by
    prediction once, and the LOESS fit and the ECCE walk reuse that order;
    the point estimates always come off the unsorted path. The measured cost
    table lives in ``docs/concepts/metrics.md``.

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
    rs = int(renamed_kwarg("evaluate", "seed", "random_state", seed, random_state, 42))  # type: ignore[call-overload]
    n_boot = _validate_n_boot(n_boot)
    if by is not None:
        return _evaluate_grouped(
            y,
            p,
            by,
            sample_weight=sample_weight,
            n_boot=n_boot,
            random_state=rs,
            metrics=metrics,
            stratify=stratify,
        )
    return _evaluate(y, p, sample_weight, n_boot, rs, metrics, stratify)


def _evaluate(
    y: object,
    p: object,
    sample_weight: object,
    n_boot: int,
    random_state: int,
    metrics: Sequence[str] | None,
    stratify: bool,
) -> MetricReport:
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
    if n_boot == 0:
        nan = np.full(len(names), np.nan)
        return MetricReport(names=names, values=values, ci_low=nan, ci_high=nan.copy())

    rng = np.random.default_rng(random_state)
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
        # One stable sort per replicate, shared by the LOESS fit and the ECCE walk.
        idx = idx[np.argsort(p_arr[idx], kind="stable")]
        pm = _point_metrics(y_arr[idx], p_arr[idx], w_arr[idx], names, presorted=True)
        boot[b] = [pm[k] for k in names]
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    return MetricReport(names=names, values=values, ci_low=ci_low, ci_high=ci_high)


def _group_rows(by: object, n: int) -> tuple[tuple[str, ...], list[np.ndarray]]:
    """Display names and row indices per group, grouped by raw label value.

    Groups are ordered by display name ``str(label)``; two distinct raw values
    with the same display name (``1`` and ``"1"``) raise.
    """
    if isinstance(by, np.ndarray) and by.dtype != object:
        by_arr = by
    else:
        seq = list(by)  # type: ignore[call-overload]
        by_arr = np.empty(len(seq), dtype=object)
        by_arr[:] = seq
    if by_arr.ndim != 1 or len(by_arr) != n:
        raise ValueError(f"by must have the same length as y ({n}), got {len(by_arr)}")
    if by_arr.dtype != object:
        uniq, inv = np.unique(by_arr, return_inverse=True)
        values = list(uniq)
    else:
        first: dict[object, int] = {}
        inv = np.empty(n, dtype=np.intp)
        for i, v in enumerate(by_arr):
            inv[i] = first.setdefault(v, len(first))
        values = list(first)
    names = [str(v) for v in values]
    seen: dict[str, object] = {}
    for v, name in zip(values, names, strict=True):
        if name in seen:
            raise ValueError(
                f"by: distinct group values {seen[name]!r} and {v!r} both display as "
                f"{name!r}; convert the labels to one type first"
            )
        seen[name] = v
    order = sorted(range(len(values)), key=names.__getitem__)
    rows = [np.flatnonzero(inv == j) for j in order]
    return tuple(names[j] for j in order), rows


def _evaluate_grouped(
    y: object,
    p: object,
    by: object,
    *,
    sample_weight: object,
    n_boot: int,
    random_state: int,
    metrics: Sequence[str] | None,
    stratify: bool,
) -> GroupedMetricReport:
    """``evaluate(..., by=...)``: a pooled report plus one report per group."""
    y_arr = np.asarray(y)
    groups, rows = _group_rows(by, len(y_arr))
    pooled = _evaluate(y, p, sample_weight, n_boot, random_state, metrics, stratify)

    p_arr = np.asarray(p)
    w_full = None if sample_weight is None else np.asarray(sample_weight)
    reports = []
    for i, (g, r) in enumerate(zip(groups, rows, strict=True)):
        sw = None if w_full is None else w_full[r]
        try:
            rep = _evaluate(
                y_arr[r], p_arr[r], sw, n_boot, random_state + 1000 * i, metrics, stratify
            )
        except ValueError as exc:
            raise ValueError(f"group {g!r}: {exc}") from exc
        reports.append(rep)

    return GroupedMetricReport(
        pooled=pooled,
        groups=groups,
        reports=tuple(reports),
        counts=np.array([len(r) for r in rows]),
    )
