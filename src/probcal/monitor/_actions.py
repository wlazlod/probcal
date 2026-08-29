"""Margin-of-conservatism (MoC) offsets derived from monitoring evidence.

Also the intended home of M4's ``AppliedAction`` (not part of this task).
Theory: ``docs/concepts/monitoring.md``.
"""

from collections.abc import Sequence

import numpy as np

from .._math import beta_ppf, expit
from ..metrics.scores import _prep
from ..offset import LogitOffset
from ._monitor import CalibrationMonitor, MonitorReport, MonitorStep


def moc_offset(
    monitor_or_report: "CalibrationMonitor | MonitorReport", *, level: float | None = None
) -> LogitOffset:
    """Margin-of-conservatism offset from a monitor's confidence sequence.

    ``CalibrationMonitor`` maintains, at every batch, a time-uniform
    confidence sequence (CS) for the current offset: the set of shifts
    ``delta`` such that ``sigma(z + delta)`` -- applying that shift to the
    monitored logits -- would itself be calibrated is covered with
    probability ``>= 1 - alpha`` *simultaneously at every stopping time*
    (``MonitorStep.delta_ci``, the surviving grid nulls' hull; ``None`` if
    every grid null has been rejected). Its upper end, ``hi``, is a
    margin-of-conservatism offset: applying ``delta=hi`` shifts the
    portfolio at least as far as the CS says drift plausibly runs, so
    (loosely) it corrects for the drift with high confidence rather than
    only for its point estimate.

    Two ways to get ``hi``:

    - ``level=None`` (default): take ``hi`` from ``steps[-1].delta_ci``
      as-is, at the monitor's own ``alpha``.
    - ``level`` given: recompute the surviving grid nulls at that
      confidence level directly from the monitor's own running state
      (``mon._cs_grid[mon._cs_max < -log(1 - level)]``) and take their
      max. This needs the live monitor object (its ``_cs_grid``/``_cs_max``
      arrays), not a frozen :class:`~probcal.monitor.MonitorReport`
      snapshot, so it raises ``TypeError`` for a report.

    The returned :class:`~probcal.offset.LogitOffset` is fit on the last
    monitored batch's probabilities (``expit(mon._z[-1])``), which fixes
    its ``pre_mean_``/``post_mean_`` audit fields and its data fingerprint
    to that batch. A :class:`~probcal.monitor.MonitorReport` retains no
    batch data at all, so in that case the offset is fit on the
    placeholder ``np.array([0.5])`` instead -- ``delta_`` is exact either
    way, but ``pre_mean_``/``post_mean_`` and the fingerprint are then
    placeholders, not a real portfolio's summary.

    Parameters
    ----------
    monitor_or_report : CalibrationMonitor or MonitorReport
        The monitor (or its report) to read the confidence sequence from.
    level : float or None, keyword-only
        Confidence level in ``(0, 1)`` to recompute the surviving grid
        nulls at; ``None`` (default) uses the last step's ``delta_ci`` at
        the monitor's own ``alpha``. Requires a ``CalibrationMonitor``.

    Returns
    -------
    LogitOffset
        Fitted offset with ``delta_`` equal to the confidence sequence's
        upper end.

    Raises
    ------
    ValueError
        If no batches have been processed yet, or the surviving grid-null
        set is empty (every null rejected) -- widen ``delta_ci_grid``.
    TypeError
        If ``level`` is given but ``monitor_or_report`` is a
        ``MonitorReport`` rather than a live ``CalibrationMonitor``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.datasets import make_pd_portfolio
    >>> from probcal.monitor import CalibrationMonitor, moc_offset
    >>> mon = CalibrationMonitor(alpha=0.05)
    >>> for seed in range(3):
    ...     d = make_pd_portfolio(n=500, random_state=seed)
    ...     rng = np.random.default_rng(seed)
    ...     y = (rng.random(500) < d.scores).astype(float)  # drift injected
    ...     _ = mon.update(y, d.scores, label=f"b{seed}")
    >>> off = moc_offset(mon)
    >>> off.delta_ >= mon.steps_[-1].delta_hat
    True
    """
    if not isinstance(monitor_or_report, (CalibrationMonitor, MonitorReport)):
        raise TypeError(
            "moc_offset requires a CalibrationMonitor or a MonitorReport, got "
            f"{type(monitor_or_report).__name__}"
        )
    mon = monitor_or_report if isinstance(monitor_or_report, CalibrationMonitor) else None

    steps: Sequence[MonitorStep]
    if isinstance(monitor_or_report, CalibrationMonitor):
        steps = monitor_or_report.steps_
    else:
        steps = monitor_or_report.steps
    if not steps:
        raise ValueError("moc_offset: no batches have been processed yet")

    if level is None:
        delta_ci = steps[-1].delta_ci
        if delta_ci is None:
            raise ValueError(
                "moc_offset: every grid null in delta_ci is rejected (delta_ci is "
                "None); widen delta_ci_grid to include the true offset"
            )
        hi = delta_ci[1]
    else:
        if mon is None:
            raise TypeError(
                "moc_offset: level requires a live CalibrationMonitor -- recomputing "
                "the surviving grid nulls at a new confidence level reads the "
                "monitor's running _cs_grid/_cs_max arrays, which a MonitorReport "
                "(a frozen snapshot) does not retain; pass the monitor itself, or "
                "omit level to use its last step's delta_ci as-is"
            )
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        threshold = -np.log(1.0 - level)
        surviving = mon._cs_grid[mon._cs_max < threshold]
        if surviving.size == 0:
            raise ValueError("moc_offset: no grid nulls survive at this level; widen delta_ci_grid")
        hi = float(surviving.max())

    batch_p = expit(mon._z[-1]) if mon is not None else np.array([0.5])
    return LogitOffset(delta=hi).fit(batch_p)


def moc_offset_from_counts(
    y: object,
    p: object,
    *,
    level: float = 0.9,
    sample_weight: object = None,
) -> LogitOffset:
    """Margin-of-conservatism offset from raw event counts (mode B, no monitor).

    The Jeffreys posterior upper bound on the observed event rate,
    ``q = beta_ppf(level, k + 0.5, n - k + 0.5)`` with ``k = sum(w * y)``
    and ``n = sum(w)`` -- the same one-sided Jeffreys quantile
    ``metrics.jeffreys_grade_test``/``metrics.jeffreys_upper_bands`` use --
    becomes the offset's target mean: ``LogitOffset(target_mean=q)`` (mode
    B) solves for the log-odds shift that re-anchors ``p``'s mean at
    ``q``, a conservative re-anchoring against the observed outcomes
    rather than a shift read off a monitor's confidence sequence.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]`` to be re-anchored.
    level : float, keyword-only
        Confidence level in ``(0, 1)`` for the Jeffreys upper quantile.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    LogitOffset
        Fitted offset with ``post_mean_`` equal to the Jeffreys upper
        quantile ``q``.

    Raises
    ------
    ValueError
        If ``y``/``p`` are invalid, or ``level`` is not in ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.monitor import moc_offset_from_counts
    >>> y = np.array([0.0] * 970 + [1.0] * 30)
    >>> p = np.full(1000, 0.02)
    >>> off = moc_offset_from_counts(y, p, level=0.9)
    >>> off.post_mean_ > 0.03
    True
    """
    y_arr, p_arr, w_arr = _prep(y, p, sample_weight)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    k = float(np.sum(w_arr * y_arr))
    n = float(np.sum(w_arr))
    q = beta_ppf(level, k + 0.5, n - k + 0.5)
    return LogitOffset(target_mean=q).fit(p_arr, sample_weight=w_arr)
