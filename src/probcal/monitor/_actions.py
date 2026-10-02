"""Margin-of-conservatism (MoC) offsets, ``apply_recommendation``, ``AppliedAction``.

Theory: ``docs/concepts/monitoring.md``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._math import beta_ppf, expit
from .._registry import load, register
from .._serialize import JsonIO, check_payload, envelope
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
    >>> from probcal import expit, logit
    >>> for seed in range(3):
    ...     d = make_pd_portfolio(n=500, random_state=seed)
    ...     rng = np.random.default_rng(seed)
    ...     true_pd = expit(logit(d.scores) + 0.3)  # +0.3 log-odds drift injected
    ...     y = (rng.random(500) < true_pd).astype(float)
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
        surviving = mon._cs.surviving(-np.log(1.0 - level))
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


def _audit(
    rep: MonitorReport,
    old_fp: str,
    old_target_fp: str | None,
    *,
    new_fp: str | None = None,
    offset_fp: str | None = None,
    new_target_fp: str | None = None,
    delta: float | None = None,
    se: float | None = None,
) -> dict[str, Any]:
    """The audit trail; defaults describe "nothing changed"."""
    return {
        "alarm_at": rep.alarm_at,
        "onset_label": rep.onset_label,
        "old_monitor_fingerprint": old_fp,
        "new_monitor_fingerprint": new_fp if new_fp is not None else old_fp,
        "offset_fingerprint": offset_fp,
        "old_target_fingerprint": old_target_fp,
        "new_target_fingerprint": new_target_fp if new_target_fp is not None else old_target_fp,
        "delta": delta,
        "se": se,
    }


def apply_recommendation(mon: CalibrationMonitor, target: object = None) -> "AppliedAction":
    """Implementation of :meth:`CalibrationMonitor.apply_recommendation` (see there)."""
    from ..chain import Chain
    from ..offset import estimate_offset
    from ..wrapper import CalibratedModel

    if target is not None and not isinstance(target, (Chain, CalibratedModel)):
        raise TypeError(
            f"target must be None, a Chain, or a CalibratedModel, got {type(target).__name__}"
        )
    old_target_fp = target.fingerprint() if target is not None else None
    rep = mon.report()
    kind = rep.recommendation
    old_fp = mon.fingerprint()
    if kind == "none":
        return AppliedAction(
            kind=kind,
            offset=None,
            composed=None,
            monitor=None,
            window=(),
            audit=_audit(rep, old_fp, old_target_fp),
        )

    # By index, not label: labels are opaque and may repeat (_onset_index).
    start = mon._recommendation_window_start(mon._onset_index())
    labels = tuple(s.label for s in mon.steps_[start:])
    if kind == "re-fit":
        return AppliedAction(
            kind=kind,
            offset=None,
            composed=None,
            monitor=None,
            window=labels,
            audit=_audit(rep, old_fp, old_target_fp),
        )

    # kind == "re-offset"
    z_w, y_w, w_w = mon._since(start)
    p_w = expit(z_w)
    est = estimate_offset(y_w, p_w, sample_weight=w_w)
    offset = LogitOffset(delta=est.delta).fit(p_w)
    composed: Any = None
    if isinstance(target, Chain):
        composed = Chain([target.calibrator_, *target.offsets_, offset])
    elif isinstance(target, CalibratedModel):
        composed = target.with_offset(offset)
    fresh = type(mon)(**mon._ctor_params())
    return AppliedAction(
        kind=kind,
        offset=offset,
        composed=composed,
        monitor=fresh,
        window=labels,
        audit=_audit(
            rep,
            old_fp,
            old_target_fp,
            new_fp=fresh.fingerprint(),
            offset_fp=offset.fingerprint(),
            new_target_fp=composed.fingerprint() if composed is not None else None,
            delta=float(est.delta),
            se=float(est.se),
        ),
    )


@register
@dataclass(frozen=True)
class AppliedAction(JsonIO):
    """The result of :meth:`CalibrationMonitor.apply_recommendation`.

    Attributes
    ----------
    kind : {"re-offset", "re-fit", "none"}
        The recommendation :meth:`CalibrationMonitor.report` produced.
    offset : LogitOffset or None
        The fitted correction; only for ``kind="re-offset"``.
    composed : object or None
        ``offset`` applied to the caller's ``target`` (a
        :class:`~probcal.chain.Chain` or a
        :class:`~probcal.wrapper.CalibratedModel`) -- ``None`` when no
        ``target`` was given or ``kind != "re-offset"``.
    monitor : CalibrationMonitor or None
        A fresh monitor with the same constructor parameters, ready to
        watch the corrected pipeline; only for ``kind="re-offset"`` (see
        the "why fresh" note on ``apply_recommendation``).
    window : tuple[str, ...]
        Batch labels the offset (or the suggested re-fit window) was
        estimated from; empty when ``kind="none"``.
    audit : dict
        Provenance: ``alarm_at``, ``onset_label``, fingerprints of the old
        and new monitor/offset/target (``None`` where not applicable), and
        the estimated ``delta``/``se`` (``None`` unless ``kind="re-offset"``).

    Examples
    --------
    >>> import numpy as np
    >>> from probcal._math import expit, logit
    >>> from probcal.datasets import make_pd_portfolio
    >>> from probcal.monitor import CalibrationMonitor
    >>> mon = CalibrationMonitor(alpha=0.05)
    >>> for k in range(6):
    ...     d = make_pd_portfolio(n=1000, random_state=k)
    ...     rng = np.random.default_rng(k + 1000)
    ...     y = (rng.random(1000) < expit(logit(d.scores) + 0.8)).astype(float)
    ...     _ = mon.update(y, d.scores, label=f"m{k}")
    >>> action = mon.apply_recommendation()
    >>> action.kind
    're-offset'
    >>> action.offset.delta_ > 0
    True
    """

    kind: str
    offset: "LogitOffset | None"
    composed: "object | None"
    monitor: "CalibrationMonitor | None"
    window: tuple[str, ...]
    audit: dict

    # ------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot; ``offset``/``composed``/``monitor`` are nested envelopes.

        Each nested field is stored via its own ``to_dict`` (``None`` stays
        ``None``): a ``CalibratedModel`` composed target stores only a
        model *reference*, reattached on load via
        ``AppliedAction.from_dict(d, model=...)`` -- see
        ``CalibratedModel.to_dict``.
        """
        return envelope(
            self,
            params={},
            state={
                "kind": self.kind,
                "offset": self.offset.to_dict() if self.offset is not None else None,
                "composed": (
                    self.composed.to_dict()  # type: ignore[attr-defined]
                    if self.composed is not None
                    else None
                ),
                "monitor": self.monitor.to_dict() if self.monitor is not None else None,
                "window": list(self.window),
                "audit": dict(self.audit),
            },
            fit_meta={},
        )

    @classmethod
    def from_dict(cls, d: dict, *, model: object = None) -> "AppliedAction":
        """Rebuild from :meth:`to_dict` output (``from_json`` forwards ``model=``).

        Parameters
        ----------
        d : dict
            Output of :meth:`to_dict`.
        model : object or None, keyword-only
            Passed through to ``CalibratedModel.from_dict`` when
            ``composed`` was a ``CalibratedModel`` (only a reference to the
            base model is serialized, never the model itself).

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_payload(cls, d)
        st = d["state"]
        offset = LogitOffset.from_dict(st["offset"]) if st["offset"] is not None else None
        composed: object | None = None
        if st["composed"] is not None:
            if st["composed"].get("class") == "CalibratedModel":
                from ..wrapper import CalibratedModel

                composed = CalibratedModel.from_dict(st["composed"], model=model)
            else:
                composed = load(st["composed"])
        monitor = CalibrationMonitor.from_dict(st["monitor"]) if st["monitor"] is not None else None
        return cls(
            kind=st["kind"],
            offset=offset,
            composed=composed,
            monitor=monitor,
            window=tuple(st["window"]),
            audit=dict(st["audit"]),
        )
