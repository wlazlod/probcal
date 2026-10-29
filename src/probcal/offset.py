"""Logit-offset (central tendency) adjustment with audit trail.

Theory — including the King–Zeng / Elkan / Tasche equivalences and the
uniqueness of the mode-B root: ``docs/concepts/offset.md``.

References
----------
King & Zeng (2001); Elkan (2001); Tasche (2013) — full records in the
documentation.
"""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np

from ._math import _LOGIT_CLIP, bisect, expit, expit1, logit, logit1
from ._registry import register
from ._results import Interpretation, OffsetEstimate, _ResultBase
from ._serialize import fit_meta, utc_now
from ._validation import validate_level, validate_scores, validate_weights
from .base import (
    UnattainableTargetError,
    _FittedObject,
    check_intersects,
    prepare_interval,
)
from .metrics._common import _prep
from .metrics.regression import GuardrailReport, calibration_guardrails

_DELTA_BRACKET = 40.0


@dataclass(frozen=True)
class AuditReport(_ResultBase):
    """Pre/post record of a logit-offset application, for validators.

    Attributes
    ----------
    delta : float
        Applied log-odds shift.
    pre_mean, post_mean : float
        Portfolio mean probability before and after the shift.
    timestamp : str
        ISO-8601 UTC time at which the offset was fitted.
    guardrails_before, guardrails_after : GuardrailReport
        The three-flag calibration health summary on the input and output
        probabilities.
    """

    delta: float
    pre_mean: float
    post_mean: float
    timestamp: str
    guardrails_before: GuardrailReport
    guardrails_after: GuardrailReport

    def __repr__(self) -> str:
        gb, ga = self.guardrails_before, self.guardrails_after
        lines = [
            f"AuditReport(delta={self.delta:+.4f}, odds factor {np.exp(self.delta):.4f}, "
            f"fitted {self.timestamp})",
            f"  portfolio mean: {self.pre_mean:.5f} -> {self.post_mean:.5f}",
            f"  slope:          {gb.slope:+.3f} -> {ga.slope:+.3f}",
            f"  intercept:      {gb.intercept:+.3f} -> {ga.intercept:+.3f}",
            f"  spiegelhalter p {gb.spiegelhalter_p:.3f} -> {ga.spiegelhalter_p:.3f}",
            f"  guardrails ok:  {gb.all_ok} -> {ga.all_ok}",
        ]
        return "\n".join(lines)


def _solve_delta(z: np.ndarray, w: np.ndarray, target: float) -> float:
    """The ``delta`` with ``mean_w(sigma(z + delta)) == target``, by bisection.

    The weighted mean is strictly increasing in ``delta``, so the root is
    unique. Shared by :meth:`LogitOffset.fit` (mode B), :func:`_offset_mle`
    and ``probcal.monitor._processes.plug_in_delta``, so they stay
    bit-identical.

    Raises
    ------
    UnattainableTargetError
        If ``target`` lies outside the means attainable for ``delta`` in
        ``[-40, 40]`` (e.g. a target of 0.5 for a portfolio whose scores
        all sit at the ``1e-12`` clip); the message names that range.
    """

    def gap(d: float) -> float:
        return float(np.average(expit(z + d), weights=w)) - target

    g_lo, g_hi = gap(-_DELTA_BRACKET), gap(_DELTA_BRACKET)
    if g_lo > 0.0 or g_hi < 0.0:
        raise UnattainableTargetError(
            f"target mean {target:.6g} is unattainable by a logit offset: shifts in "
            f"[-{_DELTA_BRACKET:g}, {_DELTA_BRACKET:g}] reach portfolio means "
            f"[{g_lo + target:.6g}, {g_hi + target:.6g}] only"
        )
    return bisect(gap, -_DELTA_BRACKET, _DELTA_BRACKET, tol=1e-14)


@register
class LogitOffset(_FittedObject):
    """Uniform log-odds shift: ``p' = sigma(logit(p) + delta)``.

    Mode A takes ``delta`` explicitly; mode B takes ``target_mean`` and
    solves ``mean(p') = target_mean`` for ``delta`` by bisection — the
    portfolio mean is strictly increasing in ``delta``, so the root is
    unique (stated in the offset chapter and unit-tested). Exactly one of
    the two arguments must be given.

    The offset is deliberately *not* folded into any calibrator's
    parameters: ``CalibratedModel.offset_to`` appends it as a separate,
    inspectable pipeline stage.

    Parameters
    ----------
    delta : float or None
        Mode A: the log-odds shift to apply directly. Mutually exclusive
        with ``target_mean`` — :meth:`fit` requires exactly one of the two.
    target_mean : float or None
        Mode B: the desired post-shift portfolio mean probability in
        ``(0, 1)``; ``delta`` is solved by bisection. Mutually exclusive
        with ``delta``.

    Attributes
    ----------
    delta_ : float
        Fitted (or given) shift in log-odds.
    pre_mean_, post_mean_ : float
        Portfolio mean before and after, recorded at fit time.
    timestamp_ : str
        ISO-8601 UTC fit time — part of the audit trail.

    Notes
    -----
    The serialized ``fit_meta`` records ``n_obs``, ``weight_sum``,
    ``fitted_at_utc``, and the ``data_fingerprint`` of the ``(p, w)`` pair —
    no ``n_events`` because the offset is fitted on probabilities alone.
    ``fingerprint`` is blind to the audit ``timestamp_``: identical fits
    fingerprint identically.
    """

    _STATE_ATTRS = ("delta_", "pre_mean_", "post_mean_", "timestamp_")

    def __init__(self, delta: float | None = None, target_mean: float | None = None) -> None:
        self.delta = delta
        self.target_mean = target_mean

    def fit(self, p: object, *, y: object = None, sample_weight: object = None) -> Self:
        """Fix ``delta`` (mode A) or solve it against the target mean (mode B).

        Parameters
        ----------
        p : array_like
            Current calibrated probabilities of the portfolio.
        y : array_like or None, keyword-only
            Ignored; accepted for compatibility with the chain fit protocol.
        sample_weight : array_like or None, keyword-only
            Weights for the portfolio mean.

        Returns
        -------
        Self
            The fitted offset.

        Raises
        ------
        ValueError
            If not exactly one of ``delta``/``target_mean`` is set, or
            ``target_mean`` is outside ``(0, 1)``.
        UnattainableTargetError
            If ``target_mean`` cannot be reached by any shift in
            ``[-40, 40]``; the message names the attainable range.
        """
        if (self.delta is None) == (self.target_mean is None):
            raise ValueError("LogitOffset: give exactly one of delta or target_mean")
        p_arr = validate_scores(p, name="p")
        w = validate_weights(sample_weight, len(p_arr))
        z = logit(p_arr)
        self.pre_mean_ = float(np.average(p_arr, weights=w))
        if self.delta is not None:
            self.delta_ = float(self.delta)
        else:
            self.delta_ = _solve_delta(z, w, validate_level(self.target_mean, "target_mean"))
        self.post_mean_ = float(np.average(expit(z + self.delta_), weights=w))
        self.timestamp_ = utc_now()
        self.fit_meta_ = fit_meta(p_arr, w, fitted_at_utc=self.timestamp_)
        self.fitted_ = True
        return self

    def transform(self, p: object) -> np.ndarray:
        """Apply the fitted shift to probabilities."""
        self._check_fitted()
        return expit(logit(validate_scores(p, name="p")) + self.delta_)

    predict_proba = transform

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float]:
        """``(1, delta)``: the offset is affine on the logit scale."""
        self._check_fitted()
        return (1.0, self.delta_)

    def interpret(self) -> Interpretation:
        """Read delta in log-odds, odds-factor, and central-tendency terms."""
        self._check_fitted()
        return self._interpretation(
            {"delta": self.delta_},
            (
                f"delta = {self.delta_:+.4f} log-odds: every observation's odds are "
                f"multiplied by exp(delta) = {np.exp(self.delta_):.4f} uniformly",
                f"portfolio mean re-anchored from {self.pre_mean_:.5f} to "
                f"{self.post_mean_:.5f} (credit-risk central tendency adjustment)",
                "equivalent to King-Zeng prior correction and to Elkan's base-rate "
                "adjustment (see the offset chapter for the derivations)",
                "ranking is untouched: the shift is strictly increasing",
            ),
        )

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Closed-form preimage: subtract delta on the logit scale.

        Same protocol as ``BaseCalibrator.interval_inverse``; the offset's
        output range is the full unit interval, so only a crossed buffer can
        make a target unattainable.

        Parameters
        ----------
        lo, hi : float
            Calibrated-probability bounds; ``lo=0``/``hi=1`` map to the full
            raw range (−inf/+inf on the logit scale).
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw bounds.
        buffer_logit : float, keyword-only
            Shrink the calibrated interval by this margin in logit space
            before inverting (must be finite and ``>= 0``).

        Returns
        -------
        tuple of float
            ``(raw_lo, raw_hi)`` preimage bounds, on the scale requested by
            ``space``.

        Raises
        ------
        UnattainableTargetError
            If a crossed ``buffer_logit`` empties the calibrated interval,
            or the (buffered) interval does not intersect the offset map's
            representable output range ``[sigma(delta - logit(1 - 1e-12)),
            sigma(delta + logit(1 - 1e-12))]`` — bounds beyond that range
            collapse to the full-range sentinels (0/1, ±inf) instead of raw
            values below the clip that ``transform`` could not round-trip.
        ValueError
            If ``lo``, ``hi`` are not ordered in ``[0, 1]``, ``space`` is
            unknown, or ``buffer_logit`` is negative.
        """
        self._check_fitted()
        lo_b, hi_b = prepare_interval(lo, hi, space, buffer_logit)
        # Representable output range: raw scores are clipped to
        # [1e-12, 1 - 1e-12] by every forward entry point, so the shifted map
        # attains only [sigma(delta - _LOGIT_CLIP), sigma(delta + _LOGIT_CLIP)].
        # Bounds beyond it collapse to the full-range sentinels (0.0/1.0,
        # -inf/+inf) exactly as in BaseCalibrator.interval_inverse; a raw
        # bound below the clip (e.g. 4.5e-14) could not round-trip through
        # transform — the silent break the no-silent-clamp doctrine forbids.
        gmin = expit1(self.delta_ - _LOGIT_CLIP)
        gmax = expit1(self.delta_ + _LOGIT_CLIP)
        check_intersects(lo_b, hi_b, gmin, gmax, what="offset map's representable output range")
        lo_z = -np.inf if lo_b <= gmin else logit1(lo_b) - self.delta_
        hi_z = np.inf if hi_b >= gmax else logit1(hi_b) - self.delta_
        if space == "logit":
            return lo_z, hi_z
        raw_lo = 0.0 if np.isneginf(lo_z) else expit1(lo_z)
        raw_hi = 1.0 if np.isposinf(hi_z) else expit1(hi_z)
        return raw_lo, raw_hi

    def audit_report(self, y: object, p: object, *, sample_weight: object = None) -> AuditReport:
        """Pre/post guardrail comparison for the validator's one-table view."""
        self._check_fitted()
        before = calibration_guardrails(y, p, sample_weight=sample_weight)
        after = calibration_guardrails(y, self.transform(p), sample_weight=sample_weight)
        return AuditReport(
            delta=self.delta_,
            pre_mean=self.pre_mean_,
            post_mean=self.post_mean_,
            timestamp=self.timestamp_,
            guardrails_before=before,
            guardrails_after=after,
        )


def _offset_mle(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Offset-only logistic MLE on already-valid inputs (logits ``z``, 0/1 ``y``, weights ``w``).

    The root of ``sum(w * (y - sigma(z + delta))) = 0``, i.e. the
    ``LogitOffset(target_mean=mean_w(y))`` shift (:func:`_solve_delta`,
    which ``probcal.monitor._processes.plug_in_delta`` solves the same way).
    Callers handle degenerate input.
    """
    return _solve_delta(z, w, float(np.average(y, weights=w)))


def estimate_offset(y: object, p: object, *, sample_weight: object = None) -> OffsetEstimate:
    """Offset-only logistic MLE of ``delta`` given ``p``, with its standard error.

    Fits the single-parameter model ``y ~ Bernoulli(sigma(logit(p) + delta))``
    by maximum likelihood. The score equation
    ``sum(w * (y - sigma(logit(p) + delta))) = 0`` is exactly the mean-matching
    condition solved by ``LogitOffset(target_mean=mean_w(y))``, so ``delta`` is
    found by the same bisection root-finder (:func:`_solve_delta`, also behind
    ``probcal.monitor._processes.plug_in_delta``). Weights are relative
    frequency weights (:mod:`probcal.metrics._common`): ``delta`` and ``se``
    do not change when every weight is multiplied by a constant. With
    ``q = sigma(logit(p) + delta)``, ``I = sum(w * q * (1 - q))`` and
    ``J = sum(w**2 * q * (1 - q))`` (the exact variance of the weighted score
    when each row is one Bernoulli draw), the standard error is
    ``sqrt(J) / I``; for unit weights that is the Fisher ``1 / sqrt(I)``.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``. Both classes must be present.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    OffsetEstimate
        The fitted ``delta``, its standard error, and the fit's ``n``,
        ``events``, and ``weight_sum``.

    Raises
    ------
    ValueError
        If ``y`` or ``p`` fail validation (shape, range, finiteness) or if
        ``y`` contains only one class — the offset MLE does not exist then.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.offset import estimate_offset
    >>> from probcal._math import expit
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(0.0, 1.0, 2000)
    >>> p = expit(z)
    >>> y = (rng.random(2000) < expit(z + 0.5)).astype(float)
    >>> est = estimate_offset(y, p)
    >>> est.n
    2000
    >>> abs(est.delta - 0.5) < 3 * est.se
    True
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    z = logit(p_arr)
    delta = _offset_mle(z, y_arr, w)
    q = expit(z + delta)
    v = w * q * (1.0 - q)
    info = float(np.sum(v))
    # sqrt(J / I) / sqrt(I) == sqrt(J) / I, written so that unit weights
    # (w * v == v, J / I == 1.0 exactly) give the Fisher 1 / sqrt(I) bit for bit.
    se = math.sqrt(float(np.sum(w * v)) / info) / math.sqrt(info)
    return OffsetEstimate(
        delta=delta,
        se=se,
        n=int(len(y_arr)),
        events=float(np.sum(w * y_arr)),
        weight_sum=float(w.sum()),
    )


def offset_from_estimate(est: OffsetEstimate, p: object) -> LogitOffset:
    """Build a fitted :class:`LogitOffset` (mode A) from an :class:`OffsetEstimate`.

    Equivalent to ``LogitOffset(delta=est.delta).fit(p)`` — a convenience for
    turning the audited MLE into the same offset object used elsewhere in
    the package (``transform``, ``interpret``, ``to_dict``, ...).

    Parameters
    ----------
    est : OffsetEstimate
        Result of :func:`estimate_offset`.
    p : array_like
        Probabilities to fit the offset's audit trail (``pre_mean_``,
        ``post_mean_``) against.

    Returns
    -------
    LogitOffset
        Fitted with ``delta = est.delta``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.offset import estimate_offset, offset_from_estimate
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.01, 0.5, 500)
    >>> y = (rng.random(500) < p).astype(float)
    >>> est = estimate_offset(y, p)
    >>> off = offset_from_estimate(est, p)
    >>> off.delta_ == est.delta
    True
    """
    return LogitOffset(delta=est.delta).fit(p)
