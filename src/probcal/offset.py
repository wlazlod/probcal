"""Logit-offset (central tendency) adjustment with audit trail.

Theory — including the King–Zeng / Elkan / Tasche equivalences and the
uniqueness of the mode-B root: ``docs/concepts/offset.md``.

References
----------
King & Zeng (2001); Elkan (2001); Tasche (2013) — full records in the
documentation.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

import numpy as np

from ._math import bisect, expit, logit
from ._results import Interpretation
from ._validation import validate_scores, validate_weights
from .metrics.regression import GuardrailReport, calibration_guardrails

_DELTA_BRACKET = 40.0


@dataclass(frozen=True)
class AuditReport:
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


class LogitOffset:
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
    """

    is_monotone_: bool = True
    fitted_: bool = False

    def __init__(self, delta: float | None = None, target_mean: float | None = None) -> None:
        self.delta = delta
        self.target_mean = target_mean

    def fit(self, p: object, sample_weight: object = None) -> Self:
        """Fix ``delta`` (mode A) or solve it against the target mean (mode B).

        Parameters
        ----------
        p : array_like
            Current calibrated probabilities of the portfolio.
        sample_weight : array_like or None
            Weights for the portfolio mean.

        Returns
        -------
        Self
            The fitted offset.
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
            target = float(self.target_mean)  # type: ignore[arg-type]
            if not 0.0 < target < 1.0:
                raise ValueError("target_mean must lie in (0, 1)")

            def gap(d: float) -> float:
                return float(np.average(expit(z + d), weights=w)) - target

            self.delta_ = bisect(gap, -_DELTA_BRACKET, _DELTA_BRACKET, tol=1e-14)
        self.post_mean_ = float(np.average(expit(z + self.delta_), weights=w))
        self.timestamp_ = datetime.now(UTC).isoformat(timespec="seconds")
        self.fitted_ = True
        return self

    def transform(self, p: object) -> np.ndarray:
        """Apply the fitted shift to probabilities."""
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        return expit(logit(validate_scores(p, name="p")) + self.delta_)

    predict_proba = transform

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float]:
        """``(1, delta)``: the offset is affine on the logit scale."""
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        return (1.0, self.delta_)

    def interpret(self) -> Interpretation:
        """Read delta in log-odds, odds-factor, and central-tendency terms."""
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        return Interpretation(
            method=type(self).__name__,
            param_names=("delta",),
            param_values=(self.delta_,),
            messages=(
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
            before inverting.

        Returns
        -------
        tuple of float
            ``(raw_lo, raw_hi)`` preimage bounds, on the scale requested by
            ``space``.

        Raises
        ------
        UnattainableTargetError
            If a crossed ``buffer_logit`` empties the calibrated interval.
        ValueError
            If ``lo``, ``hi`` are not ordered in ``[0, 1]``.
        """
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"need 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")
        from .base import UnattainableTargetError

        lo_b, hi_b = float(lo), float(hi)
        if buffer_logit > 0.0:
            if lo > 0.0:
                lo_b = float(expit(np.array([logit(np.array([lo]))[0] + buffer_logit]))[0])
            if hi < 1.0:
                hi_b = float(expit(np.array([logit(np.array([hi]))[0] - buffer_logit]))[0])
            if lo_b > hi_b:
                raise UnattainableTargetError(
                    f"buffer_logit={buffer_logit} empties the calibrated interval [{lo}, {hi}]"
                )
        lo_z = -np.inf if lo_b <= 0.0 else float(logit(np.array([lo_b]))[0]) - self.delta_
        hi_z = np.inf if hi_b >= 1.0 else float(logit(np.array([hi_b]))[0]) - self.delta_
        if space == "logit":
            return lo_z, hi_z
        raw_lo = 0.0 if np.isneginf(lo_z) else float(expit(np.array([lo_z]))[0])
        raw_hi = 1.0 if np.isposinf(hi_z) else float(expit(np.array([hi_z]))[0])
        return raw_lo, raw_hi

    def audit_report(self, y: object, p: object, *, sample_weight: object = None) -> AuditReport:
        """Pre/post guardrail comparison for the validator's one-table view."""
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
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
