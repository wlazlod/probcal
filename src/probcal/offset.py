"""Logit-offset (central tendency) adjustment with audit trail.

Theory — including the King–Zeng / Elkan / Tasche equivalences and the
uniqueness of the mode-B root: ``docs/concepts/offset.md``.

References
----------
King & Zeng (2001); Elkan (2001); Tasche (2013) — full records in the
documentation.
"""

import inspect
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

import numpy as np

from ._math import _LOGIT_CLIP, bisect, expit, logit
from ._registry import register
from ._results import Interpretation, OffsetEstimate
from ._serialize import SCHEMA_VERSION, check_schema, data_fingerprint, fingerprint_of_dict
from ._validation import validate_scores, validate_weights
from .metrics.regression import GuardrailReport, calibration_guardrails
from .metrics.scores import _prep

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


@register
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

    def fit(self, p: object, sample_weight: object = None, *, y: object = None) -> Self:
        """Fix ``delta`` (mode A) or solve it against the target mean (mode B).

        Parameters
        ----------
        p : array_like
            Current calibrated probabilities of the portfolio.
        sample_weight : array_like or None
            Weights for the portfolio mean.
        y : array_like or None, keyword-only
            Ignored; accepted for compatibility with the chain fit protocol.

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
        self.fit_meta_ = {
            "n_obs": int(p_arr.shape[0]),
            "weight_sum": float(w.sum()),
            "fitted_at_utc": self.timestamp_,
            "data_fingerprint": data_fingerprint(p_arr, w),
        }
        self.fitted_ = True
        return self

    def transform(self, p: object) -> np.ndarray:
        """Apply the fitted shift to probabilities."""
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        return expit(logit(validate_scores(p, name="p")) + self.delta_)

    predict_proba = transform

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted state for sklearn >= 1.6 (``delta_`` fixed or solved)."""
        return bool(self.fitted_)

    # ------------------------------------------------------------- parameters
    # Same manual sklearn-compatible convention as BaseCalibrator.get_params
    # / set_params: LogitOffset is not a BaseCalibrator subclass, so it is
    # duplicated here rather than shared.

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Constructor parameters as a dict (manual sklearn-compatible clone info)."""
        sig = inspect.signature(type(self).__init__)
        return {
            name: getattr(self, name)
            for name in sig.parameters
            if name not in ("self", "args", "kwargs")
        }

    def set_params(self, **params: object) -> Self:
        """Set constructor parameters; unknown names raise ``ValueError``."""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"unknown parameter {key!r} for {type(self).__name__}; "
                    f"valid: {sorted(valid)}"
                )
            setattr(self, key, value)
        return self

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
            If a crossed ``buffer_logit`` empties the calibrated interval,
            or the (buffered) interval does not intersect the offset map's
            representable output range ``[sigma(delta - logit(1 - 1e-12)),
            sigma(delta + logit(1 - 1e-12))]`` — bounds beyond that range
            collapse to the full-range sentinels (0/1, ±inf) instead of raw
            values below the clip that ``transform`` could not round-trip.
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
        # Representable output range: raw scores are clipped to
        # [1e-12, 1 - 1e-12] by every forward entry point, so the shifted map
        # attains only [sigma(delta - _LOGIT_CLIP), sigma(delta + _LOGIT_CLIP)].
        # Bounds beyond it collapse to the full-range sentinels (0.0/1.0,
        # -inf/+inf) exactly as in BaseCalibrator.interval_inverse; a raw
        # bound below the clip (e.g. 4.5e-14) could not round-trip through
        # transform — the silent break the no-silent-clamp doctrine forbids.
        gmin = float(expit(np.array([self.delta_ - _LOGIT_CLIP]))[0])
        gmax = float(expit(np.array([self.delta_ + _LOGIT_CLIP]))[0])
        if lo_b > gmax or hi_b < gmin:
            raise UnattainableTargetError(
                f"calibrated target [{lo_b:.6g}, {hi_b:.6g}] does not intersect the "
                f"offset map's representable output range [{gmin:.6g}, {gmax:.6g}]"
            )
        lo_z = -np.inf if lo_b <= gmin else float(logit(np.array([lo_b]))[0]) - self.delta_
        hi_z = np.inf if hi_b >= gmax else float(logit(np.array([hi_b]))[0]) - self.delta_
        if space == "logit":
            return lo_z, hi_z
        raw_lo = 0.0 if np.isneginf(lo_z) else float(expit(np.array([lo_z]))[0])
        raw_hi = 1.0 if np.isposinf(hi_z) else float(expit(np.array([hi_z]))[0])
        return raw_lo, raw_hi

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Raw scores whose shifted probabilities equal ``p`` (exact preimage).

        Closed form: subtract ``delta`` on the logit scale. Same protocol as
        :meth:`BaseCalibrator.point_inverse` — ``LogitOffset`` is not a
        ``BaseCalibrator`` subclass, so the fit-guard and validation are
        duplicated here rather than shared (the existing ``offset.py``
        precedent, e.g. :meth:`interval_inverse`).

        Parameters
        ----------
        p : array_like
            Shifted probabilities strictly inside ``(0, 1)``; boundary and
            out-of-range values raise ``UnattainableTargetError``
            (all-or-nothing, no silent clamp).
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw values.

        Returns
        -------
        numpy.ndarray
            Raw scores (or logits, if ``space="logit"``) whose shifted
            probability equals ``p``.

        Raises
        ------
        RuntimeError
            If not yet fitted.
        ValueError
            If ``space`` is not ``"probability"`` or ``"logit"``.
        UnattainableTargetError
            If any element of ``p`` lies outside the open interval
            ``(0, 1)``; or if ``space="probability"`` and the raw logit of
            any result exceeds ``logit(1 - 1e-12)`` in magnitude — the
            probability representation would round to 0.0/1.0 and silently
            fail to round-trip; ``space="logit"`` is exact there.
        """
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        if space not in ("probability", "logit"):
            raise ValueError(f"space must be 'probability' or 'logit', got {space!r}")
        from .base import _check_representable, _validate_point_targets

        arr = _validate_point_targets(p)
        z = logit(arr) - self.delta_
        _check_representable(z, space)
        return z if space == "logit" else expit(z)

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

    # ------------------------------------------------------------- serialization
    # LogitOffset is not a BaseCalibrator subclass, so the protocol is
    # implemented here with the shared _serialize helpers (the existing
    # offset.py duplication precedent, e.g. interval_inverse's fit guard).

    def to_dict(self) -> dict[str, object]:
        """Versioned JSON-native snapshot (see ``BaseCalibrator.to_dict``).

        ``fit_meta`` records ``n_obs``, ``weight_sum``, ``fitted_at_utc``,
        and the ``data_fingerprint`` of the ``(p, w)`` pair — no ``n_events``
        because the offset is fitted on probabilities alone.
        """
        if not self.fitted_:
            raise RuntimeError("LogitOffset is not fitted; call fit() first")
        from . import __version__

        return {
            "probcal_schema": SCHEMA_VERSION,
            "probcal_version": __version__,
            "class": type(self).__name__,
            "params": {"delta": self.delta, "target_mean": self.target_mean},
            "state": {
                "delta_": self.delta_,
                "pre_mean_": self.pre_mean_,
                "post_mean_": self.post_mean_,
                "timestamp_": self.timestamp_,
            },
            "fit_meta": dict(getattr(self, "fit_meta_", {})),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogitOffset":
        """Rebuild a fitted offset from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_schema(d)
        if d.get("class") != cls.__name__:
            raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")
        params = d.get("params", {})
        obj = cls(delta=params.get("delta"), target_mean=params.get("target_mean"))
        for key, value in d.get("state", {}).items():
            setattr(obj, key, value)
        obj.fit_meta_ = dict(d.get("fit_meta", {}))
        obj.fitted_ = True
        return obj

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> str | None:
        """Serialize to JSON text, or to ``path`` when given (returns None then)."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return None

    @classmethod
    def from_json(cls, path_or_str: object) -> "LogitOffset":
        """Load from a JSON string or a filesystem path."""
        text = str(path_or_str)
        if not text.lstrip().startswith("{"):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialized form, blind to versions and
        to the audit ``timestamp_`` — identical fits fingerprint identically."""
        return fingerprint_of_dict(self.to_dict())


def _offset_mle(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Offset-only logistic MLE: root of the score equation on already-valid inputs.

    Solves ``mean_w(sigma(z + delta)) = mean_w(y)`` by bisection in
    ``[-40, 40]`` — the mean-matching root of ``LogitOffset(target_mean=...)``
    and, equivalently, the unique root of the offset-only logistic score
    equation ``sum(w * (y - sigma(z + delta))) = 0``. Callers are
    responsible for degenerate cases (empty input, a target outside
    ``(0, 1)``); this function assumes a valid bracket and is shared,
    unchanged, by :func:`estimate_offset` and
    ``probcal.monitor._processes.plug_in_delta`` so the two stay
    bit-identical.

    Parameters
    ----------
    z : numpy.ndarray
        Logits, ``logit(p)``.
    y : numpy.ndarray
        Binary outcomes in ``{0, 1}``.
    w : numpy.ndarray
        Positive weights.

    Returns
    -------
    float
        The fitted log-odds shift ``delta``.
    """
    target = float(np.average(y, weights=w))

    def gap(d: float) -> float:
        return float(np.average(expit(z + d), weights=w)) - target

    return bisect(gap, -_DELTA_BRACKET, _DELTA_BRACKET, tol=1e-14)


def estimate_offset(y: object, p: object, *, sample_weight: object = None) -> OffsetEstimate:
    """Offset-only logistic MLE of ``delta`` given ``p``, with a Fisher standard error.

    Fits the single-parameter model ``y ~ Bernoulli(sigma(logit(p) + delta))``
    by maximum likelihood. The score equation
    ``sum(w * (y - sigma(logit(p) + delta))) = 0`` is exactly the mean-matching
    condition solved by ``LogitOffset(target_mean=mean_w(y))``, so ``delta`` is
    found by the same bisection root-finder (:func:`_offset_mle`, shared with
    ``probcal.monitor._processes.plug_in_delta``). The Fisher information for
    this one-parameter model is ``sum(w * q * (1 - q))`` at
    ``q = sigma(logit(p) + delta)``, so the standard error is its inverse
    square root. That reading of the weights is the frequency one — ``w``
    counts observations — so the SE is only valid for frequency weights;
    importance (or otherwise non-count) weights inflate the information and
    understate the SE.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``. Both classes must be present.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    OffsetEstimate
        The fitted ``delta``, its standard error, and the fit's ``n``,
        ``events``, and ``weight_sum``.

    Raises
    ------
    ValueError
        If ``y`` or ``p`` fail validation (shape, range, finiteness) or if
        ``y`` contains only one class — the offset MLE does not exist then
        (``metrics.scores._prep`` performs this check).

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
    se = 1.0 / np.sqrt(float(np.sum(w * q * (1.0 - q))))
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
