"""BaseCalibrator: the common fit / predict_proba / interpret contract."""

import inspect
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from ._math import expit, logit
from ._results import Interpretation
from ._validation import EPS, validate_binary_y, validate_scores, validate_weights


class UnattainableTargetError(ValueError):
    """The requested calibrated interval is unattainable.

    Raised when the interval does not intersect the calibrator's output
    range (or was emptied by ``buffer_logit``), instead of silently
    clamping — spec §10.
    """


class BaseCalibrator(ABC):
    """Common contract for all probcal calibrators.

    Subclasses implement ``_fit`` (estimation on validated arrays),
    ``_predict`` (the fitted map on clipped scores), and ``interpret``.
    Everything else — validation, sklearn-style parameter handling without an
    sklearn import, the 2-D probability helper — lives here.

    Attributes
    ----------
    is_monotone_ : bool
        Whether the fitted map is guaranteed non-decreasing. Class-level
        default ``True``; non-monotone calibrators (e.g. ENIR) override it.
    fitted_ : bool
        Set by :meth:`fit`.
    """

    is_monotone_: bool = True
    fitted_: bool = False

    # ------------------------------------------------------------- fitting

    def fit(self, s: object, y: object, sample_weight: object = None) -> Self:
        """Fit the calibration map on scores and binary outcomes.

        Parameters
        ----------
        s : array_like
            Raw scores/probabilities in ``[0, 1]`` (clipped to
            ``[1e-12, 1 - 1e-12]``). Users holding raw logits convert with
            :func:`probcal.expit` first.
        y : array_like
            Binary outcomes in ``{0, 1}``; both classes must be present.
        sample_weight : array_like or None
            Positive observation weights.

        Returns
        -------
        Self
            The fitted calibrator.
        """
        s_arr = validate_scores(s)
        y_arr = validate_binary_y(y)
        if s_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"s and y must have equal length, got {s_arr.shape[0]} and {y_arr.shape[0]}"
            )
        w_arr = validate_weights(sample_weight, s_arr.shape[0])
        self._fit(s_arr, y_arr, w_arr)
        self.fitted_ = True
        return self

    @abstractmethod
    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Estimate parameters from validated arrays."""

    # ------------------------------------------------------------- prediction

    def predict_proba(self, s: object) -> np.ndarray:
        """Calibrated probabilities ``P(y = 1)`` for new scores.

        Parameters
        ----------
        s : array_like
            Raw scores/probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Calibrated probabilities.
        """
        self._check_fitted()
        return self._predict(validate_scores(s))

    def predict_proba_2d(self, s: object) -> np.ndarray:
        """Sklearn-style ``(n, 2)`` probability matrix ``[P(y=0), P(y=1)]``."""
        p = self.predict_proba(s)
        return np.column_stack([1.0 - p, p])

    @abstractmethod
    def _predict(self, s: np.ndarray) -> np.ndarray:
        """Apply the fitted map to validated scores."""

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")

    # ------------------------------------------------------------- introspection

    @abstractmethod
    def interpret(self) -> Interpretation:
        """Fitted parameters with a plain-language, domain-aware reading."""

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Coefficients ``(a, b)`` of ``logit g(s) = a * logit(s) + b``, if affine.

        ``None`` for calibrators that are not affine on the logit scale.
        Consumed by the attribution adjustment (spec §9).
        """
        return None

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank for selector tie-breaks; lower wins a tie.

        Default 100.0 means "unknown — override in subclasses". Custom calibrators
        declare their place in the tie-break by overriding this property.
        """
        return 100.0

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Generalized-inverse preimage ``(raw_lo, raw_hi)`` of a calibrated interval.

        For a non-decreasing fitted map ``g``:
        ``raw_lo = inf{s : g(s) >= lo}`` and ``raw_hi = sup{s : g(s) <= hi}``.

        Parameters
        ----------
        lo, hi : float
            Calibrated-probability bounds; ``lo=0``/``hi=1`` map to the full
            raw range (−inf/+inf on the logit scale).
        space : {"probability", "logit"}
            Scale of the returned raw bounds. ``"logit"`` is what a
            SIGMOID-link raw-margin consumer (e.g. a counterfactual engine's
            ``Target.raw``) expects.
        buffer_logit : float
            Shrink the calibrated interval by this margin in logit space
            *before* inverting — robustness against future recalibration
            drift (a central-tendency update of magnitude <= buffer cannot
            invalidate the result).

        Returns
        -------
        tuple of float
            ``(raw_lo, raw_hi)`` preimage bounds, on the scale requested by
            ``space``.

        Raises
        ------
        UnattainableTargetError
            If the (buffered) interval does not intersect the output range —
            never silently clamped.
        NotImplementedError
            For non-monotone calibrators (``is_monotone_ = False``), whose
            preimage may be a union of intervals.
        """
        self._check_fitted()
        if not self.is_monotone_:
            raise NotImplementedError(
                f"{type(self).__name__} is not monotone (is_monotone_=False); its preimage "
                "may be a union of intervals. Use a monotone calibrator for thresholding "
                "and recourse."
            )
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"need 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")
        if space not in ("probability", "logit"):
            raise ValueError(f"space must be 'probability' or 'logit', got {space!r}")
        lo_b, hi_b = float(lo), float(hi)
        if buffer_logit > 0.0:
            if lo > 0.0:
                lo_b = float(expit(np.array([logit(np.array([lo]))[0] + buffer_logit]))[0])
            if hi < 1.0:
                hi_b = float(expit(np.array([logit(np.array([hi]))[0] - buffer_logit]))[0])
            if lo_b > hi_b:
                raise UnattainableTargetError(
                    f"buffer_logit={buffer_logit} empties the calibrated interval " f"[{lo}, {hi}]"
                )
        gmin, gmax = self._output_range()
        if lo_b > gmax or hi_b < gmin:
            raise UnattainableTargetError(
                f"calibrated target [{lo_b:.6g}, {hi_b:.6g}] does not intersect the "
                f"calibrator's output range [{gmin:.6g}, {gmax:.6g}]"
            )
        raw_lo = 0.0 if lo_b <= gmin else float(self._inverse_left(lo_b))
        raw_hi = 1.0 if hi_b >= gmax else float(self._inverse_right(hi_b))
        if space == "logit":
            lo_out = -np.inf if raw_lo <= 0.0 else float(logit(np.array([raw_lo]))[0])
            hi_out = np.inf if raw_hi >= 1.0 else float(logit(np.array([raw_hi]))[0])
            return lo_out, hi_out
        return raw_lo, raw_hi

    # Hooks for interval_inverse — overridden by closed-form / block calibrators.

    def _predict_scalar(self, x: float) -> float:
        return float(self._predict(np.array([x]))[0])

    def _output_range(self) -> tuple[float, float]:
        """(min, max) of the fitted map over the raw-score domain."""
        return self._predict_scalar(EPS), self._predict_scalar(1.0 - EPS)

    def _inverse_left(self, t: float) -> float:
        """inf{s : g(s) >= t} by monotone bisection (invariant: g(hi) >= t)."""
        lo_s, hi_s = EPS, 1.0 - EPS
        if self._predict_scalar(lo_s) >= t:
            return lo_s
        for _ in range(80):
            mid = 0.5 * (lo_s + hi_s)
            if self._predict_scalar(mid) >= t:
                hi_s = mid
            else:
                lo_s = mid
        return hi_s

    def _inverse_right(self, t: float) -> float:
        """sup{s : g(s) <= t} by monotone bisection (invariant: g(lo) <= t)."""
        lo_s, hi_s = EPS, 1.0 - EPS
        if self._predict_scalar(hi_s) <= t:
            return hi_s
        for _ in range(80):
            mid = 0.5 * (lo_s + hi_s)
            if self._predict_scalar(mid) <= t:
                lo_s = mid
            else:
                hi_s = mid
        return lo_s

    # ------------------------------------------------------------- parameters

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
