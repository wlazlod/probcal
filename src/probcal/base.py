"""BaseCalibrator: the common fit / predict_proba / interpret contract."""

import inspect
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

import numpy as np

from ._math import _LOGIT_CLIP, expit, expit1, logit, logit1
from ._results import Interpretation
from ._serialize import (
    JsonIO,
    check_payload,
    data_fingerprint,
    decode_value,
    encode_value,
    envelope,
)
from ._validation import (
    EPS,
    validate_binary_y,
    validate_scores,
    validate_space,
    validate_weights,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from sklearn.utils import Tags


class UnattainableTargetError(ValueError):
    """The requested calibrated target is unattainable.

    Raised instead of silently clamping when an interval does
    not intersect the calibrator's output range (or was emptied by
    ``buffer_logit``), when a point-inverse target lies outside the open
    interval ``(0, 1)``, or when a probability-space point-inverse result
    would round to 0.0/1.0 (raw logit beyond ``logit(1 - 1e-12)``).
    """


def _validate_point_targets(p: object) -> np.ndarray:
    """Validate point-inverse targets: 1-D, finite, strictly inside ``(0, 1)``.

    Unlike :func:`~probcal._validation.validate_scores` (the forward-entry
    convention), no clipping happens here: ``p = 0`` and ``p = 1`` are not
    attained by any finite raw score, so a finite "inverse" for them would be
    a silent clamp. Refusal is all-or-nothing and names the offending values.
    """
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"p must be a 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("p must contain only finite values")
    bad = (arr <= 0.0) | (arr >= 1.0)
    if np.any(bad):
        vals = np.unique(arr[bad])
        shown = vals[:5].tolist()
        more = f" (and {vals.size - 5} more)" if vals.size > 5 else ""
        raise UnattainableTargetError(
            "point-inverse targets must lie strictly inside (0, 1); got "
            f"{shown}{more} — p = 0 and p = 1 are not attained by any "
            "finite raw score (no silent clamp); use interval_inverse with lo=0/hi=1 "
            "for one-sided targets"
        )
    return arr


def _check_representable(z: np.ndarray, space: str) -> None:
    """Refuse probability-space output whose raw logit exceeds ``_LOGIT_CLIP``.

    ``sigma(z)`` for ``|z| > _LOGIT_CLIP`` rounds into the ``[1e-12, 1 - 1e-12]``
    clipping zone: ``predict_proba`` would clip it back and the round trip
    silently breaks. ``space="logit"`` is exact there, so the error names it.
    """
    if space == "probability" and not np.all(np.abs(z) <= _LOGIT_CLIP):
        worst = float(np.max(np.abs(np.asarray(z))))
        raise UnattainableTargetError(
            f"the requested target needs a raw logit of magnitude {worst:.6g} > "
            f"{_LOGIT_CLIP:.6g} = logit(1 - 1e-12); its probability-space "
            "representation rounds to 0.0/1.0 and cannot round-trip through "
            'predict_proba. Use space="logit", where the answer is exact.'
        )


def shrink_interval(lo: float, hi: float, buffer_logit: float) -> tuple[float, float]:
    """Shrink ``[lo, hi]`` by ``buffer_logit`` on the logit scale (0/1 ends stay put).

    Shared by every ``interval_inverse``: robustness against future
    recalibration drift of magnitude ``<= buffer_logit``.

    Raises
    ------
    ValueError
        If ``buffer_logit`` is negative or not finite.
    UnattainableTargetError
        If the buffer empties the interval.
    """
    buffer_logit = float(buffer_logit)
    if not buffer_logit >= 0.0 or buffer_logit == float("inf"):
        raise ValueError(f"buffer_logit must be a finite value >= 0, got {buffer_logit!r}")
    lo_b, hi_b = float(lo), float(hi)
    if buffer_logit > 0.0:
        if lo > 0.0:
            lo_b = expit1(logit1(lo) + buffer_logit)
        if hi < 1.0:
            hi_b = expit1(logit1(hi) - buffer_logit)
        if lo_b > hi_b:
            raise UnattainableTargetError(
                f"buffer_logit={buffer_logit} empties the calibrated interval [{lo}, {hi}]"
            )
    return lo_b, hi_b


def validate_interval(lo: float, hi: float) -> None:
    """``0 <= lo <= hi <= 1``."""
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"need 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")


def clone_unfitted(obj: object) -> object:
    """A fresh, unfitted copy built from ``obj.get_params()``."""
    return type(obj)(**obj.get_params())  # type: ignore[attr-defined]


_NOT_MONOTONE = (
    "{name} is not monotone (is_monotone_=False); its preimage may be a union of "
    "intervals. Use a monotone calibrator for thresholding and recourse."
)


class BaseCalibrator(JsonIO, ABC):
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
        self.fit_meta_ = {
            "n_obs": int(s_arr.shape[0]),
            "n_events": float(np.sum(w_arr * y_arr)),
            "weight_sum": float(w_arr.sum()),
            "fitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_fingerprint": data_fingerprint(s_arr, y_arr, w_arr),
        }
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

    # ------------------------------------------------------------- sklearn duck hooks

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted state for sklearn's ``check_is_fitted`` (sklearn >= 1.6).

        Returns
        -------
        bool
            ``True`` once :meth:`fit` has run.
        """
        return bool(self.fitted_)

    def __sklearn_tags__(self) -> "Tags":
        """Estimator tags for sklearn >= 1.6, built from the public constructors.

        sklearn is imported inside the body, never at module or class level:
        ``import probcal`` stays numpy-only and this hook costs nothing until
        sklearn itself calls it. Only the fields that are actually true of a
        calibrator are set — it takes 1-D scores, not a 2-D feature matrix,
        it is neither a classifier nor a regressor, and it must be fitted.

        Returns
        -------
        sklearn.utils.Tags
            The tag object sklearn's ``get_tags`` expects.

        Raises
        ------
        ImportError
            If sklearn is not installed (only reachable by calling the hook
            by hand).
        """
        from sklearn.utils import InputTags, Tags, TargetTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=True),
            requires_fit=True,
            input_tags=InputTags(two_d_array=False),
        )

    # ------------------------------------------------------------- introspection

    @abstractmethod
    def interpret(self) -> Interpretation:
        """Fitted parameters with a plain-language, domain-aware reading."""

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Coefficients ``(a, b)`` of ``logit g(s) = a * logit(s) + b``, if affine.

        ``None`` for calibrators that are not affine on the logit scale.
        Consumed by the attribution adjustment.
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
        self._check_monotone()
        validate_interval(lo, hi)
        validate_space(space)
        lo_b, hi_b = shrink_interval(lo, hi, buffer_logit)
        gmin, gmax = self._output_range()
        if lo_b > gmax or hi_b < gmin:
            raise UnattainableTargetError(
                f"calibrated target [{lo_b:.6g}, {hi_b:.6g}] does not intersect the "
                f"calibrator's output range [{gmin:.6g}, {gmax:.6g}]"
            )
        raw_lo = 0.0 if lo_b <= gmin else float(self._inverse_left(lo_b))
        raw_hi = 1.0 if hi_b >= gmax else float(self._inverse_right(hi_b))
        if space == "logit":
            lo_out = -np.inf if raw_lo <= 0.0 else logit1(raw_lo)
            hi_out = np.inf if raw_hi >= 1.0 else logit1(raw_hi)
            return lo_out, hi_out
        return raw_lo, raw_hi

    def _check_monotone(self) -> None:
        if not self.is_monotone_:
            raise NotImplementedError(_NOT_MONOTONE.format(name=type(self).__name__))

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Raw scores whose calibrated probabilities equal ``p`` (exact preimage).

        Defined only for strictly monotone calibrators with an exact
        inverse: affine-logit maps (``logit g(s) = a * logit(s) + b``)
        invert in closed form here, covering Platt scaling, temperature
        scaling, and the tied Beta variants (``"a"``, ``"ab"``);
        ``BetaCalibrator`` overrides this method with its own exact
        construction for the full ``"abm"`` variant. Others
        raise ``NotImplementedError`` and should use :meth:`interval_inverse`
        instead.

        Parameters
        ----------
        p : array_like
            Calibrated probabilities strictly inside ``(0, 1)``; boundary
            and out-of-range values raise ``UnattainableTargetError``
            (all-or-nothing, no silent clamp).
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw values. ``"logit"`` returns the raw
            logit directly; ``"probability"`` (default) returns the raw
            score.

        Returns
        -------
        numpy.ndarray
            Raw scores (or logits, if ``space="logit"``) whose calibrated
            probability equals ``p``.

        Raises
        ------
        RuntimeError
            If not yet fitted.
        ValueError
            If ``space`` is not ``"probability"`` or ``"logit"``.
        NotImplementedError
            If the calibrator is not monotone (``is_monotone_ = False``), or
            has no affine-logit closed form (``affine_logit_coeffs_`` is
            ``None``).
        UnattainableTargetError
            If any element of ``p`` lies outside the open interval
            ``(0, 1)``; or if ``space="probability"`` and the raw logit of
            any result exceeds ``logit(1 - 1e-12)`` in magnitude — the
            probability representation would round to 0.0/1.0 and silently
            fail to round-trip; ``space="logit"`` is exact there.
        """
        self._check_fitted()
        self._check_monotone()
        validate_space(space)
        arr = _validate_point_targets(p)
        z = self._point_inverse_logit(arr)
        _check_representable(z, space)
        return z if space == "logit" else expit(z)

    def _point_inverse_logit(self, p: np.ndarray) -> np.ndarray:
        """Raw logits whose calibrated probability is ``p`` (validated, inside (0, 1)).

        Default: the affine-logit closed form. Override for other exact inverses.
        """
        coeffs = self.affine_logit_coeffs_
        if coeffs is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no exact point inverse; use interval_inverse"
            )
        a, b = coeffs
        return (logit(p) - b) / a

    # Hooks for interval_inverse — overridden by closed-form / block calibrators.

    def _predict_scalar(self, x: float) -> float:
        return float(self._predict(np.array([x]))[0])

    def _output_range(self) -> tuple[float, float]:
        """(min, max) of the fitted map over the raw-score domain."""
        return self._predict_scalar(EPS), self._predict_scalar(1.0 - EPS)

    def _affine_inverse(self, t: float) -> float | None:
        """Closed-form inverse for affine-logit maps, ``None`` otherwise."""
        coeffs = self.affine_logit_coeffs_
        if coeffs is None or coeffs[0] == 0.0:
            return None
        a, b = coeffs
        return expit1((logit1(t) - b) / a)

    def _inverse_left(self, t: float) -> float:
        """inf{s : g(s) >= t}: closed form when affine, else monotone bisection."""
        closed = self._affine_inverse(t)
        if closed is not None:
            return closed
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
        """sup{s : g(s) <= t}: closed form when affine, else monotone bisection."""
        closed = self._affine_inverse(t)
        if closed is not None:
            return closed
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
                    f"unknown parameter {key!r} for {type(self).__name__}; valid: {sorted(valid)}"
                )
            setattr(self, key, value)
        return self

    def _clone(self) -> Self:
        """A fresh, unfitted copy with the same constructor parameters."""
        return type(self)(**self.get_params())

    # ------------------------------------------------------------- serialization

    _STATE_ATTRS: tuple[str, ...] = ()

    def _state(self) -> dict[str, object]:
        """Fitted state as JSON-native values (generic: reads ``_STATE_ATTRS``)."""
        return {a: encode_value(getattr(self, a)) for a in type(self)._STATE_ATTRS}

    def _set_state(self, state: dict[str, object]) -> None:
        """Inverse of :meth:`_state`."""
        for a, v in state.items():
            setattr(self, a, decode_value(v))

    def _params_for_dict(self) -> dict[str, object]:
        """Constructor parameters as stored in :meth:`to_dict` (hook for
        classes whose params are not JSON-native, e.g. the selector)."""
        return self.get_params()

    @classmethod
    def _params_from_dict(cls, params: dict[str, object]) -> dict[str, object]:
        """Inverse of :meth:`_params_for_dict`."""
        return params

    def to_dict(self) -> dict[str, object]:
        """Versioned JSON-native snapshot of the fitted object.

        Returns
        -------
        dict
            ``{"probcal_schema", "probcal_version", "class", "params",
            "state", "fit_meta"}``. ``fit_meta`` records ``n_obs``,
            ``n_events``, ``weight_sum``, ``fitted_at_utc`` (ISO 8601), the
            ``data_fingerprint`` (SHA-256 of the sorted ``(s, y, w)``
            triple), and convergence flags where they exist.

        Raises
        ------
        RuntimeError
            If not yet fitted.
        """
        self._check_fitted()
        fit_meta = dict(getattr(self, "fit_meta_", {}))
        for flag in ("converged_", "separation_fallback_"):
            if hasattr(self, flag):
                fit_meta[flag] = bool(getattr(self, flag))
        return envelope(
            self,
            params=encode_value(self._params_for_dict()),
            state=self._state(),
            fit_meta=encode_value(fit_meta),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "BaseCalibrator":
        """Rebuild a fitted object from :meth:`to_dict` output.

        Called on :class:`BaseCalibrator` itself, dispatches through the
        class registry to whatever class wrote ``d``; called on a subclass,
        requires ``d["class"]`` to match that subclass.

        Parameters
        ----------
        d : dict
            Output of :meth:`to_dict` (parsed JSON).

        Returns
        -------
        BaseCalibrator
            A fitted instance.

        Raises
        ------
        ValueError
            If the schema version is unknown (naming the writing version),
            the class is not registered, or ``d["class"]`` does not match
            the subclass this was called on.
        """
        if cls is BaseCalibrator:
            from ._registry import load

            return load(d)  # type: ignore[return-value]
        check_payload(cls, d)
        params = cls._params_from_dict(
            decode_value(d.get("params", {}), arrays=False)  # type: ignore[arg-type]
        )
        obj = cls(**params)  # type: ignore[arg-type]
        obj._set_state(dict(d.get("state", {})))  # type: ignore[arg-type]
        obj.fit_meta_ = dict(decode_value(d.get("fit_meta", {}), arrays=False))  # type: ignore[call-overload]
        obj.fitted_ = True
        return obj
