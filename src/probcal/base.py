"""BaseCalibrator: the common fit / predict_proba / interpret contract."""

import inspect
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from ._results import Interpretation
from ._validation import validate_binary_y, validate_scores, validate_weights


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

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Preimage ``(raw_lo, raw_hi)`` of a calibrated interval (spec §10).

        Implemented per calibrator in Task 11; the base contract raises until
        then.
        """
        raise NotImplementedError(
            f"interval_inverse is not yet implemented for {type(self).__name__} "
            "(arrives with Task 11)"
        )

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
