"""SklearnCalibrator: a probcal calibrator as an sklearn estimator over scores."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from ..base import BaseCalibrator
from ._base import BinaryCalibratedMixin, check_sample_weight, drop_zero_weight, score_column
from ._compat import validate_X, validate_X_y


class SklearnCalibrator(BinaryCalibratedMixin, ClassifierMixin, TransformerMixin, BaseEstimator):
    """Probability calibration over a single score column, sklearn-style.

    Wraps any probcal calibrator as a scikit-learn classifier/transformer
    whose ``X`` is the score itself — shape ``(n,)``, ``(n, 1)``, or (in
    probability mode) a two-column ``predict_proba``-style matrix. Use it
    to end a ``Pipeline`` (via :meth:`transform`) or anywhere an sklearn
    estimator is expected; the fitted probcal object stays one attribute
    away (``calibrator_``) with its full audit surface (``interpret()``,
    ``interval_inverse``, ``to_dict``, ``fingerprint()``). The prototype
    passed as ``calibrator`` may also be a :class:`~probcal.Chain`.
    The calibrator protocol (``is_monotone_``, ``interval_inverse``,
    ``point_inverse``, ``affine_logit_coeffs_``, ``interpret``) and the JSON
    surface (``to_dict``, ``to_json``, ``fingerprint``) are delegated to
    ``calibrator_``, exactly as on :class:`CalibratedClassifier`.

    Parameters
    ----------
    calibrator : BaseCalibrator or None
        Unfitted probcal prototype, cloned via ``get_params`` at fit time;
        ``None`` uses ``BetaCalibrator()``.
    input : {"probability", "logit"}, keyword-only
        Scale of the score column. ``"probability"`` requires values in
        ``[0, 1]`` (probcal's forward-entry convention) and additionally
        accepts a two-column probability matrix; ``"logit"`` stays
        single-column and accepts any reals, mapped through ``expit``
        exactly first.
    positive_column : int, keyword-only
        Which column of a two-column probability matrix holds ``P(y=1)`` —
        ``0`` or ``1`` (default ``1``, matching ``predict_proba`` output).
        Ignored for single-column input.

    Attributes
    ----------
    calibrator_ : BaseCalibrator
        The fitted probcal calibrator.
    classes_ : numpy.ndarray of shape (2,)
        Class labels in ``numpy.unique`` order; column 1 of
        :meth:`predict_proba` is ``classes_[1]``.
    n_features_in_ : int
        1 or 2, depending on the ``X`` shape seen at fit; enforced at
        predict/transform time.
    """

    def __init__(
        self,
        calibrator: BaseCalibrator | None = None,
        *,
        input: str = "probability",
        positive_column: int = 1,
    ) -> None:
        self.calibrator = calibrator
        self.input = input
        self.positive_column = positive_column

    # ------------------------------------------------------------------ helpers

    def _scores(self, X: np.ndarray) -> np.ndarray:
        return score_column(
            X,
            self.positive_column,
            logit_input=self.input == "logit",
            error=(
                "SklearnCalibrator is score-level and takes one score column, or a "
                "two-column probability matrix with input='probability'; got "
                f"{X.shape[1]} columns with input={self.input!r}; calibrate the "
                "model's score, not its features"
            ),
        )

    # ------------------------------------------------------------------ estimator API

    def fit(self, X: object, y: object, sample_weight: object = None) -> "SklearnCalibrator":
        """Fit the wrapped calibrator on the score column.

        Parameters
        ----------
        X : array_like of shape (n,), (n, 1), or (n, 2)
            Scores (probabilities, or logits with ``input="logit"``), or a
            two-column probability matrix (``input="probability"`` only).

        y : array_like of shape (n,)
            Binary target; any two label values.
        sample_weight : array_like or None
            Non-negative observation weights; zero-weight rows are excluded
            (sklearn semantics).

        Returns
        -------
        SklearnCalibrator
            The fitted adapter.

        Raises
        ------
        ValueError
            If ``X``'s column count/``input`` combination is unsupported,
            ``input`` or ``positive_column`` is invalid, ``y`` has more
            than two classes, or a weight is negative.
        """
        if self.input not in ("probability", "logit"):
            raise ValueError(f"input must be 'probability' or 'logit', got {self.input!r}")
        if self.positive_column not in (0, 1):
            raise ValueError(f"positive_column must be 0 or 1, got {self.positive_column!r}")
        X_arr, y_arr = validate_X_y(self, X, y, reset=True, allow_1d=True)
        sw = check_sample_weight(sample_weight, X_arr)
        y_bin = self._binary_target(y_arr)
        s = self._scores(X_arr)
        if s.ndim == 2:
            _, (col, y_kept) = drop_zero_weight(sw, s[:, 1], y_bin)
            both = np.unique(y_kept).size == 2
            if both and float(col[y_kept == 1.0].mean()) < float(col[y_kept == 0.0].mean()):
                warnings.warn(
                    "the selected positive-probability column has a lower mean among "
                    "events than among non-events; if the matrix is ordered the other "
                    f"way, positive_column={1 - self.positive_column} is the likely fix",
                    UserWarning,
                    stacklevel=2,
                )
        self._fit_calibrator(s, y_bin, sw)
        return self

    def _positive_proba(self, X: object) -> np.ndarray:
        X_arr = validate_X(self, X, allow_1d=True)
        return self.calibrator_.predict_proba(self._scores(X_arr))

    def transform(self, X: object) -> np.ndarray:
        """Calibrated-probability column ``(n, 1)`` — lets the adapter end a Pipeline."""
        return self.predict_proba(X)[:, [1]]
