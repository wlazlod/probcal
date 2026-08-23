"""SklearnCalibrator: a probcal calibrator as an sklearn estimator over scores."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from .._math import expit
from ..base import BaseCalibrator
from ..parametric import BetaCalibrator
from ._compat import validate_X, validate_X_y


class SklearnCalibrator(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Probability calibration over a single score column, sklearn-style.

    Wraps any probcal calibrator as a scikit-learn classifier/transformer
    whose ``X`` is the score itself — shape ``(n,)`` or ``(n, 1)``. Use it
    to end a ``Pipeline`` (via :meth:`transform`) or anywhere an sklearn
    estimator is expected; the fitted probcal object stays one attribute
    away (``calibrator_``) with its full audit surface (``interpret()``,
    ``interval_inverse``, ``to_dict``, ``fingerprint()``).

    Parameters
    ----------
    calibrator : BaseCalibrator or None
        Unfitted probcal prototype, cloned via ``get_params`` at fit time;
        ``None`` uses ``BetaCalibrator()``.
    input : {"probability", "logit"}, keyword-only
        Scale of the score column. ``"probability"`` requires values in
        ``[0, 1]`` (probcal's forward-entry convention); ``"logit"`` accepts
        any reals and maps them through ``expit`` exactly first.

    Attributes
    ----------
    calibrator_ : BaseCalibrator
        The fitted probcal calibrator.
    classes_ : numpy.ndarray of shape (2,)
        Class labels in ``numpy.unique`` order; column 1 of
        :meth:`predict_proba` is ``classes_[1]``.
    n_features_in_ : int
        Always 1 — the adapter is score-level by design.
    """

    def __init__(
        self,
        calibrator: BaseCalibrator | None = None,
        *,
        input: str = "probability",
    ) -> None:
        self.calibrator = calibrator
        self.input = input

    # ------------------------------------------------------------------ helpers

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != 1:
            raise ValueError(
                f"SklearnCalibrator is score-level and takes exactly one column, got "
                f"{X.shape[1]}; calibrate the model's score, not its features"
            )
        s = X[:, 0]
        if self.input == "logit":
            return expit(s)
        return s

    # ------------------------------------------------------------------ estimator API

    def fit(self, X: object, y: object, sample_weight: object = None) -> "SklearnCalibrator":
        """Fit the wrapped calibrator on the score column.

        Parameters
        ----------
        X : array_like of shape (n,) or (n, 1)
            Scores (probabilities, or logits with ``input="logit"``).
        y : array_like of shape (n,)
            Binary target; any two label values.
        sample_weight : array_like or None
            Positive observation weights.

        Returns
        -------
        SklearnCalibrator
            The fitted adapter.

        Raises
        ------
        ValueError
            If ``X`` has more than one column (score-level by design),
            ``input`` is unknown, or ``y`` has more than two classes.
        """
        if self.input not in ("probability", "logit"):
            raise ValueError(f"input must be 'probability' or 'logit', got {self.input!r}")
        X_arr, y_arr = validate_X_y(self, X, y, reset=True, allow_1d=True)
        sw = None if sample_weight is None else _check_sample_weight(sample_weight, X_arr)
        check_classification_targets(y_arr)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported. Got " f"{len(self.classes_)} classes."
            )
        y_bin = (y_arr == self.classes_[1]).astype(np.float64)
        s = self._scores(X_arr)
        if sw is not None:
            # Zero weight means excluded (sklearn semantics); probcal requires
            # strictly positive weights, so drop those rows here.
            keep = sw > 0.0
            s, y_bin, sw = s[keep], y_bin[keep], sw[keep]
            if np.unique(y_bin).size < 2:
                raise ValueError(
                    "Only one class remains after removing zero-weight samples; "
                    "both classes are required."
                )
        proto = self.calibrator if self.calibrator is not None else BetaCalibrator()
        self.calibrator_ = type(proto)(**proto.get_params())
        self.calibrator_.fit(s, y_bin, sample_weight=sw)
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Calibrated ``(n, 2)`` probabilities ``[P(classes_[0]), P(classes_[1])]``."""
        check_is_fitted(self, "calibrator_")
        X_arr = validate_X(self, X, allow_1d=True)
        p = self.calibrator_.predict_proba(self._scores(X_arr))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: object) -> np.ndarray:
        """Class labels at the 0.5 calibrated-probability threshold."""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def transform(self, X: object) -> np.ndarray:
        """Calibrated-probability column ``(n, 1)`` — lets the adapter end a Pipeline."""
        return self.predict_proba(X)[:, [1]]

    # ------------------------------------------------------------------ tags

    def __sklearn_tags__(self):  # noqa: ANN204 - sklearn protocol, version-dependent type
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.target_tags.required = True
        return tags

    def _more_tags(self) -> dict[str, object]:  # sklearn < 1.6
        return {"binary_only": True, "requires_y": True}
