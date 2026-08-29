"""CalibratedClassifier: CalibratedClassifierCV(ensemble=False) on probcal calibrators."""

import os
import warnings

import numpy as np
from sklearn import get_config
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.metadata_routing import MetadataRouter, get_routing_for_object
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, has_fit_parameter

from .._math import expit
from ..base import BaseCalibrator
from ..parametric import BetaCalibrator
from ._compat import CLASSIFIER_XFAIL_CHECKS, validate_X, validate_X_y


def _accepts_sample_weight(estimator: object) -> bool:
    """Whether ``estimator``'s ``fit`` can consume ``sample_weight`` at all.

    The signature check is sklearn's own precedent — ``CalibratedClassifierCV``
    does exactly this and warns when it fails. Under metadata routing a router
    (``Pipeline``, ``GridSearchCV``, …) takes weights through ``**params``
    rather than a named argument, so routers count as capable there and
    sklearn's routing machinery has the final word: an undeclared request
    raises ``UnsetMetadataPassedError`` instead of dropping the weights.
    """
    if has_fit_parameter(estimator, "sample_weight"):
        return True
    if get_config().get("enable_metadata_routing", False):
        return isinstance(get_routing_for_object(estimator), MetadataRouter)
    return False


class CalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Cross-validated probability calibration of a classifier, probcal-style.

    The drop-in for ``sklearn.calibration.CalibratedClassifierCV`` with
    ``ensemble=False``: out-of-fold scores via ``cross_val_predict``, **one**
    probcal calibrator fitted on the pooled OOF scores, and the estimator
    refit on all data (unless ``cv="prefit"``). What probcal adds on top:
    the fitted calibrator's audit surface (``interpret()``, bootstrap CIs
    via ``probcal.metrics.evaluate``), exact inverse maps, JSON
    serialization, and fingerprints — see ``guide/sklearn.md``.

    Also exposes the probcal calibrator protocol (``is_monotone_``,
    ``interval_inverse``, ``point_inverse``, ``affine_logit_coeffs_``,
    ``fingerprint``) by delegation to ``calibrator_``, so a fitted instance
    can be handed directly to consumers of that protocol (e.g. treecf's
    ``Target.calibrated``).

    Parameters
    ----------
    estimator : object or None
        Classifier to calibrate. ``None`` resolves to
        ``LogisticRegression(max_iter=1000)`` at fit time (the sklearn
        precedent for a default-constructible wrapper).
    calibrator : BaseCalibrator or None, keyword-only
        Unfitted probcal prototype (cloned via ``get_params``); ``None``
        uses ``BetaCalibrator()``.
    cv : int or "prefit", keyword-only
        Fold count for the out-of-fold protocol, or ``"prefit"`` to score
        the calibration set with the already-fitted ``estimator`` directly.
    method : {"predict_proba", "decision_function"}, keyword-only
        Score source. ``"decision_function"`` margins are mapped through
        ``expit`` before calibration — the calibrator then absorbs any
        monotone distortion this introduces.
    stratify : bool, keyword-only
        Stratify the folds by class (recommended for rare events).
    random_state : int or None, keyword-only
        Fold-assignment seed (used only when ``stratify=True``).

    Attributes
    ----------
    estimator_ : object
        The deployed classifier (input estimator for ``cv="prefit"``,
        full-data refit otherwise).
    calibrator_ : BaseCalibrator
        The fitted probcal calibrator (one map, pooled OOF scores).
    classes_ : numpy.ndarray of shape (2,)
        Class labels; column 1 of :meth:`predict_proba` is ``classes_[1]``.
    """

    def __init__(
        self,
        estimator: object = None,
        *,
        calibrator: BaseCalibrator | None = None,
        cv: object = 5,
        method: str = "predict_proba",
        stratify: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.cv = cv
        self.method = method
        self.stratify = stratify
        self.random_state = random_state

    # ------------------------------------------------------------------ internals

    def _default_estimator(self) -> object:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000)

    def _to_scores(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "decision_function":
            return expit(raw)
        return raw[:, 1] if raw.ndim == 2 else raw

    def _estimator_scores(self, estimator: object, X: np.ndarray) -> np.ndarray:
        return self._to_scores(np.asarray(getattr(estimator, self.method)(X)))

    # ------------------------------------------------------------------ estimator API

    def fit(self, X: object, y: object, sample_weight: object = None) -> "CalibratedClassifier":
        """Fit per the out-of-fold protocol (or score directly when prefit).

        Parameters
        ----------
        X : array_like of shape (n, d)
            Features, passed to the wrapped estimator.
        y : array_like of shape (n,)
            Binary target; any two label values.
        sample_weight : array_like or None
            Positive observation weights. Always used for the calibrator
            stage; also handed to the cross-validated fits and the refit
            when the estimator can take them. When it cannot, a
            ``UserWarning`` names it and those fits run unweighted.

        Returns
        -------
        CalibratedClassifier
            The fitted wrapper.

        Raises
        ------
        ValueError
            If ``method`` is unknown or ``y`` has more than two classes.

        Warns
        -----
        UserWarning
            If ``sample_weight`` is given and the estimator's ``fit`` cannot
            consume it — see ``guide/sklearn.md``.
        """
        if self.method not in ("predict_proba", "decision_function"):
            raise ValueError(
                f"method must be 'predict_proba' or 'decision_function', got {self.method!r}"
            )
        X_arr, y_arr = validate_X_y(self, X, y, reset=True)
        sw = None if sample_weight is None else _check_sample_weight(sample_weight, X_arr)
        check_classification_targets(y_arr)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported. Got " f"{len(self.classes_)} classes."
            )
        y_bin = (y_arr == self.classes_[1]).astype(np.float64)

        if self.cv == "prefit":
            check_is_fitted(self.estimator)
            self.estimator_ = self.estimator
            oof = self._estimator_scores(self.estimator_, X_arr)
        else:
            base = self.estimator if self.estimator is not None else self._default_estimator()
            n_splits = int(self.cv)  # type: ignore[call-overload]
            if self.stratify:
                splitter: object = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
            else:
                splitter = n_splits
            inner_sw = sw
            if sw is not None and not _accepts_sample_weight(base):
                inner_sw = None
                warnings.warn(
                    f"{type(base).__name__}.fit does not accept sample_weight: the "
                    "cross-validated fits and the full-data refit are unweighted, "
                    "while the calibrator is fitted with the weights. The calibration "
                    "map is still weighted, but the scores it calibrates are not.",
                    UserWarning,
                    stacklevel=2,
                )
            fit_params = {} if inner_sw is None else {"params": {"sample_weight": inner_sw}}
            raw = cross_val_predict(
                clone(base), X_arr, y_arr, cv=splitter, method=self.method, **fit_params
            )
            oof = self._to_scores(np.asarray(raw))
            refit = clone(base)
            if inner_sw is None:
                refit.fit(X_arr, y_arr)
            else:
                refit.fit(X_arr, y_arr, sample_weight=inner_sw)
            self.estimator_ = refit

        if sw is not None:
            # Zero weight means excluded (sklearn semantics); probcal requires
            # strictly positive weights, so drop those rows here.
            keep = sw > 0.0
            oof, y_bin, sw = oof[keep], y_bin[keep], sw[keep]
            if np.unique(y_bin).size < 2:
                raise ValueError(
                    "Only one class remains after removing zero-weight samples; "
                    "both classes are required."
                )
        proto = self.calibrator if self.calibrator is not None else BetaCalibrator()
        self.calibrator_ = type(proto)(**proto.get_params())
        self.calibrator_.fit(oof, y_bin, sample_weight=sw)
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Calibrated ``(n, 2)`` probabilities: estimator scores composed with the calibrator."""
        check_is_fitted(self, "calibrator_")
        X_arr = validate_X(self, X)
        p = self.calibrator_.predict_proba(self._estimator_scores(self.estimator_, X_arr))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: object) -> np.ndarray:
        """Class labels at the 0.5 calibrated-probability threshold."""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    # ------------------------------------------------------------------ probcal protocol

    @property
    def is_monotone_(self) -> bool:
        """Whether the fitted calibration map is non-decreasing (delegated)."""
        check_is_fitted(self, "calibrator_")
        return bool(self.calibrator_.is_monotone_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Affine-logit coefficients of the calibration map, if any (delegated)."""
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.affine_logit_coeffs_

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        """Preimage of a calibrated interval in the estimator's score space (delegated)."""
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.interval_inverse(lo, hi, space=space, buffer_logit=buffer_logit)

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Exact preimage of calibrated probabilities (delegated)."""
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.point_inverse(p, space=space)

    def fingerprint(self) -> str:
        """The fitted calibrator's provenance fingerprint (delegated)."""
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.fingerprint()

    def to_dict(self) -> dict[str, object]:
        """The fitted calibrator's versioned JSON envelope (delegated).

        The estimator itself follows sklearn's pickle conventions and is
        outside the JSON's scope — persist it with your model artifact.
        """
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.to_dict()

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> "str | None":
        """The fitted calibrator's JSON serialization (delegated), never pickle."""
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.to_json(path, indent=indent)

    def interpret(self):  # noqa: ANN201 - probcal Interpretation
        """The fitted calibrator's plain-language reading (delegated)."""
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.interpret()

    # ------------------------------------------------------------------ tags

    def __sklearn_tags__(self):  # noqa: ANN204 - sklearn protocol, version-dependent type
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.target_tags.required = True
        return tags

    def _more_tags(self) -> dict[str, object]:  # sklearn < 1.6
        return {
            "binary_only": True,
            "requires_y": True,
            "_xfail_checks": dict(CLASSIFIER_XFAIL_CHECKS),
        }
