"""CalibratedClassifier: CalibratedClassifierCV(ensemble=False) on probcal calibrators."""

import numbers
import warnings

import numpy as np
from sklearn import get_config
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, check_cv, cross_val_predict
from sklearn.utils import indexable
from sklearn.utils.metadata_routing import MetadataRouter, get_routing_for_object
from sklearn.utils.validation import (
    _num_samples,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
    has_fit_parameter,
)

from .._math import expit
from ..base import BaseCalibrator
from ._base import BinaryCalibratedMixin, check_sample_weight


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


class CalibratedClassifier(BinaryCalibratedMixin, ClassifierMixin, BaseEstimator):
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

    ``X`` is passed to the estimator untouched (no float conversion), like
    ``CalibratedClassifierCV``: DataFrames, mixed-dtype columns for a
    ``ColumnTransformer`` pipeline, and sparse matrices all work.

    Parameters
    ----------
    estimator : object or None
        Classifier to calibrate. ``None`` resolves to
        ``LogisticRegression(max_iter=1000)`` at fit time (the sklearn
        precedent for a default-constructible wrapper).
    calibrator : BaseCalibrator or None, keyword-only
        Unfitted probcal prototype (cloned via ``get_params``); ``None``
        uses ``BetaCalibrator()``.
    cv : int, cross-validation splitter, iterable, or "prefit", keyword-only
        An integer ``>= 2`` is a fold count (see ``stratify``); a splitter
        or an iterable of ``(train, test)`` splits is used as given
        (through ``sklearn.model_selection.check_cv``; it must partition the
        rows, as ``cross_val_predict`` requires). ``"prefit"`` scores the
        calibration set with the already-fitted ``estimator`` directly.
    method : {"predict_proba", "decision_function"}, keyword-only
        Score source. ``"decision_function"`` margins are mapped through
        ``expit`` before calibration — the calibrator then absorbs any
        monotone distortion this introduces.
    stratify : bool, keyword-only
        For an integer ``cv`` only: ``True`` uses
        ``StratifiedKFold(cv, shuffle=True, random_state=random_state)``
        (recommended for rare events), ``False`` uses
        ``KFold(cv, shuffle=True, random_state=random_state)``. Ignored for
        a splitter ``cv``.
    random_state : int or None, keyword-only
        Fold-shuffling seed for an integer ``cv``.

    Attributes
    ----------
    estimator_ : object
        The deployed classifier (input estimator for ``cv="prefit"``,
        full-data refit otherwise).
    calibrator_ : BaseCalibrator
        The fitted probcal calibrator (one map, pooled OOF scores).
    classes_ : numpy.ndarray of shape (2,)
        Class labels; column 1 of :meth:`predict_proba` is ``classes_[1]``.
    n_features_in_ : int
        Copied from ``estimator_`` when it defines one.

    Notes
    -----
    :class:`probcal.CalibratedModel` (``flow="cv"``) runs the same
    out-of-fold protocol without sklearn, with deliberate differences:
    its folds are always class-stratified and drawn with numpy's
    ``default_rng(random_state)`` (so the same seed assigns different folds
    than ``StratifiedKFold``), and it alone offers ``ensemble=True`` (one
    calibrator per fold). Both pass ``X`` to the model untouched and hand
    ``sample_weight`` to the model fits when the estimator accepts it. Use this class inside
    sklearn pipelines and searches, ``CalibratedModel`` for a numpy-only
    deployment wrapper.
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

    def _estimator_scores(self, estimator: object, X: object) -> np.ndarray:
        return self._to_scores(np.asarray(getattr(estimator, self.method)(X)))

    def _splitter(self, y: np.ndarray) -> object:
        cv = self.cv
        if isinstance(cv, numbers.Integral) and not isinstance(cv, bool):
            if int(cv) < 2:
                raise ValueError(f"cv must be an integer >= 2, a splitter, or 'prefit'; got {cv!r}")
            kind = StratifiedKFold if self.stratify else KFold
            return kind(n_splits=int(cv), shuffle=True, random_state=self.random_state)
        return check_cv(cv, y, classifier=True)

    def _is_prefit(self) -> bool:
        return isinstance(self.cv, str) and self.cv == "prefit"

    # ------------------------------------------------------------------ estimator API

    def fit(self, X: object, y: object, sample_weight: object = None) -> "CalibratedClassifier":
        """Fit per the out-of-fold protocol (or score directly when prefit).

        Parameters
        ----------
        X : array_like, DataFrame, or sparse matrix
            Features, passed to the wrapped estimator untouched.
        y : array_like of shape (n,)
            Binary target; any two label values.
        sample_weight : array_like or None
            Non-negative observation weights (zero excludes a row from the
            calibrator fit). Always used for the calibrator stage; also
            handed to the cross-validated fits and the refit when the
            estimator can take them. When it cannot, a ``UserWarning``
            names it and those fits run unweighted.

        Returns
        -------
        CalibratedClassifier
            The fitted wrapper.

        Raises
        ------
        ValueError
            If ``method`` or ``cv`` is invalid, ``y`` has more than two
            classes, a weight is negative, or (``cv="prefit"``) the
            estimator's ``classes_`` differ from the labels in ``y``.

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
        X, y = indexable(X, y)
        y_arr = column_or_1d(y, warn=True)
        check_consistent_length(X, y_arr)
        if _num_samples(X) == 0:
            raise ValueError("Found array with 0 sample(s); a minimum of 1 is required.")
        sw = check_sample_weight(sample_weight, X)
        y_bin = self._binary_target(y_arr)

        if self._is_prefit():
            check_is_fitted(self.estimator)
            est_classes = getattr(self.estimator, "classes_", None)
            if est_classes is not None and not np.array_equal(
                np.asarray(est_classes), self.classes_
            ):
                raise ValueError(
                    f"the prefit estimator was trained on classes {list(est_classes)}, but y "
                    f"has classes {list(self.classes_)}; its score columns would be "
                    "misread as P(classes_[1])"
                )
            self.estimator_ = self.estimator
            oof = self._estimator_scores(self.estimator_, X)
        else:
            base = self.estimator if self.estimator is not None else self._default_estimator()
            splitter = self._splitter(y_arr)
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
                clone(base), X, y_arr, cv=splitter, method=self.method, **fit_params
            )
            oof = self._to_scores(np.asarray(raw))
            refit = clone(base)
            if inner_sw is None:
                refit.fit(X, y_arr)
            else:
                refit.fit(X, y_arr, sample_weight=inner_sw)
            self.estimator_ = refit

        for attr in ("n_features_in_", "feature_names_in_"):
            if hasattr(self.estimator_, attr):
                setattr(self, attr, getattr(self.estimator_, attr))
        self._fit_calibrator(oof, y_bin, sw)
        return self

    def _positive_proba(self, X: object) -> np.ndarray:
        return self.calibrator_.predict_proba(self._estimator_scores(self.estimator_, X))

    def __sklearn_tags__(self):  # noqa: ANN204 - sklearn protocol, version-dependent type
        from sklearn.utils import get_tags

        tags = super().__sklearn_tags__()
        base = self.estimator if self.estimator is not None else self._default_estimator()
        # X reaches the estimator untouched, so its input capabilities are ours.
        tags.input_tags.sparse = get_tags(base).input_tags.sparse
        return tags
