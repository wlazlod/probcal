"""Shared machinery of the sklearn adapters: targets, weights, tags, delegation."""

import os

import numpy as np
from sklearn.base import clone
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from .._math import expit
from ..base import BaseCalibrator
from ..parametric import BetaCalibrator
from ._protocol import CalibratorProtocolMixin


def check_sample_weight(sample_weight: object, X: object) -> np.ndarray | None:
    """sklearn-validated weights; negative entries raise.

    Zero weights are legal (sklearn semantics: the row is excluded) and are
    dropped before the probcal fit by :func:`drop_zero_weight`; a negative
    weight has no such reading and is refused rather than silently dropped.

    Raises
    ------
    ValueError
        If any weight is negative.
    """
    if sample_weight is None:
        return None
    sw = _check_sample_weight(sample_weight, X)
    if np.any(sw < 0.0):
        raise ValueError("sample_weight must be non-negative")
    return sw


def score_column(
    X: np.ndarray, positive_column: int, *, logit_input: bool, error: str
) -> np.ndarray:
    """The score-level input of a validated 2-D ``X``.

    One column is the score itself (``expit``-mapped when ``logit_input``);
    two columns are a probability matrix, returned ``(n, 2)`` with the
    positive class in column 1 (``validate_scores`` downstream checks the
    simplex and takes that column). Anything else raises ``ValueError(error)``.
    """
    if X.shape[1] == 1:
        return expit(X[:, 0]) if logit_input else X[:, 0]
    if X.shape[1] == 2 and not logit_input:
        return X[:, ::-1] if positive_column == 0 else X
    raise ValueError(error)


def drop_zero_weight(
    sw: np.ndarray | None, *arrays: np.ndarray
) -> tuple[np.ndarray | None, tuple[np.ndarray, ...]]:
    """Drop zero-weight rows (sklearn's "excluded") — probcal needs positive weights."""
    if sw is None:
        return None, arrays
    keep = sw > 0.0
    return sw[keep], tuple(a[keep] for a in arrays)


def xfail_tags(estimator: object) -> dict[str, object]:
    """``_more_tags()["_xfail_checks"]`` for sklearn < 1.6 (lazy table import)."""
    from ._xfail import XFAIL_CHECKS

    for cls in type(estimator).__mro__:
        if cls.__name__ in XFAIL_CHECKS:
            return {"_xfail_checks": dict(XFAIL_CHECKS[cls.__name__])}
    return {}


class BinaryCalibratedMixin(CalibratorProtocolMixin):
    """Binary-target sklearn classifier around a fitted probcal calibrator.

    Subclasses implement :meth:`_positive_proba` (calibrated ``P(classes_[1])``,
    1-D) and keep the prototype in ``self.calibrator``; this mixin provides
    the target check, the calibrator fit, ``predict_proba``/``predict``, the
    calibrator protocol and its JSON surface by delegation, and the tags.
    """

    def _fitted_calibrator(self) -> BaseCalibrator:
        check_is_fitted(self, "calibrator_")
        return self.calibrator_  # type: ignore[attr-defined,no-any-return]

    def _binary_target(self, y: np.ndarray) -> np.ndarray:
        """Set ``classes_`` and return ``y == classes_[1]`` as float64.

        Raises
        ------
        ValueError
            If ``y`` is not a binary classification target.
        """
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Only binary classification is supported. Got {len(self.classes_)} classes."
            )
        return (np.asarray(y) == self.classes_[1]).astype(np.float64)

    def _fit_calibrator(self, s: np.ndarray, y_bin: np.ndarray, sw: np.ndarray | None) -> None:
        """Clone the prototype and fit it, zero-weight rows excluded."""
        sw, (s, y_bin) = drop_zero_weight(sw, s, y_bin)
        if sw is not None and np.unique(y_bin).size < 2:
            raise ValueError(
                "Only one class remains after removing zero-weight samples; "
                "both classes are required."
            )
        proto = self.calibrator if self.calibrator is not None else BetaCalibrator()  # type: ignore[attr-defined]
        self.calibrator_ = clone(proto)
        self.calibrator_.fit(s, y_bin, sample_weight=sw)

    def _positive_proba(self, X: object) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_proba(self, X: object) -> np.ndarray:
        """Calibrated ``(n, 2)`` probabilities ``[P(classes_[0]), P(classes_[1])]``."""
        check_is_fitted(self, "calibrator_")
        p = self._positive_proba(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: object) -> np.ndarray:
        """Class labels at the 0.5 calibrated-probability threshold."""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ JSON surface

    def fingerprint(self) -> str:
        """The fitted calibrator's provenance fingerprint (delegated)."""
        return self._fitted_calibrator().fingerprint()

    def to_dict(self) -> dict[str, object]:
        """The fitted calibrator's versioned JSON envelope (delegated).

        The sklearn wrapper itself follows sklearn's pickle conventions and
        is outside the JSON's scope — the envelope loads back as the inner
        probcal calibrator.
        """
        return self._fitted_calibrator().to_dict()

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> "str | None":
        """The fitted calibrator's JSON serialization (delegated), never pickle."""
        return self._fitted_calibrator().to_json(path, indent=indent)

    # ------------------------------------------------------------------ tags

    def __sklearn_tags__(self):  # noqa: ANN204 - sklearn protocol, version-dependent type
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        tags.classifier_tags.multi_class = False
        tags.target_tags.required = True
        return tags

    def _more_tags(self) -> dict[str, object]:  # sklearn < 1.6
        return {"binary_only": True, "requires_y": True, **xfail_tags(self)}
