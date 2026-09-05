"""SklearnOffset: a LogitOffset as an sklearn transformer over a probability column."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from ..offset import LogitOffset
from ._compat import OFFSET_XFAIL_CHECKS, validate_X


class SklearnOffset(TransformerMixin, BaseEstimator):
    """A logit offset over a probability column, sklearn-style.

    Wraps :class:`~probcal.offset.LogitOffset` as an sklearn transformer
    whose ``X`` is the probability itself — shape ``(n,)``, ``(n, 1)``, or
    a two-column ``predict_proba``-style matrix — so it can end (or sit
    inside) a ``Pipeline`` right after a :class:`~probcal.sklearn.SklearnCalibrator`
    step. The offset is deliberately a *separate* pipeline step rather than
    a parameter folded into the calibrator: it keeps the central-tendency
    re-anchoring inspectable and swappable on its own, exactly as
    :class:`~probcal.chain.Chain` keeps it a separate stage. No ``y`` is
    consumed (``LogitOffset`` ignores it), so there is no orientation
    check to run — the column-1 convention for two-column input is
    documented (``guide/sklearn.md``), not checked.

    Parameters
    ----------
    delta : float or None
        Mode A: the log-odds shift to apply directly. Mutually exclusive
        with ``target_mean``.
    target_mean : float or None
        Mode B: the desired post-shift portfolio mean probability,
        solved by bisection. Mutually exclusive with ``delta``.
    positive_column : int, keyword-only
        Which column of a two-column probability matrix holds ``P(y=1)`` —
        ``0`` or ``1`` (default ``1``, matching ``predict_proba`` output).
        Ignored for single-column input.

    Attributes
    ----------
    offset_ : LogitOffset
        The fitted inner offset.
    n_features_in_ : int
        1 or 2, depending on the ``X`` shape seen at fit; enforced at
        predict/transform time.
    """

    def __init__(
        self,
        delta: float | None = None,
        target_mean: float | None = None,
        *,
        positive_column: int = 1,
    ) -> None:
        self.delta = delta
        self.target_mean = target_mean
        self.positive_column = positive_column

    # ------------------------------------------------------------------ helpers

    def _probs(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == 1:
            return X[:, 0]
        if X.shape[1] == 2:
            if self.positive_column == 0:
                X = X[:, ::-1]
            return X  # (n, 2): validate_scores checks the simplex and takes column 1
        raise ValueError(
            "SklearnOffset takes one probability column or a two-column "
            f"probability matrix, got {X.shape[1]} columns"
        )

    # ------------------------------------------------------------------ estimator API

    def fit(self, X: object, y: object = None, sample_weight: object = None) -> "SklearnOffset":
        """Fit the wrapped offset on the probability column.

        Parameters
        ----------
        X : array_like of shape (n,), (n, 1), or (n, 2)
            Probabilities, or a two-column probability matrix.
        y : array_like or None
            Ignored; accepted for pipeline/estimator compatibility.
        sample_weight : array_like or None
            Positive observation weights.

        Returns
        -------
        SklearnOffset
            The fitted adapter.

        Raises
        ------
        ValueError
            If ``X``'s column count is unsupported or ``positive_column``
            is invalid.
        """
        if self.positive_column not in (0, 1):
            raise ValueError(f"positive_column must be 0 or 1, got {self.positive_column!r}")
        X_arr = validate_X(self, X, reset=True, allow_1d=True)
        sw = None if sample_weight is None else _check_sample_weight(sample_weight, X_arr)
        p = self._probs(X_arr)
        if sw is not None:
            # Zero weight means excluded (sklearn semantics); probcal requires
            # strictly positive weights, so drop those rows here.
            keep = sw > 0.0
            p, sw = p[keep], sw[keep]
        self.offset_ = LogitOffset(delta=self.delta, target_mean=self.target_mean)
        self.offset_.fit(p, sample_weight=sw, y=y)
        return self

    def transform(self, X: object) -> np.ndarray:
        """Shifted probability column ``(n, 1)`` — lets the adapter end a Pipeline."""
        return self.predict_proba(X)[:, [1]]

    def predict_proba(self, X: object) -> np.ndarray:
        """``(n, 2)`` matrix ``[1 - p_shifted, p_shifted]``."""
        check_is_fitted(self, "offset_")
        X_arr = validate_X(self, X, allow_1d=True)
        p = self.offset_.transform(self._probs(X_arr))
        return np.column_stack([1.0 - p, p])

    def to_dict(self) -> dict[str, object]:
        """The fitted inner offset's own envelope (loads back as a LogitOffset)."""
        check_is_fitted(self, "offset_")
        return self.offset_.to_dict()

    # ------------------------------------------------------------------ tags

    def __sklearn_tags__(self):  # noqa: ANN204 - sklearn protocol, version-dependent type
        return super().__sklearn_tags__()

    def _more_tags(self) -> dict[str, object]:  # sklearn < 1.6
        return {"_xfail_checks": dict(OFFSET_XFAIL_CHECKS)}
