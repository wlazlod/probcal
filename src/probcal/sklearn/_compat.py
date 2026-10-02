"""Version shims for the supported scikit-learn range (>= 1.4)."""

import warnings

import numpy as np

try:  # sklearn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:  # pragma: no cover - sklearn 1.4/1.5

    def _validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        if isinstance(y, str) and y == "no_validation":
            return estimator._validate_data(X, **kwargs)
        return estimator._validate_data(X, y, **kwargs)


def _validate_data_quietly(estimator, *args, **kwargs):
    """``_validate_data``, with sklearn's own finite-check summation warning muted.

    Sklearn's ``check_array`` probes finiteness with a fast ``sum(X)``
    before raising; a two-sided input (e.g. ``[-inf, inf]`` in the same
    row, as a rejected two-column probability matrix produces) makes that
    internal sum itself emit ``RuntimeWarning: invalid value encountered
    in reduce``. The warning is sklearn's implementation detail, not a
    signal for our callers: the resulting ``ValueError`` is unaffected and
    still raised from underneath this filter.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in reduce",
            category=RuntimeWarning,
        )
        return _validate_data(estimator, *args, **kwargs)


def validate_X_y(
    estimator, X, y, *, reset: bool, allow_1d: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """2-D float64 X plus 1-D y, with ``n_features_in_`` bookkeeping."""
    X_arr = X if hasattr(X, "toarray") else np.asarray(X)
    if allow_1d and getattr(X_arr, "ndim", 2) == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _validate_data_quietly(estimator, X_arr, y, reset=reset, dtype=np.float64)


def validate_X(estimator, X, *, reset: bool = False, allow_1d: bool = False) -> np.ndarray:
    """2-D float64 X for predict-side entry points."""
    X_arr = X if hasattr(X, "toarray") else np.asarray(X)
    if allow_1d and getattr(X_arr, "ndim", 2) == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _validate_data_quietly(estimator, X_arr, reset=reset, dtype=np.float64)
