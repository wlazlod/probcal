"""Version shims for the supported scikit-learn range (>= 1.4)."""

import numpy as np

try:  # sklearn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:  # pragma: no cover - sklearn 1.4/1.5

    def _validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        if isinstance(y, str) and y == "no_validation":
            return estimator._validate_data(X, **kwargs)
        return estimator._validate_data(X, y, **kwargs)


def validate_X_y(
    estimator, X, y, *, reset: bool, allow_1d: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """2-D float64 X plus 1-D y, with ``n_features_in_`` bookkeeping."""
    X_arr = X if hasattr(X, "toarray") else np.asarray(X)
    if allow_1d and getattr(X_arr, "ndim", 2) == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _validate_data(estimator, X_arr, y, reset=reset, dtype=np.float64)


def validate_X(estimator, X, *, reset: bool = False, allow_1d: bool = False) -> np.ndarray:
    """2-D float64 X for predict-side entry points."""
    X_arr = X if hasattr(X, "toarray") else np.asarray(X)
    if allow_1d and getattr(X_arr, "ndim", 2) == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _validate_data(estimator, X_arr, reset=reset, dtype=np.float64)
