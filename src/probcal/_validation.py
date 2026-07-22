"""Input validation: binary-target checks, probability clipping, logit/expit helpers."""

import numpy as np

EPS = 1e-12
"""Clipping bound: probabilities are confined to ``[EPS, 1 - EPS]``."""


def validate_scores(s: object, *, name: str = "s") -> np.ndarray:
    """Coerce scores/probabilities to a clipped 1-D float64 array.

    Parameters
    ----------
    s : array_like
        Scores in ``[0, 1]``. Values at the boundaries are clipped to
        ``[EPS, 1 - EPS]`` so that logits stay finite.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        1-D float64 array clipped to ``[EPS, 1 - EPS]``.

    Raises
    ------
    ValueError
        If the input is not 1-D, contains non-finite values, or lies
        outside ``[0, 1]``.
    """
    arr = np.asarray(s, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(arr, EPS, 1.0 - EPS)


def validate_binary_y(y: object) -> np.ndarray:
    """Coerce a binary target to a 1-D float64 array of 0.0 and 1.0.

    Parameters
    ----------
    y : array_like
        Binary outcomes; accepted values are ``{0, 1}`` (int, float, or bool).

    Returns
    -------
    numpy.ndarray
        1-D float64 array with values in ``{0.0, 1.0}``.

    Raises
    ------
    ValueError
        If any value is outside ``{0, 1}``, or only one class is present.
    """
    arr = np.asarray(y)
    if arr.dtype == bool:
        arr = arr.astype(np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"y must be a 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y must contain only finite values")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("y must be binary with values in {0, 1}")
    if arr.min() == arr.max():
        raise ValueError("y must contain both classes")
    return arr


def validate_weights(w: object, n: int) -> np.ndarray:
    """Coerce sample weights to a 1-D positive float64 array of length ``n``.

    Parameters
    ----------
    w : array_like or None
        Sample weights; ``None`` yields unit weights.
    n : int
        Expected length.

    Returns
    -------
    numpy.ndarray
        1-D float64 array of positive weights.

    Raises
    ------
    ValueError
        If the length differs from ``n`` or any weight is not strictly positive.
    """
    if w is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(w, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"sample_weight must have length {n}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("sample_weight must contain only positive finite values")
    return arr
