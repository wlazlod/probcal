"""Input validation: scores, binary targets, weights, CV settings, inverse-map options."""

import numpy as np

EPS = 1e-12
"""Clipping bound: probabilities are confined to ``[EPS, 1 - EPS]``."""


def validate_scores(s: object, *, name: str = "s") -> np.ndarray:
    """Coerce scores/probabilities to a clipped 1-D float64 array.

    A single-column 2-D input of shape ``(n, 1)`` — what an sklearn
    cross-validation loop hands an estimator — is ravelled before the
    checks below. A two-column 2-D input of shape ``(n, 2)`` is accepted
    as a probability-simplex matrix (each row sums to 1 within 1e-6
    absolute tolerance); the positive class column is extracted. This
    convention is shared across scikit-learn, LightGBM, XGBoost, and
    CatBoost predict_proba outputs, and the simplex check ensures that an
    arbitrary two-feature matrix will not be silently accepted as a
    probability estimate. This holds wherever the function is used:
    calibrator ``fit``/``predict_proba`` and the metrics alike. Every
    other 2-D shape is rejected, and :func:`validate_binary_y` stays
    strict.

    Parameters
    ----------
    s : array_like
        Scores in ``[0, 1]``, shape ``(n,)``, ``(n, 1)``, or ``(n, 2)``.
        A ``(n, 2)`` array is treated as a probability matrix from
        ``predict_proba``; each row must sum to 1 within ``1e-6`` absolute
        tolerance, and all entries must lie in ``[0, 1]``. Values at the
        boundaries are clipped to ``[EPS, 1 - EPS]`` so that logits stay
        finite.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        1-D float64 array clipped to ``[EPS, 1 - EPS]``.

    Raises
    ------
    ValueError
        If the input is neither 1-D nor a single column nor a valid
        two-column simplex, contains non-finite values, or lies outside
        ``[0, 1]``.
    """
    arr = np.asarray(s, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    elif arr.ndim == 2 and arr.shape[1] == 2:
        ok = (
            bool(np.all(np.isfinite(arr)))
            and bool(np.all(arr >= 0.0))
            and bool(np.all(arr <= 1.0))
            and bool(np.all(np.abs(arr.sum(axis=1) - 1.0) <= 1e-6))
        )
        if not ok:
            raise ValueError(
                f"{name}: expected 1-D scores, a single column, or a two-column "
                f"probability matrix (rows summing to 1); got shape {arr.shape}"
            )
        arr = arr[:, 1]
    if arr.ndim != 1:
        raise ValueError(
            f"{name}: expected 1-D scores, a single column, or a two-column "
            f"probability matrix (rows summing to 1); got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(arr, EPS, 1.0 - EPS)


def validate_binary_y(y: object, *, require_both_classes: bool = True) -> np.ndarray:
    """Coerce a binary target to a 1-D float64 array of 0.0 and 1.0.

    Parameters
    ----------
    y : array_like
        Binary outcomes; accepted values are ``{0, 1}`` (int, float, or bool).
    require_both_classes : bool, keyword-only
        Reject a single-class ``y`` (the default). ``False`` serves the
        low-default functions (Pluto-Tasche, Jeffreys bands), where an
        all-zero ``y`` is the motivating case.

    Returns
    -------
    numpy.ndarray
        1-D float64 array with values in ``{0.0, 1.0}``.

    Raises
    ------
    ValueError
        If any value is outside ``{0, 1}``, or only one class is present
        while ``require_both_classes`` is true.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"y must be a 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y must contain only finite values")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("y must be binary with values in {0, 1}")
    if arr.size == 0:
        raise ValueError("y must not be empty")
    if require_both_classes and arr.min() == arr.max():
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


def validate_cv(cv: object, y: np.ndarray) -> int:
    """Validate a fold count for stratified cross-fitting on binary ``y``.

    Returns
    -------
    int
        ``cv`` as an int.

    Raises
    ------
    ValueError
        If ``cv`` is not an integer ``>= 2``, or a class has fewer than two
        members (some training fold would then miss that class).
    """
    if isinstance(cv, bool) or not isinstance(cv, (int, np.integer)) or int(cv) < 2:
        raise ValueError(f"cv must be an integer >= 2, got {cv!r}")
    n_min = int(min(np.sum(y == 0.0), np.sum(y == 1.0)))
    if n_min < 2:
        raise ValueError(
            f"cross-fitting needs at least 2 observations of each class; the rarer "
            f"class has {n_min}"
        )
    return int(cv)


def stratified_folds(y: np.ndarray, cv: int, random_state: object) -> np.ndarray:
    """Fold index per row, stratified by class, shuffled with ``random_state``."""
    rng = np.random.default_rng(random_state)  # type: ignore[arg-type]
    folds = np.empty(len(y), dtype=np.int64)
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(y == cls)
        perm = rng.permutation(idx)
        folds[perm] = np.arange(len(perm)) % cv
    return folds


def validate_space(space: str) -> None:
    """``space`` must be ``"probability"`` or ``"logit"``."""
    if space not in ("probability", "logit"):
        raise ValueError(f"space must be 'probability' or 'logit', got {space!r}")


def validate_positive_int(value: object, name: str) -> int:
    """``value`` must be an integer ``>= 1`` (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
