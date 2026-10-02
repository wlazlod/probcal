"""Grade-label handling shared by the per-grade tests, Pluto-Tasche and the e-test.

One place resolves labels (plain array or ``Masterscale``), validates an
explicit best-to-worst ``order``, and aggregates per-grade counts.
"""

import numpy as np


def is_masterscale(obj: object) -> bool:
    """Duck test for :class:`probcal.Masterscale` (``assign`` + ``names``)."""
    return hasattr(obj, "assign") and hasattr(obj, "names")


def _resolve_grades(grades: object, p_arr: np.ndarray) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Labels plus grade order from either a label array or a ``Masterscale``.

    An object exposing ``assign`` and ``names`` (a :class:`probcal.Masterscale`)
    assigns the labels from ``p_arr`` and supplies its best-to-worst order,
    restricted to the grades actually present; a plain array is returned as
    strings with ``order=None`` (callers keep their current ordering rule).
    """
    if is_masterscale(grades):
        labels = np.asarray(grades.assign(p_arr)).astype(str)  # type: ignore[attr-defined]
        present = set(labels.tolist())
        order = tuple(str(g) for g in grades.names if str(g) in present)  # type: ignore[attr-defined]
        return labels, order
    return np.asarray(grades).astype(str), None


def grade_labels(
    grades: object, p_arr: np.ndarray, n: int
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """:func:`_resolve_grades` plus the 1-D / length-``n`` check."""
    labels, order = _resolve_grades(grades, p_arr)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError(
            f"grades must be a 1-D array with one label per observation ({n}), "
            f"got shape {labels.shape}"
        )
    return labels, order


def check_order(order: object, labels: np.ndarray) -> tuple[str, ...]:
    """``order`` as a tuple of str; must be a permutation of the unique labels."""
    order_t = tuple(str(g) for g in np.asarray(order).reshape(-1))
    unique_labels = tuple(sorted(np.unique(labels).tolist()))
    if len(set(order_t)) != len(order_t) or tuple(sorted(order_t)) != unique_labels:
        raise ValueError(f"order {order_t} does not match the unique grade labels {unique_labels}")
    return order_t


def resolve_order(
    labels: np.ndarray, order: object, default_order: tuple[str, ...] | None
) -> tuple[str, ...]:
    """Explicit ``order`` (validated) > masterscale order > sorted labels."""
    if order is not None:
        return check_order(order, labels)
    if default_order is not None:
        return default_order
    return tuple(sorted(np.unique(labels).tolist()))


def aggregate(
    labels: np.ndarray,
    order: tuple[str, ...],
    y: np.ndarray,
    *,
    w: np.ndarray | None = None,
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Per-grade ``(n, k, mean p, inverse index)`` in ``order``.

    ``n``/``k`` are raw counts (int64) when ``w`` is ``None``, weighted sums
    (float64) otherwise. ``mean p`` is the unweighted mean prediction per grade
    (``None`` without ``p``). The inverse index maps each observation to its
    position in ``order``.
    """
    pos = {g: i for i, g in enumerate(order)}
    uniq, inv = np.unique(labels, return_inverse=True)
    idx = np.array([pos[str(u)] for u in uniq], dtype=np.intp)[inv]
    m = len(order)
    n: np.ndarray
    k: np.ndarray
    if w is None:
        n = np.bincount(idx, minlength=m).astype(np.int64)
        k = np.bincount(idx, weights=y, minlength=m).round().astype(np.int64)
    else:
        n = np.bincount(idx, weights=w, minlength=m)
        k = np.bincount(idx, weights=w * y, minlength=m)
    pd = None
    if p is not None:
        pd = np.bincount(idx, weights=p, minlength=m) / np.bincount(idx, minlength=m)
    return n, k, pd, idx
