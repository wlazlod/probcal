"""Grade/segment label keys and the masterscale duck test, shared package-wide.

Every place that groups by a label (per-grade tests, Pluto-Tasche, the
report, ``evaluate(by=...)``, ``SegmentedCalibrator`` segments, the monitor)
keys it through :func:`grade_key`, so ``1``, ``1.0`` and ``np.int64(1)`` are
one label everywhere.
"""

import numpy as np


def is_masterscale(obj: object) -> bool:
    """Duck test for :class:`probcal.Masterscale` (``assign`` + ``names``)."""
    return hasattr(obj, "assign") and hasattr(obj, "names")


def grade_key(v: object) -> str:
    """Canonical string key for one grade label.

    Integers and integer-valued floats share a key (``1``, ``1.0``,
    ``np.int64(1)`` all map to ``"1"``), other floats use their shortest
    repr (of the value as a Python ``float``, so a ``float32`` label keys
    as the exact double it holds), bytes are decoded as ASCII, and
    everything else is ``str(v)`` as a plain ``str`` (never ``numpy.str_``).
    Strings are opaque labels: ``"1.0"`` stays ``"1.0"``.
    """
    if isinstance(v, (bytes, np.bytes_)):
        return bytes(v).decode("ascii")
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    return str(v)


def grade_keys(grade: object, *, name: str = "grade") -> np.ndarray:
    """:func:`grade_key` applied element-wise to a 1-D label array (unicode array out).

    Element-wise exactly: ``grade_keys(g)[i] == grade_key(g[i])``. A plain
    list or tuple is read as an object array, so a mixed list such as
    ``[1.0, "A"]`` keys as ``["1", "A"]`` rather than being coerced to
    strings first; typed arrays key each distinct value once.
    """
    if isinstance(grade, (list, tuple)):
        arr = np.empty(len(grade), dtype=object)
        arr[:] = list(grade)
    else:
        arr = np.asarray(grade)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if arr.dtype.kind == "U":
        return arr.astype(str)
    if arr.dtype.kind == "O":
        return np.array([grade_key(x) for x in arr.tolist()], dtype=str)
    uniq, inverse = np.unique(arr, return_inverse=True)
    keys = np.array([grade_key(x) for x in uniq.tolist()], dtype=str)
    return keys[inverse.reshape(-1)]
