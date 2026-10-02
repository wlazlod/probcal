"""Value encoding, canonical JSON, hashes, and the shared JSON-IO mixin."""

import hashlib
import json
import math
import os
from typing import Any, Self

import numpy as np

SCHEMA_VERSION = 1
"""Serialization schema version. Every 0.x release reads schema 1; a bump
requires a converter and a changelog entry (the compatibility promise)."""

_FINGERPRINT_DROP = frozenset({"fitted_at_utc", "probcal_version", "timestamp_"})

_NONFINITE = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def _encode_float(x: float) -> object:
    """Finite floats pass through; ``inf``/``-inf``/``nan`` become tagged dicts.

    Strict JSON has no token for non-finite numbers (``json.dumps`` would
    emit ``Infinity``/``NaN``, which JavaScript, jq, and Postgres reject).
    """
    if math.isfinite(x):
        return x
    return {"__float__": "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")}


def _is_probcal_object(v: object) -> bool:
    """Serializable probcal objects expose both ``to_dict`` and ``fingerprint``.

    Checking ``to_dict`` alone would also capture e.g. a pandas DataFrame.
    """
    return callable(getattr(v, "to_dict", None)) and callable(getattr(v, "fingerprint", None))


def encode_value(v: object) -> object:
    """Encode ``v`` to strict-JSON-native types.

    1-D finite float64 arrays become plain lists; any other ndarray is tagged
    with explicit ``dtype``/``shape``; non-finite floats are tagged
    ``{"__float__": "inf"|"-inf"|"nan"}``; a registered probcal object is
    embedded under ``"__probcal__"``; lists, tuples, and dicts encode
    elementwise (tuples become lists).
    """
    if isinstance(v, np.ndarray):
        if v.ndim == 1 and v.dtype == np.float64 and bool(np.all(np.isfinite(v))):
            return v.tolist()
        data = v.tolist()
        if v.dtype.kind == "f" and not bool(np.all(np.isfinite(v))):
            data = encode_value(data)
        return {"__ndarray__": data, "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float):
        return _encode_float(v)
    if _is_probcal_object(v):
        return {"__probcal__": v.to_dict()}  # type: ignore[attr-defined]
    if isinstance(v, (list, tuple)):
        return [encode_value(x) for x in v]
    if isinstance(v, dict):
        return {k: encode_value(x) for k, x in v.items()}
    return v


def decode_value(v: object, *, arrays: bool = True) -> object:
    """Inverse of :func:`encode_value`.

    With ``arrays=True`` (fitted state), plain lists of numbers decode to
    float64 arrays — the only way :func:`encode_value` emits them. With
    ``arrays=False`` (constructor parameters) they stay Python lists, so a
    ``lambdas=[1, 10]`` parameter does not come back as an ndarray. Tagged
    dicts restore dtype and shape; ``"__probcal__"`` payloads dispatch
    through the class registry.
    """
    if isinstance(v, dict):
        if "__float__" in v and len(v) == 1:
            return _NONFINITE[str(v["__float__"])]
        if "__ndarray__" in v:
            data = decode_value(v["__ndarray__"], arrays=False)
            return np.asarray(data, dtype=v["dtype"]).reshape(v["shape"])
        if "__probcal__" in v:
            from ._registry import load

            return load(v["__probcal__"])
        return {k: decode_value(x, arrays=arrays) for k, x in v.items()}
    if isinstance(v, list):
        if arrays and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return np.asarray(v, dtype=np.float64)
        return [decode_value(x, arrays=arrays) for x in v]
    return v


def canonical_json(d: object) -> str:
    """Deterministic strict JSON: sorted keys, no whitespace."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip(d: object) -> object:
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in _FINGERPRINT_DROP}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def fingerprint_of_dict(d: dict) -> str:
    """SHA-256 of the canonical JSON of ``d`` minus version/timestamp keys.

    ``fitted_at_utc``, ``probcal_version``, and ``timestamp_`` are dropped
    recursively so two identical fits produce the same fingerprint
    (``timestamp_`` covers the LogitOffset audit stamp).
    """
    return sha256_hex(canonical_json(_strip(d)))


def data_fingerprint(*arrays: np.ndarray) -> str:
    """SHA-256 of the row-sorted training data: permutation-invariant provenance."""
    cols = [np.asarray(a, dtype=np.float64) for a in arrays]
    order = np.lexsort(tuple(reversed(cols)))
    h = hashlib.sha256()
    for c in cols:
        h.update(c[order].tobytes())
    return h.hexdigest()


def check_schema(d: dict) -> None:
    """Reject payloads written under an unknown schema, naming the writing version.

    Raises
    ------
    ValueError
        If ``d["probcal_schema"]`` differs from :data:`SCHEMA_VERSION`.
    """
    schema = d.get("probcal_schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported probcal_schema {schema!r} (written by probcal "
            f"{d.get('probcal_version', 'unknown')!r}); this build reads schema "
            f"{SCHEMA_VERSION}"
        )


def check_payload(cls: type, d: dict) -> None:
    """Schema check plus "this payload was written by ``cls``".

    Raises
    ------
    ValueError
        On an unknown schema or a ``d["class"]`` other than ``cls.__name__``.
    """
    check_schema(d)
    if d.get("class") != cls.__name__:
        raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")


def envelope(obj: object, **body: object) -> dict[str, object]:
    """The versioned payload header every ``to_dict`` starts with, plus ``body``."""
    from . import __version__

    return {
        "probcal_schema": SCHEMA_VERSION,
        "probcal_version": __version__,
        "class": type(obj).__name__,
        **body,
    }


class JsonIO:
    """``to_json``/``from_json``/``fingerprint`` on top of ``to_dict``/``from_dict``.

    Serialization is JSON, never pickle: auditable, with no code execution
    on load. Output is strict JSON (no ``NaN``/``Infinity`` tokens).
    """

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> str | None:
        """Serialize to strict JSON.

        Parameters
        ----------
        path : path-like or None
            When given, write to this file and return ``None``; otherwise
            return the JSON text.
        indent : int, keyword-only
            JSON indentation.

        Returns
        -------
        str or None
            JSON text, or ``None`` when written to ``path``.
        """
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return None

    @classmethod
    def from_json(cls, path_or_str: object, **kwargs: Any) -> Self:
        """Load from JSON text or a filesystem path.

        Parameters
        ----------
        path_or_str : str or path-like
            JSON text, or a path to a JSON file. A :class:`os.PathLike`
            is always read as a path; a ``str`` is JSON text when its first
            non-blank character is ``{``.
        **kwargs
            Forwarded to ``from_dict`` (e.g. ``model=`` for objects that
            reference an external model).

        Returns
        -------
        object
            The rebuilt instance.
        """
        if isinstance(path_or_str, os.PathLike) or not str(path_or_str).lstrip().startswith("{"):
            with open(os.fspath(path_or_str), encoding="utf-8") as fh:  # type: ignore[call-overload]
                text = fh.read()
        else:
            text = str(path_or_str)
        return cls.from_dict(json.loads(text), **kwargs)  # type: ignore[attr-defined,no-any-return]

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialized form, version- and timestamp-blind.

        Returns
        -------
        str
            Hex digest.
        """
        return fingerprint_of_dict(self.to_dict())
