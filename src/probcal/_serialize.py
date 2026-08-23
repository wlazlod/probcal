"""Value encoding, canonical JSON, and hashes for versioned serialization (spec W5)."""

import hashlib
import json

import numpy as np

SCHEMA_VERSION = 1
"""Serialization schema version. Every 0.x release reads schema 1; a bump
requires a converter and a DECISIONS entry (the compatibility promise)."""

_FINGERPRINT_DROP = frozenset({"fitted_at_utc", "probcal_version", "timestamp_"})


def encode_value(v: object) -> object:
    """Encode ``v`` to JSON-native types.

    1-D float64 arrays become plain lists; any other ndarray is tagged with
    explicit ``dtype``/``shape``; an object exposing ``to_dict`` (a
    serializable probcal object) is embedded under ``"__probcal__"``; lists,
    tuples, and dicts encode elementwise (tuples become lists).
    """
    if isinstance(v, np.ndarray):
        if v.ndim == 1 and v.dtype == np.float64:
            return v.tolist()
        return {"__ndarray__": v.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if hasattr(v, "to_dict") and callable(v.to_dict):
        return {"__probcal__": v.to_dict()}
    if isinstance(v, (list, tuple)):
        return [encode_value(x) for x in v]
    if isinstance(v, dict):
        return {k: encode_value(x) for k, x in v.items()}
    return v


def decode_value(v: object) -> object:
    """Inverse of :func:`encode_value`.

    Plain lists of numbers decode to float64 arrays (the only way
    :func:`encode_value` emits them); tagged dicts restore dtype and shape;
    ``"__probcal__"`` payloads dispatch through the class registry.
    """
    if isinstance(v, dict):
        if "__ndarray__" in v:
            return np.asarray(v["__ndarray__"], dtype=v["dtype"]).reshape(v["shape"])
        if "__probcal__" in v:
            from ._registry import load

            return load(v["__probcal__"])
        return {k: decode_value(x) for k, x in v.items()}
    if isinstance(v, list):
        if v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return np.asarray(v, dtype=np.float64)
        return [decode_value(x) for x in v]
    return v


def canonical_json(d: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


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
    recursively so two identical fits produce the same fingerprint (spec W5;
    ``timestamp_`` covers the LogitOffset audit stamp — DECISIONS 73).
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
