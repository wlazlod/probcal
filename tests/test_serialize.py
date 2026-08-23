"""Serialization round-trips, fingerprints, and the class registry (spec W5)."""

import numpy as np
import pytest

from probcal._serialize import (
    SCHEMA_VERSION,
    canonical_json,
    check_schema,
    data_fingerprint,
    decode_value,
    encode_value,
    fingerprint_of_dict,
)


def test_encode_decode_scalars_and_1d_float64() -> None:
    for v in (1.5, 3, True, "x", None):
        assert decode_value(encode_value(v)) == v
    a = np.array([0.1, 0.2, 0.3])
    enc = encode_value(a)
    assert enc == [0.1, 0.2, 0.3]  # 1-D float64: plain list
    np.testing.assert_array_equal(decode_value(enc), a)


def test_encode_decode_typed_arrays() -> None:
    for a in (np.arange(6, dtype=np.int64), np.arange(6.0).reshape(2, 3)):
        enc = encode_value(a)
        assert enc["dtype"] == str(a.dtype) and enc["shape"] == list(a.shape)
        out = decode_value(enc)
        assert out.dtype == a.dtype and out.shape == a.shape
        np.testing.assert_array_equal(out, a)


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": [2, 3], "b": 1})


def test_fingerprint_ignores_version_timestamps() -> None:
    d1 = {"probcal_version": "0.1.3", "state": {"x": 1.0}, "fit_meta": {"fitted_at_utc": "A"}}
    d2 = {"probcal_version": "9.9.9", "state": {"x": 1.0}, "fit_meta": {"fitted_at_utc": "B"}}
    assert fingerprint_of_dict(d1) == fingerprint_of_dict(d2)
    d3 = {"probcal_version": "0.1.3", "state": {"x": 2.0}, "fit_meta": {"fitted_at_utc": "A"}}
    assert fingerprint_of_dict(d1) != fingerprint_of_dict(d3)


def test_data_fingerprint_permutation_invariant() -> None:
    rng = np.random.default_rng(0)
    s, y, w = rng.random(50), (rng.random(50) < 0.3).astype(float), rng.random(50)
    perm = rng.permutation(50)
    assert data_fingerprint(s, y, w) == data_fingerprint(s[perm], y[perm], w[perm])
    assert data_fingerprint(s, y, w) != data_fingerprint(s, 1.0 - y, w)


def test_check_schema_rejects_unknown_naming_writer() -> None:
    check_schema({"probcal_schema": SCHEMA_VERSION})
    with pytest.raises(ValueError, match="9.9.9"):
        check_schema({"probcal_schema": 999, "probcal_version": "9.9.9"})


@pytest.mark.xfail(strict=False, reason="registry filled from Task B2 onward")
def test_registry_load_dispatches_and_rejects_unknown_class() -> None:
    from probcal._registry import SERIALIZABLE, load

    assert "BetaCalibrator" in SERIALIZABLE
    with pytest.raises(ValueError, match="NoSuchClass"):
        load({"probcal_schema": SCHEMA_VERSION, "class": "NoSuchClass"})
