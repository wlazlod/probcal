"""Serialization round-trips, fingerprints, and the class registry."""

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


def test_registry_load_dispatches_and_rejects_unknown_class() -> None:
    from probcal._registry import SERIALIZABLE, load

    assert "BetaCalibrator" in SERIALIZABLE
    with pytest.raises(ValueError, match="NoSuchClass"):
        load({"probcal_schema": SCHEMA_VERSION, "class": "NoSuchClass"})


# ---------------------------------------------------------------- battery
# Parametrized over the registry: registering a class adds it to every test
# below. Special construction cases are handled in _fitted/_predict.

from probcal import make_pd_portfolio  # noqa: E402
from probcal._registry import SERIALIZABLE  # noqa: E402

_D = make_pd_portfolio(n=1200, random_state=7)
_Q = make_pd_portfolio(n=400, random_state=8).scores  # held-out query


class _StubModel:
    """Deterministic sklearn-free model over a single score column."""

    def fit(self, X, y):  # noqa: ARG002 - signature parity for the cv flow
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def _monitor(seed0: int = 50):
    from probcal.monitor import CalibrationMonitor

    mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 41))
    for k in range(3):
        d = make_pd_portfolio(n=300, event_rate=0.1, random_state=seed0 + k)
        rng = np.random.default_rng(seed0 + 100 + k)
        y = (rng.random(300) < d.scores).astype(float)
        mon.update(y, d.scores, label=f"b{k}")
    return mon


def _applied_action(seed0: int = 50):
    from probcal._math import expit, logit

    mon = _monitor(seed0=seed0)
    for k in range(3):
        d = make_pd_portfolio(n=300, event_rate=0.1, random_state=seed0 + k)
        p = d.scores
        rng = np.random.default_rng(seed0 + 200 + k)
        y = (rng.random(300) < expit(logit(p) + 0.8)).astype(float)
        mon.update(y, p, label=f"drift{k}")
    return mon.apply_recommendation(target=None)


def _fitted(name: str):
    cls = SERIALIZABLE[name]
    if name == "CalibrationMonitor":
        return _monitor()
    if name == "AppliedAction":
        return _applied_action()
    if name == "Chain":
        from probcal import BetaCalibrator, Chain, LogitOffset

        cal = BetaCalibrator().fit(_D.scores, _D.y)
        off = LogitOffset(delta=0.2).fit(cal.predict_proba(_D.scores))
        return Chain([cal, off])
    if name == "LogitOffset":
        return cls(delta=0.3).fit(_D.scores)
    if name == "CalibratedModel":
        from probcal import BetaCalibrator

        wrapped = cls(_StubModel(), BetaCalibrator(), flow="prefit").fit(
            _D.scores.reshape(-1, 1), _D.y
        )
        wrapped.offset_to(delta=0.15)
        return wrapped
    return cls().fit(_D.scores, _D.y)


def _predict(obj, q):
    if type(obj).__name__ == "CalibrationMonitor":
        return np.asarray([s.e_global for s in obj.steps_])
    if type(obj).__name__ == "LogitOffset":
        return obj.transform(q)
    if type(obj).__name__ == "CalibratedModel":
        return obj.predict_proba(q.reshape(-1, 1))
    if type(obj).__name__ == "AppliedAction":
        return np.asarray([obj.offset.delta_] if obj.offset is not None else [0.0])
    return obj.predict_proba(q)


@pytest.fixture(scope="module", params=sorted(SERIALIZABLE))
def fitted(request):
    return request.param, _fitted(request.param)


def test_round_trip_bit_identical(fitted) -> None:
    name, obj = fitted
    cls = SERIALIZABLE[name]
    js = obj.to_json()
    if name == "CalibratedModel":  # the model is a reference: reattach on load
        obj2 = cls.from_json(js, model=_StubModel())
    else:
        obj2 = cls.from_json(js)
    np.testing.assert_array_equal(_predict(obj, _Q), _predict(obj2, _Q))


def test_interpret_survives_round_trip(fitted) -> None:
    name, obj = fitted
    if not hasattr(obj, "interpret"):
        pytest.skip("no interpret()")
    obj2 = SERIALIZABLE[name].from_dict(obj.to_dict())
    assert repr(obj.interpret()) == repr(obj2.interpret())


def test_to_dict_idempotent(fitted) -> None:
    name, obj = fitted
    d = obj.to_dict()
    if name == "CalibratedModel":
        assert SERIALIZABLE[name].from_dict(d, model=_StubModel()).to_dict() == d
    else:
        assert SERIALIZABLE[name].from_dict(d).to_dict() == d


def test_fingerprint_stable_and_data_sensitive(fitted) -> None:
    name, obj = fitted
    twin = _fitted(name)
    assert obj.fingerprint() == twin.fingerprint()  # identical data -> identical print
    other = make_pd_portfolio(n=1200, random_state=99)
    if name == "CalibrationMonitor":
        alt = _monitor(seed0=77)
        assert obj.fingerprint() != alt.fingerprint()
        return
    if name == "AppliedAction":
        alt = _applied_action(seed0=77)
        assert obj.fingerprint() != alt.fingerprint()
        return
    if name == "Chain":
        from probcal import BetaCalibrator, Chain, LogitOffset

        cal = BetaCalibrator().fit(other.scores, other.y)
        off = LogitOffset(delta=0.2).fit(cal.predict_proba(other.scores))
        assert obj.fingerprint() != Chain([cal, off]).fingerprint()
        return
    if name == "Chain":
        from probcal import BetaCalibrator, Chain, LogitOffset

        cal = BetaCalibrator().fit(_D.scores, _D.y)
        off = LogitOffset(delta=0.2).fit(cal.predict_proba(_D.scores))
        return Chain([cal, off])
    if name == "LogitOffset":
        alt = SERIALIZABLE[name](delta=0.3).fit(other.scores)
    elif name == "CalibratedModel":
        from probcal import BetaCalibrator

        alt = SERIALIZABLE[name](_StubModel(), BetaCalibrator(), flow="prefit").fit(
            other.scores.reshape(-1, 1), other.y
        )
    else:
        alt = SERIALIZABLE[name]().fit(other.scores, other.y)
    assert obj.fingerprint() != alt.fingerprint()


def test_unknown_schema_raises(fitted) -> None:
    name, obj = fitted
    d = obj.to_dict()
    d["probcal_schema"] = 999
    with pytest.raises(ValueError, match="probcal_schema"):
        SERIALIZABLE[name].from_dict(d)


def test_wrong_class_from_dict_raises() -> None:
    from probcal import BetaCalibrator, PlattCalibrator

    d = BetaCalibrator().fit(_D.scores, _D.y).to_dict()
    with pytest.raises(ValueError, match="BetaCalibrator"):
        PlattCalibrator.from_dict(d)


def test_base_from_dict_dispatches() -> None:
    from probcal import BetaCalibrator
    from probcal.base import BaseCalibrator

    cal = BetaCalibrator().fit(_D.scores, _D.y)
    obj = BaseCalibrator.from_dict(cal.to_dict())
    assert isinstance(obj, BetaCalibrator)


def test_to_json_writes_file(tmp_path) -> None:
    from probcal import PlattCalibrator

    cal = PlattCalibrator().fit(_D.scores, _D.y)
    path = tmp_path / "cal.json"
    assert cal.to_json(path) is None
    loaded = PlattCalibrator.from_json(path)
    np.testing.assert_array_equal(cal.predict_proba(_Q), loaded.predict_proba(_Q))


def test_calibrated_model_from_dict_reattaches_model() -> None:
    from probcal import BetaCalibrator, CalibratedModel

    model = _StubModel()
    w = CalibratedModel(model, BetaCalibrator(), flow="prefit", model_id="stub-1").fit(
        _D.scores.reshape(-1, 1), _D.y
    )
    d = w.to_dict()
    assert d["state"]["model_ref"]["model_id"] == "stub-1"
    assert d["state"]["model_ref"]["class_name"] == "_StubModel"
    w2 = CalibratedModel.from_dict(d, model=model)
    np.testing.assert_array_equal(
        w.predict_proba(_Q.reshape(-1, 1)), w2.predict_proba(_Q.reshape(-1, 1))
    )
    with pytest.raises(RuntimeError, match="pass X"):
        w2.offset_to(target_mean=0.05)
    w2.offset_to(delta=0.1, X=_Q.reshape(-1, 1))  # explicit X works after reload


def test_calibrated_model_ensemble_refuses_serialization() -> None:
    from probcal import BetaCalibrator, CalibratedModel

    w = CalibratedModel(_StubModel(), BetaCalibrator(), flow="cv", ensemble=True, cv=3).fit(
        _D.scores.reshape(-1, 1), _D.y
    )
    with pytest.raises(NotImplementedError, match="ensemble"):
        w.to_dict()
