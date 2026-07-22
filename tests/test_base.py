"""Tests for probcal.base.BaseCalibrator contract."""

import numpy as np
import pytest

from probcal.base import BaseCalibrator
from probcal.parametric import BetaCalibrator, PlattCalibrator

RNG = np.random.default_rng(7)


def _calibrated_sample(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.02, 0.98, n)
    y = (RNG.random(n) < s).astype(float)
    return s, y


def test_get_params_reflects_init() -> None:
    cal = BetaCalibrator(variant="ab")
    assert cal.get_params() == {"variant": "ab"}


def test_set_params_roundtrip() -> None:
    cal = BetaCalibrator()
    assert cal.set_params(variant="a") is cal
    assert cal.get_params()["variant"] == "a"


def test_set_params_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        PlattCalibrator().set_params(nope=1)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PlattCalibrator().predict_proba([0.1, 0.2])


def test_fit_returns_self_and_predict_shapes() -> None:
    s, y = _calibrated_sample()
    cal = PlattCalibrator().fit(s, y)
    assert isinstance(cal, BaseCalibrator)
    p = cal.predict_proba(s[:10])
    assert p.shape == (10,)
    assert np.all((p > 0) & (p < 1))


def test_predict_proba_2d() -> None:
    s, y = _calibrated_sample()
    cal = PlattCalibrator().fit(s, y)
    p2 = cal.predict_proba_2d(s[:5])
    assert p2.shape == (5, 2)
    np.testing.assert_allclose(p2.sum(axis=1), 1.0)
    np.testing.assert_allclose(p2[:, 1], cal.predict_proba(s[:5]))


def test_fit_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        PlattCalibrator().fit([0.1, 0.2], [0, 1, 1])


def test_interval_inverse_not_implemented_yet() -> None:
    s, y = _calibrated_sample()
    cal = PlattCalibrator().fit(s, y)
    with pytest.raises(NotImplementedError):
        cal.interval_inverse(0.0, 0.02)


def test_is_monotone_flag_default_true() -> None:
    assert PlattCalibrator().is_monotone_ is True
