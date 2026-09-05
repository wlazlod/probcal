"""Tests for probcal._validation."""

import numpy as np
import pytest

from probcal._validation import EPS, validate_binary_y, validate_scores, validate_weights
from probcal._registry import SERIALIZABLE
from probcal.base import BaseCalibrator
from probcal.datasets import make_pd_portfolio
from probcal.offset import LogitOffset


def test_validate_scores_clips_to_eps() -> None:
    s = validate_scores(np.array([0.0, 0.5, 1.0]))
    assert s[0] == EPS
    assert s[1] == 0.5
    assert s[2] == 1.0 - EPS


def test_validate_scores_returns_float64_1d() -> None:
    s = validate_scores([0.1, 0.2])
    assert isinstance(s, np.ndarray)
    assert s.dtype == np.float64
    assert s.ndim == 1


def test_validate_scores_rejects_2d() -> None:
    with pytest.raises(ValueError, match=r"two-column probability matrix"):
        validate_scores(np.zeros((2, 2)))


def test_validate_scores_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_scores(np.array([0.1, np.nan]))


def test_validate_binary_y_accepts_01() -> None:
    y = validate_binary_y([0, 1, 1, 0])
    assert y.dtype == np.float64
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_validate_binary_y_accepts_bool() -> None:
    y = validate_binary_y(np.array([True, False]))
    assert y.tolist() == [1.0, 0.0]


def test_validate_binary_y_rejects_other_values() -> None:
    with pytest.raises(ValueError, match="binary"):
        validate_binary_y([0, 1, 2])


def test_validate_binary_y_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both classes"):
        validate_binary_y([1, 1, 1])


def test_validate_weights_default_ones() -> None:
    w = validate_weights(None, 3)
    assert w.tolist() == [1.0, 1.0, 1.0]


def test_validate_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_weights([1.0, -1.0], 2)


def test_validate_weights_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="length"):
        validate_weights([1.0], 2)


def test_validate_scores_accepts_single_column() -> None:
    col = np.linspace(0.1, 0.9, 5).reshape(-1, 1)
    out = validate_scores(col)
    assert out.ndim == 1
    assert np.array_equal(out, validate_scores(col.ravel()))


def test_validate_scores_rejects_two_columns_with_new_message() -> None:
    with pytest.raises(ValueError, match=r"expected 1-D scores, a single column, or a two-column probability matrix"):
        validate_scores(np.zeros((3, 2)))


def test_validate_binary_y_still_rejects_single_column() -> None:
    with pytest.raises(ValueError, match="1-D"):
        validate_binary_y(np.array([[0.0], [1.0]]))


def test_column_vector_fit_predict_matches_ravelled() -> None:
    from probcal import BetaCalibrator

    rng = np.random.default_rng(0)
    s = rng.uniform(0.05, 0.95, 200)
    y = (rng.random(200) < s).astype(float)
    flat = BetaCalibrator().fit(s, y)
    col = BetaCalibrator().fit(s.reshape(-1, 1), y)
    assert np.array_equal(flat.predict_proba(s), col.predict_proba(s.reshape(-1, 1)))


def test_validate_scores_accepts_two_column_simplex() -> None:
    s = np.array([0.1, 0.5, 0.9])
    m = np.column_stack([1.0 - s, s])
    np.testing.assert_array_equal(validate_scores(m), validate_scores(s))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.4, 0.5], [0.3, 0.6]]),          # rows sum to 0.9
        np.array([[1.2, -0.2], [0.3, 0.7]]),          # entry outside [0, 1]
        np.full((3, 3), 1.0 / 3.0),                   # (n, 3) stays rejected
    ],
    ids=["rows-sum-0.9", "entry-1.2", "three-columns"],
)
def test_validate_scores_rejects_non_simplex_matrices(bad) -> None:
    with pytest.raises(ValueError, match=r"two-column probability matrix"):
        validate_scores(bad)


def test_validate_scores_single_column_behaviour_unchanged() -> None:
    s = np.array([[0.2], [0.7]])
    np.testing.assert_array_equal(validate_scores(s), validate_scores(s.ravel()))


CALIBRATOR_CLASSES = sorted(
    (c for c in SERIALIZABLE.values() if isinstance(c, type) and issubclass(c, BaseCalibrator)),
    key=lambda c: c.__name__,
)


@pytest.mark.parametrize("cls", CALIBRATOR_CLASSES, ids=lambda c: c.__name__)
def test_two_column_fit_predict_matches_column_one(cls) -> None:
    d = make_pd_portfolio(n=400, random_state=7)
    q = make_pd_portfolio(n=150, random_state=8).scores
    m_fit = np.column_stack([1.0 - d.scores, d.scores])
    m_q = np.column_stack([1.0 - q, q])
    a = cls().fit(d.scores, d.y)
    b = cls().fit(m_fit, d.y)
    np.testing.assert_array_equal(a.predict_proba(q), b.predict_proba(m_q))


@pytest.mark.parametrize("kwargs", [{"delta": 0.3}, {"target_mean": 0.03}], ids=["delta", "target_mean"])
def test_logit_offset_two_column_fit_matches_column_one(kwargs) -> None:
    p = make_pd_portfolio(n=400, random_state=7).scores
    m = np.column_stack([1.0 - p, p])
    a = LogitOffset(**kwargs).fit(p)
    b = LogitOffset(**kwargs).fit(m)
    assert a.delta_ == b.delta_
    np.testing.assert_array_equal(a.transform(p), b.transform(m))
