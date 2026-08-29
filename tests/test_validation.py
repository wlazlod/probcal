"""Tests for probcal._validation."""

import numpy as np
import pytest

from probcal._validation import EPS, validate_binary_y, validate_scores, validate_weights


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
    with pytest.raises(ValueError, match="1-D"):
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
    with pytest.raises(ValueError, match=r"expected 1-D scores \(or a single column\); got shape"):
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
