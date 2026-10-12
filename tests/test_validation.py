"""Tests for probcal._validation."""

import numpy as np
import pytest

from probcal._registry import SERIALIZABLE
from probcal._validation import EPS, validate_binary_y, validate_scores, validate_weights
from probcal.base import BaseCalibrator
from probcal.datasets import make_pd_portfolio
from probcal.offset import LogitOffset


def test_validate_scores_clips_to_eps() -> None:
    s = validate_scores([0.0, 0.5, 1.0])
    assert s.dtype == np.float64 and s.ndim == 1
    assert s.tolist() == [EPS, 0.5, 1.0 - EPS]


def test_validate_scores_accepts_column_and_two_column_simplex() -> None:
    s = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(validate_scores(s.reshape(-1, 1)), validate_scores(s))
    np.testing.assert_array_equal(validate_scores(np.column_stack([1.0 - s, s])), s)


def test_validate_binary_y_accepts_bool() -> None:
    y = validate_binary_y(np.array([True, False]))
    assert y.dtype == np.float64 and y.tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: validate_scores(np.array([0.1, np.nan])), "finite"),
        (
            lambda: validate_scores(np.zeros((3, 2))),
            r"expected 1-D scores, a single column, or a two-column probability matrix",
        ),
        (lambda: validate_scores(np.array([[1.2, -0.2], [0.3, 0.7]])), "two-column probability"),
        (lambda: validate_scores(np.full((3, 3), 1.0 / 3.0)), "two-column probability"),
        (lambda: validate_binary_y([0, 1, 2]), "binary"),
        (lambda: validate_binary_y([1, 1, 1]), "both classes"),
        (lambda: validate_binary_y(np.array([[0.0], [1.0]])), "1-D"),
        (lambda: validate_weights([1.0, -1.0], 2), "positive"),
        (lambda: validate_weights([1.0], 2), "length"),
    ],
    ids=[
        "scores-nan",
        "scores-rows-sum-0",
        "scores-entry-1.2",
        "scores-three-columns",
        "y-not-binary",
        "y-single-class",
        "y-column",
        "w-negative",
        "w-length",
    ],
)
def test_validation_rejects(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


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


@pytest.mark.parametrize(
    "kwargs", [{"delta": 0.3}, {"target_mean": 0.03}], ids=["delta", "target_mean"]
)
def test_logit_offset_two_column_fit_matches_column_one(kwargs) -> None:
    p = make_pd_portfolio(n=400, random_state=7).scores
    m = np.column_stack([1.0 - p, p])
    a = LogitOffset(**kwargs).fit(p)
    b = LogitOffset(**kwargs).fit(m)
    assert a.delta_ == b.delta_
    np.testing.assert_array_equal(a.transform(p), b.transform(m))
