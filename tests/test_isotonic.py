"""Tests for probcal.isotonic: IsotonicCalibrator and CenteredIsotonicCalibrator."""

import numpy as np

from probcal._results import Interpretation
from probcal.isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator

RNG = np.random.default_rng(11)
GRID = np.linspace(0.001, 0.999, 500)


def _sample(n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.01, 0.99, n)
    y = (RNG.random(n) < s**1.4).astype(float)
    return s, y


def test_isotonic_monotone_on_grid() -> None:
    cal = IsotonicCalibrator().fit(*_sample())
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-15)


def test_isotonic_step_values_are_block_means() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    cal = IsotonicCalibrator().fit(s, y)
    np.testing.assert_allclose(cal.predict_proba(s), [0.0, 1 / 3, 1 / 3, 1 / 3, 1.0])
    assert cal.n_blocks_ == 3


def test_isotonic_out_of_range_clamps() -> None:
    s = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    cal = IsotonicCalibrator().fit(s, y)
    p = cal.predict_proba(np.array([0.001, 0.999]))
    assert p[0] == cal.block_mean_[0]
    assert p[1] == cal.block_mean_[-1]


def test_isotonic_tied_scores_get_one_prediction() -> None:
    s = np.array([0.3, 0.3, 0.3, 0.7, 0.7])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    cal = IsotonicCalibrator().fit(s, y)
    p = cal.predict_proba(np.array([0.3, 0.3]))
    assert p[0] == p[1]
    np.testing.assert_allclose(p[0], 1 / 3)


def test_isotonic_linear_interpolation_between_blocks() -> None:
    s = np.array([0.1, 0.2, 0.5, 0.6])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    cal = IsotonicCalibrator(interpolation="linear").fit(s, y)
    # Midpoints: block1 (0.1,0.2) -> 0.15 with value 0; block2 (0.5,0.6) -> 0.55 with 1.
    p_mid = cal.predict_proba(np.array([0.35]))[0]
    assert 0.0 < p_mid < 1.0
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-15)


def test_isotonic_weighted_fit_differs() -> None:
    s, y = _sample(500)
    w = np.where(s > 0.5, 4.0, 1.0)
    p_unw = IsotonicCalibrator().fit(s, y).predict_proba(GRID)
    p_w = IsotonicCalibrator().fit(s, y, sample_weight=w).predict_proba(GRID)
    assert not np.allclose(p_unw, p_w)


def test_isotonic_interpret() -> None:
    cal = IsotonicCalibrator().fit(*_sample(400))
    interp = cal.interpret()
    assert isinstance(interp, Interpretation)
    assert interp.param_names == ("n_blocks",)
    assert any("block" in m for m in interp.messages)


def test_cir_strict_inside_pooled_region() -> None:
    # Force pooling: decreasing event pattern in the middle.
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    step = IsotonicCalibrator().fit(s, y)
    cir = CenteredIsotonicCalibrator().fit(s, y)
    # Step map is flat on the pooled region [0.2, 0.4]:
    p_step = step.predict_proba(np.array([0.25, 0.35]))
    assert p_step[0] == p_step[1]
    # CIR is strictly increasing there:
    p_cir = cir.predict_proba(np.array([0.25, 0.35]))
    assert p_cir[1] > p_cir[0]


def test_cir_monotone_and_clamped() -> None:
    cal = CenteredIsotonicCalibrator().fit(*_sample())
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-15)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_cir_interpret_mentions_strictness() -> None:
    cal = CenteredIsotonicCalibrator().fit(*_sample(400))
    interp = cal.interpret()
    assert any("strict" in m.lower() for m in interp.messages)
