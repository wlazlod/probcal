"""Tests for probcal.metrics.scores."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.metrics.scores import (
    brier_score,
    brier_skill_score,
    log_loss,
    logloss_calibration_refinement,
    murphy_decomposition,
)

RNG = np.random.default_rng(53)


def _calibrated(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-1.0, 1.3, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_log_loss_hand_case() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.4])
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert abs(log_loss(y, p) - expected) < 1e-12


def test_brier_hand_case() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.4])
    expected = ((0.2) ** 2 + (0.4) ** 2) / 2
    assert abs(brier_score(y, p) - expected) < 1e-12


def test_weighted_log_loss() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.4])
    w = np.array([3.0, 1.0])
    expected = -(3 * np.log(0.8) + np.log(0.6)) / 4
    assert abs(log_loss(y, p, sample_weight=w) - expected) < 1e-12


def test_brier_skill_score_zero_at_climatology() -> None:
    y, _ = _calibrated(1000)
    p_clim = np.full_like(y, y.mean())
    assert abs(brier_skill_score(y, p_clim)) < 1e-12


def test_murphy_identity_piecewise_constant() -> None:
    # Predictions constant within bins => binned decomposition identity is exact.
    levels = np.array([0.1, 0.3, 0.5, 0.7])
    p = np.repeat(levels, 250)
    y = (RNG.random(1000) < p).astype(float)
    dec = murphy_decomposition(y, p, n_bins=4)
    total = dec.reliability - dec.resolution + dec.uncertainty
    assert abs(total - brier_score(y, p)) < 1e-12


def test_murphy_bias_corrected_reduces_reliability_when_calibrated() -> None:
    y, p = _calibrated(5000)
    naive = murphy_decomposition(y, p, n_bins=15)
    corrected = murphy_decomposition(y, p, n_bins=15, bias_corrected=True)
    assert corrected.reliability < naive.reliability


def test_logloss_decomposition_parts_positive_and_sane() -> None:
    y, p = _calibrated(3000)
    dec = logloss_calibration_refinement(y, p)
    assert dec.calibration >= 0.0
    assert dec.refinement > 0.0
    # Calibrated data: calibration part is a small fraction of the total loss.
    assert dec.calibration < 0.1 * (dec.calibration + dec.refinement)


def test_logloss_decomposition_detects_shift() -> None:
    y, p = _calibrated(3000)
    p_shifted = expit(logit(p) + 1.0)
    dec_ok = logloss_calibration_refinement(y, p)
    dec_bad = logloss_calibration_refinement(y, p_shifted)
    assert dec_bad.calibration > dec_ok.calibration


@pytest.mark.reference
def test_scores_vs_sklearn() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    y, p = _calibrated(2000)
    assert abs(log_loss(y, p) - skm.log_loss(y, p)) < 1e-10
    assert abs(brier_score(y, p) - skm.brier_score_loss(y, p)) < 1e-10
