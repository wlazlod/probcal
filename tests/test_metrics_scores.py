"""Tests for probcal.metrics.scores."""

import numpy as np
import pytest

from probcal._math import expit
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


def test_logloss_decomposition_parts_positive_and_sane() -> None:
    y, p = _calibrated(3000)
    dec = logloss_calibration_refinement(y, p)
    assert dec.calibration >= 0.0
    assert dec.refinement > 0.0
    # Calibrated data: calibration part is a small fraction of the total loss.
    assert dec.calibration < 0.1 * (dec.calibration + dec.refinement)


@pytest.mark.reference
def test_scores_vs_sklearn() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    y, p = _calibrated(2000)
    assert abs(log_loss(y, p) - skm.log_loss(y, p)) < 1e-10
    assert abs(brier_score(y, p) - skm.brier_score_loss(y, p)) < 1e-10


# ------------------------------------------------------------------ 0.3.4 fixes


def test_murphy_bias_correction_uses_per_bin_effective_counts() -> None:
    # Regression (MET-7): the correction used raw counts whatever the weights.
    rng = np.random.default_rng(9)
    p = expit(rng.normal(-1.0, 1.0, 2000))
    y = (rng.random(2000) < p).astype(float)
    w = np.where(np.arange(2000) % 20 == 0, 40.0, 1.0)
    a = murphy_decomposition(y, p, bias_corrected=True, sample_weight=w)
    b = murphy_decomposition(y, p, bias_corrected=True, sample_weight=5.0 * w)
    assert a.reliability == pytest.approx(b.reliability, rel=1e-12)
    from probcal.metrics._binstats import bin_stats

    bs = bin_stats(y, p, w, 10, "mass")
    n_eff = bs.w_sum**2 / bs.w2_sum
    var = bs.rate * (1 - bs.rate) / (n_eff - 1)
    want = np.sum(bs.w_sum / w.sum() * np.maximum((bs.p_mean - bs.rate) ** 2 - var, 0.0))
    assert a.reliability == pytest.approx(want, rel=1e-12)
