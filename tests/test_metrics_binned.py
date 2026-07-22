"""Tests for probcal.metrics.binned."""

import numpy as np
import pytest

from probcal._math import expit
from probcal.metrics.binned import adaptive_ece, ece, ece_debiased, ece_sweep, hosmer_lemeshow

RNG = np.random.default_rng(59)


def _calibrated(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_ece_two_bin_hand_case() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    # Bin 1: mean p 0.1, rate 0.5, gap 0.4; bin 2: mean p 0.9, rate 1.0, gap 0.1.
    assert abs(ece(y, p, n_bins=2) - (0.4 + 0.1) / 2) < 1e-12
    assert abs(ece(y, p, n_bins=2, norm="max") - 0.4) < 1e-12
    expected_l2 = np.sqrt((0.4**2 + 0.1**2) / 2)
    assert abs(ece(y, p, n_bins=2, norm="l2") - expected_l2) < 1e-12


def test_adaptive_ece_is_equal_mass_alias() -> None:
    y, p = _calibrated(1000)
    assert adaptive_ece(y, p, n_bins=10) == ece(y, p, n_bins=10, strategy="mass")


def test_ece_debiased_below_plain_and_near_zero_when_calibrated() -> None:
    y, p = _calibrated(8000)
    plain = ece(y, p, n_bins=15)
    debiased = ece_debiased(y, p, n_bins=15)
    assert debiased <= plain
    assert debiased < 0.02


def test_ece_positive_bias_on_calibrated_data() -> None:
    # A perfectly calibrated model still shows positive plain ECE (the bias).
    y, p = _calibrated(2000)
    assert ece(y, p, n_bins=30) > 0.0


def test_ece_sweep_returns_reasonable_value() -> None:
    y, p = _calibrated(4000)
    v = ece_sweep(y, p)
    assert 0.0 <= v < 0.1


def test_hosmer_lemeshow_result_fields() -> None:
    y, p = _calibrated(2000)
    res = hosmer_lemeshow(y, p, g=10)
    assert res.df == 8
    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0


def test_hosmer_lemeshow_rejects_gross_miscalibration() -> None:
    y, p = _calibrated(5000)
    p_bad = np.clip(p * 0.3, 1e-6, 1 - 1e-6)
    assert hosmer_lemeshow(y, p_bad).p_value < 0.001


@pytest.mark.reference
def test_hl_pvalue_vs_scipy_chi2() -> None:
    stats = pytest.importorskip("scipy.stats")
    y, p = _calibrated(2000)
    res = hosmer_lemeshow(y, p, g=10)
    expected = float(stats.chi2.sf(res.statistic, res.df))
    assert abs(res.p_value - expected) < 1e-9
