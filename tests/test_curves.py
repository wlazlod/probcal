"""Tests for probcal.curves."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal._results import BeltResult, KernelReliabilityCurve, ReliabilityCurve
from probcal.curves import (
    EcceCurve,
    calibration_belt,
    ecce_curve,
    reliability_binned,
    reliability_loess,
    reliability_smooth,
    reliability_spline,
)
from probcal.datasets import make_pd_portfolio
from probcal.metrics import smooth_ece

RNG = np.random.default_rng(83)


def _calibrated(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_reliability_binned_structure() -> None:
    y, p = _calibrated(2000)
    curve = reliability_binned(y, p, n_bins=10)
    assert isinstance(curve, ReliabilityCurve)
    assert np.all(np.diff(curve.pred_mean) > 0)
    np.testing.assert_allclose(curve.pred_mean_logit, logit(curve.pred_mean), atol=1e-12)
    assert int(curve.count.sum()) == 2000
    assert np.all(curve.ci_low <= curve.event_rate)
    assert np.all(curve.event_rate <= curve.ci_high)


def test_wilson_ci_hand_case() -> None:
    # One bin: n=100, k=10. Wilson at z=1.96.
    y = np.concatenate([np.ones(10), np.zeros(90)])
    p = np.full(100, 0.1)
    curve = reliability_binned(y, p, n_bins=1)
    z = 1.959963984540054
    n, rate = 100.0, 0.1
    denom = 1.0 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denom
    np.testing.assert_allclose(curve.ci_low, [center - half], atol=1e-12)
    np.testing.assert_allclose(curve.ci_high, [center + half], atol=1e-12)


def test_reliability_loess_near_diagonal() -> None:
    y, p = _calibrated()
    curve = reliability_loess(y, p)
    assert len(curve.grid_p) == 100
    assert np.max(np.abs(curve.event_rate - curve.grid_p)) < 0.06
    np.testing.assert_allclose(curve.grid_logit, logit(curve.grid_p), atol=1e-12)


def test_reliability_spline_near_diagonal() -> None:
    y, p = _calibrated()
    curve = reliability_spline(y, p)
    assert np.max(np.abs(curve.event_rate - curve.grid_p)) < 0.06


def test_belt_calibrated_data() -> None:
    y, p = _calibrated(8000)
    belt = calibration_belt(y, p)
    assert isinstance(belt, BeltResult)
    assert belt.p_value > 0.01
    assert 1 <= belt.degree <= 4
    inside = (belt.lower_95 <= belt.grid_p) & (belt.grid_p <= belt.upper_95)
    assert inside.mean() >= 0.9
    assert np.all(belt.lower_80 >= belt.lower_95 - 1e-12)
    assert np.all(belt.upper_80 <= belt.upper_95 + 1e-12)
    assert np.all(belt.lower_95 <= belt.upper_95)


def test_belt_rejects_distortion() -> None:
    y, p = _calibrated(8000)
    p_bad = expit(0.5 * logit(p) - 0.7)
    belt = calibration_belt(y, p_bad)
    assert belt.p_value < 1e-4
    outside = (belt.grid_p < belt.lower_95) | (belt.grid_p > belt.upper_95)
    assert outside.any()


def test_belt_grid_scales_consistent() -> None:
    y, p = _calibrated(3000)
    belt = calibration_belt(y, p)
    np.testing.assert_allclose(belt.grid_logit, logit(belt.grid_p), atol=1e-12)


def test_belt_separated_data_stops_extension() -> None:
    # Perfectly separated outcomes: the degree-1 fit separates (ridge fallback)
    # and the forward LR loop must not extend to higher degrees (IRLS_SPEC W3.3).
    rng = np.random.default_rng(9)
    n = 200
    p = np.concatenate([rng.uniform(0.02, 0.2, n // 2), rng.uniform(0.8, 0.98, n // 2)])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    with pytest.warns(UserWarning, match="[Ss]eparation"):
        belt = calibration_belt(y, p)
    assert belt.degree == 1
    assert np.all(np.isfinite(belt.lower_95)) and np.all(np.isfinite(belt.upper_95))
    assert 0.0 <= belt.p_value <= 1.0


def test_wilson_ci_contains_rate_with_empty_event_bins() -> None:
    # A zero-event bin: analytically the Wilson lower bound touches 0 exactly,
    # and floating-point noise must not push ci_low above the rate (it broke
    # errorbar rendering with negative yerr on exactly this portfolio).
    from probcal import make_pd_portfolio

    port = make_pd_portfolio(n=6000, random_state=42)
    curve = reliability_binned(port.y, port.scores)
    assert float(curve.event_rate[0]) == 0.0  # the offending zero-event bin
    assert np.all(curve.ci_low <= curve.event_rate)
    assert np.all(curve.event_rate <= curve.ci_high)


def test_ecce_curve_hand_case() -> None:
    # 4 points already sorted by p. Residuals: -0.2, -0.4, 0.6, 0.2.
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.2, 0.4, 0.4, 0.8])
    c = ecce_curve(y, p)
    assert isinstance(c, EcceCurve)
    np.testing.assert_allclose(c.frac, [0.25, 0.5, 0.75, 1.0], atol=1e-12)
    np.testing.assert_allclose(c.cumdev, np.cumsum([-0.2, -0.4, 0.6, 0.2]) / 4.0, atol=1e-12)
    var = np.cumsum([0.2 * 0.8, 0.4 * 0.6, 0.4 * 0.6, 0.8 * 0.2])
    np.testing.assert_allclose(c.sd_null, np.sqrt(var) / 4.0, atol=1e-12)
    assert abs(c.stat_max - np.max(np.abs(c.cumdev))) < 1e-15
    assert c.argmax_frac == c.frac[int(np.argmax(np.abs(c.cumdev)))]


def test_ecce_curve_stat_max_matches_metric() -> None:
    from probcal.metrics import ecce

    y, p = _calibrated(2000)
    w = RNG.uniform(0.5, 2.0, 2000)
    assert abs(ecce_curve(y, p).stat_max - ecce(y, p).stat_max) < 1e-15
    c_w = ecce_curve(y, p, sample_weight=w)
    assert abs(c_w.stat_max - ecce(y, p, sample_weight=w).stat_max) < 1e-15
    # Final cumdev equals the metric-consistent weighted mean residual.
    assert abs(c_w.cumdev[-1] - float(np.sum(w * (y - p)) / w.sum())) < 1e-12


def test_reliability_smooth_matches_smooth_ece() -> None:
    d = make_pd_portfolio(n=3000, random_state=11)
    curve = reliability_smooth(d.y, d.scores, n_boot=0)
    assert isinstance(curve, KernelReliabilityCurve)
    assert abs(curve.smooth_ece - smooth_ece(d.y, d.scores)) < 1e-12


def test_reliability_smooth_matches_smooth_ece_exact_path() -> None:
    # bins=None forces the exact O(n) path on both the metric and the curve.
    d = make_pd_portfolio(n=1500, random_state=12)
    curve = reliability_smooth(d.y, d.scores, n_boot=0, bins=None)
    assert abs(curve.smooth_ece - smooth_ece(d.y, d.scores, bins=None)) < 1e-12


def test_reliability_smooth_near_diagonal_on_calibrated_data() -> None:
    # sigma_star is tuned for the smECE aggregate, not for a low-variance
    # curve, so the pointwise kernel estimate needs a large sample to settle
    # within tolerance -- a dedicated RNG/n (not the smaller shared
    # _calibrated fixture) keeps this from being flaky at moderate n.
    rng = np.random.default_rng(0)
    n = 500_000
    p = expit(rng.normal(0.0, 1.0, n))
    y = (rng.random(n) < p).astype(float)
    curve = reliability_smooth(y, p, n_boot=0)
    m = len(curve.grid_p)
    core = slice(m // 20, m - m // 20)  # central 90% of the grid
    assert np.max(np.abs(curve.event_rate[core] - curve.grid_p[core])) < 0.05


def test_reliability_smooth_density_normalized_and_nonnegative() -> None:
    y, p = _calibrated(4000)
    curve = reliability_smooth(y, p, n_boot=0)
    assert curve.density.shape == curve.grid_p.shape
    assert np.all(curve.density >= 0.0)
    assert abs(float(curve.density.sum()) - 1.0) < 1e-10


def test_reliability_smooth_seeded_determinism() -> None:
    y, p = _calibrated(2000)
    c1 = reliability_smooth(y, p, n_boot=30, random_state=7)
    c2 = reliability_smooth(y, p, n_boot=30, random_state=7)
    np.testing.assert_array_equal(c1.event_rate, c2.event_rate)
    np.testing.assert_array_equal(c1.ci_low, c2.ci_low)
    np.testing.assert_array_equal(c1.ci_high, c2.ci_high)


def test_reliability_smooth_ci_brackets_event_rate() -> None:
    y, p = _calibrated(2000)
    curve = reliability_smooth(y, p, n_boot=30, random_state=3)
    assert np.all(curve.ci_low <= curve.event_rate + 1e-12)
    assert np.all(curve.event_rate <= curve.ci_high + 1e-12)


def test_reliability_smooth_no_bootstrap_collapses_band() -> None:
    y, p = _calibrated(1000)
    curve = reliability_smooth(y, p, n_boot=0)
    np.testing.assert_array_equal(curve.ci_low, curve.event_rate)
    np.testing.assert_array_equal(curve.ci_high, curve.event_rate)


def test_reliability_smooth_rejects_bad_level() -> None:
    y, p = _calibrated(500)
    with pytest.raises(ValueError, match="level"):
        reliability_smooth(y, p, level=0.0)
    with pytest.raises(ValueError, match="level"):
        reliability_smooth(y, p, level=1.0)


def test_reliability_smooth_grid_scales_consistent() -> None:
    y, p = _calibrated(1500)
    curve = reliability_smooth(y, p, n_boot=0)
    np.testing.assert_allclose(curve.grid_logit, logit(curve.grid_p), atol=1e-12)
    assert len(curve.grid_p) == 200
