"""Tests for probcal.binning: histogram binning and scaling-binning."""

import numpy as np

from probcal._results import Interpretation
from probcal.binning import HistogramBinningCalibrator, ScalingBinningCalibrator
from probcal.parametric import PlattCalibrator

RNG = np.random.default_rng(17)
GRID = np.linspace(0.001, 0.999, 400)


def _sample(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.01, 0.99, n)
    y = (RNG.random(n) < s).astype(float)
    return s, y


def test_histogram_two_bins_hand_computed() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    cal = HistogramBinningCalibrator(n_bins=2, shrinkage=None).fit(s, y)
    # Equal-mass: first four scores in bin 1 (rate 1/4), last four in bin 2 (rate 3/4).
    p = cal.predict_proba(np.array([0.15, 0.85]))
    np.testing.assert_allclose(p, [0.25, 0.75])


def test_histogram_jeffreys_shrinkage() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    cal = HistogramBinningCalibrator(n_bins=2).fit(s, y)  # jeffreys default
    p = cal.predict_proba(np.array([0.15, 0.85]))
    np.testing.assert_allclose(p, [(1 + 0.5) / (4 + 1), (3 + 0.5) / (4 + 1)])


def test_histogram_width_strategy_empty_bin_fallback() -> None:
    # Scores concentrated in [0, 0.3]: many width bins are empty.
    s = RNG.uniform(0.01, 0.3, 200)
    y = (RNG.random(200) < s).astype(float)
    cal = HistogramBinningCalibrator(n_bins=10, strategy="width").fit(s, y)
    p = cal.predict_proba(np.array([0.95]))  # empty region
    assert np.isfinite(p[0])
    assert 0.0 < p[0] < 1.0


def test_histogram_monotone_flag_dynamic() -> None:
    s, y = _sample()
    cal = HistogramBinningCalibrator(n_bins=5).fit(s, y)
    rates_monotone = bool(np.all(np.diff(cal.bin_rate_) >= 0))
    assert cal.is_monotone_ == rates_monotone


def test_histogram_weighted() -> None:
    s = np.array([0.1, 0.2, 0.6, 0.7])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    w = np.array([3.0, 1.0, 1.0, 3.0])
    cal = HistogramBinningCalibrator(n_bins=2, shrinkage=None).fit(s, y, sample_weight=w)
    p = cal.predict_proba(np.array([0.15, 0.65]))
    np.testing.assert_allclose(p, [0.25, 0.75])


def test_histogram_interpret() -> None:
    cal = HistogramBinningCalibrator().fit(*_sample(500))
    interp = cal.interpret()
    assert isinstance(interp, Interpretation)
    assert "n_bins" in interp.param_names


def test_scaling_binning_outputs_are_platt_bin_means() -> None:
    s, y = _sample(2000)
    cal = ScalingBinningCalibrator(n_bins=8).fit(s, y)
    platt = PlattCalibrator().fit(s, y)
    g = platt.predict_proba(s)
    # Reconstruct: equal-mass bins of g, means of g per bin.
    edges = np.quantile(g, np.linspace(0, 1, 9)[1:-1])
    idx = np.searchsorted(edges, g, side="right")
    expected_means = np.array([g[idx == b].mean() for b in range(8)])
    p = cal.predict_proba(s)
    # Every prediction must be one of the bin means.
    assert np.all(np.isin(np.round(p, 10), np.round(expected_means, 10)))


def test_scaling_binning_monotone() -> None:
    cal = ScalingBinningCalibrator().fit(*_sample())
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-15)
    assert cal.is_monotone_ is True


def test_scaling_binning_interpret_two_stages() -> None:
    cal = ScalingBinningCalibrator(n_bins=6).fit(*_sample(1000))
    interp = cal.interpret()
    assert "a" in interp.param_names and "b" in interp.param_names
    assert "n_bins" in interp.param_names
