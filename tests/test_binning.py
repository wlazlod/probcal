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


# ---------------------------------------------------------------- 0.4.0 regressions


def test_scaling_binning_decreasing_platt_is_not_monotone() -> None:
    """CAL-2: a decreasing Platt stage makes the composite decreasing; 0.3.x
    still claimed is_monotone_=True and interval_inverse returned lo > hi."""
    import pytest

    s, y = _sample(2000)
    cal = ScalingBinningCalibrator(n_bins=5).fit(s, 1.0 - y)  # labels reversed
    assert cal.platt_.a_ < 0.0
    assert cal.is_monotone_ is False
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.interval_inverse(0.2, 0.6)
    again = ScalingBinningCalibrator.from_json(cal.to_json())
    assert again.is_monotone_ is False


def test_scaling_binning_inverse_bounds_stay_in_preimage() -> None:
    """CAL-11: both bounds are inside the preimage even at bin-level targets."""
    cal = ScalingBinningCalibrator(n_bins=12).fit(*_sample(3000))
    levels = cal.bin_value_
    for j in range(1, len(levels) - 1):
        for lo, hi in ((levels[0], levels[j]), (levels[j], levels[-1]), (levels[j], levels[j])):
            raw_lo, raw_hi = cal.interval_inverse(float(lo), float(hi))
            p = cal.predict_proba(np.array([raw_lo, raw_hi]))
            assert lo <= p[0] <= hi and lo <= p[1] <= hi, (j, lo, hi, p)


def test_histogram_inverse_bounds_stay_in_preimage() -> None:
    s, y = _sample(4000)
    cal = HistogramBinningCalibrator(n_bins=4, shrinkage=None).fit(np.sort(s), np.sort(y))
    assert cal.is_monotone_
    levels = cal.bin_rate_
    for j in range(len(levels)):
        raw_lo, raw_hi = cal.interval_inverse(float(levels[j]), float(levels[j]))
        p = cal.predict_proba(np.array([raw_lo, raw_hi]))
        np.testing.assert_array_equal(p, [levels[j], levels[j]])


def test_equal_mass_edges_use_sample_weights() -> None:
    """CAL-8: "mass" bins carry equal *weight*; 0.3.x ignored the weights."""
    s = np.linspace(0.01, 0.99, 400)
    y = (RNG.random(400) < s).astype(float)
    w = np.where(s < 0.5, 9.0, 1.0)  # 90% of the weight below 0.5
    cal = HistogramBinningCalibrator(n_bins=4).fit(s, y, sample_weight=w)
    mass = cal.bin_weight_ / cal.bin_weight_.sum()
    np.testing.assert_allclose(mass, 0.25, atol=0.02)
    unit = HistogramBinningCalibrator(n_bins=4).fit(s, y)
    np.testing.assert_array_equal(unit.edges_, np.quantile(s, [0.25, 0.5, 0.75]))


def test_scaling_binning_weighted_edges() -> None:
    s = np.linspace(0.01, 0.99, 400)
    y = (RNG.random(400) < s).astype(float)
    w = np.where(s < 0.5, 9.0, 1.0)
    cal = ScalingBinningCalibrator(n_bins=4).fit(s, y, sample_weight=w)
    g = cal.platt_.predict_proba(s)
    idx = np.searchsorted(cal.edges_, g, side="right")
    mass = np.bincount(idx, weights=w, minlength=4) / w.sum()
    np.testing.assert_allclose(mass, 0.25, atol=0.02)


def test_n_bins_validated_at_fit_not_init() -> None:
    """CAL-25: sklearn convention — construction never raises."""
    import pytest

    s, y = _sample(200)
    for bad in (0, -3, 2.5, True, None):
        hist = HistogramBinningCalibrator(n_bins=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_bins"):
            hist.fit(s, y)
        sb = ScalingBinningCalibrator(n_bins=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_bins"):
            sb.fit(s, y)
