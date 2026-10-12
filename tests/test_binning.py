"""Tests for probcal.binning (histogram and scaling-binning) and the equal-mass edges."""

import numpy as np
import pytest

from probcal._steps import equal_mass_edges
from probcal.binning import HistogramBinningCalibrator, ScalingBinningCalibrator
from probcal.parametric import PlattCalibrator

RNG = np.random.default_rng(17)


def _sample(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.01, 0.99, n)
    y = (RNG.random(n) < s).astype(float)
    return s, y


def _floor_tied_sample() -> tuple[np.ndarray, np.ndarray]:
    # Low-PD scores reported to 3 dp: about a fifth of the rows sit at the floor.
    rng = np.random.default_rng(0)
    s = np.maximum(np.round(rng.beta(0.6, 40, 2000), 3), 0.001)
    y = (rng.random(2000) < s).astype(float)
    return s, y


def _assert_no_empty_bin(values: np.ndarray, edges: np.ndarray) -> None:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    assert np.all(counts > 0)


# ---------------------------------------------------------------- hand-computed fits


def test_histogram_two_bins_hand_computed() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    # Equal-mass: first four scores in bin 1 (rate 1/4), last four in bin 2 (rate 3/4).
    raw = HistogramBinningCalibrator(n_bins=2, shrinkage=None).fit(s, y)
    np.testing.assert_allclose(raw.predict_proba(np.array([0.15, 0.85])), [0.25, 0.75])
    jeffreys = HistogramBinningCalibrator(n_bins=2).fit(s, y)  # the default shrinkage
    np.testing.assert_allclose(
        jeffreys.predict_proba(np.array([0.15, 0.85])), [(1 + 0.5) / (4 + 1), (3 + 0.5) / (4 + 1)]
    )


def test_histogram_weighted() -> None:
    s = np.array([0.1, 0.2, 0.6, 0.7])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    w = np.array([3.0, 1.0, 1.0, 3.0])
    cal = HistogramBinningCalibrator(n_bins=2, shrinkage=None).fit(s, y, sample_weight=w)
    np.testing.assert_allclose(cal.predict_proba(np.array([0.15, 0.65])), [0.25, 0.75])


def test_scaling_binning_outputs_are_platt_bin_means() -> None:
    s, y = _sample(2000)
    cal = ScalingBinningCalibrator(n_bins=8).fit(s, y)
    g = PlattCalibrator().fit(s, y).predict_proba(s)
    # Reconstruct: equal-mass bins of g, means of g per bin.
    edges = np.quantile(g, np.linspace(0, 1, 9)[1:-1])
    idx = np.searchsorted(edges, g, side="right")
    expected_means = np.array([g[idx == b].mean() for b in range(8)])
    p = cal.predict_proba(s)
    assert np.all(np.isin(np.round(p, 10), np.round(expected_means, 10)))


# ---------------------------------------------------------------- empty width bins (0.3.5)


def test_histogram_width_empty_bins_take_neighbour_rate() -> None:
    rng = np.random.default_rng(5)
    s = rng.uniform(0.01, 0.3, 300)
    y = (rng.random(300) < s).astype(float)
    cal = HistogramBinningCalibrator(n_bins=10, strategy="width").fit(s, y)
    filled = cal.bin_weight_ > 0
    assert not filled.all()
    last = np.flatnonzero(filled)[-1]
    # Trailing empty bins carry the last filled bin's rate, not the global rate.
    np.testing.assert_array_equal(cal.bin_rate_[last:], cal.bin_rate_[last])
    assert cal.predict_proba([0.95])[0] == cal.bin_rate_[last]


def test_histogram_width_interior_empty_bin_between_neighbours() -> None:
    s = np.array([0.01, 0.02, 0.05, 0.08, 0.35, 0.36, 0.38])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    cal = HistogramBinningCalibrator(n_bins=10, strategy="width").fit(s, y)
    r = cal.bin_rate_
    assert cal.bin_weight_[1] == 0 and cal.bin_weight_[2] == 0
    assert min(r[0], r[3]) <= r[1] <= r[2] <= max(r[0], r[3])
    assert cal.is_monotone_


# ---------------------------------------------------------------- monotonicity / inverse


def test_scaling_binning_decreasing_platt_is_not_monotone() -> None:
    """CAL-2: a decreasing Platt stage makes the composite decreasing; 0.3.x
    still claimed is_monotone_=True and interval_inverse returned lo > hi."""
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


# ---------------------------------------------------------------- equal-mass edges


def test_equal_mass_edges_use_sample_weights() -> None:
    """CAL-8: "mass" bins carry equal *weight*; 0.3.x ignored the weights."""
    s = np.linspace(0.01, 0.99, 400)
    y = (RNG.random(400) < s).astype(float)
    w = np.where(s < 0.5, 9.0, 1.0)  # 90% of the weight below 0.5
    cal = HistogramBinningCalibrator(n_bins=4).fit(s, y, sample_weight=w)
    mass = cal.bin_weight_ / cal.bin_weight_.sum()
    np.testing.assert_allclose(mass, 0.25, atol=0.02)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_equal_mass_edges_no_empty_bin_with_ties_at_minimum(weighted: bool) -> None:
    """0.3.5: a fifth of the rows tied at the floor left the first bins empty."""
    s, _ = _floor_tied_sample()
    w = np.random.default_rng(3).uniform(0.5, 2.0, len(s)) if weighted else None
    for n_bins in (2, 7, 20):
        edges = equal_mass_edges(s, n_bins, w)
        assert np.all(edges > s.min())
        _assert_no_empty_bin(s, edges)


def test_equal_mass_edges_constant_values_single_bin() -> None:
    assert equal_mass_edges(np.full(30, 0.2), 5).size == 0


def test_equal_mass_edges_unchanged_without_ties() -> None:
    s = np.random.default_rng(4).uniform(0.0, 1.0, 999)
    for n_bins in (2, 10, 31):
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        np.testing.assert_array_equal(equal_mass_edges(s, n_bins), np.unique(np.quantile(s, qs)))


def test_equal_mass_edges_drops_gap_bin_between_ties() -> None:
    # The linear quantile rule can put an edge inside a gap with a tied value
    # as the next edge: the bin between them holds no value.
    s = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5])
    qs = np.linspace(0.0, 1.0, 6)[1:-1]
    np.testing.assert_allclose(np.unique(np.quantile(s, qs)), [0.1, 0.12, 0.2, 0.34])
    edges = equal_mass_edges(s, 5)
    np.testing.assert_allclose(edges, [0.2, 0.34])
    _assert_no_empty_bin(s, edges)


def test_histogram_mass_below_tied_minimum() -> None:
    s, y = _floor_tied_sample()
    cal = HistogramBinningCalibrator(n_bins=10).fit(s, y)
    assert np.all(cal.bin_weight_ > 0)
    below, at_min = cal.predict_proba([0.0005, 0.001])
    assert below == at_min
