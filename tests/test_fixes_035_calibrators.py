"""Regression tests for 0.3.5: no empty equal-mass bins, ulp-exact raw band edges."""

import math
import warnings

import numpy as np
import pytest

import probcal as pc
from probcal._steps import equal_mass_edges, refine_lower, refine_upper
from probcal.thresholds import calibrated_bands_to_raw


def _floor_tied_sample() -> tuple[np.ndarray, np.ndarray]:
    # Low-PD scores reported to 3 dp: about a fifth of the rows sit at the floor.
    rng = np.random.default_rng(0)
    s = np.maximum(np.round(rng.beta(0.6, 40, 2000), 3), 0.001)
    y = (rng.random(2000) < s).astype(float)
    return s, y


def _sample(kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.beta(1, 8, 400)
    if kind == "noisy":
        y = (rng.random(400) < s * 1.3).astype(float)
    else:  # labels ordered by score: every binning (hence BBQ) is monotone
        y = (s > np.quantile(s, 0.7)).astype(float)
    return s, y


# ------------------------------------------------------------ equal-mass edges


def _assert_no_empty_bin(values: np.ndarray, edges: np.ndarray) -> None:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    assert np.all(counts > 0)


@pytest.mark.parametrize("n_bins", [2, 5, 7, 20])
def test_equal_mass_edges_no_empty_bin_with_ties_at_minimum(n_bins: int) -> None:
    s, _ = _floor_tied_sample()
    edges = equal_mass_edges(s, n_bins)
    assert np.all(edges > s.min())
    _assert_no_empty_bin(s, edges)


def test_equal_mass_edges_no_empty_bin_weighted() -> None:
    s, _ = _floor_tied_sample()
    w = np.random.default_rng(3).uniform(0.5, 2.0, len(s))
    for n_bins in (3, 6, 12):
        _assert_no_empty_bin(s, equal_mass_edges(s, n_bins, w))


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


# ------------------------------------------------------------ BBQ / histogram


def test_bbq_below_tied_minimum_is_not_the_prior() -> None:
    s, y = _floor_tied_sample()
    cal = pc.BBQCalibrator().fit(s, y)
    below, at_min = cal.predict_proba([0.0005, 0.001])
    assert below == at_min
    assert below < 0.01
    for edges, rates in zip(cal.model_edges_, cal.model_rates_, strict=True):
        assert np.all(edges > s.min())
        assert len(rates) == len(edges) + 1


def test_bbq_constant_scores_constant_map() -> None:
    s = np.full(50, 0.3)
    y = (np.arange(50) < 17).astype(float)
    cal = pc.BBQCalibrator().fit(s, y)
    p = cal.predict_proba([0.0, 0.3, 1.0])
    np.testing.assert_allclose(p, (17 + 0.5) / (50 + 1.0))
    assert cal.is_monotone_


def test_histogram_mass_below_tied_minimum() -> None:
    s, y = _floor_tied_sample()
    cal = pc.HistogramBinningCalibrator(n_bins=10).fit(s, y)
    assert np.all(cal.bin_weight_ > 0)
    below, at_min = cal.predict_proba([0.0005, 0.001])
    assert below == at_min


def test_histogram_width_empty_bins_take_neighbour_rate() -> None:
    rng = np.random.default_rng(5)
    s = rng.uniform(0.01, 0.3, 300)
    y = (rng.random(300) < s).astype(float)
    cal = pc.HistogramBinningCalibrator(n_bins=10, strategy="width").fit(s, y)
    filled = cal.bin_weight_ > 0
    assert not filled.all()
    last = np.flatnonzero(filled)[-1]
    # Trailing empty bins carry the last filled bin's rate, not the global rate.
    np.testing.assert_array_equal(cal.bin_rate_[last:], cal.bin_rate_[last])
    assert cal.predict_proba([0.95])[0] == cal.bin_rate_[last]


def test_histogram_width_interior_empty_bin_between_neighbours() -> None:
    s = np.array([0.01, 0.02, 0.05, 0.08, 0.35, 0.36, 0.38])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    cal = pc.HistogramBinningCalibrator(n_bins=10, strategy="width").fit(s, y)
    r = cal.bin_rate_
    assert cal.bin_weight_[1] == 0 and cal.bin_weight_[2] == 0
    assert min(r[0], r[3]) <= r[1] <= r[2] <= max(r[0], r[3])
    assert cal.is_monotone_


def test_scaling_binning_tied_minimum_no_empty_bin() -> None:
    s, y = _floor_tied_sample()
    cal = pc.ScalingBinningCalibrator(n_bins=10).fit(s, y)
    g = cal.platt_.predict_proba(s)
    assert np.all(cal.edges_ > g.min())
    _assert_no_empty_bin(g, cal.edges_)


# ------------------------------------------------------------ ulp refinement


def test_refine_lower_upper_exact() -> None:
    t = 0.3
    x0 = 0.3
    for start in (x0, math.nextafter(x0, 0.0), math.nextafter(math.nextafter(x0, 1.0), 1.0)):
        assert refine_lower(start, lambda v: v >= t) == t
        assert refine_upper(start, lambda v: v <= t) == t


_INVERTIBLE = [
    pc.PlattCalibrator,
    pc.TemperatureCalibrator,
    lambda: pc.BetaCalibrator("ab"),
    lambda: pc.BetaCalibrator("a"),
    pc.BetaCalibrator,
    pc.IsotonicCalibrator,
    lambda: pc.IsotonicCalibrator("linear"),
    pc.CenteredIsotonicCalibrator,
    pc.HistogramBinningCalibrator,
    pc.ScalingBinningCalibrator,
    pc.BBQCalibrator,
    lambda: pc.BBQCalibrator(min_bins=2, max_bins=3),
    pc.SplineCalibrator,
    pc.VennAbersCalibrator,
    pc.CrossVennAbersCalibrator,
]


@pytest.mark.parametrize("kind", ["noisy", "ordered"])
@pytest.mark.parametrize("factory", _INVERTIBLE)
def test_raw_band_edges_partition_like_assign(factory, kind: str) -> None:
    s, y = _sample(kind, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = factory().fit(s, y)
    if not cal.is_monotone_:
        pytest.skip("non-monotone fit has no interval inverse")
    lo, hi = cal._output_range()
    rng = np.random.default_rng(2)
    for e in rng.uniform(lo, hi, 60):
        ms = pc.Masterscale.from_edges([e])
        edge = calibrated_bands_to_raw(cal, ms)["G2"][0]
        below = math.nextafter(edge, 0.0)
        assert ms.assign(cal.predict_proba([edge]))[0] == "G2"
        assert ms.assign(cal.predict_proba([below]))[0] == "G1"


@pytest.mark.parametrize("kind", ["noisy", "ordered"])
@pytest.mark.parametrize("factory", _INVERTIBLE)
def test_interval_inverse_bounds_in_preimage_and_tight(factory, kind: str) -> None:
    s, y = _sample(kind, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = factory().fit(s, y)
    if not cal.is_monotone_:
        pytest.skip("non-monotone fit has no interval inverse")
    lo, hi = cal._output_range()
    rng = np.random.default_rng(7)
    for t in rng.uniform(lo, hi, 40):
        raw_lo, _ = cal.interval_inverse(t, 1.0)
        _, raw_hi = cal.interval_inverse(0.0, t)
        g = cal.predict_proba
        assert g([raw_lo])[0] >= t > g([math.nextafter(raw_lo, 0.0)])[0]
        assert g([raw_hi])[0] <= t < g([math.nextafter(raw_hi, 1.0)])[0]
