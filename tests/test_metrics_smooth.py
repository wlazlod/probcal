"""Tests for probcal.metrics.smooth."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.metrics.smooth import (
    _ici_distances,
    e50,
    e90,
    ecce,
    emax,
    ici,
    smooth_ece,
    spiegelhalter_z,
)

RNG = np.random.default_rng(61)

_GRID_CONFIGS = ({}, {"slope": 1.0, "asymmetry": 0.0}, {"event_rate": 0.10})


def _calibrated(n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_smooth_ece_small_when_calibrated_larger_when_shifted() -> None:
    y, p = _calibrated()
    v_ok = smooth_ece(y, p)
    v_bad = smooth_ece(y, expit(logit(p) + 1.2))
    assert v_ok < 0.05
    assert v_bad > 2 * v_ok


def test_ecce_hand_case() -> None:
    p = np.array([0.2, 0.4, 0.6])
    y = np.array([0.0, 1.0, 1.0])
    # Sorted by p already. Cumulative (y - p)/n: (-0.2, 0.4, 0.8)/3.
    res = ecce(y, p)
    np.testing.assert_allclose(res.stat_max, 0.8 / 3)
    np.testing.assert_allclose(res.stat_mean, (0.2 + 0.4 + 0.8) / 9)


def test_ecce_small_when_calibrated() -> None:
    y, p = _calibrated()
    assert ecce(y, p).stat_max < 0.05


def test_ici_family_ordering() -> None:
    y, p = _calibrated()
    v_ici = ici(y, p)
    assert 0.0 <= v_ici < 0.05
    assert e50(y, p) <= e90(y, p) <= emax(y, p)


def test_ici_detects_shift() -> None:
    y, p = _calibrated()
    assert ici(y, expit(logit(p) + 1.0)) > 5 * ici(y, p)


def test_spiegelhalter_near_zero_when_calibrated() -> None:
    y, p = _calibrated(8000)
    res = spiegelhalter_z(y, p)
    assert abs(res.z) < 3.0
    assert 0.0 < res.p_value <= 1.0


def test_spiegelhalter_rejects_overconfidence() -> None:
    y, p = _calibrated(8000)
    p_over = expit(2.0 * logit(p))  # spread out: overconfident
    res = spiegelhalter_z(y, p_over)
    assert res.p_value < 0.001


@pytest.mark.parametrize("kw", _GRID_CONFIGS)
def test_ici_family_grid_default_close_to_exact(kw: dict) -> None:
    d = make_pd_portfolio(n=5000, **kw)
    for fn, tol in ((ici, 1e-4), (e50, 1e-4), (e90, 1e-4), (emax, 1e-3)):
        assert abs(fn(d.y, d.scores) - fn(d.y, d.scores, grid_size=None)) <= tol


@pytest.mark.parametrize("kw", _GRID_CONFIGS)
def test_smooth_ece_binned_close_to_exact(kw: dict) -> None:
    d = make_pd_portfolio(n=5000, **kw)
    assert abs(smooth_ece(d.y, d.scores, bins=1024) - smooth_ece(d.y, d.scores, bins=None)) <= 1e-3


def test_smooth_ece_default_exact_below_bin_count() -> None:
    d = make_pd_portfolio(n=2000)  # n <= 8192: default must be bit-identical to exact
    assert smooth_ece(d.y, d.scores) == smooth_ece(d.y, d.scores, bins=None)


def test_e50_unweighted_equals_all_equal_weight() -> None:
    d = make_pd_portfolio(n=1500)
    assert e50(d.y, d.scores) == e50(d.y, d.scores, sample_weight=np.ones(len(d.y)))


def test_e50_weighted_moves_when_tail_upweighted() -> None:
    d = make_pd_portfolio(n=2000)
    y, p = d.y, d.scores
    dist = _ici_distances(y, p, 0.75, 512)
    tail_idx = np.argsort(dist)[-100:]  # top 5% largest ICI distances
    w = np.ones(len(y))
    w[tail_idx] = 50.0
    baseline = e50(y, p)
    weighted = e50(y, p, sample_weight=w)
    assert weighted > baseline


def test_smooth_ece_guard_falls_back_to_exact() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.uniform(0.45, 0.55, n)
    p[:5] = 1e-12  # clipped scores: logit range ~55 wide -> huge bin width
    y = (rng.uniform(size=n) < p).astype(float)
    assert smooth_ece(y, p, bins=64) == smooth_ece(y, p, bins=None)
