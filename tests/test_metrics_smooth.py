"""Tests for probcal.metrics.smooth."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.metrics.smooth import (
    _ici_distances,
    _smece_at_sigma,
    _smece_at_sigma_lattice,
    _smece_fixed_point,
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


def test_smooth_ece_default_close_to_exact_small_n() -> None:
    # n <= 8192 now takes the lattice path too (0.1.3 ran exact here — the
    # "size cliff"); values may differ from bins=None at the ~1e-4 level
    # because the lattice integrator resolves the kernel at >= 8 samples per
    # sigma vs the exact path's 257-point grid (DECISIONS 68).
    d = make_pd_portfolio(n=2000)
    assert abs(smooth_ece(d.y, d.scores) - smooth_ece(d.y, d.scores, bins=None)) <= 1e-3


def test_smooth_ece_small_n_wide_range_uses_lattice() -> None:
    # The lattice path engages at every n (no small-n exact carve-out): on a
    # wide clipped-logit range the exact 257-point grid misses the isolated
    # small-sigma kernels entirely and spuriously early-exits at ~1e-13
    # (the DECISIONS 66 aliasing mechanism); the lattice integrator reports
    # the real total variation. bins=None remains the exact escape hatch.
    rng = np.random.default_rng(5)
    n = 50
    p = rng.uniform(0.4, 0.6, n)
    p[:3] = 1e-12  # clipped scores: logit range ~28 wide
    y = (rng.uniform(size=n) < p).astype(float)
    assert smooth_ece(y, p) > 0.01
    assert smooth_ece(y, p, bins=None) < 1e-3  # the aliasing-prone exact value


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


def test_smooth_ece_guard_falls_back_to_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    # Wide clipped-score logit range + tiny bin budget: the refinement size
    # b2 = ceil(range / (sigma/8)) exceeds the (patched) lattice cap, so the
    # exact path must run — and must equal bins=None bit-for-bit. (Under the
    # real 2**20 cap this data would refine successfully instead; the patch
    # keeps the infeasible-refinement fallback reachable at test scale.)
    import probcal.metrics.smooth as smooth_mod

    monkeypatch.setattr(smooth_mod, "_SMECE_MAX_BINS", 1024)
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.uniform(0.45, 0.55, n)
    p[:5] = 1e-12  # clipped scores: logit range ~28 wide -> huge refinement size
    y = (rng.uniform(size=n) < p).astype(float)
    assert smooth_ece(y, p, bins=64) == smooth_ece(y, p, bins=None)


def test_smooth_ece_second_guard_falls_back_to_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the refined solve to stay under-resolved: a fake lattice solver
    # returning sigma = 4*width fails the sigma >= 8*width acceptance at the
    # initial and the refined binning alike (b2 = 2*bins stays far below the
    # 2**20 cap), so the last-resort exact fallback must run and match
    # bins=None bit-for-bit.
    import probcal.metrics.smooth as smooth_mod

    monkeypatch.setattr(
        smooth_mod, "_smece_fixed_point_lattice", lambda m, width: (4.0 * width, 4.0 * width)
    )
    d = make_pd_portfolio(n=2000)
    assert smooth_ece(d.y, d.scores) == smooth_ece(d.y, d.scores, bins=None)


def test_smece_lattice_no_spurious_early_exit() -> None:
    # Pre-fix, the 257-point grid aliased against the 8192-bin lattice and
    # reported ~1.7e-7 at sigma=1e-4 (spurious "perfectly calibrated" exit).
    d = make_pd_portfolio(n=10_000)
    y, p = d.y.astype(float), d.scores
    t = logit(np.clip(p, 1e-12, 1 - 1e-12))
    mass = (np.ones(t.size) / t.size) * (y - p)
    t_lo, t_hi = float(t.min()), float(t.max())
    width = (t_hi - t_lo) / 8192
    idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, 8191)
    m = np.bincount(idx, weights=mass, minlength=8192)
    tv = float(np.sum(np.abs(m)))
    assert _smece_at_sigma_lattice(m, width, 1e-4) >= 0.5 * tv  # old path: ~2e-6 * tv


def test_smece_lattice_evaluator_beats_257_grid_accuracy() -> None:
    # Both integrators vs a dense (sigma/16-spaced trapezoid on the exact
    # measure) reference at the exact path's own fixed point; the lattice must
    # be at least as close as the 257-point grid.
    d = make_pd_portfolio(n=10_000)
    y, p = d.y.astype(float), d.scores
    t = logit(np.clip(p, 1e-12, 1 - 1e-12))
    mass = (np.ones(t.size) / t.size) * (y - p)
    t_lo, t_hi = float(t.min()), float(t.max())
    sigma_star = _smece_fixed_point(t, mass)[1]

    # Chunked over grid blocks (rather than one (~4871, 10000) kernel matrix)
    # to keep peak memory in the tens of MB instead of ~780MB; the reference
    # value is unchanged (verified equal to the unchunked computation to
    # machine precision during development).
    dense_grid = np.arange(t_lo - 5.0 * sigma_star, t_hi + 5.0 * sigma_star, sigma_star / 16.0)
    smoothed = np.empty_like(dense_grid)
    block = 256
    for i in range(0, dense_grid.size, block):
        chunk = dense_grid[i : i + block]
        diff = (chunk[:, None] - t[None, :]) / sigma_star
        kern = np.exp(-0.5 * diff**2) / (sigma_star * np.sqrt(2.0 * np.pi))
        smoothed[i : i + block] = kern @ mass
    ref = float(np.trapezoid(np.abs(smoothed), dense_grid))

    grid_value = _smece_at_sigma(t, mass, sigma_star)

    width = (t_hi - t_lo) / 8192
    idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, 8191)
    m = np.bincount(idx, weights=mass, minlength=8192)
    lattice_value = _smece_at_sigma_lattice(m, width, sigma_star)

    assert abs(lattice_value - ref) <= abs(grid_value - ref)
