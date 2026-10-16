"""Tests for probcal.metrics.smooth."""

import math
import warnings

import numpy as np
import pytest

from probcal._math import expit, loess, logit, weighted_quantile
from probcal.datasets import make_pd_portfolio
from probcal.metrics import evaluate, logloss_calibration_refinement
from probcal.metrics.smooth import (
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


def _trapezoid(f: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule (np.trapezoid needs numpy >= 2.0)."""
    return float((np.diff(x) * (f[1:] + f[:-1]) / 2.0).sum())


def _calibrated(n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_ecce_hand_case() -> None:
    p = np.array([0.2, 0.4, 0.6])
    y = np.array([0.0, 1.0, 1.0])
    # Sorted by p already. Cumulative (y - p)/n: (-0.2, 0.4, 0.8)/3.
    res = ecce(y, p)
    np.testing.assert_allclose(res.stat_max, 0.8 / 3)
    np.testing.assert_allclose(res.stat_mean, (0.2 + 0.4 + 0.8) / 9)


def test_ici_family_grid_default_close_to_exact() -> None:
    d = make_pd_portfolio(n=5000)
    for fn, tol in ((ici, 1e-4), (e50, 1e-4), (e90, 1e-4), (emax, 1e-3)):
        assert abs(fn(d.y, d.scores) - fn(d.y, d.scores, grid_size=None)) <= tol


def test_smooth_ece_binned_close_to_exact() -> None:
    d = make_pd_portfolio(n=5000, event_rate=0.10)
    assert abs(smooth_ece(d.y, d.scores, bins=1024) - smooth_ece(d.y, d.scores, bins=None)) <= 1e-3


def test_smooth_ece_small_n_wide_range_uses_lattice() -> None:
    # The lattice path engages at every n (no small-n exact carve-out): on a
    # wide clipped-logit range the exact 257-point grid misses the isolated
    # small-sigma kernels entirely and spuriously early-exits at ~1e-13
    # (the same aliasing mechanism the binned-path fix addressed); the lattice integrator reports
    # the real total variation. bins=None remains the exact escape hatch.
    rng = np.random.default_rng(5)
    n = 50
    p = rng.uniform(0.4, 0.6, n)
    p[:3] = 1e-12  # clipped scores: logit range ~28 wide
    y = (rng.uniform(size=n) < p).astype(float)
    assert smooth_ece(y, p) > 0.01
    assert smooth_ece(y, p, bins=None) < 1e-3  # the aliasing-prone exact value


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
    ref = float(_trapezoid(np.abs(smoothed), dense_grid))

    grid_value = _smece_at_sigma(t, mass, sigma_star)

    width = (t_hi - t_lo) / 8192
    idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, 8191)
    m = np.bincount(idx, weights=mass, minlength=8192)
    lattice_value = _smece_at_sigma_lattice(m, width, sigma_star)

    assert abs(lattice_value - ref) <= abs(grid_value - ref)


# ------------------------------------------------------------------ 0.3.4 fixes


def test_ecce_invariant_to_tie_order() -> None:
    # Regression (MET-1): with tied scores the stable sort kept input order,
    # so permuting tied rows moved stat_max/stat_mean.
    rng = np.random.default_rng(0)
    p = np.round(rng.uniform(0.05, 0.6, 2000), 1)
    y = (rng.random(2000) < p).astype(float)
    perm = rng.permutation(2000)
    a, b = ecce(y, p), ecce(y[perm], p[perm])
    assert a.stat_max == pytest.approx(b.stat_max, rel=1e-12)
    assert a.stat_mean == pytest.approx(b.stat_mean, rel=1e-12)
    # Sorting events first vs last inside every block used to change the walk.
    order_ev_first = np.lexsort((-y, p))
    order_ev_last = np.lexsort((y, p))
    first = ecce(y[order_ev_first], p[order_ev_first])
    last = ecce(y[order_ev_last], p[order_ev_last])
    assert first.stat_max == pytest.approx(last.stat_max, rel=1e-12)


def test_ecce_unchanged_without_ties_and_unit_weights() -> None:
    y, p = _calibrated(3000)
    order = np.argsort(p, kind="stable")
    c = np.cumsum(y[order] - p[order]) / len(p)
    res = ecce(y, p)
    assert res.stat_max == float(np.max(np.abs(c)))
    assert res.stat_mean == float(np.mean(np.abs(c)))
    assert ecce(y, p, sample_weight=np.ones(len(p))) == res


def test_ecce_stat_mean_is_weight_averaged() -> None:
    # Regression (MET-14): stat_mean averaged over row indices, ignoring weights.
    p = np.array([0.1, 0.3, 0.5, 0.7])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    w = np.array([1.0, 3.0, 1.0, 5.0])
    c = np.cumsum(w * (y - p)) / w.sum()
    res = ecce(y, p, sample_weight=w)
    assert res.stat_mean == pytest.approx(float(np.sum(w * np.abs(c)) / w.sum()))
    # Duplicating rows equals integer weights.
    rep = np.repeat(np.arange(4), w.astype(int))
    dup = ecce(y[rep], p[rep])
    assert res.stat_max == pytest.approx(dup.stat_max)
    assert res.stat_mean == pytest.approx(dup.stat_mean)


def test_spiegelhalter_zero_variance_is_nan() -> None:
    # Regression (MET-3): all p = 0.5 gives a zero null variance -> 0/0.
    y = np.array([0.0, 1.0, 1.0, 0.0])
    res = spiegelhalter_z(y, np.full(4, 0.5))
    assert math.isnan(res.z) and math.isnan(res.p_value)


def test_spiegelhalter_weighted_variance() -> None:
    y, p = _calibrated(4000)
    base = spiegelhalter_z(y, p)
    assert spiegelhalter_z(y, p, sample_weight=np.ones(len(y))) == base
    # Constant weights are unit weights (0.3.x divided z by sqrt(c): w in the
    # numerator, w**2 in the variance).
    assert spiegelhalter_z(y, p, sample_weight=np.full(len(y), 5.0)).z == pytest.approx(
        base.z, rel=1e-12
    )
    w = np.random.default_rng(1).uniform(0.2, 3.0, len(y))
    a = 1.0 - 2.0 * p
    z_want = np.sum(w * (y - p) * a) / np.sqrt(np.sum(w**2 * a**2 * p * (1.0 - p)))
    assert spiegelhalter_z(y, p, sample_weight=w).z == pytest.approx(z_want, rel=1e-12)
    assert spiegelhalter_z(y, p, sample_weight=10 * w).z == pytest.approx(z_want, rel=1e-12)


def test_spiegelhalter_p_value_tail_accurate() -> None:
    # Regression (MET-5): 2 * (1 - Phi(|z|)) rounds to 0 beyond |z| ~ 8.3.
    rng = np.random.default_rng(5)
    p = expit(rng.normal(-0.8, 1.2, 8000))
    y = (rng.random(8000) < p).astype(float)
    res = spiegelhalter_z(y, expit(1.3 * logit(p)))
    assert abs(res.z) > 9.0
    assert 0.0 < res.p_value < 1e-17


def test_emax_validates_weights() -> None:
    # Regression (MET-28): emax accepted any sample_weight without looking at it.
    y, p = _calibrated(500)
    with pytest.raises(ValueError, match="sample_weight"):
        emax(y, p, sample_weight=np.ones(3))
    assert emax(y, p, sample_weight=np.full(len(y), 2.0)) == emax(y, p)


def test_spiegelhalter_weighted_type_one_error() -> None:
    # 10% of rows near p=0.15 with weight 10, the rest near p=0.5 with weight 1.
    rng = np.random.default_rng(1)
    hi = np.arange(4000) < 400
    p = np.where(hi, 0.15, 0.5) + rng.uniform(-0.01, 0.01, 4000)
    w = np.where(hi, 10.0, 1.0)
    rng = np.random.default_rng(7)
    rej = [
        spiegelhalter_z((rng.random(len(p)) < p).astype(float), p, sample_weight=w).p_value < 0.05
        for _ in range(600)
    ]
    # The Kish-rescaled variance rejected ~14% of calibrated samples here.
    assert 0.02 <= np.mean(rej) <= 0.085


# ------------------------------------------------------------------- LOESS ties


def test_loess_tied_x_order_invariant() -> None:
    rng = np.random.default_rng(0)
    p = np.r_[np.full(5000, 0.02), np.full(500, 0.2)]
    y = (rng.random(5500) < p).astype(float)
    vals = []
    for _ in range(4):
        o = rng.permutation(len(p))
        vals.append(
            (
                ici(y[o], p[o]),
                ici(y[o], p[o], grid_size=None),
                e50(y[o], p[o]),
                logloss_calibration_refinement(y[o], p[o]).calibration,
            )
        )
    # Equal up to the summation order of the final weighted mean (it moved
    # by up to ~40% between row orders before).
    np.testing.assert_allclose(vals, [vals[0]] * 4, rtol=1e-12)


def test_loess_constant_x_is_the_mean() -> None:
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0.0])
    pc = np.full(10, 0.3)
    np.testing.assert_allclose(loess(pc, y, frac=0.75), 0.3, atol=1e-15)
    assert all(ici(np.roll(y, k), pc) == pytest.approx(0.0, abs=1e-15) for k in range(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = evaluate(y, pc, n_boot=20)
    i = rep.names.index("ici")
    assert rep.ci_low[i] <= rep.values[i] <= rep.ci_high[i]
    # Two PD levels: the bootstrap interval brackets the order-free estimate.
    rng = np.random.default_rng(0)
    p2 = np.r_[np.full(5000, 0.02), np.full(500, 0.2)]
    y2 = (rng.random(5500) < p2).astype(float)
    rep = evaluate(y2, p2, n_boot=100, random_state=0, metrics=("ici",))
    assert rep.ci_low[0] <= rep.values[0] <= rep.ci_high[0]


def test_weighted_quantile_continuous_at_equal_weights() -> None:
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0.0])
    p = np.linspace(0.05, 0.6, 10)
    w = np.ones(10)
    w[0] = 1.0 + 1e-12
    assert e90(y, p, sample_weight=w) == pytest.approx(e90(y, p), abs=1e-10)
    assert e90(y, p, sample_weight=np.ones(10)) == e90(y, p)
    x = np.random.default_rng(4).normal(size=101)
    np.testing.assert_array_equal(weighted_quantile(x, 0.3, np.full(101, 2.0)), np.quantile(x, 0.3))
    assert float(weighted_quantile(x, 0.0, np.arange(1.0, 102.0))) == x.min()
    assert float(weighted_quantile(x, 1.0, np.arange(1.0, 102.0))) == x.max()
