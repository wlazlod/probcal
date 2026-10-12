"""Tests for probcal.metrics.binned."""

import numpy as np
import pytest

from probcal._math import expit
from probcal.datasets import make_pd_portfolio
from probcal.metrics._binstats import bin_stats
from probcal.metrics._common import _prep
from probcal.metrics.binned import (
    _sweep_best_b,
    adaptive_ece,
    ece,
    ece_debiased,
    ece_sweep,
    hosmer_lemeshow,
)

RNG = np.random.default_rng(59)


def _calibrated(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def _heavy_tail_design(n: int = 4000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """10% of rows near p=0.15 with weight 10, the rest near p=0.5 with weight 1."""
    rng = np.random.default_rng(seed)
    hi = np.arange(n) < n // 10
    p = np.where(hi, 0.15, 0.5) + rng.uniform(-0.01, 0.01, n)
    return p, np.where(hi, 10.0, 1.0)


def test_ece_two_bin_hand_case() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    # Bin 1: mean p 0.1, rate 0.5, gap 0.4; bin 2: mean p 0.9, rate 1.0, gap 0.1.
    assert abs(ece(y, p, n_bins=2) - (0.4 + 0.1) / 2) < 1e-12
    assert abs(ece(y, p, n_bins=2, norm="max") - 0.4) < 1e-12
    expected_l2 = np.sqrt((0.4**2 + 0.1**2) / 2)
    assert abs(ece(y, p, n_bins=2, norm="l2") - expected_l2) < 1e-12


_BAD_ARGS = [
    (lambda y, p: ece(y, p, n_bins=0), "n_bins"),
    (lambda y, p: hosmer_lemeshow(y, p, n_bins=2.5), "n_bins"),
    (lambda y, p: hosmer_lemeshow(y, p, ddof=-1), "ddof"),
    (lambda y, p: hosmer_lemeshow(y, p, ddof=1.5), "ddof"),
    (lambda y, p: hosmer_lemeshow(y, p, ddof=True), "ddof"),
    (lambda y, p: hosmer_lemeshow(y, p, ddof="2"), "ddof"),
    (lambda y, p: ece_sweep(y, p, rule="nope"), "rule"),
    (lambda y, p: ece_sweep(y, p, max_bins=0), "max_bins"),
]


def test_bad_arguments_raise() -> None:
    y, p = _calibrated(200)
    for call, match in _BAD_ARGS:
        with pytest.raises(ValueError, match=match):
            call(y, p)


def test_deprecated_aliases() -> None:
    y, p = _calibrated(1000)
    with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
        v = adaptive_ece(y, p, n_bins=10)
    assert v == ece(y, p, n_bins=10, strategy="mass")
    with pytest.warns(DeprecationWarning, match="n_bins"):
        res = hosmer_lemeshow(y, p, g=8)
    assert res == hosmer_lemeshow(y, p, n_bins=8)
    with pytest.raises(TypeError, match="only n_bins"):
        hosmer_lemeshow(y, p, g=8, n_bins=5)


# ------------------------------------------------------------------ Hosmer-Lemeshow


@pytest.mark.reference
def test_hl_pvalue_vs_scipy_chi2() -> None:
    stats = pytest.importorskip("scipy.stats")
    y, p = _calibrated(2000)
    res = hosmer_lemeshow(y, p, n_bins=10)
    expected = float(stats.chi2.sf(res.statistic, res.df))
    assert abs(res.p_value - expected) < 1e-9


def test_hl_too_few_groups_gives_nan_not_fake_df() -> None:
    # Two distinct scores -> at most two non-empty groups: none left with ddof=2.
    p = np.array([0.2] * 50 + [0.6] * 50)
    y = np.array(([1.0] * 10 + [0.0] * 40) + ([1.0] * 30 + [0.0] * 20))
    res = hosmer_lemeshow(y, p, ddof=2)
    assert res.df == 0
    assert np.isnan(res.p_value)
    assert hosmer_lemeshow(y, p).df == 2


def test_hl_out_of_sample_null_size() -> None:
    # p fixed in advance: chi2(groups) holds; the classical groups - 2 over-rejects.
    rng = np.random.default_rng(11)
    reps = 600
    rej0 = rej2 = 0
    for _ in range(reps):
        p = rng.uniform(0.01, 0.5, 2000)
        y = (rng.random(p.size) < p).astype(float)
        rej0 += hosmer_lemeshow(y, p).p_value < 0.05
        rej2 += hosmer_lemeshow(y, p, ddof=2).p_value < 0.05
    assert rej0 / reps < 0.08
    assert rej2 / reps > 0.08


def test_hl_pvalue_tail_accurate() -> None:
    y, p = _calibrated(5000)
    res = hosmer_lemeshow(y, np.clip(p * 0.5, 1e-6, 1 - 1e-6))
    assert 0.0 < res.p_value < 1e-200  # 1 - cdf rounded this to exactly 0


def test_hosmer_lemeshow_weighted_variance() -> None:
    rng = np.random.default_rng(0)
    p = 1.0 / (1.0 + np.exp(-rng.normal(-1.5, 1.0, 3000)))
    y = (rng.random(3000) < p).astype(float)
    base = hosmer_lemeshow(y, p)
    assert hosmer_lemeshow(y, p, sample_weight=np.ones(len(y))) == base
    # A weight constant within each risk group cancels: O - E scales by c_g,
    # the variance by c_g**2.
    q = np.quantile(p, np.linspace(0, 1, 11)[1:-1])
    g = np.searchsorted(q, p, side="right")
    w = (1.0 + g).astype(float)
    assert hosmer_lemeshow(y, p, sample_weight=w).statistic == pytest.approx(
        base.statistic, rel=1e-10
    )
    # Hand formula for arbitrary weights.
    w = np.random.default_rng(3).uniform(0.2, 5.0, len(y))
    stat = 0.0
    for k in range(10):
        m = g == k
        s2, s2p = np.sum(w[m] ** 2), np.sum(w[m] ** 2 * p[m])
        stat += np.sum(w[m] * (y[m] - p[m])) ** 2 / (s2p * (1 - s2p / s2))
    got = hosmer_lemeshow(y, p, sample_weight=w)
    assert got.statistic == pytest.approx(stat, rel=1e-10)
    assert hosmer_lemeshow(y, p, sample_weight=9.0 * w).statistic == pytest.approx(
        got.statistic, rel=1e-10
    )


def test_hosmer_lemeshow_weighted_matches_unweighted_null_distribution() -> None:
    # Heavy rows where Var(y) differs: the weighted statistic has the same
    # null mean as the unweighted one (about the number of groups).
    p, w = _heavy_tail_design()
    rng = np.random.default_rng(11)
    stats = [
        hosmer_lemeshow((rng.random(len(p)) < p).astype(float), p, sample_weight=w).statistic
        for _ in range(400)
    ]
    assert 8.8 < np.mean(stats) < 11.2


# ------------------------------------------------------------------ ece_debiased


def test_ece_debiased_unit_weights_unchanged_and_per_bin_effective_counts() -> None:
    y, p = _calibrated(3000)
    assert ece_debiased(y, p, sample_weight=np.ones(len(y))) == ece_debiased(y, p)
    # Concentrated weights shrink the effective bin counts, so the variance
    # correction grows; 0.3.x used the raw counts and ignored this.
    w = np.where(np.arange(len(y)) % 10 == 0, 50.0, 1.0)
    yw, pw, ww = _prep(y, p, w)
    bs = bin_stats(yw, pw, ww, 15, "mass")
    idx = np.searchsorted(np.unique(np.quantile(pw, np.linspace(0, 1, 16)[1:-1])), pw, "right")
    n_eff = np.bincount(idx, weights=ww) ** 2 / np.bincount(idx, weights=ww**2)
    gaps = np.abs(bs.p_mean - bs.rate)
    var_b = bs.rate * (1 - bs.rate) / (n_eff - 1)
    want = np.sum(bs.w_sum / ww.sum() * np.sqrt(np.maximum(gaps**2 - var_b, 0.0)))
    assert ece_debiased(y, p, sample_weight=w) == pytest.approx(want, rel=1e-12)
    assert ece_debiased(y, p, sample_weight=3.0 * w) == pytest.approx(want, rel=1e-12)


# --------------------------------------------------- ece_sweep: Roelofs et al. stopping rule


def _sweep_reference(y: np.ndarray, p: np.ndarray, w: np.ndarray, stop: bool) -> int:
    """Bin-count scan written out: ``stop`` = paper rule, else 0.3.x rule."""
    best_b = 1
    for b in range(2, min(len(p), 100) + 1):
        rates = bin_stats(y, p, w, b, "mass").rate
        if np.all(np.diff(rates) >= 0.0):
            best_b = b
        elif stop:
            break
    return best_b


def _sweep_fixtures() -> list[object]:
    d = make_pd_portfolio(n=2000, random_state=4)
    rng = np.random.default_rng(17)
    return [
        pytest.param(d.y, np.round(d.scores, 2), np.ones(2000), id="tied-scores"),
        pytest.param(d.y, d.scores, rng.uniform(0.5, 2.0, 2000), id="weighted"),
    ]


@pytest.mark.parametrize("y,p,w", _sweep_fixtures())
def test_sweep_rules_match_reference(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> None:
    ya, pa, wa = _prep(y, p, w)
    assert _sweep_best_b(ya, pa, wa, 100, "first_violation") == _sweep_reference(ya, pa, wa, True)
    assert _sweep_best_b(ya, pa, wa, 100, "largest") == _sweep_reference(ya, pa, wa, False)


def test_ece_sweep_stops_at_first_violation() -> None:
    # Continuous scores, seed chosen so that a non-monotone bin count sits
    # below a larger monotone one: 0.3.x jumped over it, the paper stops.
    rng = np.random.default_rng(0)
    p = expit(rng.normal(-1.5, 1.0, 500))
    y = (rng.random(500) < expit(1.1 * np.log(p / (1 - p)) + 0.1)).astype(float)
    ya, pa, wa = _prep(y, p, None)
    first = _sweep_best_b(ya, pa, wa, 100, "first_violation")
    largest = _sweep_best_b(ya, pa, wa, 100, "largest")
    assert first < largest
    rates = bin_stats(ya, pa, wa, first + 1, "mass").rate
    assert np.any(np.diff(rates) < 0.0)
    assert ece_sweep(y, p) == ece(y, p, n_bins=first)
    assert ece_sweep(y, p, rule="largest") == ece(y, p, n_bins=largest)
    assert ece_sweep(y, p, max_bins=1) == pytest.approx(abs(np.mean(p) - np.mean(y)))
