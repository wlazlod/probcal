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


def test_ece_two_bin_hand_case() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    # Bin 1: mean p 0.1, rate 0.5, gap 0.4; bin 2: mean p 0.9, rate 1.0, gap 0.1.
    assert abs(ece(y, p, n_bins=2) - (0.4 + 0.1) / 2) < 1e-12
    assert abs(ece(y, p, n_bins=2, norm="max") - 0.4) < 1e-12
    expected_l2 = np.sqrt((0.4**2 + 0.1**2) / 2)
    assert abs(ece(y, p, n_bins=2, norm="l2") - expected_l2) < 1e-12


def test_adaptive_ece_is_deprecated_equal_mass_alias() -> None:
    y, p = _calibrated(1000)
    with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
        v = adaptive_ece(y, p, n_bins=10)
    assert v == ece(y, p, n_bins=10, strategy="mass")


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
    res = hosmer_lemeshow(y, p, n_bins=10)
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
    res = hosmer_lemeshow(y, p, n_bins=10)
    expected = float(stats.chi2.sf(res.statistic, res.df))
    assert abs(res.p_value - expected) < 1e-9


# ------------------------------------------------------------------ 0.3.4 fixes


def _old_hosmer_lemeshow(y: np.ndarray, p: np.ndarray, g: int = 10) -> tuple[float, int]:
    """0.3.x HL statistic (unweighted), verbatim loop."""
    from probcal.metrics._binstats import bin_index

    idx, m = bin_index(p, g, "mass")
    stat, used = 0.0, 0
    for b in range(m):
        mask = idx == b
        if not np.any(mask):
            continue
        nb = float(mask.sum())
        obs = float(np.sum(y[mask]))
        exp = float(np.sum(p[mask]))
        denom = exp * (1.0 - exp / nb)
        if denom > 0:
            stat += (obs - exp) ** 2 / denom
        used += 1
    return stat, max(used - 2, 1)


def test_hl_vectorized_matches_old_loop_and_unit_weights() -> None:
    y, p = _calibrated(3000)
    old_stat, old_df = _old_hosmer_lemeshow(y, p)
    res = hosmer_lemeshow(y, p)
    assert res.df == old_df
    assert res.statistic == pytest.approx(old_stat, rel=1e-12)
    assert hosmer_lemeshow(y, p, sample_weight=np.ones(len(y))) == res


def test_hl_g_keyword_is_deprecated() -> None:
    y, p = _calibrated(1000)
    with pytest.warns(DeprecationWarning, match="n_bins"):
        res = hosmer_lemeshow(y, p, g=8)
    assert res == hosmer_lemeshow(y, p, n_bins=8)
    with pytest.raises(TypeError, match="only n_bins"):
        hosmer_lemeshow(y, p, g=8, n_bins=5)


def test_hl_too_few_groups_gives_nan_not_fake_df() -> None:
    # Two distinct scores -> at most two non-empty groups: no degrees of freedom.
    p = np.array([0.2] * 50 + [0.6] * 50)
    y = np.array(([1.0] * 10 + [0.0] * 40) + ([1.0] * 30 + [0.0] * 20))
    res = hosmer_lemeshow(y, p)
    assert res.df == 0
    assert np.isnan(res.p_value)


def test_hl_weights_are_relative() -> None:
    y, p = _calibrated(2000)
    w = np.random.default_rng(3).uniform(0.5, 2.0, len(y))
    a = hosmer_lemeshow(y, p, sample_weight=w)
    b = hosmer_lemeshow(y, p, sample_weight=1000.0 * w)
    assert b.statistic == pytest.approx(a.statistic, rel=1e-10)
    # Constant weights are unit weights.
    assert hosmer_lemeshow(y, p, sample_weight=np.full(len(y), 7.0)).statistic == pytest.approx(
        hosmer_lemeshow(y, p).statistic, rel=1e-12
    )


def test_hl_pvalue_tail_accurate() -> None:
    y, p = _calibrated(5000)
    res = hosmer_lemeshow(y, np.clip(p * 0.5, 1e-6, 1 - 1e-6))
    assert 0.0 < res.p_value < 1e-200  # 1 - cdf rounded this to exactly 0


def test_n_bins_validated() -> None:
    y, p = _calibrated(200)
    with pytest.raises(ValueError, match="n_bins"):
        ece(y, p, n_bins=0)
    with pytest.raises(ValueError, match="n_bins"):
        hosmer_lemeshow(y, p, n_bins=2.5)  # type: ignore[arg-type]


def _old_ece_debiased(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bs = bin_stats(y, p, np.ones(len(y)), n_bins, "mass")
    shares = bs.w_sum / len(y)
    gaps = np.abs(bs.p_mean - bs.rate)
    out = 0.0
    for i in range(len(gaps)):
        if bs.n_obs[i] > 1:
            var_b = bs.rate[i] * (1.0 - bs.rate[i]) / (bs.n_obs[i] - 1)
            out += shares[i] * np.sqrt(max(gaps[i] ** 2 - var_b, 0.0))
        else:
            out += shares[i] * gaps[i]
    return float(out)


def test_ece_debiased_unit_weights_unchanged_and_kish_weighted() -> None:
    y, p = _calibrated(3000)
    assert ece_debiased(y, p) == pytest.approx(_old_ece_debiased(y, p), rel=1e-12, abs=1e-15)
    assert ece_debiased(y, p, sample_weight=np.ones(len(y))) == ece_debiased(y, p)
    # Concentrated weights shrink the effective bin counts, so the variance
    # correction grows; 0.3.x used the raw counts and ignored this.
    w = np.where(np.arange(len(y)) % 10 == 0, 50.0, 1.0)
    yw, pw, ww = _prep(y, p, w)
    bs = bin_stats(yw, pw, ww, 15, "mass")
    n_eff = bs.w_sum * ww.sum() / np.dot(ww, ww)
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
    out: list[object] = []
    for n in (500, 5000):
        d = make_pd_portfolio(n=n, random_state=3)
        out.append(pytest.param(d.y, d.scores, np.ones(n), id=f"portfolio-{n}"))
    d = make_pd_portfolio(n=2000, random_state=4)
    out.append(pytest.param(d.y, np.round(d.scores, 2), np.ones(2000), id="tied-scores"))
    rng = np.random.default_rng(17)
    out.append(pytest.param(d.y, d.scores, rng.uniform(0.5, 2.0, 2000), id="weighted"))
    return out


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


def test_ece_sweep_options_validated() -> None:
    y, p = _calibrated(300)
    with pytest.raises(ValueError, match="rule"):
        ece_sweep(y, p, rule="nope")
    with pytest.raises(ValueError, match="max_bins"):
        ece_sweep(y, p, max_bins=0)
    assert ece_sweep(y, p, max_bins=1) == pytest.approx(abs(np.mean(p) - np.mean(y)))
