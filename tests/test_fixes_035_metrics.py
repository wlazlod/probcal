"""Regression tests for the 0.3.5 metrics fixes: weighted variances, LOESS ties,
the weighted quantile, Jeffreys band merging, constant-``p`` regression, and
value equality of grade/ECCE results."""

import math
import warnings

import numpy as np
import pytest

from probcal import IsotonicCalibrator
from probcal._math import _tie_means, loess, weighted_quantile
from probcal.curves import calibration_belt, ecce_curve
from probcal.metrics import (
    binomial_grade_test,
    calibration_guardrails,
    calibration_slope,
    calibration_test,
    e50,
    e90,
    ece_debiased,
    evaluate,
    hosmer_lemeshow,
    ici,
    jeffreys_grade_test,
    jeffreys_upper_bands,
    logloss_calibration_refinement,
    murphy_decomposition,
    spiegelhalter_z,
)
from probcal.metrics._binstats import bin_stats
from probcal.thresholds import calibrated_bands_to_raw


def _heavy_tail_design(n: int = 4000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """10% of rows near p=0.15 with weight 10, the rest near p=0.5 with weight 1."""
    rng = np.random.default_rng(seed)
    hi = np.arange(n) < n // 10
    p = np.where(hi, 0.15, 0.5) + rng.uniform(-0.01, 0.01, n)
    return p, np.where(hi, 10.0, 1.0)


def _portfolio(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + np.exp(-rng.normal(-1.5, 1.0, n)))
    return (rng.random(n) < p).astype(float), p


# ------------------------------------------------------------- weighted variances


def test_spiegelhalter_weighted_type_one_error() -> None:
    p, w = _heavy_tail_design()
    rng = np.random.default_rng(7)
    rej = [
        spiegelhalter_z((rng.random(len(p)) < p).astype(float), p, sample_weight=w).p_value < 0.05
        for _ in range(600)
    ]
    # The Kish-rescaled variance rejected ~14% of calibrated samples here.
    assert 0.02 <= np.mean(rej) <= 0.085


def test_ecce_null_sd_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    n = 600
    p = np.sort(rng.uniform(0.01, 0.5, n))
    w = np.where(p < 0.1, 20.0, 1.0)
    walks = np.array(
        [
            ecce_curve((rng.random(n) < p).astype(float), p, sample_weight=w).cumdev
            for _ in range(1500)
        ]
    )
    sd = ecce_curve(np.r_[0.0, np.ones(n - 1)], p, sample_weight=w).sd_null
    ratio = sd[-1] / walks.std(axis=0)[-1]
    assert 0.93 < ratio < 1.07  # Kish gave ~1.22 here


def test_per_bin_effective_counts() -> None:
    # Weights constant within each bin: the effective count is the raw count,
    # whatever the bin's weight (a global Kish factor shrank the light bins).
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.3, 3000)
    w = np.where(p < 0.175, 1.0, 10.0)
    y = (rng.random(3000) < p).astype(float)
    bs = bin_stats(y, p, w, 15, "mass")
    edge = np.quantile(p, np.linspace(0, 1, 16)[1:-1])
    straddles = np.searchsorted(edge, 0.175)  # the one bin mixing both weights
    same = np.arange(len(bs.n_obs)) != straddles
    np.testing.assert_allclose(bs.n_eff[same], bs.n_obs[same], rtol=1e-12)
    a = ece_debiased(y, p, sample_weight=w)
    assert a == pytest.approx(ece_debiased(y, p, sample_weight=7.0 * w), rel=1e-12)
    m = murphy_decomposition(y, p, bias_corrected=True, sample_weight=w)
    m7 = murphy_decomposition(y, p, bias_corrected=True, sample_weight=7.0 * w)
    assert m.reliability == pytest.approx(m7.reliability, rel=1e-12)


def test_hosmer_lemeshow_weighted_variance() -> None:
    y, p = _portfolio()
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


def test_calibration_test_rao_scott_type_one_error_and_units() -> None:
    p, w = _heavy_tail_design()
    rng = np.random.default_rng(5)
    rej = [
        calibration_test((rng.random(len(p)) < p).astype(float), p, sample_weight=w).p_value < 0.05
        for _ in range(400)
    ]
    assert 0.02 <= np.mean(rej) <= 0.09
    y, pp = _portfolio()
    unit = calibration_test(y, pp)
    assert unit.df == 2.0
    const = calibration_test(y, pp, sample_weight=np.full(len(y), 4.0))
    assert const.statistic == pytest.approx(unit.statistic, rel=1e-10)
    assert const.df == 2.0
    weighted = calibration_test(y, pp, sample_weight=np.where(pp < 0.2, 5.0, 1.0))
    assert 0.0 < weighted.df < 2.0


def test_calibration_belt_weights_scale_invariant() -> None:
    y, p = _portfolio(4000)
    w = np.where(p < 0.2, 5.0, 1.0)
    a = calibration_belt(y, p, sample_weight=w)
    b = calibration_belt(y, p, sample_weight=3.0 * w)
    assert a.degree == b.degree
    assert a.p_value == pytest.approx(b.p_value, rel=1e-8)
    np.testing.assert_allclose(a.bands[0.95][1], b.bands[0.95][1], rtol=1e-8)
    assert calibration_belt(y, p, sample_weight=np.ones(len(y))) == calibration_belt(y, p)


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
    rep = evaluate(y2, p2, n_boot=100, random_state=0)
    assert rep.ci_low[i] <= rep.values[i] <= rep.ci_high[i]


def test_loess_untied_input_untouched() -> None:
    xs = np.sort(np.random.default_rng(1).uniform(size=50))
    ys = np.random.default_rng(2).uniform(size=50)
    assert _tie_means(xs, ys) is ys


# ---------------------------------------------------------- weighted quantile


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


# ------------------------------------------------------------ Jeffreys bands


def test_jeffreys_bands_strictly_increasing() -> None:
    grades = np.array(["A"] * 100 + ["B"] * 1000)
    y = np.r_[np.zeros(97), np.ones(3), np.zeros(995), np.ones(5)]
    p = np.r_[np.full(100, 0.01), np.full(1000, 0.02)]
    with pytest.warns(UserWarning, match="'B'.*merged into 'A\\+B'"):
        bands = jeffreys_upper_bands(y, p, grades)
    assert list(bands) == ["A+B"]
    lo, hi = bands["A+B"]
    assert lo == 0.0 < hi
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 3000)
    cal = IsotonicCalibrator().fit(s, (rng.random(3000) < s * 0.2).astype(float))
    raw = calibrated_bands_to_raw(cal, bands)
    assert list(raw) == ["A+B"]


def test_jeffreys_bands_merge_only_violators() -> None:
    grades = np.array(["A"] * 200 + ["B"] * 50 + ["C"] * 2000 + ["D"] * 300)
    y = np.r_[np.zeros(200), np.ones(1), np.zeros(49), np.ones(4), np.zeros(1996), np.ones(30)]
    y = np.r_[y, np.zeros(300 - 30)]
    p = np.r_[np.full(200, 0.001), np.full(50, 0.01), np.full(2000, 0.02), np.full(300, 0.1)]
    with pytest.warns(UserWarning, match="merged"):
        bands = jeffreys_upper_bands(y, p, grades)
    assert list(bands) == ["A", "B+C", "D"]
    his = [hi for _, hi in bands.values()]
    los = [lo for lo, _ in bands.values()]
    assert los[0] == 0.0 and los[1:] == his[:-1]
    assert all(lo < hi for lo, hi in bands.values())


# ---------------------------------------------------- constant p / single level


@pytest.mark.parametrize("c", [0.3, 0.05, 0.6])
def test_constant_p_slope_nan_and_intercept_only_test(c: float) -> None:
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0.0])
    p = np.full(10, c)
    with pytest.warns(UserWarning, match="not identified"):
        assert math.isnan(calibration_slope(y, p))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="not identified"):
            calibration_slope(y, p)  # the warning is not about separation
    with pytest.warns(UserWarning, match="not identified"):
        res = calibration_test(y, p)
    assert res.df == 1.0 and math.isnan(res.beta)
    # Intercept-only LR with slope fixed at 1: the fitted rate is the observed rate.
    rate = y.mean()
    lr = 2 * np.sum(y * np.log(rate / c) + (1 - y) * np.log((1 - rate) / (1 - c)))
    assert res.statistic == pytest.approx(max(lr, 0.0), abs=1e-9)
    assert res.alpha == pytest.approx(np.log(rate / (1 - rate)) - np.log(c / (1 - c)), abs=1e-7)
    with pytest.warns(UserWarning, match="not identified"):
        g = calibration_guardrails(y, p)
    assert math.isnan(g.slope) and not g.slope_ok


# --------------------------------------------------------------- equality


def test_grade_and_ecce_results_compare_by_value() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.3, 500)
    y = (rng.random(500) < p).astype(float)
    g = np.where(p < 0.1, "A", "B")
    assert binomial_grade_test(y, p, g) == binomial_grade_test(y, p, g)
    assert jeffreys_grade_test(y, p, g) == jeffreys_grade_test(y, p, g)
    assert jeffreys_grade_test(y, p, g) != jeffreys_grade_test(y, 0.9 * p, g)
    assert ecce_curve(y, p) == ecce_curve(y, p)
    assert ecce_curve(y, p) != ecce_curve(y, 0.99 * p)
    with pytest.raises(TypeError):
        hash(ecce_curve(y, p))


def test_hl_e_test_constant_weights_equal_unweighted() -> None:
    from probcal.metrics import hl_e_test

    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.2, 2000)
    y = (rng.random(2000) < p).astype(float)
    grades = np.where(p < 0.1, "A", "B")
    base = hl_e_test(y, p, grades).e_value
    assert hl_e_test(y, p, grades, sample_weight=np.full(2000, 10.0)).e_value == base


def test_hl_e_test_weighted_null_stays_valid() -> None:
    from probcal.metrics import hl_e_test

    rng = np.random.default_rng(4)
    n, reps = 400, 300
    rejections = 0
    for _ in range(reps):
        p = rng.uniform(0.02, 0.3, n)
        y = (rng.random(n) < p).astype(float)
        w = rng.lognormal(0.0, 1.0, n)
        grades = np.where(p < 0.1, "A", "B")
        rejections += hl_e_test(y, p, grades, sample_weight=w).e_value >= 20.0
    assert rejections / reps <= 0.05


def test_corp_zero_resamples_skips_bands() -> None:
    from probcal.curves import corp_reliability

    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.5, 300)
    y = (rng.random(300) < p).astype(float)
    r = corp_reliability(y, p, n_resamples=0)
    assert r.band_grid.size == 0
    with pytest.raises(ValueError, match="n_resamples"):
        corp_reliability(y, p, n_resamples=-1)
