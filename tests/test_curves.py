"""Tests for probcal.curves."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal._results import BeltResult, KernelReliabilityCurve, ReliabilityCurve
from probcal.curves import (
    EcceCurve,
    calibration_belt,
    ecce_curve,
    reliability_binned,
    reliability_loess,
    reliability_smooth,
    reliability_spline,
)
from probcal.datasets import make_pd_portfolio
from probcal.metrics import smooth_ece

RNG = np.random.default_rng(83)


def _calibrated(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_reliability_binned_structure() -> None:
    y, p = _calibrated(2000)
    curve = reliability_binned(y, p, n_bins=10)
    assert isinstance(curve, ReliabilityCurve)
    assert np.all(np.diff(curve.pred_mean) > 0)
    np.testing.assert_allclose(curve.pred_mean_logit, logit(curve.pred_mean), atol=1e-12)
    assert int(curve.count.sum()) == 2000
    assert np.all(curve.ci_low <= curve.event_rate)
    assert np.all(curve.event_rate <= curve.ci_high)


def test_wilson_ci_hand_case() -> None:
    # One bin: n=100, k=10. Wilson at z=1.96.
    y = np.concatenate([np.ones(10), np.zeros(90)])
    p = np.full(100, 0.1)
    curve = reliability_binned(y, p, n_bins=1)
    z = 1.959963984540054
    n, rate = 100.0, 0.1
    denom = 1.0 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denom
    np.testing.assert_allclose(curve.ci_low, [center - half], atol=1e-12)
    np.testing.assert_allclose(curve.ci_high, [center + half], atol=1e-12)


def test_reliability_loess_near_diagonal() -> None:
    y, p = _calibrated()
    curve = reliability_loess(y, p)
    assert len(curve.grid_p) == 100
    assert np.max(np.abs(curve.event_rate - curve.grid_p)) < 0.06
    np.testing.assert_allclose(curve.grid_logit, logit(curve.grid_p), atol=1e-12)


def test_reliability_spline_near_diagonal() -> None:
    y, p = _calibrated()
    curve = reliability_spline(y, p)
    assert np.max(np.abs(curve.event_rate - curve.grid_p)) < 0.06


def test_belt_calibrated_data() -> None:
    y, p = _calibrated(8000)
    belt = calibration_belt(y, p)
    assert isinstance(belt, BeltResult)
    assert belt.p_value > 0.01
    assert 1 <= belt.degree <= 4
    inside = (belt.lower_95 <= belt.grid_p) & (belt.grid_p <= belt.upper_95)
    assert inside.mean() >= 0.9
    assert np.all(belt.lower_80 >= belt.lower_95 - 1e-12)
    assert np.all(belt.upper_80 <= belt.upper_95 + 1e-12)
    assert np.all(belt.lower_95 <= belt.upper_95)


def test_belt_rejects_distortion() -> None:
    y, p = _calibrated(8000)
    p_bad = expit(0.5 * logit(p) - 0.7)
    belt = calibration_belt(y, p_bad)
    assert belt.p_value < 1e-4
    outside = (belt.grid_p < belt.lower_95) | (belt.grid_p > belt.upper_95)
    assert outside.any()


def test_belt_grid_scales_consistent() -> None:
    y, p = _calibrated(3000)
    belt = calibration_belt(y, p)
    np.testing.assert_allclose(belt.grid_logit, logit(belt.grid_p), atol=1e-12)


def test_belt_separated_data_stops_extension() -> None:
    # Perfectly separated outcomes: the degree-1 fit separates (ridge fallback)
    # and the forward LR loop must not extend to higher degrees.
    rng = np.random.default_rng(9)
    n = 200
    p = np.concatenate([rng.uniform(0.02, 0.2, n // 2), rng.uniform(0.8, 0.98, n // 2)])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    with pytest.warns(UserWarning, match="[Ss]eparation"):
        belt = calibration_belt(y, p)
    assert belt.degree == 1
    assert np.all(np.isfinite(belt.lower_95)) and np.all(np.isfinite(belt.upper_95))
    assert 0.0 <= belt.p_value <= 1.0


def test_wilson_ci_contains_rate_with_empty_event_bins() -> None:
    # A zero-event bin: analytically the Wilson lower bound touches 0 exactly,
    # and floating-point noise must not push ci_low above the rate (it broke
    # errorbar rendering with negative yerr on exactly this portfolio).
    from probcal import make_pd_portfolio

    port = make_pd_portfolio(n=6000, random_state=42)
    curve = reliability_binned(port.y, port.scores)
    assert float(curve.event_rate[0]) == 0.0  # the offending zero-event bin
    assert np.all(curve.ci_low <= curve.event_rate)
    assert np.all(curve.event_rate <= curve.ci_high)


def test_ecce_curve_hand_case() -> None:
    # 4 points already sorted by p. Residuals: -0.2, -0.4, 0.6, 0.2; the two
    # tied p=0.4 rows form one block, so the walk is read at indices 0, 2, 3.
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.2, 0.4, 0.4, 0.8])
    c = ecce_curve(y, p)
    assert isinstance(c, EcceCurve)
    ends = [0, 2, 3]
    np.testing.assert_allclose(c.frac, [0.25, 0.75, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        c.cumdev, (np.cumsum([-0.2, -0.4, 0.6, 0.2]) / 4.0)[ends], atol=1e-12
    )
    var = np.cumsum([0.2 * 0.8, 0.4 * 0.6, 0.4 * 0.6, 0.8 * 0.2])[ends]
    np.testing.assert_allclose(c.sd_null, np.sqrt(var) / 4.0, atol=1e-12)
    assert abs(c.stat_max - np.max(np.abs(c.cumdev))) < 1e-15
    assert c.argmax_frac == c.frac[int(np.argmax(np.abs(c.cumdev)))]


def test_ecce_curve_stat_max_matches_metric() -> None:
    from probcal.metrics import ecce

    y, p = _calibrated(2000)
    w = RNG.uniform(0.5, 2.0, 2000)
    assert abs(ecce_curve(y, p).stat_max - ecce(y, p).stat_max) < 1e-15
    c_w = ecce_curve(y, p, sample_weight=w)
    assert abs(c_w.stat_max - ecce(y, p, sample_weight=w).stat_max) < 1e-15
    # Final cumdev equals the metric-consistent weighted mean residual.
    assert abs(c_w.cumdev[-1] - float(np.sum(w * (y - p)) / w.sum())) < 1e-12


def test_reliability_smooth_matches_smooth_ece() -> None:
    d = make_pd_portfolio(n=3000, random_state=11)
    curve = reliability_smooth(d.y, d.scores, n_boot=0)
    assert isinstance(curve, KernelReliabilityCurve)
    assert abs(curve.smooth_ece - smooth_ece(d.y, d.scores)) < 1e-12


def test_reliability_smooth_matches_smooth_ece_exact_path() -> None:
    # bins=None forces the exact O(n) path on both the metric and the curve.
    d = make_pd_portfolio(n=1500, random_state=12)
    curve = reliability_smooth(d.y, d.scores, n_boot=0, bins=None)
    assert abs(curve.smooth_ece - smooth_ece(d.y, d.scores, bins=None)) < 1e-12


def test_reliability_smooth_near_diagonal_on_calibrated_data() -> None:
    # sigma_star is tuned for the smECE aggregate, not for a low-variance
    # curve, so the pointwise kernel estimate needs a large sample to settle
    # within tolerance -- a dedicated RNG/n (not the smaller shared
    # _calibrated fixture) keeps this from being flaky at moderate n.
    rng = np.random.default_rng(0)
    n = 500_000
    p = expit(rng.normal(0.0, 1.0, n))
    y = (rng.random(n) < p).astype(float)
    curve = reliability_smooth(y, p, n_boot=0)
    m = len(curve.grid_p)
    core = slice(m // 20, m - m // 20)  # central 90% of the grid
    assert np.max(np.abs(curve.event_rate[core] - curve.grid_p[core])) < 0.05


def test_reliability_smooth_density_normalized_and_nonnegative() -> None:
    y, p = _calibrated(4000)
    curve = reliability_smooth(y, p, n_boot=0)
    assert curve.density.shape == curve.grid_p.shape
    assert np.all(curve.density >= 0.0)
    assert abs(float(curve.density.sum()) - 1.0) < 1e-10


def test_reliability_smooth_seeded_determinism() -> None:
    y, p = _calibrated(2000)
    c1 = reliability_smooth(y, p, n_boot=30, random_state=7)
    c2 = reliability_smooth(y, p, n_boot=30, random_state=7)
    np.testing.assert_array_equal(c1.event_rate, c2.event_rate)
    np.testing.assert_array_equal(c1.ci_low, c2.ci_low)
    np.testing.assert_array_equal(c1.ci_high, c2.ci_high)


def test_reliability_smooth_ci_brackets_event_rate() -> None:
    y, p = _calibrated(2000)
    curve = reliability_smooth(y, p, n_boot=30, random_state=3)
    assert np.all(curve.ci_low <= curve.event_rate + 1e-12)
    assert np.all(curve.event_rate <= curve.ci_high + 1e-12)


def test_reliability_smooth_no_bootstrap_collapses_band() -> None:
    y, p = _calibrated(1000)
    curve = reliability_smooth(y, p, n_boot=0)
    np.testing.assert_array_equal(curve.ci_low, curve.event_rate)
    np.testing.assert_array_equal(curve.ci_high, curve.event_rate)


def test_reliability_smooth_rejects_bad_level() -> None:
    y, p = _calibrated(500)
    with pytest.raises(ValueError, match="confidence"):
        reliability_smooth(y, p, confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        reliability_smooth(y, p, confidence=1.0)
    with pytest.warns(DeprecationWarning, match="confidence"):
        a = reliability_smooth(y, p, level=0.8, n_boot=5)
    b = reliability_smooth(y, p, confidence=0.8, n_boot=5)
    np.testing.assert_array_equal(a.ci_low, b.ci_low)


def test_reliability_smooth_grid_scales_consistent() -> None:
    y, p = _calibrated(1500)
    curve = reliability_smooth(y, p, n_boot=0)
    np.testing.assert_allclose(curve.grid_logit, logit(curve.grid_p), atol=1e-12)
    assert len(curve.grid_p) == 200


# ------------------------------------------------------------------ 0.4.0 fixes


def _gap_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    p = np.concatenate([rng.uniform(0.01, 0.03, 1500), rng.uniform(0.5, 0.9, 1500)])
    y = (rng.random(3000) < p).astype(float)
    return y, p


@pytest.mark.parametrize("bins", [8192, None])
def test_reliability_smooth_is_nan_in_data_gaps(bins: int | None) -> None:
    # Regression (MET-9): the lattice path reported rate 0 with a [0, 0]
    # ribbon where no data were within reach of the kernel.
    y, p = _gap_data()
    curve = reliability_smooth(y, p, n_boot=20, bins=bins)
    t = logit(p)
    near = np.min(np.abs(curve.grid_logit[:, None] - t[None, :]), axis=1)
    gap = near > 5.0 * curve.sigma_star + 0.05
    assert gap.any()
    assert np.all(np.isnan(curve.event_rate[gap]))
    assert np.all(np.isnan(curve.ci_low[gap])) and np.all(np.isnan(curve.ci_high[gap]))
    covered = near < 1.0 * curve.sigma_star
    assert np.all(np.isfinite(curve.event_rate[covered]))
    ok = np.isfinite(curve.event_rate)
    assert np.all(curve.ci_low[ok] <= curve.event_rate[ok])
    assert np.all(curve.event_rate[ok] <= curve.ci_high[ok])


def test_reliability_smooth_without_gaps_has_no_nan() -> None:
    y, p = _calibrated(2000)
    curve = reliability_smooth(y, p, n_boot=10)
    assert np.all(np.isfinite(curve.event_rate))
    assert np.all(np.isfinite(curve.ci_low)) and np.all(np.isfinite(curve.ci_high))


def _old_belt(y: np.ndarray, p: np.ndarray) -> tuple[int, float]:
    """0.3.x degree selection and p-value (uncentred design)."""
    from probcal._math import chi2_sf, irls_logistic

    z = logit(p)

    def design(deg: int) -> np.ndarray:
        return np.column_stack([z**k for k in range(deg + 1)])

    def ll(beta: np.ndarray, deg: int) -> float:
        prob = np.clip(expit(design(deg) @ beta), 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(prob) + (1 - y) * np.log1p(-prob)))

    degree, fit = 1, irls_logistic(design(1), y)
    cur = ll(fit.beta, 1)
    while degree < 4:
        cand = irls_logistic(design(degree + 1), y)
        new = ll(cand.beta, degree + 1)
        if chi2_sf(max(2 * (new - cur), 0.0), 1.0) >= 0.05:
            break
        degree, cur = degree + 1, new
    ll0 = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    return degree, chi2_sf(max(2 * (cur - ll0), 0.0), degree + 1.0)


def test_belt_centring_keeps_the_model() -> None:
    # MET-13: centring/scaling logit(p) is a reparametrisation of the same
    # polynomial; degree and p-value agree with the raw design.
    y, p = _calibrated(4000)
    p_bad = expit(0.8 * logit(p) + 0.3 * logit(p) ** 2 / 4)
    for pp in (p, p_bad):
        belt = calibration_belt(y, pp)
        degree, p_value = _old_belt(y, pp)
        assert belt.degree == degree
        assert belt.p_value == pytest.approx(p_value, rel=1e-6, abs=1e-300)


def test_belt_bands_follow_requested_levels() -> None:
    # MET-13: lower_80/upper_95 hold confidence[0]/[1] whatever their values.
    y, p = _calibrated(3000)
    wide = calibration_belt(y, p, confidence=(0.9, 0.99))
    std = calibration_belt(y, p)
    assert np.all(wide.upper_80 >= std.upper_80 - 1e-12)  # 90% >= 80%
    assert np.all(wide.upper_95 >= std.upper_95 - 1e-12)  # 99% >= 95%
    if hasattr(wide, "bands"):
        np.testing.assert_array_equal(wide.bands[0.9][1], wide.upper_80)
    with pytest.raises(ValueError, match="confidence"):
        calibration_belt(y, p, confidence=(0.8, 1.2))


def test_belt_weights_are_relative() -> None:
    y, p = _calibrated(3000)
    w = np.random.default_rng(3).uniform(0.5, 2.0, len(y))
    a = calibration_belt(y, p, sample_weight=w)
    b = calibration_belt(y, p, sample_weight=10.0 * w)
    assert a.degree == b.degree
    assert a.p_value == pytest.approx(b.p_value, rel=1e-6)
    np.testing.assert_allclose(a.upper_95, b.upper_95, rtol=1e-6)


def test_reliability_loess_uses_weights() -> None:
    # Regression (MET-28): sample_weight was validated and then ignored.
    y, p = _calibrated(3000)
    base = reliability_loess(y, p)
    same = reliability_loess(y, p, sample_weight=np.full(len(y), 3.0))
    np.testing.assert_array_equal(same.event_rate, base.event_rate)
    w = np.where(y == 1.0, 5.0, 1.0)
    up = reliability_loess(y, p, sample_weight=w)
    assert np.mean(up.event_rate) > np.mean(base.event_rate) + 0.02


def test_weighted_local_linear_matches_loess_on_unit_weights() -> None:
    from probcal._math import loess
    from probcal.curves import _weighted_local_linear

    rng = np.random.default_rng(6)
    x = np.sort(rng.uniform(0, 1, 400))
    yv = (rng.random(400) < x).astype(float)
    grid = np.linspace(0.05, 0.95, 25)
    np.testing.assert_allclose(
        _weighted_local_linear(x, yv, np.ones(400), grid, 0.5),
        loess(x, yv, frac=0.5, xeval=grid),
        atol=1e-10,
    )


def test_ecce_curve_weights_sd_kish() -> None:
    y, p = _calibrated(1000)
    w = np.random.default_rng(2).uniform(0.5, 2.0, len(y))
    a, b = ecce_curve(y, p, sample_weight=w), ecce_curve(y, p, sample_weight=4.0 * w)
    np.testing.assert_allclose(a.sd_null, b.sd_null, rtol=1e-12)
    np.testing.assert_allclose(a.cumdev, b.cumdev, rtol=1e-12, atol=1e-15)
    unit = ecce_curve(y, p)
    np.testing.assert_array_equal(
        ecce_curve(y, p, sample_weight=np.ones(len(y))).sd_null, unit.sd_null
    )


def test_corp_level_keyword_deprecated() -> None:
    from probcal.curves import corp_reliability

    y, p = _calibrated(300)
    with pytest.warns(DeprecationWarning, match="confidence"):
        old = corp_reliability(y, p, level=0.8, n_resamples=10)
    new = corp_reliability(y, p, confidence=0.8, n_resamples=10)
    assert old.level == new.level == 0.8
    np.testing.assert_array_equal(old.band_low, new.band_low)
