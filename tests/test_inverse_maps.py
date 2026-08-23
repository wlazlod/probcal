"""Tests for interval_inverse across the calibrator catalog (spec §10)."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.base import UnattainableTargetError
from probcal.bayesian import ENIRCalibrator
from probcal.isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
from probcal.offset import LogitOffset
from probcal.parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from probcal.spline import SplineCalibrator
from probcal.vennabers import VennAbersCalibrator

RNG = np.random.default_rng(103)


def _sample(n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    s = expit(RNG.normal(-1.0, 1.4, n))
    y = (RNG.random(n) < expit(0.8 * logit(s) - 0.4)).astype(float)
    return s, y


@pytest.mark.parametrize(
    "factory",
    [
        PlattCalibrator,
        TemperatureCalibrator,
        BetaCalibrator,
        SplineCalibrator,
        CenteredIsotonicCalibrator,
    ],
    ids=["platt", "temperature", "beta", "spline", "cir"],
)
def test_round_trip_strictly_monotone(factory) -> None:
    cal = factory().fit(*_sample())
    for tau in (0.02, 0.1, 0.4):
        lo_s, hi_s = cal.interval_inverse(tau, tau)
        assert abs(hi_s - lo_s) < 1e-6
        p = cal.predict_proba(np.array([np.clip(0.5 * (lo_s + hi_s), 1e-9, 1 - 1e-9)]))[0]
        assert abs(p - tau) < 1e-4


def test_isotonic_block_edge_semantics() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    cal = IsotonicCalibrator().fit(s, y)  # blocks: 0 @0.1 | 1/3 @0.2-0.4 | 1 @0.5
    raw_lo, raw_hi = cal.interval_inverse(0.2, 0.9)
    assert raw_lo == 0.2  # left edge of the first block with mean >= 0.2
    assert raw_hi == 0.5  # boundary where the map exceeds 0.9 (last block start)


def test_isotonic_unattainable_raises() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4])
    # Pooling caps the top block mean at 2/3: targets above it are unattainable.
    y = np.array([0.0, 1.0, 1.0, 0.0])
    cal = IsotonicCalibrator().fit(s, y)
    gmax = float(cal.block_mean_[-1])
    assert gmax < 0.9
    with pytest.raises(UnattainableTargetError, match="does not intersect"):
        cal.interval_inverse(gmax + 0.05, 1.0)


def test_offset_shifts_raw_bounds_by_minus_delta() -> None:
    p = expit(RNG.normal(-2.0, 1.0, 500))
    off0 = LogitOffset(delta=0.0).fit(p)
    off1 = LogitOffset(delta=0.5).fit(p)
    lo0, hi0 = off0.interval_inverse(0.02, 0.10, space="logit")
    lo1, hi1 = off1.interval_inverse(0.02, 0.10, space="logit")
    np.testing.assert_allclose(lo1, lo0 - 0.5, atol=1e-10)
    np.testing.assert_allclose(hi1, hi0 - 0.5, atol=1e-10)


def test_space_consistency() -> None:
    cal = PlattCalibrator().fit(*_sample())
    lo_p, hi_p = cal.interval_inverse(0.02, 0.10)
    lo_z, hi_z = cal.interval_inverse(0.02, 0.10, space="logit")
    np.testing.assert_allclose(logit(np.array([lo_p, hi_p])), [lo_z, hi_z], atol=1e-9)


def test_boundary_targets() -> None:
    cal = PlattCalibrator().fit(*_sample())
    lo_p, hi_p = cal.interval_inverse(0.0, 1.0)
    assert lo_p == 0.0 and hi_p == 1.0
    lo_z, hi_z = cal.interval_inverse(0.0, 1.0, space="logit")
    assert lo_z == -np.inf and hi_z == np.inf


def test_buffer_monotonicity() -> None:
    cal = PlattCalibrator().fit(*_sample())
    plain = cal.interval_inverse(0.02, 0.10)
    small = cal.interval_inverse(0.02, 0.10, buffer_logit=0.1)
    large = cal.interval_inverse(0.02, 0.10, buffer_logit=0.3)
    assert plain[0] <= small[0] <= large[0]
    assert large[1] <= small[1] <= plain[1]


def test_buffer_crossing_raises() -> None:
    cal = PlattCalibrator().fit(*_sample())
    with pytest.raises(UnattainableTargetError, match="buffer"):
        cal.interval_inverse(0.05, 0.055, buffer_logit=2.0)


def test_enir_not_implemented() -> None:
    cal = ENIRCalibrator().fit(*_sample(500))
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.interval_inverse(0.1, 0.2)


def test_vennabers_bisection_inverse() -> None:
    cal = VennAbersCalibrator().fit(*_sample(400))
    lo_s, hi_s = cal.interval_inverse(0.1, 0.5)
    p_lo = cal.predict_proba(np.array([min(lo_s + 1e-6, 1 - 1e-9)]))[0]
    p_hi = cal.predict_proba(np.array([max(hi_s - 1e-6, 1e-9)]))[0]
    assert p_lo >= 0.1 - 1e-3
    assert p_hi <= 0.5 + 1e-3


def test_thresholds_wrappers() -> None:
    from probcal.thresholds import calibrated_bands_to_raw, calibrated_interval_to_raw

    cal = PlattCalibrator().fit(*_sample())
    direct = cal.interval_inverse(0.0, 0.02, space="logit")
    wrapped = calibrated_interval_to_raw(cal, 0.0, 0.02, space="logit")
    assert wrapped == direct

    bands = {"A": (0.0, 0.01), "B": (0.01, 0.05), "C": (0.05, 1.0)}
    raw = calibrated_bands_to_raw(cal, bands, space="logit")
    assert set(raw) == {"A", "B", "C"}
    assert raw["A"][1] == pytest.approx(raw["B"][0], abs=1e-9)  # shared edge
    assert raw["A"][0] == -np.inf and raw["C"][1] == np.inf


def test_point_inverse_platt_round_trip() -> None:
    cal = PlattCalibrator().fit(*_sample())
    p = np.linspace(0.02, 0.9, 25)
    s = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(s), p, atol=1e-12)
    s2 = np.linspace(0.05, 0.9, 25)
    p2 = cal.predict_proba(s2)
    np.testing.assert_allclose(cal.point_inverse(p2), s2, atol=1e-9)


def test_point_inverse_temperature_round_trip() -> None:
    cal = TemperatureCalibrator().fit(*_sample())
    p = np.linspace(0.02, 0.9, 25)
    s = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(s), p, atol=1e-12)
    s2 = np.linspace(0.05, 0.9, 25)
    p2 = cal.predict_proba(s2)
    np.testing.assert_allclose(cal.point_inverse(p2), s2, atol=1e-9)


def test_point_inverse_space_consistency() -> None:
    cal = PlattCalibrator().fit(*_sample())
    p = np.linspace(0.02, 0.9, 15)
    z = cal.point_inverse(p, space="logit")
    np.testing.assert_allclose(expit(z), cal.point_inverse(p), atol=1e-12)


def test_point_inverse_non_monotone_platt_raises() -> None:
    cal = PlattCalibrator().fit(*_sample())
    cal.a_ = -1.0
    cal.is_monotone_ = False
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.point_inverse(np.array([0.3]))


def test_point_inverse_unfitted_raises() -> None:
    cal = PlattCalibrator()
    with pytest.raises(RuntimeError):
        cal.point_inverse(np.array([0.3]))


def test_point_inverse_non_affine_monotone_raises() -> None:
    cal = SplineCalibrator().fit(*_sample())
    assert cal.is_monotone_
    with pytest.raises(NotImplementedError, match="interval_inverse"):
        cal.point_inverse(np.array([0.3]))


def _beta_with_params(a: float, b: float, c: float) -> BetaCalibrator:
    cal = BetaCalibrator().fit(*_sample())
    cal.a_, cal.b_, cal.c_ = a, b, c
    return cal


@pytest.mark.parametrize("ratio", [1, 3, 10, 50])
def test_beta_point_inverse_round_trip_ratios(ratio: int) -> None:
    a = 1.0
    b = a * ratio
    cal = _beta_with_params(a, b, 0.2)
    p = np.concatenate([np.linspace(1e-5, 1 - 1e-5, 41), [1e-4, 1 - 1e-4]])
    s = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(s), p, atol=1e-10)


def test_beta_point_inverse_abm_fit_round_trip() -> None:
    s_train, y_train = _sample()
    cal = BetaCalibrator().fit(s_train, y_train)
    p = np.linspace(0.02, 0.9, 25)
    s = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(s), p, atol=1e-10)


def test_beta_point_inverse_a_equals_b_matches_affine() -> None:
    cal = _beta_with_params(0.7, 0.7, 0.3)
    p = np.linspace(0.01, 0.99, 21)
    z = cal.point_inverse(p, space="logit")
    z_affine = (logit(p) - 0.3) / 0.7
    np.testing.assert_allclose(z, z_affine, atol=1e-14)


def test_beta_variant_a_point_inverse_round_trip() -> None:
    cal = BetaCalibrator(variant="a").fit(*_sample())
    p = np.linspace(0.02, 0.9, 15)
    out = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(out), p, atol=1e-10)


def test_beta_variant_ab_point_inverse_round_trip() -> None:
    cal = BetaCalibrator(variant="ab").fit(*_sample())
    p = np.linspace(0.02, 0.9, 15)
    out = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(out), p, atol=1e-10)


def test_beta_variant_ab_override_matches_base_affine_path() -> None:
    # The "ab" variant is affine (a_ == b_), so the base class's generic
    # affine_logit_coeffs_ path and BetaCalibrator's override must agree
    # exactly (not just up to round-trip tolerance) on the same instance.
    from probcal.base import BaseCalibrator

    cal = BetaCalibrator(variant="ab").fit(*_sample())
    p = np.linspace(0.02, 0.9, 15)
    np.testing.assert_allclose(
        BaseCalibrator.point_inverse(cal, p), cal.point_inverse(p), atol=1e-14
    )


def test_beta_point_inverse_degenerate_a_zero() -> None:
    cal = _beta_with_params(0.0, 2.0, 0.1)
    lo = float(expit(np.array([0.1]))[0])
    p_ok = np.linspace(lo + 1e-4, 1 - 1e-6, 10)
    s = cal.point_inverse(p_ok)
    # atol 1e-8, not 1e-10+: expm1/log lose precision as K -> 0+ (p -> lo+),
    # not slack in the closed form itself.
    np.testing.assert_allclose(cal.predict_proba(s), p_ok, atol=1e-8)
    with pytest.raises(UnattainableTargetError, match="attainable"):
        cal.point_inverse(np.array([max(lo - 1e-4, 1e-9)]))


def test_beta_point_inverse_degenerate_b_zero() -> None:
    cal = _beta_with_params(2.0, 0.0, -0.1)
    hi = float(expit(np.array([-0.1]))[0])
    p_ok = np.linspace(1e-6, hi - 1e-4, 10)
    s = cal.point_inverse(p_ok)
    # atol 1e-8, not 1e-10+: expm1/log lose precision as K -> 0- (p -> hi-),
    # not slack in the closed form itself.
    np.testing.assert_allclose(cal.predict_proba(s), p_ok, atol=1e-8)
    with pytest.raises(UnattainableTargetError, match="attainable"):
        cal.point_inverse(np.array([min(hi + 1e-4, 1 - 1e-9)]))


def test_beta_point_inverse_constant_map_raises() -> None:
    cal = _beta_with_params(0.0, 0.0, 0.2)
    with pytest.raises(NotImplementedError):
        cal.point_inverse(np.array([0.5]))


def test_beta_point_inverse_certificate_ratio_50() -> None:
    from probcal.parametric import _beta_point_inverse_z

    a, b = 0.1, 5.0
    K = np.linspace(-25, 25, 51)
    z = _beta_point_inverse_z(K, a, b)
    resid = a * z + (b - a) * np.logaddexp(0.0, z) - K
    assert np.max(np.abs(resid)) <= 1e-12


def test_beta_point_inverse_pathological_certifies_or_raises() -> None:
    # Ratio 50,000 (a=1e-4, b=5) is far outside the numerically verified
    # domain (a, b in (0, 5], ratio <= 50). The post-loop certificate must
    # never let an uncertified root through silently: either the round trip
    # holds at 1e-8, or point_inverse raises RuntimeError naming the
    # residual. No third outcome (silently wrong output) is acceptable.
    cal = _beta_with_params(1e-4, 5.0, 0.2)
    p = np.linspace(0.02, 0.9, 25)
    try:
        s = cal.point_inverse(p)
    except RuntimeError as err:
        assert "residual" in str(err)
    else:
        np.testing.assert_allclose(cal.predict_proba(s), p, atol=1e-8)


def test_offset_point_inverse_round_trip() -> None:
    p = expit(RNG.normal(-1.0, 1.0, 500))
    off = LogitOffset(delta=0.35).fit(p)
    targets = np.linspace(0.02, 0.9, 25)
    s = off.point_inverse(targets)
    np.testing.assert_allclose(off.transform(s), targets, atol=1e-12)
    s2 = np.linspace(0.05, 0.9, 25)
    t2 = off.transform(s2)
    np.testing.assert_allclose(off.point_inverse(t2), s2, atol=1e-9)


def test_offset_point_inverse_matches_interval_inverse_midpoint() -> None:
    p = expit(RNG.normal(-1.0, 1.0, 500))
    off = LogitOffset(delta=-0.2).fit(p)
    for tau in (0.02, 0.1, 0.4):
        raw = off.point_inverse(np.array([tau]))[0]
        lo_s, hi_s = off.interval_inverse(tau, tau)
        np.testing.assert_allclose(raw, 0.5 * (lo_s + hi_s), atol=1e-9)


def test_offset_point_inverse_space_consistency() -> None:
    p = expit(RNG.normal(-1.0, 1.0, 500))
    off = LogitOffset(delta=0.1).fit(p)
    targets = np.linspace(0.02, 0.9, 15)
    z = off.point_inverse(targets, space="logit")
    np.testing.assert_allclose(expit(z), off.point_inverse(targets), atol=1e-12)


def test_offset_point_inverse_unfitted_raises() -> None:
    off = LogitOffset(delta=0.0)
    with pytest.raises(RuntimeError):
        off.point_inverse(np.array([0.3]))


def test_exports() -> None:
    import probcal

    for name in (
        "UnattainableTargetError",
        "calibrated_interval_to_raw",
        "calibrated_bands_to_raw",
    ):
        assert name in probcal.__all__


def test_point_inverse_boundary_p_raises() -> None:
    # p = 0 and p = 1 are not attained by any finite raw score: refusal is
    # all-or-nothing (one bad element fails the whole call), never a silent
    # clip to [1e-12, 1 - 1e-12] followed by a finite "inverse" (W2 doctrine).
    platt = PlattCalibrator().fit(*_sample())
    beta = BetaCalibrator().fit(*_sample())
    off = LogitOffset(delta=0.3).fit(expit(RNG.normal(-1.0, 1.0, 200)))
    for cal in (platt, beta, off):
        for bad in (np.array([0.0]), np.array([1.0]), np.array([0.5, 1.0])):
            with pytest.raises(UnattainableTargetError, match="strictly inside"):
                cal.point_inverse(bad)
