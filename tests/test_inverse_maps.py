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


def test_exports() -> None:
    import probcal

    for name in (
        "UnattainableTargetError",
        "calibrated_interval_to_raw",
        "calibrated_bands_to_raw",
    ):
        assert name in probcal.__all__
