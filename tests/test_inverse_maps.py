"""Tests for interval_inverse / point_inverse across the calibrator catalog."""

import math
import warnings

import numpy as np
import pytest

import probcal as pc
from probcal._math import expit, logit
from probcal._steps import refine_lower, refine_upper
from probcal.base import UnattainableTargetError
from probcal.bayesian import ENIRCalibrator
from probcal.isotonic import IsotonicCalibrator
from probcal.offset import LogitOffset
from probcal.parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from probcal.spline import SplineCalibrator

RNG = np.random.default_rng(103)


def _sample(n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    s = expit(RNG.normal(-1.0, 1.4, n))
    y = (RNG.random(n) < expit(0.8 * logit(s) - 0.4)).astype(float)
    return s, y


def test_isotonic_block_edge_semantics() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    cal = IsotonicCalibrator().fit(s, y)  # blocks: 0 @0.1 | 1/3 @0.2-0.4 | 1 @0.5
    raw_lo, raw_hi = cal.interval_inverse(0.2, 0.9)
    assert raw_lo == 0.2  # left edge of the first block with mean >= 0.2
    assert raw_hi == np.nextafter(0.5, 0.0)  # one float below the last block start:
    # the bound is itself in the preimage, so closed-bound consumers are safe


def test_offset_shifts_raw_bounds_by_minus_delta() -> None:
    p = expit(RNG.normal(-2.0, 1.0, 500))
    off0 = LogitOffset(delta=0.0).fit(p)
    off1 = LogitOffset(delta=0.5).fit(p)
    lo0, hi0 = off0.interval_inverse(0.02, 0.10, space="logit")
    lo1, hi1 = off1.interval_inverse(0.02, 0.10, space="logit")
    np.testing.assert_allclose(lo1, lo0 - 0.5, atol=1e-10)
    np.testing.assert_allclose(hi1, hi0 - 0.5, atol=1e-10)


def test_enir_not_implemented() -> None:
    cal = ENIRCalibrator().fit(*_sample(500))
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.interval_inverse(0.1, 0.2)


@pytest.mark.parametrize(
    "cls",
    [PlattCalibrator, TemperatureCalibrator],
    ids=["PlattCalibrator", "TemperatureCalibrator"],
)
def test_point_inverse_affine_round_trip(cls: type) -> None:
    cal = cls().fit(*_sample())
    p = np.linspace(0.02, 0.9, 25)
    s = cal.point_inverse(p)
    np.testing.assert_allclose(cal.predict_proba(s), p, atol=1e-12)
    s2 = np.linspace(0.05, 0.9, 25)
    p2 = cal.predict_proba(s2)
    np.testing.assert_allclose(cal.point_inverse(p2), s2, atol=1e-9)
    z = cal.point_inverse(p, space="logit")
    np.testing.assert_allclose(expit(z), s, atol=1e-12)


def test_point_inverse_non_monotone_platt_raises() -> None:
    cal = PlattCalibrator().fit(*_sample())
    cal.a_ = -1.0
    cal.is_monotone_ = False
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.point_inverse(np.array([0.3]))
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.interval_inverse(0.0, 0.5, space="logit", buffer_logit=0.0)


def test_point_inverse_non_affine_monotone_raises() -> None:
    cal = SplineCalibrator().fit(*_sample())
    assert cal.is_monotone_
    with pytest.raises(NotImplementedError, match="interval_inverse"):
        cal.point_inverse(np.array([0.3]))


def _beta_with_params(a: float, b: float, c: float) -> BetaCalibrator:
    cal = BetaCalibrator().fit(*_sample())
    cal.a_, cal.b_, cal.c_ = a, b, c
    return cal


def test_beta_point_inverse_a_equals_b_matches_affine() -> None:
    cal = _beta_with_params(0.7, 0.7, 0.3)
    p = np.linspace(0.01, 0.99, 21)
    z = cal.point_inverse(p, space="logit")
    z_affine = (logit(p) - 0.3) / 0.7
    np.testing.assert_allclose(z, z_affine, atol=1e-14)


@pytest.mark.parametrize(
    ("a", "b", "c", "bad"),
    [(0.0, 2.0, 0.1, "below"), (2.0, 0.0, -0.1, "above")],
    ids=["a-zero", "b-zero"],
)
def test_beta_point_inverse_degenerate_exponent(a: float, b: float, c: float, bad: str) -> None:
    # One exponent at 0: the attainable range is (sigma(c), 1) or (0, sigma(c)).
    cal = _beta_with_params(a, b, c)
    edge = float(expit(np.array([c]))[0])
    p_ok = (
        np.linspace(edge + 1e-4, 1 - 1e-6, 10)
        if bad == "below"
        else np.linspace(1e-6, edge - 1e-4, 10)
    )
    s = cal.point_inverse(p_ok)
    # atol 1e-8, not 1e-10+: expm1/log lose precision as p -> sigma(c),
    # not slack in the closed form itself.
    np.testing.assert_allclose(cal.predict_proba(s), p_ok, atol=1e-8)
    p_bad = max(edge - 1e-4, 1e-9) if bad == "below" else min(edge + 1e-4, 1 - 1e-9)
    with pytest.raises(UnattainableTargetError, match="attainable"):
        cal.point_inverse(np.array([p_bad]))


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
    z = off.point_inverse(targets, space="logit")
    np.testing.assert_allclose(expit(z), s, atol=1e-12)


def test_point_inverse_boundary_p_raises() -> None:
    # p = 0 and p = 1 are not attained by any finite raw score: refusal is
    # all-or-nothing (one bad element fails the whole call), never a silent
    # clip to [1e-12, 1 - 1e-12] followed by a finite "inverse".
    platt = PlattCalibrator().fit(*_sample())
    beta = BetaCalibrator().fit(*_sample())
    off = LogitOffset(delta=0.3).fit(expit(RNG.normal(-1.0, 1.0, 200)))
    for cal in (platt, beta, off):
        for bad in (np.array([0.0]), np.array([1.0]), np.array([0.5, 1.0])):
            with pytest.raises(UnattainableTargetError, match="strictly inside"):
                cal.point_inverse(bad)
    # The error names at most 5 offending values, never the whole array.
    with pytest.raises(UnattainableTargetError, match=r"and 2 more"):
        platt.point_inverse(np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]))


def test_point_inverse_probability_space_overflow_raises() -> None:
    # A certified root with |z| > logit(1 - 1e-12) ~ 27.63 is exact in logit
    # space but sigma(z) rounds into the clipping zone: predict_proba clips it
    # back and the round trip silently breaks (0.1.3: target 0.998 -> raw 1.0
    # -> forward 0.17, no warning). Probability space must refuse and point at
    # space="logit".
    cal = PlattCalibrator().fit(*_sample())
    cal.a_, cal.b_ = 0.2, 0.0  # shallow map: p = 0.998 needs raw logit ~31
    with pytest.raises(UnattainableTargetError, match="logit"):
        cal.point_inverse(np.array([0.998]))
    z = cal.point_inverse(np.array([0.998]), space="logit")  # exact there
    np.testing.assert_allclose(z, logit(np.array([0.998])) / 0.2, atol=1e-12)

    beta = _beta_with_params(1.0, 0.25, 0.0)  # abm Halley path, z ~ K/b ~ 37
    with pytest.raises(UnattainableTargetError, match="logit"):
        beta.point_inverse(np.array([0.9999]))
    assert np.isfinite(beta.point_inverse(np.array([0.9999]), space="logit")).all()

    off = LogitOffset(delta=10.0).fit(expit(RNG.normal(-1.0, 1.0, 200)))
    with pytest.raises(UnattainableTargetError, match="logit"):
        off.point_inverse(np.array([1e-9]))  # z = logit(1e-9) - 10 ~ -30.7
    assert np.isfinite(off.point_inverse(np.array([1e-9]), space="logit")).all()


def test_offset_interval_inverse_respects_representable_range() -> None:
    # A bound whose preimage would fall below the 1e-12 clip cannot round-trip
    # through transform (raw_lo=4.5e-14 for target 1e-9 at delta=10 was the
    # silent break); such a bound collapses to the full-range sentinel (0.0 /
    # -inf), matching BaseCalibrator.interval_inverse's convention, and an
    # interval entirely outside the representable output range raises.
    off = LogitOffset(delta=10.0).fit(expit(RNG.normal(-1.0, 1.0, 200)))
    raw_lo, raw_hi = off.interval_inverse(1e-9, 0.5)
    assert raw_lo == 0.0  # no lower constraint: every representable raw qualifies
    assert 1e-12 < raw_hi < 1.0 - 1e-12  # in-range bound still exact
    np.testing.assert_allclose(off.transform(np.array([raw_hi])), 0.5, atol=1e-12)
    lo_z, hi_z = off.interval_inverse(1e-9, 0.5, space="logit")
    assert np.isneginf(lo_z) and np.isfinite(hi_z)
    with pytest.raises(UnattainableTargetError, match="intersect"):
        off.interval_inverse(1e-10, 1e-9)  # entirely below the representable range
    # Moderate targets are untouched by the range guard.
    off2 = LogitOffset(delta=0.35).fit(expit(RNG.normal(-1.0, 1.0, 200)))
    lo_r, hi_r = off2.interval_inverse(0.02, 0.1)
    np.testing.assert_allclose(off2.transform(np.array([lo_r, hi_r])), [0.02, 0.1], atol=1e-12)


def test_point_and_interval_inverse_agree_on_attainability() -> None:
    # The two inverse maps must classify the same target the same way.
    # (a) Degenerate beta a=0: attainable range is (sigma(c), 1).
    cal = _beta_with_params(0.0, 2.0, 0.1)
    lo = float(expit(np.array([0.1]))[0])
    p_bad = lo - 0.05
    with pytest.raises(UnattainableTargetError, match="attainable"):
        cal.point_inverse(np.array([p_bad]))
    with pytest.raises(UnattainableTargetError, match="intersect"):
        cal.interval_inverse(0.0, p_bad)
    p_ok = lo + 0.1
    cal.point_inverse(np.array([p_ok]))  # must not raise
    cal.interval_inverse(p_ok, p_ok)  # must not raise
    # (b) Affine map: p above g(1 - 1e-12) <=> z above logit(1 - 1e-12), so the
    # representability check aligns the two methods exactly at the range edge.
    platt = PlattCalibrator().fit(*_sample())
    gmax = float(platt.predict_proba(np.array([1.0 - 1e-12]))[0])
    p_above = 0.5 * (gmax + 1.0)
    with pytest.raises(UnattainableTargetError, match="logit"):
        platt.point_inverse(np.array([p_above]))
    with pytest.raises(UnattainableTargetError, match="intersect"):
        platt.interval_inverse(p_above, 1.0)


# ------------------------------------------------------------ ulp-exact bounds (0.3.5)


def test_refine_lower_upper_exact() -> None:
    t = 0.3
    for start in (t, math.nextafter(t, 0.0), math.nextafter(math.nextafter(t, 1.0), 1.0)):
        assert refine_lower(start, lambda v: v >= t) == t
        assert refine_upper(start, lambda v: v <= t) == t


def _tight_sample(kind: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    s = rng.beta(1, 8, 400)
    if kind == "noisy":
        y = (rng.random(400) < s * 1.3).astype(float)
    else:  # labels ordered by score: the histogram rates come out monotone
        y = (s > np.quantile(s, 0.7)).astype(float)
    return s, y


_INVERTIBLE = [
    pytest.param(pc.PlattCalibrator, "noisy", id="PlattCalibrator"),
    pytest.param(pc.TemperatureCalibrator, "noisy", id="TemperatureCalibrator"),
    pytest.param(lambda: pc.BetaCalibrator("ab"), "noisy", id="Beta-ab"),
    pytest.param(lambda: pc.BetaCalibrator("a"), "noisy", id="Beta-a"),
    pytest.param(pc.BetaCalibrator, "noisy", id="BetaCalibrator"),
    pytest.param(pc.IsotonicCalibrator, "noisy", id="IsotonicCalibrator"),
    pytest.param(lambda: pc.IsotonicCalibrator("linear"), "noisy", id="Isotonic-linear"),
    pytest.param(pc.CenteredIsotonicCalibrator, "noisy", id="CenteredIsotonicCalibrator"),
    pytest.param(pc.HistogramBinningCalibrator, "ordered", id="HistogramBinningCalibrator"),
    pytest.param(pc.ScalingBinningCalibrator, "noisy", id="ScalingBinningCalibrator"),
    pytest.param(lambda: pc.BBQCalibrator(min_bins=2, max_bins=3), "noisy", id="BBQ-2-3"),
    pytest.param(lambda: pc.BBQCalibrator(monotone=True), "noisy", id="BBQ-monotone"),
    pytest.param(pc.SplineCalibrator, "noisy", id="SplineCalibrator"),
    pytest.param(pc.VennAbersCalibrator, "noisy", id="VennAbersCalibrator"),
    pytest.param(pc.CrossVennAbersCalibrator, "noisy", id="CrossVennAbersCalibrator"),
]


@pytest.mark.parametrize(("factory", "kind"), _INVERTIBLE)
def test_interval_inverse_bounds_in_preimage_and_tight(factory, kind: str) -> None:
    """Both one-sided bounds lie in the preimage and are tight to one float step
    (CAL-1/CAL-11 and the 0.3.5 ulp refinement): the next float outward leaves it."""
    s, y = _tight_sample(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = factory().fit(s, y)
    assert cal.is_monotone_  # the fixture is chosen so that no case is skipped
    lo, hi = cal._output_range()
    g = cal.predict_proba
    for t in np.random.default_rng(7).uniform(lo, hi, 40):
        raw_lo, _ = cal.interval_inverse(t, 1.0)
        _, raw_hi = cal.interval_inverse(0.0, t)
        assert g([raw_lo])[0] >= t > g([math.nextafter(raw_lo, 0.0)])[0]
        assert g([raw_hi])[0] <= t < g([math.nextafter(raw_hi, 1.0)])[0]
