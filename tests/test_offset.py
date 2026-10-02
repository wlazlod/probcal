"""Tests for probcal.offset.LogitOffset and probcal.offset.estimate_offset."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.monitor._processes import plug_in_delta
from probcal.offset import LogitOffset, _offset_mle, estimate_offset, offset_from_estimate

RNG = np.random.default_rng(97)


def _pd_portfolio(n: int = 5000, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    p_true = expit(RNG.normal(-3.4, 1.0, n))
    y = (RNG.random(n) < p_true).astype(float)
    p_model = expit(logit(p_true) + shift)
    return y, p_model


def test_mode_a_explicit_delta() -> None:
    p = np.array([0.01, 0.05, 0.2])
    off = LogitOffset(delta=-0.3).fit(p)
    np.testing.assert_allclose(off.transform(p), expit(logit(p) - 0.3), atol=1e-14)
    assert off.delta_ == -0.3


def test_mode_a_zero_delta_is_identity() -> None:
    p = np.linspace(0.01, 0.9, 50)
    off = LogitOffset(delta=0.0).fit(p)
    np.testing.assert_allclose(off.transform(p), p, atol=1e-12)


def test_mode_b_hits_target_mean() -> None:
    _, p = _pd_portfolio(shift=0.4)  # inflated portfolio
    target = 0.031
    off = LogitOffset(target_mean=target).fit(p)
    assert abs(off.transform(p).mean() - target) < 1e-10
    assert abs(off.post_mean_ - target) < 1e-10
    assert off.pre_mean_ == pytest.approx(p.mean())


def test_mode_b_root_unique_vs_grid() -> None:
    _, p = _pd_portfolio(1000, shift=0.5)
    target = 0.03
    off = LogitOffset(target_mean=target).fit(p)
    # Brute-force check: the mean is strictly increasing in delta, so the
    # solved delta must be the unique grid minimizer of |mean - target|.
    grid = np.linspace(off.delta_ - 0.5, off.delta_ + 0.5, 2001)
    means = np.array([expit(logit(p) + d).mean() for d in grid])
    assert np.all(np.diff(means) > 0)
    assert abs(grid[np.argmin(np.abs(means - target))] - off.delta_) < 1e-3


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        LogitOffset(delta=0.1, target_mean=0.05).fit(np.array([0.1]))
    with pytest.raises(ValueError, match="exactly one"):
        LogitOffset().fit(np.array([0.1]))


def test_affine_coeffs_and_interpret() -> None:
    p = np.linspace(0.01, 0.5, 100)
    off = LogitOffset(delta=-0.42).fit(p)
    a, b = off.affine_logit_coeffs_
    assert (a, b) == (1.0, -0.42)
    interp = off.interpret()
    assert "delta" in interp.param_names
    assert any("odds" in m for m in interp.messages)


def test_audit_report_shows_repair() -> None:
    y, p = _pd_portfolio(20000, shift=0.5)  # pure level error
    off = LogitOffset(target_mean=float(y.mean())).fit(p)
    report = off.audit_report(y, p)
    assert abs(report.guardrails_after.intercept) < abs(report.guardrails_before.intercept)
    assert report.delta == off.delta_
    assert report.timestamp  # non-empty ISO string
    assert not report.guardrails_before.intercept_ok
    assert report.guardrails_after.intercept_ok


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        LogitOffset(delta=0.1).transform(np.array([0.1]))


def test_export() -> None:
    import probcal

    assert "LogitOffset" in probcal.__all__
    assert "estimate_offset" in probcal.__all__
    assert "offset_from_estimate" in probcal.__all__
    assert "OffsetEstimate" in probcal.__all__


def test_estimate_offset_recovers_delta_within_se() -> None:
    rng = np.random.default_rng(1234)
    n = 20000
    z = rng.normal(0.0, 1.0, n)
    p = expit(z)
    true_delta = 0.5
    y = (rng.random(n) < expit(z + true_delta)).astype(float)

    est = estimate_offset(y, p)

    assert abs(est.delta - true_delta) < 3.0 * est.se
    q = expit(logit(p) + est.delta)
    expected_se = 1.0 / np.sqrt(np.sum(q * (1.0 - q)))
    assert est.se == pytest.approx(expected_se, abs=1e-12)
    assert est.n == n
    assert est.events == pytest.approx(float(y.sum()))
    assert est.weight_sum == pytest.approx(float(n))


def test_estimate_offset_single_class_raises() -> None:
    p = np.linspace(0.1, 0.9, 20)
    y = np.zeros(20)
    with pytest.raises(ValueError, match="both classes"):
        estimate_offset(y, p)


def test_estimate_offset_weights_equal_duplication() -> None:
    rng = np.random.default_rng(5)
    n = 200
    p = expit(rng.normal(0.0, 1.0, n))
    y = (rng.random(n) < expit(logit(p) + 0.3)).astype(float)
    weights = rng.integers(1, 4, n).astype(float)

    est_weighted = estimate_offset(y, p, sample_weight=weights)

    y_dup = np.repeat(y, weights.astype(int))
    p_dup = np.repeat(p, weights.astype(int))
    est_dup = estimate_offset(y_dup, p_dup)

    assert est_weighted.delta == pytest.approx(est_dup.delta, abs=1e-10)
    assert est_weighted.se == pytest.approx(est_dup.se, abs=1e-10)


def test_offset_from_estimate_matches_delta() -> None:
    p = np.linspace(0.05, 0.6, 500)
    y = (RNG.random(500) < p).astype(float)
    est = estimate_offset(y, p)
    off = offset_from_estimate(est, p)
    assert off.delta_ == est.delta
    np.testing.assert_allclose(off.transform(p), expit(logit(p) + est.delta), atol=1e-14)


def test_plug_in_delta_matches_offset_mle() -> None:
    rng = np.random.default_rng(77)
    n = 500
    z = rng.normal(0.0, 1.0, n)
    y = (rng.random(n) < expit(z + 0.2)).astype(float)
    w = rng.uniform(0.5, 2.0, n)

    assert plug_in_delta(z, y, w) == _offset_mle(z, y, w)


def test_fit_accepts_and_ignores_y() -> None:
    p = np.array([0.1, 0.2, 0.4])
    y = np.array([0.0, 1.0, 1.0])
    a = LogitOffset(delta=0.3).fit(p)
    b = LogitOffset(delta=0.3).fit(p, y=y)
    assert a.delta_ == b.delta_
    np.testing.assert_array_equal(a.transform(p), b.transform(p))


def test_interval_inverse_validates_space_and_buffer() -> None:
    # OFF-11 / CORE-5: an unknown space silently returned probabilities; a
    # negative buffer was silently ignored.
    off = LogitOffset(delta=0.3).fit(np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="space"):
        off.interval_inverse(0.1, 0.2, space="logits")
    with pytest.raises(ValueError, match="buffer_logit"):
        off.interval_inverse(0.1, 0.2, buffer_logit=-0.1)
    lo, hi = off.interval_inverse(0.1, 0.2, space="logit", buffer_logit=0.05)
    assert lo == pytest.approx(logit(np.array([0.1]))[0] + 0.05 - 0.3, abs=1e-12)
    assert hi == pytest.approx(logit(np.array([0.2]))[0] - 0.05 - 0.3, abs=1e-12)


def test_unattainable_target_mean_names_the_attainable_range() -> None:
    # OFF-17: used to surface as a bare bisection-bracket error.
    from probcal import UnattainableTargetError

    p = np.full(5, 1e-12)
    with pytest.raises(UnattainableTargetError, match="reach portfolio means"):
        LogitOffset(target_mean=1.0 - 1e-9).fit(p)


def test_positional_sample_weight_is_deprecated_but_works() -> None:
    # OFF-26: sample_weight is keyword-only; the 0.3 positional form warns.
    p = np.array([0.05, 0.1, 0.3])
    w = np.array([1.0, 2.0, 3.0])
    ref = LogitOffset(target_mean=0.12).fit(p, sample_weight=w)
    with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
        old = LogitOffset(target_mean=0.12).fit(p, w)
    assert old.delta_ == ref.delta_
    with pytest.raises(TypeError):
        LogitOffset(target_mean=0.12).fit(p, w, sample_weight=w)


def test_fit_mode_b_and_offset_mle_share_the_root_finder() -> None:
    # OFF-20: one _solve_delta behind both entry points.
    rng = np.random.default_rng(3)
    p = expit(rng.normal(-2.0, 1.0, 300))
    y = (rng.random(300) < p).astype(float)
    w = np.ones(300)
    off = LogitOffset(target_mean=float(y.mean())).fit(p)
    assert off.delta_ == _offset_mle(logit(p), y, w)


def test_json_round_trip_uses_shared_protocol() -> None:
    off = LogitOffset(delta=0.2).fit(np.array([0.1, 0.4]))
    back = LogitOffset.from_json(off.to_json())
    assert back.delta_ == off.delta_ and back.fingerprint() == off.fingerprint()
    assert back.get_params() == {"delta": 0.2, "target_mean": None}
    assert back.set_params(delta=0.5).delta == 0.5
    with pytest.raises(ValueError, match="unknown parameter"):
        back.set_params(nope=1)
