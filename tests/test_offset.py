"""Tests for probcal.offset.LogitOffset and probcal.offset.estimate_offset."""

import numpy as np
import pytest

from probcal import UnattainableTargetError
from probcal._math import expit, logit
from probcal.monitor._processes import plug_in_delta
from probcal.offset import LogitOffset, _offset_mle, estimate_offset, offset_from_estimate

RNG = np.random.default_rng(97)


def _pd_portfolio(n: int = 5000, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    p_true = expit(RNG.normal(-3.4, 1.0, n))
    y = (RNG.random(n) < p_true).astype(float)
    p_model = expit(logit(p_true) + shift)
    return y, p_model


def test_mode_a_closed_form_transform_affine_and_inverse() -> None:
    p = np.array([0.01, 0.05, 0.2])
    off = LogitOffset(delta=-0.3).fit(p, y=np.array([0.0, 1.0, 1.0]))  # y is ignored
    np.testing.assert_allclose(off.transform(p), expit(logit(p) - 0.3), atol=1e-14)
    assert off.delta_ == -0.3
    assert off.affine_logit_coeffs_ == (1.0, -0.3)
    assert "delta" in off.interpret().param_names
    # OFF-11: the logit-space inverse is the closed form, buffer included.
    lo, hi = off.interval_inverse(0.1, 0.2, space="logit", buffer_logit=0.05)
    assert lo == pytest.approx(logit(np.array([0.1]))[0] + 0.05 + 0.3, abs=1e-12)
    assert hi == pytest.approx(logit(np.array([0.2]))[0] - 0.05 + 0.3, abs=1e-12)


def test_mode_b_hits_target_and_root_is_unique_vs_grid() -> None:
    _, p = _pd_portfolio(1000, shift=0.5)
    target = 0.03
    off = LogitOffset(target_mean=target).fit(p)
    assert abs(off.transform(p).mean() - target) < 1e-10
    assert abs(off.post_mean_ - target) < 1e-10
    assert off.pre_mean_ == pytest.approx(p.mean())
    # Brute-force check: the mean is strictly increasing in delta, so the
    # solved delta must be the unique grid minimizer of |mean - target|.
    grid = np.linspace(off.delta_ - 0.5, off.delta_ + 0.5, 2001)
    means = np.array([expit(logit(p) + d).mean() for d in grid])
    assert np.all(np.diff(means) > 0)
    assert abs(grid[np.argmin(np.abs(means - target))] - off.delta_) < 1e-3


_FITTED = LogitOffset(delta=0.3).fit(np.array([0.1, 0.2]))


@pytest.mark.parametrize(
    ("call", "exc", "match"),
    [
        (
            lambda: LogitOffset(delta=0.1, target_mean=0.05).fit(np.array([0.1])),
            ValueError,
            "exactly one",
        ),
        (lambda: LogitOffset().fit(np.array([0.1])), ValueError, "exactly one"),
        (lambda: LogitOffset(delta=0.1).transform(np.array([0.1])), RuntimeError, "not fitted"),
        (
            lambda: estimate_offset(np.zeros(20), np.linspace(0.1, 0.9, 20)),
            ValueError,
            "both classes",
        ),
        # OFF-11 / CORE-5: an unknown space / a negative buffer used to be ignored.
        (lambda: _FITTED.interval_inverse(0.1, 0.2, space="logits"), ValueError, "space"),
        (lambda: _FITTED.interval_inverse(0.1, 0.2, buffer_logit=-0.1), ValueError, "buffer_logit"),
        # OFF-17: used to surface as a bare bisection-bracket error.
        (
            lambda: LogitOffset(target_mean=1.0 - 1e-9).fit(np.full(5, 1e-12)),
            UnattainableTargetError,
            "reach portfolio means",
        ),
    ],
    ids=["both", "neither", "unfitted", "single-class", "space", "buffer", "unattainable"],
)
def test_validation(call, exc, match) -> None:
    with pytest.raises(exc, match=match):
        call()


def test_audit_report_shows_repair() -> None:
    y, p = _pd_portfolio(20000, shift=0.5)  # pure level error
    off = LogitOffset(target_mean=float(y.mean())).fit(p)
    report = off.audit_report(y, p)
    assert abs(report.guardrails_after.intercept) < abs(report.guardrails_before.intercept)
    assert report.delta == off.delta_
    assert report.timestamp  # non-empty ISO string
    assert not report.guardrails_before.intercept_ok
    assert report.guardrails_after.intercept_ok


def test_estimate_offset_recovers_delta_with_closed_form_se() -> None:
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
    off = offset_from_estimate(est, p)
    assert off.delta_ == est.delta
    np.testing.assert_allclose(off.transform(p), q, atol=1e-14)


def test_estimate_offset_weights_scale_invariant_with_exact_variance() -> None:
    rng = np.random.default_rng(5)
    n = 200
    p = expit(rng.normal(0.0, 1.0, n))
    y = (rng.random(n) < expit(logit(p) + 0.3)).astype(float)
    weights = rng.integers(1, 4, n).astype(float)

    est = estimate_offset(y, p, sample_weight=weights)
    # delta solves the same score equation as row duplication.
    est_dup = estimate_offset(np.repeat(y, weights.astype(int)), np.repeat(p, weights.astype(int)))
    assert est.delta == pytest.approx(est_dup.delta, abs=1e-10)
    for c in (0.1, 3.7, 10.0):
        scaled = estimate_offset(y, p, sample_weight=c * weights)
        assert scaled.delta == est.delta
        assert scaled.se == pytest.approx(est.se, rel=1e-12)
    q = expit(logit(p) + est.delta)
    expected = np.sqrt(np.sum(weights**2 * q * (1 - q))) / np.sum(weights * q * (1 - q))
    assert est.se == pytest.approx(expected, rel=1e-12)
    assert estimate_offset(y, p, sample_weight=np.ones(n)).se == estimate_offset(y, p).se


def test_fit_mode_b_monitor_and_offset_mle_share_the_root_finder() -> None:
    # OFF-20: one _solve_delta behind every entry point.
    rng = np.random.default_rng(3)
    p = expit(rng.normal(-2.0, 1.0, 300))
    y = (rng.random(300) < p).astype(float)
    w = rng.uniform(0.5, 2.0, 300)
    off = LogitOffset(target_mean=float(y.mean())).fit(p)
    assert off.delta_ == _offset_mle(logit(p), y, np.ones(300))
    assert plug_in_delta(logit(p), y, w) == _offset_mle(logit(p), y, w)
