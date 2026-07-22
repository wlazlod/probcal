"""Tests for probcal.offset.LogitOffset."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.offset import LogitOffset

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
