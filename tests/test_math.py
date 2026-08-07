"""Unit tests for probcal._math (numpy-only; reference comparisons live in
test_math_reference.py)."""

import numpy as np
import pytest

from probcal._math import (
    beta_ppf,
    betainc,
    bisect,
    chi2_ppf,
    erf_vec,
    expit,
    gammainc_lower,
    irls_logistic,
    lgamma_vec,
    loess,
    logit,
    natural_cubic_basis,
    newton_1d,
    norm_cdf,
    norm_ppf,
    pava,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- logit/expit


def test_logit_expit_roundtrip() -> None:
    p = np.linspace(1e-9, 1 - 1e-9, 101)
    np.testing.assert_allclose(expit(logit(p)), p, atol=1e-12)


def test_expit_overflow_safe() -> None:
    z = np.array([-800.0, 800.0])
    out = expit(z)
    assert np.all(np.isfinite(out))
    assert out[0] >= 0.0 and out[1] <= 1.0


def test_logit_clips_boundaries() -> None:
    out = logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- solvers


def test_bisect_finds_root() -> None:
    root = bisect(lambda x: x**2 - 2.0, 0.0, 2.0)
    assert abs(root - np.sqrt(2.0)) < 1e-10


def test_newton_1d_finds_root() -> None:
    root = newton_1d(lambda x: x**3 - 8.0, lambda x: 3 * x**2, x0=1.0, lo=0.0, hi=10.0)
    assert abs(root - 2.0) < 1e-10


def test_newton_1d_bisection_fallback() -> None:
    # Newton from x0 with a tiny derivative would jump out of [lo, hi]; must still converge.
    root = newton_1d(lambda x: np.tanh(x - 3.0), lambda x: 1e-12, x0=0.5, lo=0.0, hi=10.0)
    assert abs(root - 3.0) < 1e-8


# ---------------------------------------------------------------- special functions


def test_lgamma_vec_known_values() -> None:
    np.testing.assert_allclose(lgamma_vec(np.array([1.0, 2.0, 5.0])), [0.0, 0.0, np.log(24.0)])


def test_erf_vec_known_values() -> None:
    out = erf_vec(np.array([0.0, 10.0, -10.0]))
    np.testing.assert_allclose(out, [0.0, 1.0, -1.0], atol=1e-15)
    assert out.dtype == np.float64


def test_betainc_identity_parameters() -> None:
    x = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(betainc(1.0, 1.0, x), x, atol=1e-14)


def test_betainc_symmetry() -> None:
    x = np.linspace(0.01, 0.99, 50)
    left = betainc(2.5, 4.0, x)
    right = 1.0 - betainc(4.0, 2.5, 1.0 - x)
    np.testing.assert_allclose(left, right, atol=1e-13)


def test_betainc_monotone_in_x() -> None:
    x = np.linspace(0.0, 1.0, 200)
    v = betainc(3.0, 7.0, x)
    assert np.all(np.diff(v) >= -1e-15)


def test_gammainc_lower_exponential_case() -> None:
    x = np.linspace(0.0, 20.0, 50)
    np.testing.assert_allclose(gammainc_lower(1.0, x), 1.0 - np.exp(-x), atol=1e-13)


def test_chi2_ppf_roundtrip() -> None:
    for df in (1.0, 2.0, 5.0, 30.0):
        x = 7.3
        q = gammainc_lower(df / 2.0, x / 2.0)
        assert abs(chi2_ppf(q, df) - x) < 1e-8


def test_beta_ppf_hand_anchors() -> None:
    assert abs(beta_ppf(0.5, 2.0, 2.0) - 0.5) < 1e-10
    assert abs(beta_ppf(0.5, 1.0, 1.0) - 0.5) < 1e-10
    # Beta(1,2): CDF = 1 - (1-x)^2 -> x = 1 - sqrt(1-q); q=0.19 -> 0.1.
    assert abs(beta_ppf(0.19, 1.0, 2.0) - 0.1) < 1e-10


def test_beta_ppf_monotone_in_q() -> None:
    qs = np.linspace(0.01, 0.99, 25)
    vals = [beta_ppf(float(q), 2.5, 7.0) for q in qs]
    assert np.all(np.diff(vals) > 0)


def test_beta_ppf_validation() -> None:
    with pytest.raises(ValueError, match="beta_ppf"):
        beta_ppf(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="beta_ppf"):
        beta_ppf(1.0, 1.0, 1.0)


def test_norm_cdf_symmetry() -> None:
    x = np.linspace(-8.0, 8.0, 81)
    np.testing.assert_allclose(norm_cdf(x) + norm_cdf(-x), np.ones_like(x), atol=1e-14)


def test_norm_ppf_known_quantile() -> None:
    assert abs(norm_ppf(np.array([0.975]))[0] - 1.959963984540054) < 1e-11


def test_norm_ppf_roundtrip() -> None:
    q = np.linspace(1e-10, 1 - 1e-10, 101)
    np.testing.assert_allclose(norm_cdf(norm_ppf(q)), q, atol=1e-12)


# ---------------------------------------------------------------- PAVA


def test_pava_worked_example() -> None:
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    w = np.ones(5)
    res = pava(y, w)
    np.testing.assert_allclose(res.fitted, [0.0, 1 / 3, 1 / 3, 1 / 3, 1.0])
    assert res.block_start.tolist() == [0, 1, 4]
    np.testing.assert_allclose(res.block_mean, [0.0, 1 / 3, 1.0])
    np.testing.assert_allclose(res.block_weight, [1.0, 3.0, 1.0])


def test_pava_monotone_output() -> None:
    y = RNG.random(500)
    res = pava(y, np.ones(500))
    assert np.all(np.diff(res.fitted) >= -1e-15)


def test_pava_already_monotone_unchanged() -> None:
    y = np.array([0.1, 0.2, 0.3])
    res = pava(y, np.ones(3))
    np.testing.assert_allclose(res.fitted, y)


def test_pava_weighted() -> None:
    # Heavier weight on the later observation pulls the pooled mean toward it.
    y = np.array([1.0, 0.0])
    res = pava(y, np.array([1.0, 3.0]))
    np.testing.assert_allclose(res.fitted, [0.25, 0.25])


# ---------------------------------------------------------------- IRLS logistic


def _make_logistic_data(n: int = 5000, beta=(-2.0, 1.5)) -> tuple[np.ndarray, np.ndarray]:
    x = RNG.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    p = expit(X @ np.asarray(beta))
    y = (RNG.random(n) < p).astype(float)
    return X, y


def test_irls_recovers_coefficients() -> None:
    X, y = _make_logistic_data()
    res = irls_logistic(X, y)
    assert res.converged
    assert not res.separation
    np.testing.assert_allclose(res.beta, [-2.0, 1.5], atol=0.15)


def test_irls_offset_shifts_intercept() -> None:
    X, y = _make_logistic_data()
    res0 = irls_logistic(X, y)
    res1 = irls_logistic(X, y, offset=np.full(len(y), 0.7))
    np.testing.assert_allclose(res1.beta[0], res0.beta[0] - 0.7, atol=1e-6)
    np.testing.assert_allclose(res1.beta[1], res0.beta[1], atol=1e-6)


def test_irls_separation_warns_and_returns_finite() -> None:
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    X = np.column_stack([np.ones(4), x])
    y = np.array([0.0, 0.0, 1.0, 1.0])  # perfectly separated
    with pytest.warns(UserWarning, match="[Ss]eparation"):
        res = irls_logistic(X, y)
    assert np.all(np.isfinite(res.beta))
    assert res.separation


# ---------------------------------------------------------------- LOESS


def test_loess_reproduces_linear_trend() -> None:
    x = np.linspace(0.0, 1.0, 60)
    y = 2.0 * x + 1.0
    fitted = loess(x, y, frac=0.5)
    np.testing.assert_allclose(fitted, y, atol=1e-10)


def test_loess_xeval() -> None:
    x = np.linspace(0.0, 1.0, 60)
    y = 2.0 * x + 1.0
    grid = np.array([0.25, 0.75])
    np.testing.assert_allclose(loess(x, y, xeval=grid), 2.0 * grid + 1.0, atol=1e-10)


# ---------------------------------------------------------------- natural cubic basis


def test_natural_cubic_basis_shape() -> None:
    x = np.linspace(0.0, 1.0, 40)
    knots = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    B = natural_cubic_basis(x, knots)
    assert B.shape == (40, 5)


def test_natural_cubic_basis_linear_beyond_boundary() -> None:
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x = np.array([1.5, 2.0, 2.5, 3.0])  # all beyond the last knot
    B = natural_cubic_basis(x, knots)
    # Linear in x out there: second differences vanish column by column.
    second_diff = np.diff(B, n=2, axis=0)
    np.testing.assert_allclose(second_diff, 0.0, atol=1e-9)
