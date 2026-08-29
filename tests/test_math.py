"""Unit tests for probcal._math (numpy-only; reference comparisons live in
test_math_reference.py)."""

import math
import warnings

import numpy as np
import pytest

from probcal._math import (
    _FPMIN,
    _loess_fit_sorted,
    _loess_fit_sorted_vec,
    _loess_window_starts,
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
    weighted_quantile,
)
from probcal.datasets import make_pd_portfolio

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
    with pytest.warns(UserWarning, match="[Ss]eparation") as rec:
        res = irls_logistic(X, y)
    assert sum("eparation" in str(r.message) for r in rec) == 1
    assert np.all(np.isfinite(res.beta))
    assert res.separation
    assert res.converged  # the ridge fallback is coercive: it must converge
    assert res.beta[1] > 0.0  # fitted map monotone in x


@pytest.mark.parametrize("size", ["small", "large"])
def test_irls_quasi_separation_warns_and_converges(size: str) -> None:
    if size == "small":
        # Detected via the singular Hessian: far-point weights underflow.
        x = np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # class ranges touch at x = 3
    else:
        # Realistic size: the Hessian stays numerically nonsingular and the
        # tied pair keeps the margin rule from firing, so detection comes from
        # the unconverged-divergence signature (max_iter exhaustion).
        x = np.concatenate([np.linspace(-4.0, -0.2, 50), [0.0, 0.0], np.linspace(0.2, 4.0, 50)])
        y = np.concatenate([np.zeros(50), [0.0, 1.0], np.ones(50)])
    X = np.column_stack([np.ones(len(x)), x])
    with pytest.warns(UserWarning, match="[Ss]eparation") as rec:
        res = irls_logistic(X, y)
    assert sum("eparation" in str(r.message) for r in rec) == 1
    assert np.all(np.isfinite(res.beta))
    assert res.separation
    assert res.converged
    assert res.beta[1] > 0.0


def test_irls_offset_alone_is_not_separation() -> None:
    # A strongly discriminating offset with a mixed-outcome design: the MLE for
    # the intercept exists, so no separation may be declared (the margin rule
    # must measure the design's own contribution, eta - offset).
    n1, n0 = 900, 100
    off = np.concatenate([np.full(n1, 12.0), np.full(n0, -12.0)])
    y = np.concatenate([np.ones(n1), np.zeros(n0)])
    X = np.ones((n1 + n0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = irls_logistic(X, y, offset=off)
    assert not res.separation
    assert res.converged
    np.testing.assert_allclose(res.beta[0], np.log(3.0), atol=1e-3)


def test_irls_soft_targets_never_separate() -> None:
    # Lin-Lin-Weng smoothing of the perfectly-split design: interior targets
    # make the objective coercive, so separation is a category error.
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    X = np.column_stack([np.ones(4), x])
    targets = np.array([0.25, 0.25, 0.75, 0.75])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = irls_logistic(X, targets)
    assert not res.separation
    assert res.converged


def test_irls_monotone_descent() -> None:
    rng = np.random.default_rng(7)
    n = 20_000
    z = rng.normal(0.0, 8.0, n)
    y = (rng.random(n) < expit(1.5 * z + 0.5)).astype(float)
    n_pos, n_neg = float(y.sum()), float(n - y.sum())
    targets = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    X = np.column_stack([np.ones(n), logit(expit(z))])  # the design the Platt pipeline sees

    def platt_nll(beta: np.ndarray) -> float:
        eta = X @ beta
        softplus = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
        return float(np.sum(softplus - targets * eta))

    res = irls_logistic(X, targets)
    assert res.converged
    assert not res.separation
    assert res.nll <= platt_nll(np.zeros(2))
    # v0.1.1's false-separation abort returned (b, a) ~ (0.3518, 1.1825); the
    # monotone-descent fit must do at least as well on the objective.
    assert res.nll <= platt_nll(np.array([0.3518, 1.1825]))


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


def _loess_v012(
    x: object,
    y: object,
    frac: float = 0.75,
    degree: int = 1,
    xeval: object = None,
) -> np.ndarray:
    """Verbatim copy of the v0.1.2 ``loess`` body (argpartition search).

    Frozen as an equivalence reference for the sorted two-pointer rewrite.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    ev = x_arr if xeval is None else np.asarray(xeval, dtype=np.float64)
    n = x_arr.shape[0]
    r = max(int(math.ceil(frac * n)), degree + 1)
    r = min(r, n)
    out = np.empty(ev.shape[0], dtype=np.float64)
    for j, x0 in enumerate(ev):
        dist = np.abs(x_arr - x0)
        idx = np.argpartition(dist, r - 1)[:r]
        h = dist[idx].max()
        if h == 0.0:
            out[j] = y_arr[idx].mean()
            continue
        u = dist[idx] / h
        wts = np.clip(1.0 - u**3, 0.0, None) ** 3
        if degree == 0:
            out[j] = float(np.average(y_arr[idx], weights=np.maximum(wts, _FPMIN)))
            continue
        xc = x_arr[idx] - x0
        sw = wts.sum()
        swx = (wts * xc).sum()
        swxx = (wts * xc * xc).sum()
        swy = (wts * y_arr[idx]).sum()
        swxy = (wts * xc * y_arr[idx]).sum()
        det = sw * swxx - swx * swx
        if abs(det) < _FPMIN:
            out[j] = swy / sw
        else:
            out[j] = (swxx * swy - swx * swxy) / det
    return out


def test_loess_matches_v012_on_tie_free_data() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)  # continuous draws: tie-free a.s.
    y = np.sin(2.0 * x) + rng.normal(scale=0.1, size=400)
    for frac in (0.3, 0.75):
        np.testing.assert_allclose(loess(x, y, frac=frac), _loess_v012(x, y, frac=frac), atol=1e-9)
    xe = rng.uniform(x.min() - 0.5, x.max() + 0.5, 50)  # includes out-of-range evals
    np.testing.assert_allclose(
        loess(x, y, frac=0.5, xeval=xe), _loess_v012(x, y, frac=0.5, xeval=xe), atol=1e-9
    )


def test_loess_tie_window_uses_min_width_rule() -> None:
    # r=2, eval at the tied pair: min-width window is the zero-width [1.0, 1.0]
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert loess(x, y, frac=0.5, xeval=np.array([1.0]))[0] == 1.5  # h==0 fallback: mean


def test_loess_grid_matches_exact_within_tolerance() -> None:
    d = make_pd_portfolio(n=5000)
    yf = d.y.astype(np.float64)
    assert np.max(np.abs(loess(d.scores, yf) - loess(d.scores, yf, grid_size=512))) <= 5e-3


def test_loess_grid_size_none_and_oversized_are_exact() -> None:
    d = make_pd_portfolio(n=1000)
    yf = d.y.astype(np.float64)
    base = loess(d.scores, yf)
    assert np.array_equal(loess(d.scores, yf, grid_size=None), base)
    assert np.array_equal(loess(d.scores, yf, grid_size=5000), base)  # threshold not crossed


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


def test_weighted_quantile_unit_weights_matches_hazen() -> None:
    # Same Hazen positions as np.quantile(..., method="hazen"), computed via a
    # different arithmetic path (cumsum + division vs. numpy's virtual-index
    # formula), so equality holds to double precision rather than bit-for-bit.
    rng = np.random.default_rng(17)
    x = rng.normal(size=200)
    w = np.ones(200)
    q = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    np.testing.assert_allclose(
        weighted_quantile(x, q, w), np.quantile(x, q, method="hazen"), rtol=0.0, atol=1e-12
    )


def test_weighted_quantile_integer_weights_matches_repeat_within_one_gap() -> None:
    rng = np.random.default_rng(29)
    x = rng.normal(size=25)
    w = rng.integers(1, 6, size=25)
    q = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    expanded = np.repeat(x, w)
    expected = np.quantile(expanded, q, method="hazen")
    actual = weighted_quantile(x, q, w)
    tol = float(np.diff(np.sort(expanded)).max())
    np.testing.assert_allclose(actual, expected, atol=tol)


# ------------------------------------------- vectorized presorted anchor fit (0.3.0)


def _loess_anchor_case(
    p: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Sorted arrays, the window size ``loess`` would use, and the 512-anchor grid."""
    order = np.argsort(p, kind="stable")
    xs, ys = p[order], y[order]
    n = len(xs)
    r = min(max(int(np.ceil(0.75 * n)), 2), n)
    anchors = np.unique(np.quantile(xs, np.linspace(0.0, 1.0, 512)))
    return xs, ys, r, anchors


def _loess_fixtures() -> list[object]:
    out = []
    for n in (1000, 10_000):
        d = make_pd_portfolio(n=n, random_state=3)
        out.append(pytest.param(d.scores, d.y, id=f"portfolio-{n}"))
    d = make_pd_portfolio(n=2000, random_state=4)
    out.append(pytest.param(np.round(d.scores, 2), d.y, id="tied-scores"))
    return out


@pytest.mark.parametrize("p,y", _loess_fixtures())
def test_loess_window_starts_match_the_two_pointer_loop(p: np.ndarray, y: np.ndarray) -> None:
    """The vectorized window search must land on the loop's exact index.

    Not an approximation: the loop's comparison is reproduced verbatim, so a
    tied-score fixture (where the sum form and the difference form of the
    comparison round differently) has to agree too.
    """
    xs, _, r, anchors = _loess_anchor_case(p, y)
    n = len(xs)
    expected, i = [], 0
    for x0 in anchors:
        while i + r < n and xs[i + r] - x0 < x0 - xs[i]:
            i += 1
        expected.append(i)
    np.testing.assert_array_equal(_loess_window_starts(xs, anchors, r), np.array(expected))


@pytest.mark.parametrize("p,y", _loess_fixtures())
@pytest.mark.parametrize("degree", [0, 1])
def test_loess_vectorized_anchors_match_the_loop(p: np.ndarray, y: np.ndarray, degree: int) -> None:
    """Every anchor agrees with the scalar loop to within the tricube cube's ulp.

    ``_loess_fit_sorted_vec`` cubes by multiplication where the loop calls
    ``** 3``; that is the only intended difference, and it is a sub-ulp one.
    """
    xs, ys, r, anchors = _loess_anchor_case(p, y)
    np.testing.assert_allclose(
        _loess_fit_sorted_vec(xs, ys, anchors, r, degree),
        _loess_fit_sorted(xs, ys, anchors, r, degree),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize("p,y", _loess_fixtures())
def test_loess_presorted_matches_the_default_path(p: np.ndarray, y: np.ndarray) -> None:
    """End-to-end: ``presorted=True`` only changes throughput, not the fit."""
    order = np.argsort(p, kind="stable")
    xs, ys = p[order], y[order]
    np.testing.assert_allclose(
        loess(xs, ys, frac=0.75, grid_size=512, presorted=True),
        loess(xs, ys, frac=0.75, grid_size=512),
        rtol=1e-9,
        atol=1e-12,
    )


def test_loess_presorted_without_grid_still_uses_the_scalar_loop() -> None:
    """The per-observation path is O(n * r) to gather, so it must stay on the loop."""
    d = make_pd_portfolio(n=400, random_state=3)
    order = np.argsort(d.scores, kind="stable")
    xs, ys = d.scores[order], d.y[order]
    assert np.array_equal(loess(xs, ys, frac=0.75, presorted=True), loess(xs, ys, frac=0.75))


def _two_score_fixture() -> tuple[np.ndarray, np.ndarray, int]:
    """1025 rows carrying only two distinct scores, 837 low / 188 high."""
    rng = np.random.default_rng(0)
    n, n_low = 1025, 837
    xs = np.concatenate([np.full(n_low, 0.1), np.full(n - n_low, 0.9)])
    ys = (rng.random(n) < 0.3).astype(float)
    return xs, ys, min(max(int(np.ceil(0.75 * n)), 2), n)


def test_loess_vectorized_rank_deficient_window() -> None:
    """The sub-ulp agreement bound does not cover rank-deficient windows.

    With only two distinct scores, an eval point above their midpoint puts the
    far group at exactly the bandwidth ``h``, so its tricube weight is exactly
    zero and every surviving row shares one ``x``. The local *linear* system is
    then singular and ``det = sw * swxx - swx * swx`` is pure cancellation,
    computed as ~1e-23 rather than 0 — small enough that the ulp-level weight
    difference between ``u ** 3`` and ``u * u * u`` can land the loop and the
    vectorized routine on opposite sides of the ``abs(det) < _FPMIN`` guard.
    Here the vectorized path takes the ``swy / sw`` branch, which for a
    single-``x`` window is the plain mean of ``y`` over the surviving rows and
    the only well-defined answer; the loop divides by the cancellation noise and
    returns an arbitrary value. Which path lands where is a property of the
    rounding, not of the algorithm — the loop's own value on such a window is
    already arbitrary, so this is a characterization test of a known corner, not
    a correctness bound. The guard is deliberately not changed: that would move
    the point-estimate path.
    """
    xs, ys, r = _two_score_fixture()
    x0 = 0.50148
    start = _loess_window_starts(xs, np.array([x0]), r)[0]
    xw, yw = xs[start : start + r], ys[start : start + r]
    h = max(x0 - xs[start], xs[start + r - 1] - x0)
    tri = np.clip(1.0 - (np.abs(xw - x0) / h) ** 3, 0.0, None) ** 3
    surviving = tri > 0
    assert len(np.unique(xw[surviving])) == 1  # rank-deficient by construction

    with np.errstate(invalid="ignore", divide="ignore"):
        vec = _loess_fit_sorted_vec(xs, ys, np.array([x0]), r, 1)[0]
        scalar = _loess_fit_sorted(xs, ys, np.array([x0]), r, 1)[0]
    assert vec == float(np.mean(yw[surviving]))  # the swy/sw branch
    assert abs(scalar - vec) > 0.1  # the loop's cancellation-noise branch


def test_loess_rank_deficient_windows_do_not_reach_reported_values() -> None:
    """Anchors are data quantiles, so the corner above stays off the fit.

    ``loess`` evaluates at ``np.unique(np.quantile(ev, ...))``, which for this
    fixture is just the two data values — neither of which sits above the
    midpoint with the far group excluded. The end-to-end presorted result is
    therefore exactly the default one, which is the guarantee that matters.
    """
    xs, ys, _ = _two_score_fixture()
    fast = loess(xs, ys, frac=0.75, grid_size=512, presorted=True)
    slow = loess(xs, ys, frac=0.75, grid_size=512)
    np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-12)
    assert np.max(np.abs(fast - slow)) <= 1e-12
