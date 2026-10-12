"""Unit tests for probcal._math (numpy-only; reference comparisons live in
test_math_reference.py)."""

import math
import warnings

import numpy as np
import pytest

from probcal._math import (
    _FPMIN,
    _loess_fit_sorted_vec,
    beta_ppf,
    bisect,
    expit,
    irls_logistic,
    loess,
    logit,
    natural_cubic_basis,
    norm_cdf,
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


# ---------------------------------------------------------------- solvers


def test_bisect_finds_root() -> None:
    root = bisect(lambda x: x**2 - 2.0, 0.0, 2.0)
    assert abs(root - np.sqrt(2.0)) < 1e-10


# ---------------------------------------------------------------- special functions


def test_beta_ppf_validation() -> None:
    with pytest.raises(ValueError, match="beta_ppf"):
        beta_ppf(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="beta_ppf"):
        beta_ppf(1.0, 1.0, 1.0)


def test_norm_cdf_symmetry() -> None:
    x = np.linspace(-8.0, 8.0, 81)
    np.testing.assert_allclose(norm_cdf(x) + norm_cdf(-x), np.ones_like(x), atol=1e-14)


# ---------------------------------------------------------------- PAVA


def test_pava_worked_example() -> None:
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    w = np.ones(5)
    res = pava(y, w)
    np.testing.assert_allclose(res.fitted, [0.0, 1 / 3, 1 / 3, 1 / 3, 1.0])
    assert res.block_start.tolist() == [0, 1, 4]
    np.testing.assert_allclose(res.block_mean, [0.0, 1 / 3, 1.0])
    np.testing.assert_allclose(res.block_weight, [1.0, 3.0, 1.0])


# ---------------------------------------------------------------- IRLS logistic


def _make_logistic_data(n: int = 5000, beta=(-2.0, 1.5)) -> tuple[np.ndarray, np.ndarray]:
    x = RNG.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    p = expit(X @ np.asarray(beta))
    y = (RNG.random(n) < p).astype(float)
    return X, y


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


# ---------------------------------------------------------------- LOESS


def test_loess_reproduces_linear_trend() -> None:
    x = np.linspace(0.0, 1.0, 60)
    y = 2.0 * x + 1.0
    fitted = loess(x, y, frac=0.5)
    np.testing.assert_allclose(fitted, y, atol=1e-10)
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


def test_natural_cubic_basis_linear_beyond_boundary() -> None:
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x = np.array([1.5, 2.0, 2.5, 3.0])  # all beyond the last knot
    B = natural_cubic_basis(x, knots)
    # Linear in x out there: second differences vanish column by column.
    second_diff = np.diff(B, n=2, axis=0)
    np.testing.assert_allclose(second_diff, 0.0, atol=1e-9)


def test_weighted_quantile_equal_weights_is_numpy_default() -> None:
    # Equal weights return np.quantile's default (linear) rule bit for bit,
    # and nearly equal weights stay within rounding of it (no rule switch).
    rng = np.random.default_rng(17)
    x = rng.normal(size=200)
    q = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    for c in (1.0, 3.5):
        np.testing.assert_array_equal(weighted_quantile(x, q, np.full(200, c)), np.quantile(x, q))
    w = np.ones(200)
    w[0] = 1.0 + 1e-12
    np.testing.assert_allclose(weighted_quantile(x, q, w), np.quantile(x, q), rtol=0, atol=1e-10)


def test_weighted_quantile_integer_weights_matches_repeat_within_one_gap() -> None:
    rng = np.random.default_rng(29)
    x = rng.normal(size=25)
    w = rng.integers(1, 6, size=25)
    q = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    expanded = np.repeat(x, w)
    expected = np.quantile(expanded, q)
    actual = weighted_quantile(x, q, w)
    tol = float(np.diff(np.sort(expanded)).max())
    np.testing.assert_allclose(actual, expected, atol=tol)


# ------------------------------------------- vectorized presorted anchor fit (0.3.0)


def _loess_fit_sorted(
    xs: np.ndarray, ys: np.ndarray, evs: np.ndarray, r: int, degree: int
) -> np.ndarray:
    """Retired (0.3.x) per-point LOESS loop, frozen as the engine's reference.

    LOESS at sorted eval points via the contiguous min-width r-window.

    For 1-D x the r-nearest-neighbor window is contiguous in sorted order; the
    two-pointer rule advances the window start while the point entering on the
    right is strictly closer than the one leaving on the left. Distance ties
    resolve to the leftmost minimal-width window (strict `<`).

    ``xs`` and ``evs`` must already be sorted ascending.
    """
    n = xs.shape[0]
    out = np.empty(evs.shape[0], dtype=np.float64)
    i = 0
    for j in range(evs.shape[0]):
        x0 = evs[j]
        while i + r < n and xs[i + r] - x0 < x0 - xs[i]:
            i += 1
        xw = xs[i : i + r]
        yw = ys[i : i + r]
        h = max(x0 - xs[i], xs[i + r - 1] - x0)
        if h == 0.0:
            out[j] = yw.mean()
            continue
        u = np.abs(xw - x0) / h
        wts = np.clip(1.0 - u**3, 0.0, None) ** 3
        if degree == 0:
            out[j] = float(np.average(yw, weights=np.maximum(wts, _FPMIN)))
            continue
        xc = xw - x0
        sw = wts.sum()
        swx = (wts * xc).sum()
        swxx = (wts * xc * xc).sum()
        swy = (wts * yw).sum()
        swxy = (wts * xc * yw).sum()
        det = sw * swxx - swx * swx
        out[j] = swy / sw if abs(det) < _FPMIN else (swxx * swy - swx * swxy) / det
    return out


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
    d = make_pd_portfolio(n=1000, random_state=3)
    out.append(pytest.param(d.scores, d.y, id="portfolio-1000"))
    d = make_pd_portfolio(n=2000, random_state=4)
    out.append(pytest.param(np.round(d.scores, 2), d.y, id="tied-scores"))
    return out


def _two_score_fixture() -> object:
    """1025 rows carrying only two distinct scores, 837 low / 188 high.

    An anchor above their midpoint would give a rank-deficient local-linear
    window (``det`` is pure cancellation), where the loop and the vectorized
    routine may take different branches; anchors are data quantiles, so the
    corner never reaches a reported value and presorted == default still holds.
    """
    rng = np.random.default_rng(0)
    xs = np.concatenate([np.full(837, 0.1), np.full(188, 0.9)])
    ys = (rng.random(1025) < 0.3).astype(float)
    return pytest.param(xs, ys, id="two-scores")


def test_loess_per_observation_matches_the_retired_loop() -> None:
    """0.3.4 routes every evaluation through the blocked engine; on data points
    it agrees with the retired loop to a few ulps."""
    for n, rnd in ((300, None), (2000, 2)):
        d = make_pd_portfolio(n=n, random_state=5)
        p = d.scores if rnd is None else np.round(d.scores, rnd)
        order = np.argsort(p, kind="stable")
        xs, ys = p[order], d.y[order].astype(float)
        for frac in (0.2, 0.75):
            r = min(max(int(np.ceil(frac * n)), 2), n)
            for degree in (0, 1):
                np.testing.assert_allclose(
                    loess(xs, ys, frac=frac, degree=degree, presorted=True),
                    _loess_fit_sorted(xs, ys, xs, r, degree),
                    rtol=0,
                    atol=1e-13,
                )


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


@pytest.mark.parametrize("p,y", [*_loess_fixtures(), _two_score_fixture()])
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


def test_loess_presorted_without_grid_matches_default() -> None:
    """``presorted`` is a throughput switch on the per-observation path too."""
    d = make_pd_portfolio(n=400, random_state=3)
    order = np.argsort(d.scores, kind="stable")
    xs, ys = d.scores[order], d.y[order]
    assert np.array_equal(loess(xs, ys, frac=0.75, presorted=True), loess(xs, ys, frac=0.75))


# ------------------------------------------------- IRLS convergence criterion (0.3.4)


def test_irls_unconverged_iteration_cap_is_reported() -> None:
    """Convergence is judged on the full Newton step: two iterations from zero
    are not enough on a well-posed problem, and soft targets cannot hide it."""
    X, y = _make_logistic_data()
    n_pos = float(y.sum())
    soft = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (len(y) - n_pos + 2.0))
    assert not irls_logistic(X, soft, max_iter=2).converged
    assert irls_logistic(X, soft).converged


def test_irls_quasi_separation_stall_is_not_convergence() -> None:
    """CAL-12: on a quasi-separated design the line search eventually stalls
    with a vanishing gradient; 0.3.x called that ``converged`` and returned a
    runaway slope without a warning."""
    x = np.concatenate([np.linspace(-4.0, -0.2, 50), [0.0, 0.0], np.linspace(0.2, 4.0, 50)])
    y = np.concatenate([np.zeros(50), [0.0, 1.0], np.ones(50)])
    X = np.column_stack([np.ones(len(x)), x])
    with pytest.warns(UserWarning, match="separation"):
        res = irls_logistic(X, y, max_iter=1000)
    assert res.separation and res.converged
    assert res.beta[1] < 100.0  # the ridge fit, not the runaway iterate


def test_irls_penalty_matrix_generalizes_ridge() -> None:
    X, y = _make_logistic_data()
    a = irls_logistic(X, y, ridge=3.0)
    b = irls_logistic(X, y, penalty=3.0 * np.eye(2))
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-12)
    assert a.converged and b.converged


def test_irls_penalty_null_space_still_detects_separation() -> None:
    """A penalty that leaves the slope free cannot stop a separating slope."""
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    X = np.column_stack([np.ones(4), x])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    with pytest.warns(UserWarning, match="separation"):
        res = irls_logistic(X, y, penalty=np.diag([1.0, 0.0]))
    assert res.separation and res.converged
