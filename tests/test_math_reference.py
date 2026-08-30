"""Reference tests: probcal._math vs scipy / scikit-learn / statsmodels.

These libraries are dev/test-only dependencies (spec §1.2); every test here is
marked `reference` and skipped when the library is absent.
"""

import numpy as np
import pytest

from probcal._math import (
    beta_ppf,
    betainc,
    chi2_ppf,
    erf_vec,
    expit,
    gammainc_lower,
    irls_logistic,
    lgamma_vec,
    loess,
    logit,
    norm_ppf,
    pava,
)

pytestmark = pytest.mark.reference

RNG = np.random.default_rng(2026)


def test_betainc_vs_scipy() -> None:
    sp = pytest.importorskip("scipy.special")
    x = np.linspace(1e-6, 1.0 - 1e-6, 2001)
    for a in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
        for b in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
            err = np.max(np.abs(betainc(a, b, x) - sp.betainc(a, b, x)))
            assert err < 1e-12, f"betainc(a={a}, b={b}): max abs err {err:.2e}"


def test_gammainc_lower_vs_scipy() -> None:
    sp = pytest.importorskip("scipy.special")
    x = np.linspace(1e-8, 200.0, 2001)
    for s in (0.5, 1.0, 2.5, 10.0, 50.0, 150.0):
        err = np.max(np.abs(gammainc_lower(s, x) - sp.gammainc(s, x)))
        assert err < 1e-12, f"gammainc_lower(s={s}): max abs err {err:.2e}"


def test_beta_ppf_vs_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for a in (0.5, 1.0, 2.0, 5.0, 20.0):
        for b in (0.5, 1.0, 2.0, 5.0, 20.0):
            for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                ours, ref = beta_ppf(q, a, b), float(stats.beta.ppf(q, a, b))
                assert abs(ours - ref) < 1e-8 + abs(ref) * 1e-8, f"a={a}, b={b}, q={q}"


def test_chi2_ppf_vs_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    q = np.array([0.001, 0.05, 0.5, 0.8, 0.95, 0.999])
    for df in (1.0, 2.0, 4.0, 10.0, 100.0):
        ours = np.array([chi2_ppf(qi, df) for qi in q])
        np.testing.assert_allclose(ours, stats.chi2.ppf(q, df), atol=1e-8, rtol=1e-10)


def test_norm_ppf_vs_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    q = np.concatenate(
        [np.array([1e-12, 1e-9, 1e-4]), np.linspace(0.01, 0.99, 99), np.array([1 - 1e-9])]
    )
    np.testing.assert_allclose(norm_ppf(q), stats.norm.ppf(q), atol=1e-11)


def test_lgamma_erf_vs_scipy() -> None:
    sp = pytest.importorskip("scipy.special")
    x = np.linspace(0.05, 30.0, 500)
    np.testing.assert_allclose(lgamma_vec(x), sp.gammaln(x), rtol=1e-13)
    z = np.linspace(-6.0, 6.0, 500)
    np.testing.assert_allclose(erf_vec(z), sp.erf(z), rtol=1e-13, atol=1e-15)


def test_pava_vs_sklearn() -> None:
    iso_mod = pytest.importorskip("sklearn.isotonic")
    y = RNG.random(400)
    w = RNG.uniform(0.5, 2.0, 400)
    ours = pava(y, w).fitted
    theirs = iso_mod.IsotonicRegression().fit_transform(np.arange(400), y, sample_weight=w)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_irls_vs_statsmodels() -> None:
    sm = pytest.importorskip("statsmodels.api")
    n = 2000
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    p = expit(-1.0 + 0.8 * x1 - 0.5 * x2)
    y = (RNG.random(n) < p).astype(float)
    ours = irls_logistic(X, y).beta
    theirs = sm.GLM(y, X, family=sm.families.Binomial()).fit().params
    np.testing.assert_allclose(ours, theirs, rtol=1e-8)


def test_irls_offset_vs_statsmodels() -> None:
    sm = pytest.importorskip("statsmodels.api")
    n = 2000
    x = RNG.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    off = RNG.normal(scale=0.3, size=n)
    p = expit(0.5 + 1.2 * x + off)
    y = (RNG.random(n) < p).astype(float)
    ours = irls_logistic(X, y, offset=off).beta
    theirs = sm.GLM(y, X, family=sm.families.Binomial(), offset=off).fit().params
    np.testing.assert_allclose(ours, theirs, rtol=1e-8)


def test_irls_soft_targets_vs_statsmodels() -> None:
    # GLM(Binomial) accepts interior-valued targets: the smoothed Platt targets
    # on wide-score data must give matching coefficients.
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(11)  # local rng: keep the module RNG draw order intact
    n = 20_000
    z = rng.normal(0.0, 8.0, n)
    y = (rng.random(n) < expit(1.5 * z + 0.5)).astype(float)
    n_pos, n_neg = float(y.sum()), float(n - y.sum())
    targets = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    X = np.column_stack([np.ones(n), logit(expit(z))])
    ours = irls_logistic(X, targets).beta
    theirs = sm.GLM(targets, X, family=sm.families.Binomial()).fit().params
    np.testing.assert_allclose(ours, theirs, rtol=1e-6)


def test_loess_vs_statsmodels_lowess() -> None:
    # Loose comparison, documented: statsmodels lowess uses a different local scheme
    # (robustifying iterations off, but delta handling and window details differ).
    lowess_mod = pytest.importorskip("statsmodels.nonparametric.smoothers_lowess")
    x = np.sort(RNG.uniform(0.0, 1.0, 300))
    y = np.sin(2.0 * np.pi * x) + RNG.normal(scale=0.15, size=300)
    ours = loess(x, y, frac=0.4)
    theirs = lowess_mod.lowess(y, x, frac=0.4, it=0, return_sorted=False)
    # Agreement within 5% of the response range.
    scale = np.ptp(theirs)
    assert np.max(np.abs(ours - theirs)) < 0.05 * scale
