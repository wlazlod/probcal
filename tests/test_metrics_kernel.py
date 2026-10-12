"""Tests for probcal.metrics.kernel."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.metrics.kernel import SkceTestResult, skce, skce_test

RNG = np.random.default_rng(79)


def _calibrated(n: int = 300, seed: int = 79) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(-0.8, 1.2, n))
    y = (rng.random(n) < p).astype(float)
    return y, p


def _h_matrix(
    y: np.ndarray, p: np.ndarray, kernel: str, bandwidth: float, scale: str
) -> np.ndarray:
    """Independent double-loop reference for the h-matrix."""
    s = logit(p) if scale == "logit" else p
    r = y - p
    n = len(p)
    h = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(s[i] - s[j])
            if kernel == "laplace":
                k = np.exp(-d / bandwidth)
            else:
                k = np.exp(-(d**2) / (2 * bandwidth**2))
            h[i, j] = 2.0 * k * r[i] * r[j]
    return h


def _centered_h(h: np.ndarray) -> np.ndarray:
    c = h.mean(axis=1)
    return h - c[:, None] - c[None, :] + h.mean()


def test_skce_three_point_anchor() -> None:
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.8, 0.4, 0.6])
    # Residuals (0.2, -0.4, 0.4); |dists| d12=0.4, d13=0.2, d23=0.2.
    h12 = 2.0 * np.exp(-0.4) * 0.2 * (-0.4)
    h13 = 2.0 * np.exp(-0.2) * 0.2 * 0.4
    h23 = 2.0 * np.exp(-0.2) * (-0.4) * 0.4
    h11 = 2.0 * 0.2**2
    h22 = 2.0 * 0.4**2
    h33 = 2.0 * 0.4**2
    uq_expected = (h12 + h13 + h23) / 3.0  # 2*(h12+h13+h23)/(3*2)
    biased_expected = (h11 + h22 + h33 + 2.0 * (h12 + h13 + h23)) / 9.0
    uq = skce(y, p, estimator="uq", kernel="laplace", bandwidth=1.0)
    biased = skce(y, p, estimator="biased", kernel="laplace", bandwidth=1.0)
    assert abs(uq - uq_expected) < 1e-12
    assert abs(biased - biased_expected) < 1e-12
    # Anchors computed to 10 significant digits, verified independently.
    assert abs(uq - (-0.0794160426)) < 1e-9
    assert abs(biased - 0.0270559716) < 1e-9


@pytest.mark.parametrize(
    "kernel,scale", [("laplace", "probability"), ("gaussian", "logit")], ids=["laplace", "gaussian"]
)
def test_skce_matches_brute_force(kernel: str, scale: str) -> None:
    y, p = _calibrated(40)
    h = _h_matrix(y, p, kernel, 0.5, scale)
    n = len(p)
    uq_ref = (h.sum() - np.trace(h)) / (n * (n - 1))
    biased_ref = h.sum() / n**2
    uq = skce(y, p, estimator="uq", kernel=kernel, bandwidth=0.5, scale=scale)
    biased = skce(y, p, estimator="biased", kernel=kernel, bandwidth=0.5, scale=scale)
    assert abs(uq - uq_ref) < 1e-12 + abs(uq_ref) * 1e-12
    assert abs(biased - biased_ref) < 1e-12 + abs(biased_ref) * 1e-12


@pytest.mark.parametrize("n", [40, 41], ids=["even", "odd"])
def test_skce_ul_matches_seeded_pairing(n: int) -> None:
    y, p = _calibrated(n)
    h = _h_matrix(y, p, "laplace", 0.5, "probability")
    # Reconstruct the documented pairing: permute with default_rng(seed), pair 0::2 with 1::2.
    perm = np.random.default_rng(7).permutation(n)
    m = n // 2
    ref = float(np.mean([h[perm[2 * i], perm[2 * i + 1]] for i in range(m)]))
    val = skce(y, p, estimator="ul", kernel="laplace", bandwidth=0.5, random_state=7)
    assert abs(val - ref) < 1e-12


@pytest.mark.slow
def test_level_both_methods() -> None:
    # Empirical rejection rate at alpha=0.05 over 200 calibrated replications.
    # Reference run measured 0.045 (bootstrap) / 0.040 (asymptotic).
    rej_boot = rej_asym = 0
    for k in range(200):
        y, p = _calibrated(150, seed=1000 + k)
        if skce_test(y, p, method="bootstrap", n_boot=99).p_value < 0.05:
            rej_boot += 1
        if skce_test(y, p, method="asymptotic").p_value < 0.05:
            rej_asym += 1
    assert 0.005 <= rej_boot / 200 <= 0.12
    assert 0.005 <= rej_asym / 200 <= 0.12


def test_skce_test_determinism_and_fields() -> None:
    y, p = _calibrated(60)
    a = skce_test(y, p, n_boot=49)
    b = skce_test(y, p, n_boot=49)
    assert a == b  # frozen dataclass equality: fully deterministic
    assert a.method == "bootstrap" and a.estimator == "uq" and a.n_boot == 49
    assert 1.0 / 50.0 <= a.p_value <= 1.0  # add-one estimate range
    assert abs(a.statistic - skce(y, p, estimator="uq")) < 1e-12
    c = skce_test(y, p, method="asymptotic")
    assert c.method == "asymptotic" and c.estimator == "ul" and c.n_boot is None
    assert abs(c.statistic - skce(y, p, estimator="ul")) < 1e-12


def test_p_value_bound_formula() -> None:
    y, p = _calibrated(60)
    res = skce_test(y, p, method="bootstrap", n_boot=49)
    t = res.statistic
    if t > 0:
        expected = min(1.0, np.exp(-(60 // 2) * t**2 / 8.0))
        assert abs(res.p_value_bound - expected) < abs(expected) * 1e-12
    else:
        assert res.p_value_bound == 1.0
    # A negative statistic must give the trivial bound 1.0; find one via seeds.
    for k in range(50):
        yk, pk = _calibrated(30, seed=5000 + k)
        r = skce_test(yk, pk, n_boot=9)
        if r.statistic <= 0:
            assert r.p_value_bound == 1.0
            break
    else:  # pragma: no cover - calibrated data yields negative uq ~half the time
        raise AssertionError("no negative statistic found in 50 calibrated draws")


_Y4, _P4 = np.array([0.0, 1.0, 0.0, 1.0]), np.full(4, 0.5)


_BAD_ARGS = [
    (lambda y, p: skce(y, p, estimator="linear"), "estimator"),
    (lambda y, p: skce(y, p, kernel="rbf"), "kernel"),
    (lambda y, p: skce(y, p, scale="log"), "scale"),
    (lambda y, p: skce(y, p, bandwidth=0.0), "bandwidth"),
    (lambda y, p: skce(y, p, bandwidth=float("nan")), "bandwidth"),
    (lambda y, p: skce(np.array([2.0, 0.0]), np.array([0.5, 0.5])), None),
    (lambda y, p: skce(np.array([0.0, 1.0]), np.array([0.5, 0.5, 0.5])), "length"),
    (lambda y, p: skce(_Y4, _P4), "identical"),
    (lambda y, p: skce_test(y, p, method="wild"), "method"),
    (lambda y, p: skce_test(y, p, n_boot=0), "n_boot"),
    (lambda y, p: skce_test(np.array([0.0, 1.0, 1.0]), np.array([0.2, 0.5, 0.7])), "4"),
    (lambda y, p: skce_test(_Y4, _P4), "identical"),
]


def test_bad_arguments_raise() -> None:
    y, p = _calibrated(10)
    for call, match in _BAD_ARGS:
        with pytest.raises(ValueError, match=match):
            call(y, p)


def test_identical_scores_work_with_explicit_bandwidth() -> None:
    assert np.isfinite(skce(_Y4, _P4, bandwidth=0.1))
    assert isinstance(skce_test(_Y4, _P4, bandwidth=0.1), SkceTestResult)


# ------------------------------------------------------------------ 0.3.4 fixes


def test_bootstrap_replicates_on_the_statistic_scale() -> None:
    # Regression (MET-8): t_obs = n * U_n = S / (n - 1) but replicates were
    # S_b / n, shrinking every replicate by (n - 1) / n.
    y, p = _calibrated(40)
    n = len(p)
    res = skce_test(y, p, method="bootstrap", n_boot=199, bandwidth=0.5, random_state=3)
    h = _h_matrix(y, p, "laplace", 0.5, "probability")
    ht = _centered_h(h)
    t_obs = (h.sum() - np.trace(h)) / (n - 1)
    rng = np.random.default_rng(3)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=199).astype(float)
    t_b = (np.einsum("bi,ij,bj->b", counts, ht, counts) - counts @ np.diag(ht)) / (n - 1)
    assert res.p_value == (1 + int(np.sum(t_b >= t_obs))) / 200
    assert res.statistic * n == pytest.approx(t_obs)
