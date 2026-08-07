"""Kernel calibration error (SKCE) and its calibration tests.

Theory: ``docs/concepts/metrics.md``. Widmann, Lindsten & Zachariah (2019),
"Calibration tests in multi-class classification: A unifying framework",
NeurIPS 32 (arXiv:1910.11385).

Binary specialization: with the identity-matrix kernel construction and a
prediction represented as the 2-vector ``(1 - p, p)``, the paper's kernel term
reduces to ``h_ij = 2 * k(s_i, s_j) * (y_i - p_i) * (y_j - p_j)`` — the factor
2 keeps values comparable with the paper's framework. Residuals always stay on
the probability scale; only the kernel input ``s`` may be logit-transformed.

No ``sample_weight``: the cited U-statistic theory (unbiasedness, the
degenerate limit, the distribution-free bounds) is stated for unweighted
i.i.d. samples. Refusing the argument is honest; improvising weighted
inference is not.

Complexity: ``"uq"``, ``"biased"``, and the bootstrap test are O(n^2) memory
and O(n_boot * n^2) time; prefer ``method="asymptotic"`` for n >~ 20 000.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._math import logit, norm_cdf
from .scores import _prep

_MAX_MEDIAN_POINTS = 4096


def _kernel_input(p: np.ndarray, scale: str) -> np.ndarray:
    if scale == "probability":
        return p
    if scale == "logit":
        return logit(p)
    raise ValueError(f"scale must be 'probability' or 'logit', got {scale!r}")


def _kernel_values(d: np.ndarray, kernel: str, bw: float) -> np.ndarray:
    if kernel == "laplace":
        return np.exp(-d / bw)
    if kernel == "gaussian":
        return np.exp(-(d**2) / (2.0 * bw**2))
    raise ValueError(f"kernel must be 'laplace' or 'gaussian', got {kernel!r}")


def _resolve_bandwidth(s: np.ndarray, bandwidth: float | None) -> float:
    if bandwidth is not None:
        bw = float(bandwidth)
        if not math.isfinite(bw) or bw <= 0.0:
            raise ValueError(f"bandwidth must be positive and finite, got {bandwidth!r}")
        return bw
    n = len(s)
    if n > _MAX_MEDIAN_POINTS:
        # Evenly strided, RNG-free subsample: bit-reproducible, and keeps the
        # O(n) "ul" path off an O(n^2) distance matrix.
        s = s[:: math.ceil(n / _MAX_MEDIAN_POINTS)]
    d = np.abs(s[:, None] - s[None, :])[np.triu_indices(len(s), k=1)]
    med = float(np.median(d))
    if med > 0.0:
        return med
    mean = float(np.mean(d))
    if mean > 0.0:
        return mean
    raise ValueError("all scores are identical; pass bandwidth explicitly")


def _h_full(y: np.ndarray, p: np.ndarray, s: np.ndarray, kernel: str, bw: float) -> np.ndarray:
    r = y - p
    d = np.abs(s[:, None] - s[None, :])
    return 2.0 * _kernel_values(d, kernel, bw) * np.outer(r, r)


def _ul_terms(
    y: np.ndarray, p: np.ndarray, s: np.ndarray, kernel: str, bw: float, random_state: int
) -> np.ndarray:
    # Seeded permutation first: a score-sorted input would silently break the
    # independence of pair terms that the linear test's CLT requires.
    perm = np.random.default_rng(random_state).permutation(len(p))
    m = len(p) // 2
    a, b = perm[0::2][:m], perm[1::2][:m]
    r = y - p
    return 2.0 * _kernel_values(np.abs(s[a] - s[b]), kernel, bw) * r[a] * r[b]


def skce(
    y: object,
    p: object,
    *,
    estimator: str = "uq",
    kernel: str = "laplace",
    bandwidth: float | None = None,
    scale: str = "probability",
    random_state: int = 42,
) -> float:
    """Squared kernel calibration error (Widmann et al., 2019, Table 1).

    ``"uq"`` (default) is the unbiased quadratic estimator (may be negative);
    ``"ul"`` the unbiased linear O(n) estimator over seeded disjoint pairs
    (``random_state`` controls the pairing); ``"biased"`` the nonnegative
    V-statistic. ``bandwidth=None`` uses the deterministic median heuristic;
    ``scale="logit"`` transforms the kernel input only (the low-PD option).
    """
    if estimator not in ("uq", "ul", "biased"):
        raise ValueError(f"estimator must be 'uq', 'ul', or 'biased', got {estimator!r}")
    y_arr, p_arr, _ = _prep(y, p, None)
    n = len(p_arr)
    if n < 2:
        raise ValueError(f"skce needs at least 2 observations, got {n}")
    s = _kernel_input(p_arr, scale)
    bw = _resolve_bandwidth(s, bandwidth)
    if estimator == "ul":
        return float(np.mean(_ul_terms(y_arr, p_arr, s, kernel, bw, random_state)))
    h = _h_full(y_arr, p_arr, s, kernel, bw)
    if estimator == "biased":
        return float(h.sum() / n**2)
    return float((h.sum() - np.trace(h)) / (n * (n - 1)))


@dataclass(frozen=True)
class SkceTestResult:
    """One-sided SKCE calibration test (H0: calibrated; large positive rejects)."""

    statistic: float
    estimator: str
    method: str
    p_value: float
    p_value_bound: float
    bandwidth: float
    n_boot: int | None


def _p_value_bound(stat: float, n: int) -> float:
    # Theorems H.3/H.4 with B = 2 (scalar kernel bounded by 1, identity
    # construction): min(1, exp(-floor(n/2) * t^2 / 8)). Valid without
    # asymptotics but loose — decide with p_value, report this as worst case.
    if stat <= 0.0:
        return 1.0
    return float(min(1.0, math.exp(-(n // 2) * stat**2 / 8.0)))


def skce_test(
    y: object,
    p: object,
    *,
    method: str = "bootstrap",
    n_boot: int = 999,
    kernel: str = "laplace",
    bandwidth: float | None = None,
    scale: str = "probability",
    random_state: int = 42,
) -> SkceTestResult:
    """Calibration test on the SKCE (Widmann et al., 2019, Sec. 6 / App. G).

    ``"bootstrap"`` (default): quadratic statistic with Arcones–Giné centered
    resampling; O(n_boot * n^2) — the more powerful choice. ``"asymptotic"``:
    linear statistic, normal approximation (Corollary G.3); O(n), preferred
    for n >~ 20 000, but a single random pairing can miss slope-type
    miscalibration that the bootstrap test rejects (the paper's documented
    power gap). ``p_value_bound`` is the distribution-free worst case.
    """
    if method not in ("bootstrap", "asymptotic"):
        raise ValueError(f"method must be 'bootstrap' or 'asymptotic', got {method!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    y_arr, p_arr, _ = _prep(y, p, None)
    n = len(p_arr)
    if n < 4:
        raise ValueError(f"skce_test needs at least 4 observations, got {n}")
    s = _kernel_input(p_arr, scale)
    bw = _resolve_bandwidth(s, bandwidth)

    if method == "asymptotic":
        terms = _ul_terms(y_arr, p_arr, s, kernel, bw, random_state)
        stat = float(np.mean(terms))
        sd = float(np.std(terms, ddof=1))
        if sd == 0.0:
            p_value = 1.0 if stat <= 0.0 else 0.0
        else:
            z = math.sqrt(len(terms)) * stat / sd
            p_value = float(1.0 - norm_cdf(np.array([z]))[0])
        return SkceTestResult(
            statistic=stat,
            estimator="ul",
            method="asymptotic",
            p_value=p_value,
            p_value_bound=_p_value_bound(stat, n),
            bandwidth=bw,
            n_boot=None,
        )

    h = _h_full(y_arr, p_arr, s, kernel, bw)
    stat = float((h.sum() - np.trace(h)) / (n * (n - 1)))
    t_obs = n * stat
    c = h.mean(axis=1)
    h_tilde = h - c[:, None] - c[None, :] + h.mean()
    rng = np.random.default_rng(random_state)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    quad = np.einsum("bi,ij,bj->b", counts, h_tilde, counts)
    t_b = (quad - counts @ np.diag(h_tilde)) / n
    p_value = float((1 + int(np.sum(t_b >= t_obs))) / (n_boot + 1))
    return SkceTestResult(
        statistic=stat,
        estimator="uq",
        method="bootstrap",
        p_value=p_value,
        p_value_bound=_p_value_bound(stat, n),
        bandwidth=bw,
        n_boot=n_boot,
    )
