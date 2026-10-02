"""Numerical core: PAVA, IRLS logistic regression, special functions, LOESS, spline basis.

Pure numpy + stdlib. Special functions are hand-rolled (continued fractions, series,
``math.erfc``/``math.lgamma``) and verified against scipy in
``tests/test_math_reference.py``.
"""

import math
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._validation import EPS

_FPMIN = 1e-300
_CF_EPS = 1e-16
_CF_MAX_ITER = 500

# ------------------------------------------------------------------ logit / expit


def logit(p: object) -> np.ndarray:
    """Log-odds of ``p``, clipped to keep the output finite.

    Parameters
    ----------
    p : array_like
        Probabilities; values outside ``[1e-12, 1 - 1e-12]`` are clipped.

    Returns
    -------
    numpy.ndarray
        ``log(p / (1 - p))`` elementwise.
    """
    arr = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(arr) - np.log1p(-arr)


_LOGIT_CLIP = float(logit(1.0 - EPS))
"""``logit`` of the upper clipping bound, ~27.631 (== ``-logit(1e-12)``).

A probability-space result whose raw logit exceeds this magnitude rounds into
the package-wide ``[1e-12, 1 - 1e-12]`` clipping zone and cannot round-trip
through any forward entry point; inverse maps refuse rather than return it.
"""


def expit(z: object) -> np.ndarray:
    """Logistic sigmoid ``1 / (1 + exp(-z))``, overflow-safe.

    Parameters
    ----------
    z : array_like
        Logits; any real values, including large magnitudes.

    Returns
    -------
    numpy.ndarray
        Probabilities in ``[0, 1]``.
    """
    arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ez = np.exp(arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit1(p: float) -> float:
    """Scalar :func:`logit` (same clipping), returning a Python float."""
    q = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(q) - math.log1p(-q)


def expit1(z: float) -> float:
    """Scalar :func:`expit`, overflow-safe, returning a Python float."""
    z = float(z)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def bern_log_lr(y: np.ndarray, p_null: np.ndarray, q_alt: np.ndarray, w: np.ndarray) -> float:
    """Weighted log Bernoulli likelihood ratio of ``q_alt`` against ``p_null``."""
    q = np.clip(q_alt, EPS, 1.0 - EPS)
    p = np.clip(p_null, EPS, 1.0 - EPS)
    terms = y * (np.log(q) - np.log(p)) + (1.0 - y) * (np.log1p(-q) - np.log1p(-p))
    return float(np.sum(w * terms))


def logsumexp(values: np.ndarray) -> float:
    """Overflow-safe ``log(sum(exp(values)))``."""
    m = float(np.max(values))
    return m + float(np.log(np.sum(np.exp(values - m))))


# ------------------------------------------------------------------ 1-D solvers


def bisect(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Root of ``f`` on ``[lo, hi]`` by bisection.

    Parameters
    ----------
    f : callable
        Continuous function with ``f(lo)`` and ``f(hi)`` of opposite sign.
    lo, hi : float
        Bracketing interval.
    tol : float
        Absolute interval tolerance.
    max_iter : int
        Iteration cap.

    Returns
    -------
    float
        The bracketed root.

    Raises
    ------
    ValueError
        If the bracket does not straddle a sign change.
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("bisect: f(lo) and f(hi) must have opposite signs")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0 or (hi - lo) < tol:
            return mid
        if flo * fmid < 0.0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ special functions

_lgamma_ufunc = np.frompyfunc(math.lgamma, 1, 1)
_erfc_ufunc = np.frompyfunc(math.erfc, 1, 1)


def lgamma_vec(x: object) -> np.ndarray:
    """Vectorized ``math.lgamma`` cast to float64."""
    return np.asarray(_lgamma_ufunc(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            break
    return h


def _betainc_scalar(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def betainc(a: float, b: float, x: object) -> np.ndarray:
    """Regularized incomplete beta function ``I_x(a, b)``.

    Computed by the Lentz continued fraction with the standard symmetry
    split at ``x = (a + 1) / (a + b + 2)``.

    Parameters
    ----------
    a, b : float
        Positive shape parameters.
    x : array_like
        Evaluation points in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``I_x(a, b)`` elementwise.
    """
    arr = np.asarray(x, dtype=np.float64)
    flat = arr.reshape(-1)
    out = np.array([_betainc_scalar(float(a), float(b), float(v)) for v in flat])
    return out.reshape(arr.shape)


def _gammainc_pq_scalar(s: float, x: float) -> tuple[float, float]:
    """``(P(s, x), Q(s, x))``, each computed directly where it is accurate.

    The series converges for ``x < s + 1`` and yields ``P``; the continued
    fraction yields ``Q``. The complement is formed by subtraction only on
    the side where it is not small, so neither tail loses relative precision.
    """
    if x <= 0.0:
        return 0.0, 1.0
    ln_scale = -x + s * math.log(x) - math.lgamma(s)
    if x < s + 1.0:
        # Series representation of P(s, x).
        term = 1.0 / s
        total = term
        denom = s
        for _ in range(_CF_MAX_ITER):
            denom += 1.0
            term *= x / denom
            total += term
            if abs(term) < abs(total) * _CF_EPS:
                break
        p = total * math.exp(ln_scale)
        return p, 1.0 - p
    # Continued fraction for Q(s, x) (modified Lentz).
    b0 = x + 1.0 - s
    c = 1.0 / _FPMIN
    d = 1.0 / b0
    h = d
    for i in range(1, _CF_MAX_ITER + 1):
        an = -i * (i - s)
        b0 += 2.0
        d = an * d + b0
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b0 + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            break
    q = math.exp(ln_scale) * h
    return 1.0 - q, q


def _gammainc_lower_scalar(s: float, x: float) -> float:
    return _gammainc_pq_scalar(s, x)[0]


def gammainc_lower(s: float, x: object) -> np.ndarray:
    """Regularized lower incomplete gamma function ``P(s, x)``.

    Series representation for ``x < s + 1``, Lentz continued fraction for the
    complement otherwise. Gives the chi-square CDF as
    ``chi2_cdf(x, df) = P(df / 2, x / 2)``.

    Parameters
    ----------
    s : float
        Positive shape parameter.
    x : array_like
        Non-negative evaluation points.

    Returns
    -------
    numpy.ndarray
        ``P(s, x)`` elementwise.
    """
    arr = np.asarray(x, dtype=np.float64)
    flat = arr.reshape(-1)
    out = np.array([_gammainc_lower_scalar(float(s), float(v)) for v in flat])
    return out.reshape(arr.shape)


def chi2_ppf(q: float, df: float) -> float:
    """Chi-square quantile function via bisection on :func:`gammainc_lower`.

    Parameters
    ----------
    q : float
        Probability level in ``(0, 1)``.
    df : float
        Degrees of freedom.

    Returns
    -------
    float
        ``x`` such that ``P(df/2, x/2) = q``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("chi2_ppf: q must lie in (0, 1)")
    hi = df + 10.0 * math.sqrt(2.0 * df) + 10.0
    while float(gammainc_lower(df / 2.0, hi / 2.0)) < q:
        hi *= 2.0
    return bisect(lambda x: float(gammainc_lower(df / 2.0, x / 2.0)) - q, 0.0, hi, tol=1e-12)


def chi2_sf(x: float, df: float) -> float:
    """Chi-square upper tail ``P(X > x)``, accurate far into the tail.

    Uses the regularized *upper* incomplete gamma ``Q(df/2, x/2)`` directly
    rather than ``1 - P``, so p-values below ``1e-16`` do not round to 0.

    Parameters
    ----------
    x : float
        Statistic value (``<= 0`` gives 1.0).
    df : float
        Positive degrees of freedom.

    Returns
    -------
    float
        Upper-tail probability.
    """
    return _gammainc_pq_scalar(float(df) / 2.0, float(x) / 2.0)[1]


def beta_ppf(q: float, a: float, b: float) -> float:
    """Beta(a, b) quantile function via bisection on :func:`betainc`.

    Parameters
    ----------
    q : float
        Probability level in ``(0, 1)``.
    a, b : float
        Positive shape parameters.

    Returns
    -------
    float
        ``x`` such that ``I_x(a, b) = q``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("beta_ppf: q must lie in (0, 1)")
    return bisect(lambda x: float(betainc(a, b, x)) - q, 0.0, 1.0, tol=1e-12)


def norm_cdf(x: object) -> np.ndarray:
    """Standard normal CDF via the complementary error function.

    ``erfc`` is used instead of ``erf`` for relative accuracy in the tails
    (a deliberate deviation from the literal spec wording).

    Parameters
    ----------
    x : array_like
        Evaluation points.

    Returns
    -------
    numpy.ndarray
        ``Phi(x)`` elementwise.
    """
    arr = np.asarray(x, dtype=np.float64)
    return 0.5 * np.asarray(_erfc_ufunc(-arr / math.sqrt(2.0)), dtype=np.float64)


def norm_sf(x: float) -> float:
    """Standard normal upper tail ``1 - Phi(x)`` via ``erfc`` (tail-accurate)."""
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


# ------------------------------------------------------------------ PAVA


class PavaResult(NamedTuple):
    """Pool-adjacent-violators fit with its block structure."""

    fitted: np.ndarray
    block_start: np.ndarray
    block_mean: np.ndarray
    block_weight: np.ndarray


def pava(y: object, w: object) -> PavaResult:
    """Weighted isotonic regression by pool-adjacent-violators.

    Solves ``min sum w_i (y_i - m_i)^2`` subject to ``m`` non-decreasing, in
    amortized O(n) with preallocated block arrays.

    Parameters
    ----------
    y : array_like
        Response values in their given (sorted-by-score) order.
    w : array_like
        Positive weights, same length as ``y``.

    Returns
    -------
    PavaResult
        ``fitted`` (block means expanded to observations), ``block_start``
        (index of each block's first observation), ``block_mean`` and
        ``block_weight`` (pooled means and total weights per block).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    n = y_arr.shape[0]
    start = np.empty(n, dtype=np.int64)
    mean = np.empty(n, dtype=np.float64)
    weight = np.empty(n, dtype=np.float64)
    k = -1
    for i in range(n):
        k += 1
        start[k] = i
        mean[k] = y_arr[i]
        weight[k] = w_arr[i]
        while k > 0 and mean[k - 1] > mean[k]:
            total = weight[k - 1] + weight[k]
            mean[k - 1] = (weight[k - 1] * mean[k - 1] + weight[k] * mean[k]) / total
            weight[k - 1] = total
            k -= 1
    n_blocks = k + 1
    fitted = np.empty(n, dtype=np.float64)
    for j in range(n_blocks):
        end = start[j + 1] if j + 1 < n_blocks else n
        fitted[start[j] : end] = mean[j]
    return PavaResult(
        fitted=fitted,
        block_start=start[:n_blocks].copy(),
        block_mean=mean[:n_blocks].copy(),
        block_weight=weight[:n_blocks].copy(),
    )


# ------------------------------------------------------------------ IRLS logistic


class IrlsResult(NamedTuple):
    """IRLS logistic regression fit."""

    beta: np.ndarray
    converged: bool
    separation: bool
    nll: float


_STALL_STEP_TOL = 1e-5
"""Relative full-Newton-step bound under which a line-search stall still counts
as convergence: at a genuine optimum the objective stops decreasing in floating
point once the step is ~sqrt(eps) relative, well below this; along a
quasi-separation ray the full step stays O(1 / margin) and never gets there."""


def irls_logistic(
    X: object,
    y: object,
    w: object = None,
    ridge: float = 0.0,
    offset: object = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    *,
    penalty: object = None,
) -> IrlsResult:
    """Logistic regression by Newton/IRLS with monotone descent.

    Minimizes ``sum w * (softplus(eta) - y * eta) + 0.5 * beta' Q beta`` with
    ``eta = X beta + offset`` and ``Q = ridge * I + penalty``. Each Newton step
    is halved until it does not increase the objective, so the iteration never
    diverges.

    **Convergence** is judged on the *full* Newton step, never on a halved one:
    the fit is ``converged`` when ``max|H^-1 g| < tol * (1 + max|beta|)``. If
    the line search stalls (no step strictly decreases the objective in
    floating point) the fit counts as converged only when the full step is
    already below ``1e-5`` relative — the floating-point floor at a genuine
    optimum; along a quasi-separation ray the full step stays O(1 / margin)
    while the gradient vanishes, so a stall there is never convergence.

    **Separation** is declared only for effectively binary targets (every
    ``y`` within ``1e-9`` of 0 or 1) on fits without a ridge term, when every
    observation sits on the correct side by more than 10 log-odds from the
    design's own contribution (``eta - offset``) while the gradient is still
    large (the signature of a nonexistent MLE), when the Hessian is singular,
    or when the iteration ends unconverged (the divergence signature of
    quasi-separation, whose full Newton step stays O(1) while the gradient
    vanishes); the function then warns and returns a ridge-regularized refit
    (``ridge=1e-6`` on top of ``penalty``), which is coercive and expected to
    report ``converged=True``. A ``penalty`` matrix with a null space (e.g. a
    roughness penalty that leaves linear terms free) does not prevent
    separation along that null space, so it does not switch detection off.
    Soft targets (as produced by Platt target smoothing) make the objective
    coercive, so ``separation`` is never ``True`` for them; failures surface
    only as ``converged=False`` or a singular-Hessian ridge fallback.

    Parameters
    ----------
    X : array_like of shape (n, k)
        Design matrix (include an intercept column if wanted).
    y : array_like of shape (n,)
        Binary responses in ``{0, 1}`` (soft targets in ``[0, 1]`` are
        accepted, as required by Platt target smoothing).
    w : array_like or None
        Observation weights; ``None`` means unit weights.
    ridge : float
        L2 penalty ``0.5 * ridge * |beta|^2``.
    offset : array_like or None
        Fixed additive term of the linear predictor (coefficient 1).
    max_iter : int
        Newton iteration cap.
    tol : float
        Convergence tolerance on the max absolute full Newton step, relative
        to ``1 + max|beta|``.
    penalty : array_like of shape (k, k) or None, keyword-only
        Symmetric positive semi-definite quadratic penalty
        ``0.5 * beta' penalty beta`` (e.g. ``lam * D'D`` for a penalized
        spline), added to the ridge term.

    Returns
    -------
    IrlsResult
        Coefficients plus ``converged``, ``separation`` and the final
        penalized objective value ``nll``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n, k = X_arr.shape
    w_arr = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    pen = np.zeros((k, k)) if penalty is None else np.asarray(penalty, dtype=np.float64)
    quad = ridge * np.eye(k) + pen

    # Separation is a binary-target concept: soft targets make the objective
    # coercive, so a finite minimizer always exists.
    binary = bool(np.all((y_arr < 1e-9) | (y_arr > 1.0 - 1e-9)))
    detect = binary and ridge == 0.0
    sign = 2.0 * y_arr - 1.0

    def nll_at(beta: np.ndarray, eta: np.ndarray) -> float:
        # Overflow-safe softplus(eta) - y*eta; never log(mu) on saturated mu.
        softplus = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
        return float(np.sum(w_arr * (softplus - y_arr * eta))) + 0.5 * float(beta @ quad @ beta)

    beta = np.zeros(k)
    eta = off.copy()
    obj = nll_at(beta, eta)
    separation = False
    singular = False
    converged = False
    for _ in range(max_iter):
        mu = expit(eta)
        grad = X_arr.T @ (w_arr * (y_arr - mu)) - quad @ beta
        grad_inf = float(np.max(np.abs(grad)))
        tol_grad = 1e-8 * (1.0 + abs(obj))
        if detect and float(np.min(sign * (eta - off))) > 10.0 and grad_inf > tol_grad:
            # Every point classified correctly by > 10 log-odds *by the design
            # itself* (offset excluded: it is not under the coefficients'
            # control) yet the likelihood still improves by pushing beta
            # outward: no MLE.
            separation = True
            break
        wt = w_arr * mu * (1.0 - mu)
        hess = (X_arr * wt[:, None]).T @ X_arr + quad
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            singular = True
            separation = binary
            break
        if not np.all(np.isfinite(step)):
            singular = True
            separation = binary
            break
        step_rel = float(np.max(np.abs(step))) / (1.0 + float(np.max(np.abs(beta))))
        eta_step = X_arr @ step
        accepted = False
        for _ in range(31):  # step-halving: accept beta + step / 2**j, j in {0, ..., 30}
            beta_new = beta + step
            eta_new = eta + eta_step
            obj_new = nll_at(beta_new, eta_new)
            if obj_new <= obj:
                accepted = True
                break
            step = 0.5 * step
            eta_step = 0.5 * eta_step
        stalled = not accepted or obj_new == obj
        if accepted:
            beta, eta, obj = beta_new, eta_new, obj_new
        if stalled:
            # Line-search stall (no strict decrease left in floating point):
            # converged only if the *full* step was already negligible — the
            # floor at a genuine optimum — never merely because the gradient
            # vanished along a runaway (quasi-separation) direction.
            converged = step_rel < _STALL_STEP_TOL
            break
        if step_rel < tol:
            converged = True
            break

    if detect and not (converged or separation or singular):
        # Ending unconverged on an unpenalized binary fit is the divergence
        # signature of quasi-separation: tied boundary points keep the
        # per-point margin near zero, so the margin rule cannot fire, while
        # the Hessian stays numerically nonsingular as beta runs away.
        separation = True

    if (separation or singular) and ridge == 0.0:
        reason = "separation detected" if separation else "singular Hessian"
        warnings.warn(
            f"irls_logistic: {reason}; returning ridge-regularized fit (ridge=1e-6)",
            UserWarning,
            stacklevel=2,
        )
        ridged = irls_logistic(
            X_arr, y_arr, w=w_arr, ridge=1e-6, offset=off, max_iter=max_iter, tol=tol, penalty=pen
        )
        return IrlsResult(
            beta=ridged.beta, converged=ridged.converged, separation=separation, nll=ridged.nll
        )
    return IrlsResult(beta=beta, converged=converged, separation=separation, nll=obj)


# ------------------------------------------------------------------ LOESS


# Elements per gathered window block in ``_loess_fit_sorted_vec``. Anchors are
# processed in blocks of ``_LOESS_BLOCK // r`` so the (block x r) window matrix
# and its temporaries stay cache-resident regardless of n: at n=1e4 (r=7500)
# that is 4 anchors per block, at n=1e6 (r=750000) one. Measured at n=1e4,
# 512 anchors: 0.045s here against 0.083s at 1 << 20 and 0.062s at 1 << 17 --
# a locality effect, not an arithmetic one. Block size does not affect the
# result: each anchor's row is reduced independently.
_LOESS_BLOCK = 1 << 15


def _loess_window_starts(xs: np.ndarray, evs: np.ndarray, r: int) -> np.ndarray:
    """Window start index per eval point: the leftmost minimal-width r-window.

    For 1-D x the r-nearest-neighbor window is contiguous in sorted order. The
    reference two-pointer loop (kept in ``tests/test_math.py``) advances its
    window while ``xs[i + r] - x0 < x0 - xs[i]``; distance ties resolve to the
    leftmost minimal-width window. Both
    sides of that comparison are monotone in ``i`` (IEEE subtraction is), so the
    windows that fail it are upward-closed and the loop's answer is the *first*
    ``i`` that fails — found here without the two-pointer walk. ``xs[i] +
    xs[i + r]`` is likewise non-decreasing, so one ``searchsorted`` against
    ``2 * x0`` lands on that index or within a few of it; the two exact-
    comparison sweeps then move to the loop's index, since the sum form and the
    difference form round differently (``0.3 - 0.2 < 0.2 - 0.1`` while
    ``0.1 + 0.3 == 2 * 0.2``). Both sweeps are monotone and bounded by ``n - r``.

    The forward sweep does fire, routinely, on quantized scores — 156 index
    steps across the 42 distinct anchors of 2-dp-rounded ``make_pd_portfolio``
    at n=4000/frac=0.75, 538 at frac=0.2, 3987 at n=1e5 — and not at all on the
    tie-free portfolios (0 steps at both sizes). The backward sweep has not been
    observed to fire. The result is exact either way; the cost is bounded, and
    measured at 20% of ``_loess_fit_sorted_vec`` at n=1e5 on 2-dp scores against
    a negligible share on continuous ones.
    """
    n = xs.shape[0]
    limit = n - r
    if limit <= 0:
        return np.zeros(evs.shape[0], dtype=np.intp)
    start = np.minimum(np.searchsorted(xs[:limit] + xs[r:], 2.0 * evs, side="left"), limit).astype(
        np.intp
    )
    while True:
        right = xs[np.minimum(start + r, n - 1)]
        advance = (start < limit) & ((right - evs) < (evs - xs[start]))
        if not advance.any():
            break
        start = start + advance
    while True:
        prev = np.maximum(start - 1, 0)
        retreat = (start > 0) & ~((xs[prev + r] - evs) < (evs - xs[prev]))
        if not retreat.any():
            break
        start = start - retreat
    return start


def _loess_fit_sorted_vec(
    xs: np.ndarray, ys: np.ndarray, evs: np.ndarray, r: int, degree: int
) -> np.ndarray:
    """LOESS at sorted eval points: tricube-weighted local fits on r-windows.

    The single LOESS engine (0.3.4 retired the per-point Python loop it was
    once the "vectorized twin" of). Anchors are processed in blocks of
    ``_LOESS_BLOCK // r`` so the gathered (block x r) window matrix stays
    cache-resident; every window has exactly ``r`` points, so the gather is
    rectangular. ``xs`` and ``evs`` must already be sorted ascending.

    Against the retired loop (frozen in ``tests/test_math.py`` as a
    reference) the only intended difference is the tricube cube, taken by
    multiplication (``u * u * u``) rather than ``** 3`` (libm ``pow``, ~10x the
    cost): at most one ulp per weight on a well-conditioned window, measured
    <= 4.3e-15 absolute on fitted values over 720 portfolio configurations
    (continuous, 2-dp and 3-dp scores; frac 0.2/0.75; degree 0/1).

    On a *rank-deficient* window — every non-zero tricube weight on one
    distinct ``x``, which happens when the far half of the window lies at
    exactly the bandwidth ``h`` — ``det`` is pure cancellation (~1e-23 rather
    than 0), and the ulp-level weight difference can put the two
    implementations on opposite sides of the ``abs(det) < _FPMIN`` guard. Such
    a window's local *linear* fit is undefined; the ``swy / sw`` branch (the
    weighted mean, what it degenerates to) is the well-defined answer, and the
    loop's alternative divided by cancellation noise. See
    ``tests/test_math.py::test_loess_vectorized_rank_deficient_window``.
    """
    m = evs.shape[0]
    start = _loess_window_starts(xs, evs, r)
    out = np.empty(m, dtype=np.float64)
    windows_x = sliding_window_view(xs, r)
    windows_y = sliding_window_view(ys, r)
    block = max(1, _LOESS_BLOCK // max(r, 1))
    for lo in range(0, m, block):
        hi = min(lo + block, m)
        s = start[lo:hi]
        x0 = evs[lo:hi]
        xw = windows_x[s]
        yw = windows_y[s]
        xc = xw - x0[:, None]
        h = np.maximum(x0 - xs[s], xs[s + r - 1] - x0)
        degenerate = h == 0.0
        u = np.abs(xc) / np.where(degenerate, 1.0, h)[:, None]
        cube = u * u * u
        tri = np.clip(1.0 - cube, 0.0, None)
        wts = tri * tri * tri
        if degree == 0:
            ww = np.maximum(wts, _FPMIN)
            vals = (ww * yw).sum(axis=1) / ww.sum(axis=1)
        else:
            wxc = wts * xc
            sw = wts.sum(axis=1)
            swx = wxc.sum(axis=1)
            swxx = (wxc * xc).sum(axis=1)
            swy = (wts * yw).sum(axis=1)
            swxy = (wxc * yw).sum(axis=1)
            det = sw * swxx - swx * swx
            singular = np.abs(det) < _FPMIN
            vals = np.where(
                singular,
                swy / np.maximum(sw, _FPMIN),
                (swxx * swy - swx * swxy) / np.where(singular, 1.0, det),
            )
        # ``yw.mean(axis=1)`` is another full pass over the block; only pay it
        # when some window actually collapsed to a single distinct x.
        if degenerate.any():
            vals = np.where(degenerate, yw.mean(axis=1), vals)
        out[lo:hi] = vals
    return out


def loess(
    x: object,
    y: object,
    frac: float = 0.75,
    degree: int = 1,
    xeval: object = None,
    *,
    grid_size: int | None = None,
    presorted: bool = False,
) -> np.ndarray:
    """Tricube-weighted local polynomial regression (LOESS).

    Parameters
    ----------
    x, y : array_like
        Data points.
    frac : float
        Fraction of observations in each local window.
    degree : int
        Local polynomial degree, 0 (mean) or 1 (linear).
    xeval : array_like or None
        Points at which to evaluate the smoother; ``None`` evaluates at ``x``
        (the ICI use case).
    grid_size : int or None, keyword-only
        When set and ``xeval`` has more than ``grid_size`` points, fit only at
        ``grid_size`` equal-mass anchors (quantiles of the eval points, endpoints
        included, no extrapolation) and linearly interpolate between them.
        Windows and bandwidths are still computed against the full ``x``. This
        mirrors R ``stats::lowess``, whose ``delta`` parameter (default
        ``0.01 * diff(range(x))``) likewise fits at spaced points and
        interpolates. ``None`` (default) evaluates exactly at every point.
    presorted : bool, keyword-only
        Declare that ``x`` is already sorted ascending, so the internal
        ``argsort`` (and the inverse permutation of the fitted values) can be
        skipped. Purely a throughput switch for callers that already hold a
        sorted copy — ``evaluate``'s bootstrap sorts each replicate once and
        shares that order across metrics. Output is unchanged when the
        declaration holds and meaningless when it does not; nothing checks it.

    Returns
    -------
    numpy.ndarray
        Fitted values at ``xeval``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    ev = x_arr if xeval is None else np.asarray(xeval, dtype=np.float64)
    n = x_arr.shape[0]
    r = min(max(int(math.ceil(frac * n)), degree + 1), n)
    order = None if presorted else np.argsort(x_arr, kind="stable")
    xs, ys = (x_arr, y_arr) if order is None else (x_arr[order], y_arr[order])
    if grid_size is not None and ev.shape[0] > grid_size:
        anchors = np.unique(np.quantile(ev, np.linspace(0.0, 1.0, grid_size)))
        return np.interp(ev, anchors, _loess_fit_sorted_vec(xs, ys, anchors, r, degree))
    if xeval is None:
        fit = _loess_fit_sorted_vec(xs, ys, xs, r, degree)
        if order is None:
            return fit
        out = np.empty_like(fit)
        out[order] = fit
    else:
        ev_order = np.argsort(ev, kind="stable")
        fit = _loess_fit_sorted_vec(xs, ys, ev[ev_order], r, degree)
        out = np.empty_like(fit)
        out[ev_order] = fit
    return out


# ------------------------------------------------------------------ natural cubic basis


def natural_cubic_basis(x: object, knots: object, *, deriv: int = 0) -> np.ndarray:
    """Natural cubic spline basis (Hastie–Tibshirani–Friedman §5.2.1).

    With knots ``xi_1 < ... < xi_K``, returns the K columns
    ``N_1 = 1``, ``N_2 = x`` and ``N_{k+2} = d_k - d_{K-1}`` where
    ``d_k(x) = [(x - xi_k)_+^3 - (x - xi_K)_+^3] / (xi_K - xi_k)``.
    The basis is linear beyond the boundary knots.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    knots : array_like
        Strictly increasing interior knots, at least 3 of them.
    deriv : {0, 1, 2}, keyword-only
        Derivative order of the returned columns (0: the basis itself).

    Returns
    -------
    numpy.ndarray of shape (n, K)
        Basis matrix (or its ``deriv``-th derivative).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    kn = np.asarray(knots, dtype=np.float64)
    n_knots = kn.shape[0]
    if n_knots < 3:
        raise ValueError("natural_cubic_basis: need at least 3 knots")
    if deriv not in (0, 1, 2):
        raise ValueError(f"natural_cubic_basis: deriv must be 0, 1 or 2, got {deriv!r}")
    power = 3 - deriv
    factor = (1.0, 3.0, 6.0)[deriv]

    def d(k: int) -> np.ndarray:
        num = (
            np.clip(x_arr - kn[k], 0.0, None) ** power - np.clip(x_arr - kn[-1], 0.0, None) ** power
        )
        return factor * num / (kn[-1] - kn[k])

    lead = (
        [np.ones_like(x_arr), x_arr],
        [np.zeros_like(x_arr), np.ones_like(x_arr)],
        [np.zeros_like(x_arr), np.zeros_like(x_arr)],
    )[deriv]
    cols = list(lead)
    d_last = d(n_knots - 2)
    for k in range(n_knots - 2):
        cols.append(d(k) - d_last)
    return np.column_stack(cols)


# ------------------------------------------------------------------ weighted quantile


def weighted_quantile(x: object, q: object, w: object) -> np.ndarray:
    """Weighted quantiles by linear interpolation of the empirical weighted CDF.

    Positions are ``p_i = (C_i - w_i / 2) / W``, where ``C_i`` is the
    cumulative weight through the ``i``-th sorted observation and ``W`` is the
    total weight; ``q`` outside ``[p_1, p_n]`` is clipped to the extremes
    (``np.interp`` clamps). With equal weights these are the Hazen positions
    ``(i - 0.5) / n`` — numpy's ``method="hazen"`` — *not* numpy's default
    quantile method. Callers that must stay bit-identical to ``np.quantile``
    on unweighted or equal-weight input should short-circuit to
    ``np.quantile`` themselves rather than rely on this function.

    Parameters
    ----------
    x : array_like
        Sample values.
    q : array_like
        Probability level(s) in ``[0, 1]``.
    w : array_like
        Positive weights, same length as ``x``.

    Returns
    -------
    numpy.ndarray
        Interpolated weighted quantile(s) at ``q``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    order = np.argsort(x_arr, kind="stable")
    xs, ws = x_arr[order], w_arr[order]
    cw = np.cumsum(ws)
    pos = (cw - 0.5 * ws) / cw[-1]
    return np.interp(np.asarray(q, dtype=np.float64), pos, xs)
