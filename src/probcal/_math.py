"""Numerical core: PAVA, IRLS logistic regression, special functions, LOESS, spline basis.

Pure numpy + stdlib. Special functions are hand-rolled (continued fractions, series,
rational approximations) and verified against scipy in ``tests/test_math_reference.py``.
"""

import math
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

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
    arr = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(arr) - np.log1p(-arr)


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


def newton_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    lo: float,
    hi: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Safeguarded 1-D Newton root finder with bisection fallback.

    Maintains the bracket ``[lo, hi]`` (which must straddle a sign change);
    any Newton step that leaves the bracket, or meets a vanishing derivative,
    is replaced by a bisection step.

    Parameters
    ----------
    f, df : callable
        Function and its derivative.
    x0 : float
        Initial point inside the bracket.
    lo, hi : float
        Bracketing interval with a sign change.
    tol : float
        Absolute tolerance on the root.
    max_iter : int
        Iteration cap.

    Returns
    -------
    float
        The bracketed root.
    """
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("newton_1d: f(lo) and f(hi) must have opposite signs")
    x = float(min(max(x0, lo), hi))
    for _ in range(max_iter):
        fx = f(x)
        if fx == 0.0:
            return x
        if flo * fx < 0.0:
            hi = x
        else:
            lo, flo = x, fx
        if (hi - lo) < tol:
            return 0.5 * (lo + hi)
        d = df(x)
        step_ok = d != 0.0 and math.isfinite(d)
        if step_ok:
            x_new = x - fx / d
            step_ok = lo < x_new < hi
        x = x_new if step_ok else 0.5 * (lo + hi)
    return x


# ------------------------------------------------------------------ special functions

_lgamma_ufunc = np.frompyfunc(math.lgamma, 1, 1)
_erf_ufunc = np.frompyfunc(math.erf, 1, 1)
_erfc_ufunc = np.frompyfunc(math.erfc, 1, 1)


def lgamma_vec(x: object) -> np.ndarray:
    """Vectorized ``math.lgamma`` cast to float64."""
    return np.asarray(_lgamma_ufunc(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def erf_vec(x: object) -> np.ndarray:
    """Vectorized ``math.erf`` cast to float64."""
    return np.asarray(_erf_ufunc(np.asarray(x, dtype=np.float64)), dtype=np.float64)


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


def _gammainc_lower_scalar(s: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
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
        return total * math.exp(ln_scale)
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
    return 1.0 - math.exp(ln_scale) * h


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


def norm_cdf(x: object) -> np.ndarray:
    """Standard normal CDF via the complementary error function.

    ``erfc`` is used instead of ``erf`` for relative accuracy in the tails
    (DECISIONS entry: deviation from the literal spec wording).

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


# Acklam's rational approximation coefficients for the normal quantile.
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _norm_ppf_acklam(q: np.ndarray) -> np.ndarray:
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    p_low = 0.02425
    x = np.empty_like(q)

    low = q < p_low
    high = q > 1.0 - p_low
    mid = ~(low | high)

    if np.any(low):
        u = np.sqrt(-2.0 * np.log(q[low]))
        num = ((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]
        den = (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        x[low] = num / den
    if np.any(high):
        u = np.sqrt(-2.0 * np.log(1.0 - q[high]))
        num = ((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]
        den = (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        x[high] = -(num / den)
    if np.any(mid):
        u = q[mid] - 0.5
        r = u * u
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        x[mid] = num / den
    return x


def norm_ppf(q: object) -> np.ndarray:
    """Standard normal quantile function.

    Acklam's rational approximation refined by two Halley steps on the
    erfc-based CDF; accurate to well below ``1e-11`` absolute over
    ``(1e-12, 1 - 1e-12)`` (reference-tested against scipy).

    Parameters
    ----------
    q : array_like
        Probability levels in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        ``Phi^{-1}(q)`` elementwise.
    """
    arr = np.asarray(q, dtype=np.float64)
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError("norm_ppf: q must lie strictly inside (0, 1)")
    x = _norm_ppf_acklam(arr.reshape(-1)).reshape(arr.shape)
    for _ in range(2):
        e = norm_cdf(x) - arr
        u = e * math.sqrt(2.0 * math.pi) * np.exp(0.5 * x * x)
        x = x - u / (1.0 + 0.5 * x * u)
    return x


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
    n_iter: int


_ETA_SEPARATION = 30.0


def irls_logistic(
    X: object,
    y: object,
    w: object = None,
    ridge: float = 0.0,
    offset: object = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> IrlsResult:
    """Logistic regression by Newton/IRLS with ridge stabilization.

    Detects (quasi-)separation — fitted linear predictors running away or a
    singular Hessian — and in that case warns and returns a ridge-regularized
    refit instead of diverging.

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
        L2 penalty added to the Hessian diagonal (and gradient).
    offset : array_like or None
        Fixed additive term of the linear predictor (coefficient 1).
    max_iter : int
        Newton iteration cap.
    tol : float
        Convergence tolerance on the max absolute Newton step.

    Returns
    -------
    IrlsResult
        Coefficients plus ``converged``, ``separation`` and ``n_iter`` flags.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n, k = X_arr.shape
    w_arr = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)

    beta = np.zeros(k)
    separation = False
    converged = False
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        eta = X_arr @ beta + off
        if np.max(np.abs(eta)) > _ETA_SEPARATION:
            separation = True
            break
        mu = expit(eta)
        wt = w_arr * mu * (1.0 - mu)
        grad = X_arr.T @ (w_arr * (y_arr - mu)) - ridge * beta
        hess = (X_arr * wt[:, None]).T @ X_arr + ridge * np.eye(k)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            separation = True
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol * (1.0 + np.max(np.abs(beta))):
            converged = True
            break

    if separation and ridge == 0.0:
        warnings.warn(
            "irls_logistic: separation detected; returning ridge-regularized fit " "(ridge=1e-6)",
            UserWarning,
            stacklevel=2,
        )
        ridged = irls_logistic(
            X_arr, y_arr, w=w_arr, ridge=1e-6, offset=off, max_iter=max_iter, tol=tol
        )
        return IrlsResult(
            beta=ridged.beta, converged=ridged.converged, separation=True, n_iter=ridged.n_iter
        )
    return IrlsResult(beta=beta, converged=converged, separation=separation, n_iter=n_iter)


# ------------------------------------------------------------------ LOESS


def loess(
    x: object,
    y: object,
    frac: float = 0.75,
    degree: int = 1,
    xeval: object = None,
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

    Returns
    -------
    numpy.ndarray
        Fitted values at ``xeval``.
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


# ------------------------------------------------------------------ natural cubic basis


def natural_cubic_basis(x: object, knots: object) -> np.ndarray:
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

    Returns
    -------
    numpy.ndarray of shape (n, K)
        Basis matrix.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    kn = np.asarray(knots, dtype=np.float64)
    n_knots = kn.shape[0]
    if n_knots < 3:
        raise ValueError("natural_cubic_basis: need at least 3 knots")

    def d(k: int) -> np.ndarray:
        num = np.clip(x_arr - kn[k], 0.0, None) ** 3 - np.clip(x_arr - kn[-1], 0.0, None) ** 3
        return num / (kn[-1] - kn[k])

    cols = [np.ones_like(x_arr), x_arr]
    d_last = d(n_knots - 2)
    for k in range(n_knots - 2):
        cols.append(d(k) - d_last)
    return np.column_stack(cols)
