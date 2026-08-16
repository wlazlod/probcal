"""Binning-free calibration metrics: smoothECE, ECCE, ICI family, Spiegelhalter z.

Theory: ``docs/concepts/metrics.md``.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._math import loess, logit, norm_cdf, weighted_quantile
from .scores import _prep


def _smece_at_sigma(loc: np.ndarray, mass: np.ndarray, sigma: float) -> float:
    """smECE at bandwidth sigma: integral of |kernel-smoothed signed residual measure|."""
    grid = np.linspace(loc.min() - 5.0 * sigma, loc.max() + 5.0 * sigma, 257)
    diff = (grid[:, None] - loc[None, :]) / sigma
    kern = np.exp(-0.5 * diff**2) / (sigma * math.sqrt(2.0 * math.pi))
    return float(np.trapezoid(np.abs(kern @ mass), grid))


def _smece_fixed_point(loc: np.ndarray, mass: np.ndarray) -> tuple[float, float]:
    """Solve smECE(sigma) = sigma by bisection; return (value, sigma_used)."""
    lo, hi = 1e-4, 2.0
    if _smece_at_sigma(loc, mass, lo) - lo <= 0.0:  # near-perfectly calibrated
        return _smece_at_sigma(loc, mass, lo), lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = _smece_at_sigma(loc, mass, mid) - mid
        if abs(hi - lo) < 1e-4:
            break
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma, sigma


_SMECE_MAX_BINS = 1 << 20


def _smece_at_sigma_lattice(m: np.ndarray, width: float, sigma: float) -> float:
    """smECE of a lattice-binned measure at bandwidth sigma, by direct convolution.

    The measure is first coarsened to spacing ``max(width, ~sigma/8)`` (integer
    factor, mass-conserving), then convolved with a truncated Gaussian
    (+-5 sigma, at most ~81 taps) and integrated by the midpoint rule on the
    lattice. Cost is O(len(m)) per call, independent of n and of sigma. For
    ``5 * sigma <= spacing`` the kernels are isolated and the integral is
    exactly the total variation ``sum(|m|)`` (also its upper bound for every
    sigma), which replaces the aliasing-prone coarse-grid evaluation that made
    the pre-fix binned path spuriously report ~0 at small sigma.
    """
    factor = max(1, int(sigma / (8.0 * width)))
    if factor > 1:
        pad = (-m.shape[0]) % factor
        mp = np.concatenate([m, np.zeros(pad)]) if pad else m
        mc = mp.reshape(-1, factor).sum(axis=1)
        w2 = width * factor
    else:
        mc, w2 = m, width
    if 5.0 * sigma <= w2:  # isolated masses: integral is the total variation
        return float(np.sum(np.abs(mc)))
    k = int(math.ceil(5.0 * sigma / w2))
    offs = np.arange(-k, k + 1) * w2
    taps = np.exp(-0.5 * (offs / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    f = np.convolve(mc, taps, mode="full")  # spans +-k cells beyond the lattice
    return float(w2 * np.sum(np.abs(f)))


def _smece_fixed_point_lattice(m: np.ndarray, width: float) -> tuple[float, float]:
    """Bisection twin of ``_smece_fixed_point`` on the lattice evaluator.

    Deliberately duplicates the 12-line skeleton (including the discarded
    final-mid quirk) instead of parametrizing it, so the exact path's
    bit-behavior is untouchable by construction.
    """
    lo, hi = 1e-4, 2.0
    if _smece_at_sigma_lattice(m, width, lo) - lo <= 0.0:
        return _smece_at_sigma_lattice(m, width, lo), lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = _smece_at_sigma_lattice(m, width, mid) - mid
        if abs(hi - lo) < 1e-4:
            break
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma, sigma


def smooth_ece(
    y: object, p: object, *, sample_weight: object = None, bins: int | None = 8192
) -> float:
    """Kernel-smoothed ECE with a self-consistent bandwidth (Błasiok–Nakkiran).

    Residuals are smoothed with a Gaussian kernel on the logit scale (the
    paper's reflected kernel is a boundary device for [0, 1]; on the
    unbounded logit scale no reflection is needed — DECISIONS entry), and the
    reported value is the fixed point ``smECE(sigma) = sigma`` found by
    bisection.

    ``bins`` pre-aggregates the weighted residual measure onto a regular grid
    over the logit range before solving the fixed point; the binned measure
    is then evaluated in closed form on its own lattice by direct Gaussian
    convolution, at a cost independent of n and of sigma. With ``bins=None``,
    or whenever ``n <= bins``, or the logit range is degenerate
    (``t.max() == t.min()``), this reproduces the exact 0.1.2 computation
    bit-for-bit. Otherwise, if the found ``sigma`` is smaller than 8 bin
    widths (the kernel would be under-resolved by the bins), the solve is
    repeated once on an adaptively refined binning (``bins <- ceil(range /
    (sigma/8))``); when that refined bin count would reach or exceed ``n``,
    or exceed ``2**20``, the exact computation is used directly instead of
    refining that far; otherwise the guard is retried once on the refined
    binning, and the exact computation is used as a silent fallback if it
    still trips. That fallback is
    O(n) per bisection step, matching the pre-0.1.3 cost, and can be reached
    for near-perfectly-calibrated data spread over a wide logit range (e.g.
    extreme/clipped scores), so worst-case cost is unchanged from 0.1.2. The
    lattice integrator resolves the kernel at >= 8 samples per sigma, which
    is more accurate than the pre-fix 257-point grid whenever sigma < range /
    256.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.
    bins : int or None, keyword-only
        Number of lattice bins for the fast path (default 8192); ``None``
        forces the exact O(n) computation.

    Returns
    -------
    float
        smECE: the fixed point ``sigma`` solving ``smECE(sigma) = sigma``.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    t = logit(p_arr)
    mass = (w / w.sum()) * (y_arr - p_arr)
    t_lo, t_hi = float(t.min()), float(t.max())
    if bins is None or t.size <= bins or t_hi == t_lo:
        return _smece_fixed_point(t, mass)[0]

    def _binned_solve(b: int) -> tuple[float, float, float]:
        width = (t_hi - t_lo) / b
        idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, b - 1)
        m = np.bincount(idx, weights=mass, minlength=b)
        value, sigma = _smece_fixed_point_lattice(m, width)
        return value, sigma, width

    value, sigma, width = _binned_solve(bins)
    if sigma >= 8.0 * width:
        return value
    # Under-resolved: one adaptive refinement sized so 8 bins span sigma.
    b2 = math.ceil((t_hi - t_lo) / (sigma / 8.0))
    if b2 >= t.size or b2 > _SMECE_MAX_BINS:
        return _smece_fixed_point(t, mass)[0]  # exact is cheaper or required
    value, sigma, width = _binned_solve(b2)
    if sigma >= 8.0 * width:
        return value
    return _smece_fixed_point(t, mass)[0]  # O(n) worst case, no warning


@dataclass(frozen=True)
class EcceResult:
    """Empirical cumulative calibration error: Kolmogorov-style max and mean
    of the cumulative deviation over sorted predictions.

    Attributes
    ----------
    stat_max : float
        Maximum absolute cumulative deviation.
    stat_mean : float
        Mean absolute cumulative deviation.
    """

    stat_max: float
    stat_mean: float


def ecce(y: object, p: object, *, sample_weight: object = None) -> EcceResult:
    """Cumulative-deviation calibration error (Arrieta-Ibarra et al., 2022).

    Sort by prediction and walk the cumulative sum of weighted residuals;
    under calibration the walk hovers near zero, and drift localizes
    miscalibration without any smoothing parameter.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    EcceResult
        Max and mean absolute cumulative deviation.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    order = np.argsort(p_arr, kind="stable")
    c = np.cumsum(w[order] * (y_arr[order] - p_arr[order])) / w.sum()
    return EcceResult(stat_max=float(np.max(np.abs(c))), stat_mean=float(np.mean(np.abs(c))))


def ici(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Integrated calibration index: weighted mean |LOESS(y|p) - p|
    (Austin & Steyerberg, 2019).

    The LOESS stage itself is unweighted (DECISIONS entry).
    ``grid_size=None`` recovers 0.1.2 values exactly.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``; weights only the
        final averaging step, not the LOESS fit.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

    Returns
    -------
    float
        Weighted mean absolute LOESS-to-prediction distance.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    c = loess(p_arr, y_arr, frac=frac, grid_size=grid_size)
    return float(np.average(np.abs(c - p_arr), weights=w))


def _ici_distances(y: object, p: object, frac: float, grid_size: int | None) -> np.ndarray:
    y_arr, p_arr, _ = _prep(y, p, None)
    return np.abs(loess(p_arr, y_arr, frac=frac, grid_size=grid_size) - p_arr)


def _ici_quantile(d: np.ndarray, q: float, y: object, p: object, sample_weight: object) -> float:
    """Quantile of the (always-unweighted) LOESS distances ``d``.

    ``sample_weight is None`` or all-equal weights use ``np.quantile``
    unchanged, so every unweighted/equal-weight caller stays bit-identical to
    0.1.2; otherwise the quantile step (only) is weighted via
    :func:`weighted_quantile`.
    """
    if sample_weight is None:
        return float(np.quantile(d, q))
    _, _, w = _prep(y, p, sample_weight)
    if np.all(w == w[0]):
        return float(np.quantile(d, q))
    return float(weighted_quantile(d, q, w))


def e50(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Median of the |LOESS(y|p) - p| distances.

    ``grid_size=None`` recovers 0.1.2 values exactly. The LOESS distances are
    always unweighted (DECISIONS entry); ``sample_weight``, when given and
    not uniform, weights only the quantile step (see
    :func:`weighted_quantile`).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights; used only for the quantile step.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

    Returns
    -------
    float
        Median of the LOESS distances.
    """
    d = _ici_distances(y, p, frac, grid_size)
    return _ici_quantile(d, 0.5, y, p, sample_weight)


def e90(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """90th percentile of the |LOESS(y|p) - p| distances.

    ``grid_size=None`` recovers 0.1.2 values exactly. The LOESS distances are
    always unweighted (DECISIONS entry); ``sample_weight``, when given and
    not uniform, weights only the quantile step (see
    :func:`weighted_quantile`).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights; used only for the quantile step.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

    Returns
    -------
    float
        90th percentile of the LOESS distances.
    """
    d = _ici_distances(y, p, frac, grid_size)
    return _ici_quantile(d, 0.9, y, p, sample_weight)


def emax(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Maximum of the |LOESS(y|p) - p| distances.

    ``grid_size=None`` recovers 0.1.2 values exactly.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Accepted for signature parity with the other ICI-family metrics but
        not used: the maximum is a weight-independent order statistic.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

    Returns
    -------
    float
        Maximum of the LOESS distances.
    """
    return float(np.max(_ici_distances(y, p, frac, grid_size)))


@dataclass(frozen=True)
class SpiegelhalterResult:
    """Spiegelhalter's z test of forecast unbiasedness (two-sided).

    Attributes
    ----------
    z : float
        Standardized test statistic.
    p_value : float
        Two-sided p-value under the standard normal approximation.
    """

    z: float
    p_value: float


def spiegelhalter_z(y: object, p: object, *, sample_weight: object = None) -> SpiegelhalterResult:
    """Spiegelhalter (1986) z statistic built on the Brier score.

    The numerator has expectation zero under calibration; the statistic is
    asymptotically standard normal. No binning, no smoothing; aggregates the
    whole range, so compensating regional errors can cancel.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    SpiegelhalterResult
        Z statistic and two-sided p-value.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    num = float(np.sum(w * (y_arr - p_arr) * (1.0 - 2.0 * p_arr)))
    var = float(np.sum(w**2 * (1.0 - 2.0 * p_arr) ** 2 * p_arr * (1.0 - p_arr)))
    z = num / math.sqrt(var)
    p_value = float(2.0 * (1.0 - norm_cdf(np.array([abs(z)]))[0]))
    return SpiegelhalterResult(z=z, p_value=p_value)
