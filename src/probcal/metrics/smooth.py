"""Binning-free calibration metrics: smoothECE, ECCE, ICI family, Spiegelhalter z.

Theory: ``docs/concepts/metrics.md``.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._math import loess, logit, norm_cdf
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
    over the logit range before solving the fixed point, cutting the cost of
    the 257 x n kernel matrix built at every bisection step down to 257 x
    bins. With ``bins=None``, or whenever ``n <= bins``, or the logit range is
    degenerate (``t.max() == t.min()``), this reproduces the exact 0.1.2
    computation bit-for-bit. Otherwise the solve is repeated once on an 8x
    finer binning if the found ``sigma`` is smaller than 8 bin widths (the
    kernel would then be under-resolved by the bins); if that still holds,
    the exact computation is used as a silent fallback. That fallback is
    O(n) per bisection step, matching the pre-0.1.3 cost, and can be reached
    for near-perfectly-calibrated data spread over a wide logit range (e.g.
    extreme/clipped scores), so worst-case cost is unchanged from 0.1.2.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    t = logit(p_arr)
    mass = (w / w.sum()) * (y_arr - p_arr)
    t_lo, t_hi = float(t.min()), float(t.max())
    if bins is None or t.size <= bins or t_hi == t_lo:
        return _smece_fixed_point(t, mass)[0]
    b = bins
    for _ in range(2):  # initial bin count, then one 8x refinement
        width = (t_hi - t_lo) / b
        idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, b - 1)
        m = np.bincount(idx, weights=mass, minlength=b)
        centers = t_lo + (np.arange(b) + 0.5) * width
        value, sigma = _smece_fixed_point(centers, m)
        if sigma >= 8.0 * width:
            return value
        b *= 8
    return _smece_fixed_point(t, mass)[0]  # exact fallback; O(n) worst case, no warning


@dataclass(frozen=True)
class EcceResult:
    """Empirical cumulative calibration error: Kolmogorov-style max and mean
    of the cumulative deviation over sorted predictions."""

    stat_max: float
    stat_mean: float


def ecce(y: object, p: object, *, sample_weight: object = None) -> EcceResult:
    """Cumulative-deviation calibration error (Arrieta-Ibarra et al., 2022).

    Sort by prediction and walk the cumulative sum of weighted residuals;
    under calibration the walk hovers near zero, and drift localizes
    miscalibration without any smoothing parameter.
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
    (Austin & Steyerberg, 2019). The LOESS stage itself is unweighted
    (DECISIONS entry). ``grid_size=None`` recovers 0.1.2 values exactly."""
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    c = loess(p_arr, y_arr, frac=frac, grid_size=grid_size)
    return float(np.average(np.abs(c - p_arr), weights=w))


def _ici_distances(y: object, p: object, frac: float, grid_size: int | None) -> np.ndarray:
    y_arr, p_arr, _ = _prep(y, p, None)
    return np.abs(loess(p_arr, y_arr, frac=frac, grid_size=grid_size) - p_arr)


def e50(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Median of the |LOESS(y|p) - p| distances. ``grid_size=None`` recovers
    0.1.2 values exactly."""
    return float(np.quantile(_ici_distances(y, p, frac, grid_size), 0.5))


def e90(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """90th percentile of the |LOESS(y|p) - p| distances. ``grid_size=None``
    recovers 0.1.2 values exactly."""
    return float(np.quantile(_ici_distances(y, p, frac, grid_size), 0.9))


def emax(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Maximum of the |LOESS(y|p) - p| distances. ``grid_size=None`` recovers
    0.1.2 values exactly."""
    return float(np.max(_ici_distances(y, p, frac, grid_size)))


@dataclass(frozen=True)
class SpiegelhalterResult:
    """Spiegelhalter's z test of forecast unbiasedness (two-sided)."""

    z: float
    p_value: float


def spiegelhalter_z(y: object, p: object, *, sample_weight: object = None) -> SpiegelhalterResult:
    """Spiegelhalter (1986) z statistic built on the Brier score.

    The numerator has expectation zero under calibration; the statistic is
    asymptotically standard normal. No binning, no smoothing; aggregates the
    whole range, so compensating regional errors can cancel.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    num = float(np.sum(w * (y_arr - p_arr) * (1.0 - 2.0 * p_arr)))
    var = float(np.sum(w**2 * (1.0 - 2.0 * p_arr) ** 2 * p_arr * (1.0 - p_arr)))
    z = num / math.sqrt(var)
    p_value = float(2.0 * (1.0 - norm_cdf(np.array([abs(z)]))[0]))
    return SpiegelhalterResult(z=z, p_value=p_value)
