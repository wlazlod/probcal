"""Binning-free calibration metrics: smoothECE, ECCE, ICI family, Spiegelhalter z.

Theory: ``docs/concepts/metrics.md``.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._math import loess, logit, norm_cdf
from .scores import _prep


def _smece_at_sigma(t: np.ndarray, r: np.ndarray, wn: np.ndarray, sigma: float) -> float:
    """smECE at bandwidth sigma: integral of |kernel-smoothed signed residual measure|."""
    grid = np.linspace(t.min() - 5.0 * sigma, t.max() + 5.0 * sigma, 257)
    diff = (grid[:, None] - t[None, :]) / sigma
    kern = np.exp(-0.5 * diff**2) / (sigma * math.sqrt(2.0 * math.pi))
    f = kern @ (wn * r)
    return float(np.trapezoid(np.abs(f), grid))


def smooth_ece(y: object, p: object, *, sample_weight: object = None) -> float:
    """Kernel-smoothed ECE with a self-consistent bandwidth (Błasiok–Nakkiran).

    Residuals are smoothed with a Gaussian kernel on the logit scale (the
    paper's reflected kernel is a boundary device for [0, 1]; on the
    unbounded logit scale no reflection is needed — DECISIONS entry), and the
    reported value is the fixed point ``smECE(sigma) = sigma`` found by
    bisection.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    t = logit(p_arr)
    r = y_arr - p_arr
    wn = w / w.sum()
    lo, hi = 1e-4, 2.0
    f_lo = _smece_at_sigma(t, r, wn, lo) - lo
    if f_lo <= 0.0:  # essentially perfectly calibrated at the finest scale
        return _smece_at_sigma(t, r, wn, lo)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = _smece_at_sigma(t, r, wn, mid) - mid
        if abs(hi - lo) < 1e-4:
            break
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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


def ici(y: object, p: object, *, frac: float = 0.75, sample_weight: object = None) -> float:
    """Integrated calibration index: weighted mean |LOESS(y|p) - p|
    (Austin & Steyerberg, 2019). The LOESS stage itself is unweighted
    (DECISIONS entry)."""
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    c = loess(p_arr, y_arr, frac=frac)
    return float(np.average(np.abs(c - p_arr), weights=w))


def _ici_distances(y: object, p: object, frac: float) -> np.ndarray:
    y_arr, p_arr, _ = _prep(y, p, None)
    return np.abs(loess(p_arr, y_arr, frac=frac) - p_arr)


def e50(y: object, p: object, *, frac: float = 0.75, sample_weight: object = None) -> float:
    """Median of the |LOESS(y|p) - p| distances."""
    return float(np.quantile(_ici_distances(y, p, frac), 0.5))


def e90(y: object, p: object, *, frac: float = 0.75, sample_weight: object = None) -> float:
    """90th percentile of the |LOESS(y|p) - p| distances."""
    return float(np.quantile(_ici_distances(y, p, frac), 0.9))


def emax(y: object, p: object, *, frac: float = 0.75, sample_weight: object = None) -> float:
    """Maximum of the |LOESS(y|p) - p| distances."""
    return float(np.max(_ici_distances(y, p, frac)))


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
