"""Binning-free calibration metrics: smoothECE, ECCE, ICI family, Spiegelhalter z.

Theory: ``docs/concepts/metrics.md``. Sample weights follow the package
convention in :mod:`probcal.metrics._common`.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .._math import loess, logit, norm_sf, weighted_quantile
from ._common import _prep, is_uniform


def _smece_at_sigma(loc: np.ndarray, mass: np.ndarray, sigma: float) -> float:
    """smECE at bandwidth sigma: integral of |kernel-smoothed signed residual measure|."""
    grid = np.linspace(loc.min() - 5.0 * sigma, loc.max() + 5.0 * sigma, 257)
    diff = (grid[:, None] - loc[None, :]) / sigma
    kern = np.exp(-0.5 * diff**2) / (sigma * math.sqrt(2.0 * math.pi))
    f = np.abs(kern @ mass)
    # Trapezoid rule, written out: np.trapezoid needs numpy >= 2.0.
    return float((np.diff(grid) * (f[1:] + f[:-1]) / 2.0).sum())


def _bisect_fixed_point(f: Callable[[float], float]) -> tuple[float, float]:
    """Solve ``f(sigma) = sigma`` on ``[1e-4, 2]`` by bisection; return (value, sigma).

    Shared by the exact and the lattice smECE evaluators. If ``f(1e-4) <=
    1e-4`` (near-perfect calibration) the value at the lower end is returned.
    """
    lo, hi = 1e-4, 2.0
    if f(lo) - lo <= 0.0:
        return f(lo), lo
    for _ in range(40):
        if abs(hi - lo) < 1e-4:
            break
        mid = 0.5 * (lo + hi)
        if f(mid) - mid > 0.0:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma, sigma


def _smece_fixed_point(loc: np.ndarray, mass: np.ndarray) -> tuple[float, float]:
    """Exact-path fixed point (257-point grid evaluator)."""
    return _bisect_fixed_point(lambda sigma: _smece_at_sigma(loc, mass, sigma))


_SMECE_MAX_BINS = 1 << 20


def _coarsen(m: np.ndarray, width: float, sigma: float) -> tuple[np.ndarray, float]:
    """Mass-conserving coarsening of a lattice measure to spacing ``>= ~sigma/8``.

    The integer factor ``max(1, int(sigma / (8 * width)))`` merges adjacent
    cells; returns the coarsened vector and its cell width.
    """
    factor = max(1, int(sigma / (8.0 * width)))
    if factor == 1:
        return m, width
    pad = (-m.shape[0]) % factor
    mp = np.concatenate([m, np.zeros(pad)]) if pad else m
    return mp.reshape(-1, factor).sum(axis=1), width * factor


def _gauss_taps(sigma: float, w2: float) -> np.ndarray:
    """Gaussian kernel taps on a ``w2``-spaced lattice, truncated at +-5 sigma."""
    k = int(math.ceil(5.0 * sigma / w2))
    offs = np.arange(-k, k + 1) * w2
    return np.exp(-0.5 * (offs / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _smece_at_sigma_lattice(m: np.ndarray, width: float, sigma: float) -> float:
    """smECE of a lattice-binned measure at bandwidth sigma, by direct convolution.

    The measure is first coarsened to spacing ``max(width, ~sigma/8)`` (integer
    factor, mass-conserving), then convolved with a truncated Gaussian
    (+-5 sigma, at most ~81 taps) and integrated by the midpoint rule on the
    lattice. Cost is O(len(m)) per call, independent of n and of sigma. For
    ``5 * sigma <= spacing`` the kernels are isolated and the integral is
    exactly the total variation ``sum(|m|)`` (also its upper bound for every
    sigma), which avoids the aliasing of a coarse-grid evaluation (which can
    spuriously report ~0 at small sigma).
    """
    mc, w2 = _coarsen(m, width, sigma)
    if 5.0 * sigma <= w2:  # isolated masses: integral is the total variation
        return float(np.sum(np.abs(mc)))
    f = np.convolve(mc, _gauss_taps(sigma, w2), mode="full")  # spans +-k cells beyond
    return float(w2 * np.sum(np.abs(f)))


def _smece_fixed_point_lattice(m: np.ndarray, width: float) -> tuple[float, float]:
    """Lattice-path fixed point (same bisection as the exact path)."""
    return _bisect_fixed_point(lambda sigma: _smece_at_sigma_lattice(m, width, sigma))


def _lattice_index(t: np.ndarray, t_lo: float, width: float, bins: int) -> np.ndarray:
    """Cell index of each ``t`` on the ``bins``-cell lattice starting at ``t_lo``.

    Shared by ``smooth_ece``'s lattice solve and ``curves.reliability_smooth``
    (which re-bins bootstrap resamples on the point estimate's lattice).
    """
    return np.clip(((t - t_lo) / width).astype(np.int64), 0, bins - 1)


Lattice = tuple[float, float, int]
"""``(width, t_lo, bins)`` of the equal-width logit lattice a smECE was solved on."""


def _smece_solve(
    t: np.ndarray, mass: np.ndarray, bins: int | None
) -> tuple[float, float, Lattice | None]:
    """``smooth_ece``'s path selection and fixed-point solve.

    Returns ``(value, sigma, lattice)``; ``lattice`` is ``None`` when the
    exact O(n) path ran (``bins=None``, a degenerate logit range, or an
    infeasible/under-resolved refinement). ``curves.reliability_smooth``
    shares this solve, so its ``sigma_star`` is ``smooth_ece``'s by
    construction.
    """
    t_lo, t_hi = float(t.min()), float(t.max())
    if bins is None or t_hi == t_lo:
        return (*_smece_fixed_point(t, mass), None)

    def _binned_solve(b: int) -> tuple[float, float, Lattice]:
        width = (t_hi - t_lo) / b
        m = np.bincount(_lattice_index(t, t_lo, width, b), weights=mass, minlength=b)
        return (*_smece_fixed_point_lattice(m, width), (width, t_lo, b))

    value, sigma, lattice = _binned_solve(bins)
    if sigma >= 8.0 * lattice[0]:
        return value, sigma, lattice
    # Under-resolved: one adaptive refinement sized so 8 bins span sigma.
    b2 = math.ceil((t_hi - t_lo) / (sigma / 8.0))
    if b2 <= _SMECE_MAX_BINS:
        value, sigma, lattice = _binned_solve(b2)
        if sigma >= 8.0 * lattice[0]:
            return value, sigma, lattice
    # Refinement infeasible or still under-resolved: exact O(n) path, no warning.
    return (*_smece_fixed_point(t, mass), None)


def _lattice_kernel_smooth(
    m: np.ndarray, width: float, sigma: float, t_lo: float
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-smoothed value of a lattice mass vector at each (coarsened) cell centre.

    Same coarsening and +-5 sigma truncated kernel as
    :func:`_smece_at_sigma_lattice`, but always convolved (no total-variation
    shortcut): a curve needs pointwise values. Returns ``(centers, smoothed)``
    on the logit scale, ``smoothed`` in ``m``'s mass-per-cell units (so a
    ratio of two such vectors is free of the coarsening factor).
    """
    mc, w2 = _coarsen(m, width, sigma)
    smoothed = np.convolve(mc, _gauss_taps(sigma, w2), mode="same")
    centers = t_lo + (np.arange(mc.shape[0]) + 0.5) * w2
    return centers, smoothed


def smooth_ece(
    y: object, p: object, *, sample_weight: object = None, bins: int | None = 8192
) -> float:
    """Kernel-smoothed ECE with a self-consistent bandwidth (Błasiok–Nakkiran).

    Residuals are smoothed with a Gaussian kernel on the logit scale (the
    paper's reflected kernel is a boundary device for [0, 1]; on the
    unbounded logit scale no reflection is needed), and the
    reported value is the fixed point ``smECE(sigma) = sigma`` found by
    bisection.

    ``bins`` aggregates the weighted residual measure onto an equal-width
    logit lattice and solves the fixed point there by direct convolution
    (cost independent of n and sigma). If the solved ``sigma`` spans fewer
    than 8 cells, the solve is repeated once on a lattice refined to 8 cells
    per sigma; if that would exceed ``2**20`` cells or is still
    under-resolved -- or with ``bins=None`` or a degenerate logit range --
    the exact computation (a 257-point grid evaluator, O(n) per bisection
    step) runs instead. The lattice value differs from the exact one by
    ~1e-4 on typical portfolios; on wide clipped-logit data the exact path's
    fixed grid under-resolves small-sigma kernels and the lattice value is
    the better one.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
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
    return _smece_solve(t, mass, bins)[0]


@dataclass(frozen=True)
class EcceResult:
    """Empirical cumulative calibration error: Kolmogorov-style max and mean
    of the cumulative deviation over sorted predictions.

    Attributes
    ----------
    stat_max : float
        Maximum absolute cumulative deviation.
    stat_mean : float
        Weighted mean absolute cumulative deviation (each tied-score block
        weighted by its total sample weight).
    """

    stat_max: float
    stat_mean: float


class _Walk(NamedTuple):
    """The ECCE cumulative walk, evaluated at the end of every tied-``p`` block."""

    frac: np.ndarray
    """Cumulative weight fraction at each block end (``1/n .. 1`` without ties)."""
    cumdev: np.ndarray
    """Cumulative weighted residual ``sum(w * (y - p)) / sum(w)`` at each block end."""
    block_w: np.ndarray
    """Total weight of each block."""
    sd_null: np.ndarray | None
    """Pointwise SD of the walk under calibration, if asked:
    ``sqrt(cumsum(w**2 * p * (1 - p))) / sum(w)``."""


def _ecce_walk(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, *, presorted: bool = False, sd: bool = False
) -> _Walk:
    """Cumulative-deviation walk shared by :func:`ecce` and ``curves.ecce_curve``.

    Within a block of tied scores the sort order is arbitrary, so the walk is
    only read at block ends: the result does not depend on how ties are
    ordered in the input. ``presorted=True`` declares ``p`` ascending (the
    bootstrap fast path); nothing checks it.
    """
    if presorted:
        ys, ps, ws = y, p, w
    else:
        order = np.argsort(p, kind="stable")
        ys, ps, ws = y[order], p[order], w[order]
    w_total = w.sum()
    n = ps.shape[0]
    ends = np.append(np.flatnonzero(ps[1:] != ps[:-1]), n - 1)
    cumdev = (np.cumsum(ws * (ys - ps)) / w_total)[ends]
    cum_w = np.cumsum(ws)[ends]
    block_w = np.diff(cum_w, prepend=0.0)
    sd_null = None
    if sd:
        # Exact null SD of the walk: Var(sum w (y - p)) = sum w**2 p (1 - p).
        sd_null = np.sqrt(np.cumsum(ws * ws * ps * (1.0 - ps))[ends]) / w_total
    return _Walk(frac=cum_w / cum_w[-1], cumdev=cumdev, block_w=block_w, sd_null=sd_null)


def _ecce_stats(walk: _Walk) -> EcceResult:
    a = np.abs(walk.cumdev)
    return EcceResult(
        stat_max=float(np.max(a)), stat_mean=float(np.average(a, weights=walk.block_w))
    )


def ecce(y: object, p: object, *, sample_weight: object = None) -> EcceResult:
    """Cumulative-deviation calibration error (Arrieta-Ibarra et al., 2022).

    Sort by prediction and walk the cumulative sum of weighted residuals,
    normalized by the total weight; under calibration the walk hovers near
    zero, and drift localizes miscalibration without any smoothing
    parameter. The walk is read only at the end of each block of tied
    predictions, so the result does not depend on the input order of tied
    rows; ``stat_mean`` weights each block by its total sample weight.
    Without ties and with unit weights the walk is read after every row.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    EcceResult
        Max and mean absolute cumulative deviation.
    """
    return _ecce_stats(_ecce_walk(*_prep(y, p, sample_weight)))


def _ici_distances(
    y: np.ndarray, p: np.ndarray, frac: float, grid_size: int | None, *, presorted: bool = False
) -> np.ndarray:
    """``|LOESS(y | p) - p|`` per observation (the LOESS fit is unweighted)."""
    return np.abs(loess(p, y, frac=frac, grid_size=grid_size, presorted=presorted) - p)


def _ici_family(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    names: Iterable[str],
    *,
    frac: float = 0.75,
    grid_size: int | None = 512,
    presorted: bool = False,
) -> dict[str, float]:
    """The ICI family off one shared LOESS fit.

    ``ici`` is the ``w``-weighted mean distance; ``e50``/``e90`` are the
    distance quantiles — ``np.quantile`` for all-equal weights (bit-identical
    to the unweighted values), otherwise
    :func:`~probcal._math.weighted_quantile`; ``emax`` is the maximum, which
    positive weights cannot move.
    """
    sel = set(names)
    d = _ici_distances(y, p, frac, grid_size, presorted=presorted)
    uniform = is_uniform(w)
    out: dict[str, float] = {}
    if "ici" in sel:
        out["ici"] = float(np.average(d, weights=w))
    for name, q in (("e50", 0.5), ("e90", 0.9)):
        if name in sel:
            out[name] = float(np.quantile(d, q)) if uniform else float(weighted_quantile(d, q, w))
    if "emax" in sel:
        out["emax"] = float(np.max(d))
    return out


def _ici_public(
    name: str, y: object, p: object, frac: float, sample_weight: object, grid_size: int | None
) -> float:
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    return _ici_family(y_arr, p_arr, w, (name,), frac=frac, grid_size=grid_size)[name]


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

    The LOESS stage itself is unweighted.
    ``grid_size=None`` fits the LOESS exactly at every prediction.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``; weights only the
        final averaging step, not the LOESS fit.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` fits exactly at every prediction.

    Returns
    -------
    float
        Weighted mean absolute LOESS-to-prediction distance.
    """
    return _ici_public("ici", y, p, frac, sample_weight, grid_size)


def e50(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Median of the |LOESS(y|p) - p| distances.

    ``grid_size=None`` fits the LOESS exactly at every prediction. The LOESS distances are
    always unweighted; ``sample_weight``, when given and
    not uniform, weights only the quantile step (see
    :func:`~probcal._math.weighted_quantile`).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional positive weights; used only for the quantile step.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` fits exactly at every prediction.

    Returns
    -------
    float
        Median of the LOESS distances.
    """
    return _ici_public("e50", y, p, frac, sample_weight, grid_size)


def e90(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """90th percentile of the |LOESS(y|p) - p| distances.

    ``grid_size=None`` fits the LOESS exactly at every prediction. The LOESS distances are
    always unweighted; ``sample_weight``, when given and
    not uniform, weights only the quantile step (see
    :func:`~probcal._math.weighted_quantile`).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional positive weights; used only for the quantile step.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` fits exactly at every prediction.

    Returns
    -------
    float
        90th percentile of the LOESS distances.
    """
    return _ici_public("e90", y, p, frac, sample_weight, grid_size)


def emax(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> float:
    """Maximum of the |LOESS(y|p) - p| distances.

    ``grid_size=None`` fits the LOESS exactly at every prediction.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, validated like everywhere else. The
        weighted maximum over strictly positive weights is the plain maximum,
        so the value does not depend on them.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` fits exactly at every prediction.

    Returns
    -------
    float
        Maximum of the LOESS distances.
    """
    return _ici_public("emax", y, p, frac, sample_weight, grid_size)


@dataclass(frozen=True)
class SpiegelhalterResult:
    """Spiegelhalter's z test of forecast unbiasedness (two-sided).

    Attributes
    ----------
    z : float
        Standardized test statistic (``nan`` when its null variance is 0).
    p_value : float
        Two-sided p-value under the standard normal approximation (``nan``
        when ``z`` is).
    """

    z: float
    p_value: float


def spiegelhalter_z(y: object, p: object, *, sample_weight: object = None) -> SpiegelhalterResult:
    """Spiegelhalter (1986) z statistic built on the Brier score.

    The numerator has expectation zero under calibration; the statistic is
    asymptotically standard normal. No binning, no smoothing; aggregates the
    whole range, so compensating regional errors can cancel.

    With weights the numerator is ``sum(w (y - p) (1 - 2p))`` and the null
    variance its exact variance under calibration, ``sum(w**2 (1 - 2p)^2 p
    (1 - p))`` (package convention, :mod:`probcal.metrics._common`): ``z``
    is invariant to rescaling the weights and unit weights give the
    unweighted statistic. A Kish-rescaled effective sample size is not used
    because it misstates the variance when the weights vary with ``p (1 - p)``.

    The null variance is zero when every prediction is exactly 0.5 (or
    clipped to 0/1); the statistic is then undefined and ``z`` and
    ``p_value`` are ``nan``.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    SpiegelhalterResult
        Z statistic and two-sided p-value.
    """
    return _spiegelhalter(*_prep(y, p, sample_weight))


def _spiegelhalter(y_arr: np.ndarray, p_arr: np.ndarray, w: np.ndarray) -> SpiegelhalterResult:
    """:func:`spiegelhalter_z` on validated arrays."""
    num = float(np.sum(w * (y_arr - p_arr) * (1.0 - 2.0 * p_arr)))
    var = float(np.sum(w * w * (1.0 - 2.0 * p_arr) ** 2 * p_arr * (1.0 - p_arr)))
    if not var > 0.0:
        return SpiegelhalterResult(z=math.nan, p_value=math.nan)
    z = num / math.sqrt(var)
    return SpiegelhalterResult(z=z, p_value=min(1.0, 2.0 * norm_sf(abs(z))))
