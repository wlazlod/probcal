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
from ._common import _prep, effective_weights, is_uniform
from ._deprecation import UNSET, deprecated


def _smece_at_sigma(loc: np.ndarray, mass: np.ndarray, sigma: float) -> float:
    """smECE at bandwidth sigma: integral of |kernel-smoothed signed residual measure|."""
    grid = np.linspace(loc.min() - 5.0 * sigma, loc.max() + 5.0 * sigma, 257)
    diff = (grid[:, None] - loc[None, :]) / sigma
    kern = np.exp(-0.5 * diff**2) / (sigma * math.sqrt(2.0 * math.pi))
    return float(np.trapezoid(np.abs(kern @ mass), grid))


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
    sigma), which replaces the aliasing-prone coarse-grid evaluation that made
    the pre-fix binned path spuriously report ~0 at small sigma.
    """
    mc, w2 = _coarsen(m, width, sigma)
    if 5.0 * sigma <= w2:  # isolated masses: integral is the total variation
        return float(np.sum(np.abs(mc)))
    f = np.convolve(mc, _gauss_taps(sigma, w2), mode="full")  # spans +-k cells beyond
    return float(w2 * np.sum(np.abs(f)))


def _smece_fixed_point_lattice(m: np.ndarray, width: float) -> tuple[float, float]:
    """Lattice-path fixed point (same bisection as the exact path)."""
    return _bisect_fixed_point(lambda sigma: _smece_at_sigma_lattice(m, width, sigma))


def _lattice(t: np.ndarray, bins: int) -> tuple[np.ndarray, float, float]:
    """Equal-width logit lattice: per-sample bin index, bin width, and lower
    edge over ``[t.min(), t.max()]``.

    Shared by ``smooth_ece``'s binned solve (via ``_smece_solve``) and
    ``curves.reliability_smooth`` so the two lattices cannot drift apart.
    """
    t_lo, t_hi = float(t.min()), float(t.max())
    width = (t_hi - t_lo) / bins
    idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, bins - 1)
    return idx, width, t_lo


def _smece_solve(
    t: np.ndarray, mass: np.ndarray, bins: int | None
) -> tuple[float, float, np.ndarray | None, float, float, int | None]:
    """Core smECE fixed-point solve behind ``smooth_ece``, exposing the
    lattice state a caller needs to build a numerically consistent curve.

    Literally the pre-refactor body of ``smooth_ece`` (path selection at
    module docstring / historical lines ~152-172), factored out so
    ``curves.reliability_smooth`` can share it without re-deriving
    ``sigma_star`` — the two are therefore identical by construction, not by
    agreement.

    Returns
    -------
    value : float
        The smECE value (``smooth_ece``'s return value).
    sigma : float
        The fixed-point bandwidth ``sigma_star``.
    m : numpy.ndarray or None
        The binned residual-mass vector actually solved on, when the
        lattice path was used; ``None`` when the exact O(n) path ran
        instead (``bins=None``, a degenerate logit range, or an
        infeasible/under-resolved refinement) — a caller then has to fall
        back to direct O(n * grid) kernel smoothing at ``sigma``.
    width : float
        Bin width of ``m``'s lattice (``0.0`` when ``m`` is ``None``).
    t_lo : float
        Lower edge of the logit lattice (``t.min()``), always meaningful.
    b : int or None
        Bin count of ``m``'s lattice (``None`` when ``m`` is ``None``).
    """
    t_lo, t_hi = float(t.min()), float(t.max())
    if bins is None or t_hi == t_lo:
        value, sigma = _smece_fixed_point(t, mass)
        return value, sigma, None, 0.0, t_lo, None

    def _binned_solve(b: int) -> tuple[float, float, np.ndarray, float]:
        idx, width, _ = _lattice(t, b)
        m = np.bincount(idx, weights=mass, minlength=b)
        value, sigma = _smece_fixed_point_lattice(m, width)
        return value, sigma, m, width

    value, sigma, m, width = _binned_solve(bins)
    if sigma >= 8.0 * width:
        return value, sigma, m, width, t_lo, bins
    # Under-resolved: one adaptive refinement sized so 8 bins span sigma.
    b2 = math.ceil((t_hi - t_lo) / (sigma / 8.0))
    if b2 > _SMECE_MAX_BINS:
        value, sigma = _smece_fixed_point(t, mass)  # refinement infeasible: exact
        return value, sigma, None, 0.0, t_lo, None
    value, sigma, m, width = _binned_solve(b2)
    if sigma >= 8.0 * width:
        return value, sigma, m, width, t_lo, b2
    value, sigma = _smece_fixed_point(t, mass)  # O(n) worst case, no warning
    return value, sigma, None, 0.0, t_lo, None


def _lattice_kernel_smooth(
    m: np.ndarray, width: float, sigma: float, t_lo: float
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated-Gaussian convolution of a lattice-binned mass vector,
    returning the pointwise smoothed values (not just their integral).

    Coarsening (mass-conserving factor ``max(1, int(sigma/(8*width)))``) and
    kernel truncation (+-5 sigma) mirror ``_smece_at_sigma_lattice``'s
    general branch exactly, so ``curves.reliability_smooth``'s kernel
    matches the one that produced ``sigma_star``. Unlike
    ``_smece_at_sigma_lattice``, this always convolves (no isolated-mass
    total-variation shortcut): a curve needs the smoothed value at every
    point, not only the integral of its absolute value.

    Parameters
    ----------
    m : numpy.ndarray
        Lattice-binned mass vector (e.g. ``bincount(w)`` or
        ``bincount(w * y)``), on the same lattice ``sigma_star`` was solved
        on.
    width : float
        That lattice's bin width.
    sigma : float
        Kernel bandwidth (``sigma_star``).
    t_lo : float
        Lower edge of the (uncoarsened) lattice.

    Returns
    -------
    centers, smoothed : numpy.ndarray
        Logit-scale centers of the (possibly coarsened) lattice cells, and
        the kernel-smoothed value at each center, in ``m``'s own mass-per-
        cell scale (a ratio of two such vectors, e.g. ``num / den``, cancels
        the coarsening factor and is scale-free).
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

    ``bins`` pre-aggregates the weighted residual measure onto a regular grid
    over the logit range before solving the fixed point; the binned measure
    is then evaluated in closed form on its own lattice by direct Gaussian
    convolution, at a cost independent of n and of sigma. The lattice path
    engages for every call with a non-degenerate logit range
    (0.1.3 engaged it only for ``n > bins``, leaving typical calibration-set
    sizes on the exact O(n)-per-step path — the "size cliff").
    With ``bins=None``, or a degenerate range
    (``t.max() == t.min()``), the exact 0.1.2 computation runs bit-for-bit.
    Otherwise, if the found ``sigma`` is smaller than 8 bin widths (the
    kernel would be under-resolved by the bins), the solve is repeated once
    on an adaptively refined binning (``bins <- ceil(range / (sigma/8))``);
    the exact computation is used only when that refinement is infeasible
    (refined bin count above ``2**20``) or still under-resolved — reachable
    for near-perfectly-calibrated data spread over a wide logit range (e.g.
    extreme/clipped scores), so the worst case matches the pre-0.1.3 O(n)
    cost. For ``n <= bins`` the lattice value may differ from the exact
    grid at the ~1e-4 level on typical portfolios (measured <= 2.4e-4 on
    ``make_pd_portfolio``); on wide clipped-logit-range data the gap can be
    much larger because the exact path's fixed 257-point grid under-resolves
    small-sigma kernels there — in that regime the lattice value is the
    better one (>= 8 samples per sigma). ``bins=None`` recovers the old
    values.

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
    """Pointwise SD of the walk under calibration (Kish-rescaled weights), if asked."""


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
        we = effective_weights(ws)
        sd_null = np.sqrt(np.cumsum(we * ps * (1.0 - ps))[ends]) / we.sum()
    return _Walk(frac=cum_w / cum_w[-1], cumdev=cumdev, block_w=block_w, sd_null=sd_null)


def _ecce_stats(walk: _Walk) -> EcceResult:
    a = np.abs(walk.cumdev)
    return EcceResult(
        stat_max=float(np.max(a)), stat_mean=float(np.average(a, weights=walk.block_w))
    )


def ecce(
    y: object, p: object, *, sample_weight: object = None, presorted: object = UNSET
) -> EcceResult:
    """Cumulative-deviation calibration error (Arrieta-Ibarra et al., 2022).

    Sort by prediction and walk the cumulative sum of weighted residuals,
    normalized by the total weight; under calibration the walk hovers near
    zero, and drift localizes miscalibration without any smoothing
    parameter. The walk is read only at the end of each block of tied
    predictions, so the result does not depend on the input order of tied
    rows; ``stat_mean`` weights each block by its total sample weight.
    Without ties and with unit weights both statistics equal the 0.3.x values.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
    presorted : bool, keyword-only
        Deprecated (removed in 0.4.0): a throughput switch declaring ``p``
        already sorted ascending. Still honoured.

    Returns
    -------
    EcceResult
        Max and mean absolute cumulative deviation.
    """
    if presorted is not UNSET:
        deprecated("ecce(presorted=...) is internal; drop the argument")
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    return _ecce_stats(_ecce_walk(y_arr, p_arr, w, presorted=bool(presorted)))


_ICI_FAMILY = ("ici", "e50", "e90", "emax")


def _ici_distances(
    y: np.ndarray, p: np.ndarray, frac: float, grid_size: int | None, *, presorted: bool = False
) -> np.ndarray:
    """``|LOESS(y | p) - p|`` per observation (the LOESS fit is unweighted)."""
    return np.abs(loess(p, y, frac=frac, grid_size=grid_size, presorted=presorted) - p)


def _ici_family(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray | None,
    names: Iterable[str],
    *,
    frac: float = 0.75,
    grid_size: int | None = 512,
    presorted: bool = False,
) -> dict[str, float]:
    """The ICI family off one shared LOESS fit.

    ``ici`` is the ``w``-weighted mean distance; ``e50``/``e90`` are the
    distance quantiles — ``np.quantile`` for ``w=None`` or all-equal weights
    (bit-identical to the unweighted values), otherwise
    :func:`~probcal._math.weighted_quantile`; ``emax`` is the maximum, which
    positive weights cannot move.
    """
    sel = set(names)
    d = _ici_distances(y, p, frac, grid_size, presorted=presorted)
    uniform = is_uniform(w)
    out: dict[str, float] = {}
    if "ici" in sel:
        out["ici"] = float(np.average(d, weights=np.ones(len(p)) if w is None else w))
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
    wq = None if sample_weight is None else w
    return _ici_family(y_arr, p_arr, wq, (name,), frac=frac, grid_size=grid_size)[name]


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
        Optional positive weights, same length as ``y``; weights only the
        final averaging step, not the LOESS fit.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

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

    ``grid_size=None`` recovers 0.1.2 values exactly. The LOESS distances are
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
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

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

    ``grid_size=None`` recovers 0.1.2 values exactly. The LOESS distances are
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
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

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
        Optional positive weights, validated like everywhere else. The
        weighted maximum over strictly positive weights is the plain maximum,
        so the value does not depend on them.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size; ``None`` recovers 0.1.2 values exactly.

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

    With weights, numerator and null variance use the Kish-rescaled weights
    (package convention, :mod:`probcal.metrics._common`): ``z`` is invariant
    to rescaling the weights and unit weights give the unweighted statistic.
    (0.3.x used ``w`` in the numerator and ``w**2`` in the variance.)

    The null variance ``sum(w (1 - 2p)^2 p (1 - p))`` is zero when every
    prediction is exactly 0.5 (or clipped to 0/1); the statistic is then
    undefined and ``z`` and ``p_value`` are ``nan``.

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
    we = effective_weights(w)
    num = float(np.sum(we * (y_arr - p_arr) * (1.0 - 2.0 * p_arr)))
    var = float(np.sum(we * (1.0 - 2.0 * p_arr) ** 2 * p_arr * (1.0 - p_arr)))
    if not var > 0.0:
        return SpiegelhalterResult(z=math.nan, p_value=math.nan)
    z = num / math.sqrt(var)
    return SpiegelhalterResult(z=z, p_value=min(1.0, 2.0 * norm_sf(abs(z))))
