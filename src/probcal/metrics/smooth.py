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
    factor = max(1, int(sigma / (8.0 * width)))
    if factor > 1:
        pad = (-m.shape[0]) % factor
        mp = np.concatenate([m, np.zeros(pad)]) if pad else m
        mc = mp.reshape(-1, factor).sum(axis=1)
        w2 = width * factor
    else:
        mc, w2 = m, width
    n = mc.shape[0]
    k = int(math.ceil(5.0 * sigma / w2))
    offs = np.arange(-k, k + 1) * w2
    taps = np.exp(-0.5 * (offs / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    smoothed = np.convolve(mc, taps, mode="same")
    centers = t_lo + (np.arange(n) + 0.5) * w2
    return centers, smoothed


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
    convolution, at a cost independent of n and of sigma. The lattice path
    engages for every call with a non-degenerate logit range
    (0.1.3 engaged it only for ``n > bins``, leaving typical calibration-set
    sizes on the exact O(n)-per-step path — the "size cliff", DECISIONS 68).
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
