"""Reliability-curve builders and the GiViTI-style calibration belt.

Numpy-only; every result is a frozen dataclass carrying both probability- and
logit-scale coordinates, plotting-backend-agnostic (rendering lives in
``probcal.plots``). Theory: ``docs/concepts/visualization.md``.

References
----------
Austin & Steyerberg (2014); Nattino, Finazzi & Bertolini (2014); Nattino,
Lemeshow, Phillips, Finazzi & Bertolini (2017) — full records in the
documentation. The belt is reimplemented from the papers; no GPL code is used.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ._corp import corp_bands, corp_fit, decompose
from ._math import bern_loglik, chi2_ppf, chi2_sf, expit, irls_logistic, loess, logit
from ._results import (
    BeltResult,
    CorpResult,
    KernelReliabilityCurve,
    ReliabilityCurve,
    SmoothReliabilityCurve,
    _ResultBase,
)
from ._validation import validate_level, validate_nonneg_int, validate_positive_int
from .metrics._binstats import bin_stats
from .metrics._common import _prep, gram, is_uniform, rao_scott
from .metrics.smooth import (
    Lattice,
    _ecce_walk,
    _lattice_index,
    _lattice_kernel_smooth,
    _smece_solve,
)

_Z_95 = 1.959963984540054


def _wilson(rate: np.ndarray, n: np.ndarray, z: float = _Z_95) -> tuple[np.ndarray, np.ndarray]:
    denom = 1.0 + z**2 / n
    center = (rate + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(rate * (1.0 - rate) / n + z**2 / (4.0 * n**2)) / denom
    return center - half, center + half


def reliability_binned(
    y: object,
    p: object,
    *,
    n_bins: int = 10,
    strategy: str = "mass",
    sample_weight: object = None,
) -> ReliabilityCurve:
    """Binned reliability curve with Wilson confidence intervals.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    n_bins : int
        Requested bin count.
    strategy : {"mass", "width"}
        Equal-count (default) or equal-width bins.
    sample_weight : array_like or None
        Weights for the bin means; Wilson CIs use raw counts.

    Returns
    -------
    ReliabilityCurve
        Per-bin mean prediction, event rate, count, Wilson CI, and the
        logit-scale coordinates.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    bs = bin_stats(y_arr, p_arr, w, n_bins, strategy)
    counts = bs.n_obs
    pred_mean = bs.p_mean
    event_rate = bs.rate
    ci_low, ci_high = _wilson(event_rate, counts.astype(np.float64))
    # The Wilson interval contains the point estimate analytically; enforce it
    # against floating-point noise at 0/1-rate bins (negative yerr otherwise).
    ci_low = np.minimum(np.clip(ci_low, 0.0, 1.0), event_rate)
    ci_high = np.maximum(np.clip(ci_high, 0.0, 1.0), event_rate)
    return ReliabilityCurve(
        pred_mean=pred_mean,
        event_rate=event_rate,
        count=counts.astype(np.int64),
        ci_low=ci_low,
        ci_high=ci_high,
        pred_mean_logit=logit(pred_mean),
    )


def _grid(p: np.ndarray, grid_size: int) -> np.ndarray:
    return np.linspace(float(np.quantile(p, 0.005)), float(np.quantile(p, 0.995)), grid_size)


def reliability_loess(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    grid_size: int = 100,
    sample_weight: object = None,
) -> SmoothReliabilityCurve:
    """LOESS-smoothed reliability curve on a grid (Austin & Steyerberg, 2014).

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    frac : float, keyword-only
        LOESS smoothing fraction.
    grid_size : int, keyword-only
        Number of evaluation points, spanning the 0.5th to 99.5th percentile
        of ``p``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights. Non-uniform weights multiply the tricube
        kernel weights of each local-linear fit (same nearest-neighbour
        windows); ``None`` or all-equal weights give the unweighted LOESS
        curve unchanged.

    Returns
    -------
    SmoothReliabilityCurve
        Grid coordinates (probability and logit scale) and the smoothed
        event rate at each point.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    grid = _grid(p_arr, grid_size)
    if is_uniform(w):
        fitted = loess(p_arr, y_arr, frac=frac, xeval=grid)
    else:
        fitted = _weighted_local_linear(p_arr, y_arr, w, grid, frac)
    rate = np.clip(fitted, 0.0, 1.0)
    return SmoothReliabilityCurve(grid_p=grid, grid_logit=logit(grid), event_rate=rate)


def _weighted_local_linear(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, xeval: np.ndarray, frac: float
) -> np.ndarray:
    """Sample-weighted LOESS (degree 1) at ``xeval``.

    Each fit uses the ``r = ceil(frac * n)`` nearest observations, tricube
    kernel weights over the window radius times the sample weights, and a
    weighted least-squares line; a window whose points share one ``x`` (or
    whose design is singular) returns the weighted mean.
    """
    n = x.shape[0]
    r = min(max(int(math.ceil(frac * n)), 2), n)
    out = np.empty(xeval.shape[0])
    for i, x0 in enumerate(xeval):
        dist = np.abs(x - x0)
        win = np.argpartition(dist, r - 1)[:r]
        h = float(dist[win].max())
        u = dist[win] / h if h > 0.0 else np.zeros(r)
        k = np.clip(1.0 - u**3, 0.0, None) ** 3 * w[win]
        xc = x[win] - x0
        sw, swx, swxx = k.sum(), (k * xc).sum(), (k * xc * xc).sum()
        swy, swxy = (k * y[win]).sum(), (k * xc * y[win]).sum()
        det = sw * swxx - swx * swx
        if h == 0.0 or abs(det) <= 1e-12 * max(sw * swxx, 1e-300):
            out[i] = swy / sw if sw > 0.0 else float(np.average(y[win], weights=w[win]))
        else:
            out[i] = (swxx * swy - swx * swxy) / det
    return out


def reliability_spline(
    y: object,
    p: object,
    *,
    grid_size: int = 100,
    sample_weight: object = None,
) -> SmoothReliabilityCurve:
    """Spline-smoothed reliability curve on a grid.

    Penalized natural cubic spline of the outcome on the logit prediction.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    grid_size : int, keyword-only
        Number of evaluation points, spanning the 0.5th to 99.5th percentile
        of ``p``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights passed to the spline fit.

    Returns
    -------
    SmoothReliabilityCurve
        Grid coordinates (probability and logit scale) and the smoothed
        event rate at each point.
    """
    from .spline import SplineCalibrator

    y_arr, p_arr, w = _prep(y, p, sample_weight)
    cal = SplineCalibrator()
    cal.fit(p_arr, y_arr, sample_weight=w)
    grid = _grid(p_arr, grid_size)
    return SmoothReliabilityCurve(
        grid_p=grid, grid_logit=logit(grid), event_rate=cal.predict_proba(grid)
    )


def _kernel_rate_density(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    sigma: float,
    grid_logit: np.ndarray,
    lattice: Lattice | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the Nadaraya-Watson kernel rate/density at ``sigma`` on
    ``grid_logit``, sharing the smECE lattice and kernel when available.

    ``lattice`` is ``(width, t_lo, b)`` from the point-estimate's
    ``_smece_solve`` (fixed across bootstrap resamples, per the module
    docstring of ``reliability_smooth``); ``None`` selects the exact
    O(n * grid) direct-smoothing path (mirrors ``smooth_ece``'s own
    lattice/exact selection so the two stay consistent).
    """
    if lattice is not None:
        width, t_lo, b = lattice
        idx = _lattice_index(t, t_lo, width, b)
        num = np.bincount(idx, weights=w * y, minlength=b)
        den = np.bincount(idx, weights=w, minlength=b)
        centers, num_s = _lattice_kernel_smooth(num, width, sigma, t_lo)
        _, den_s = _lattice_kernel_smooth(den, width, sigma, t_lo)
        # Cells farther than the kernel's +-5 sigma truncation from every
        # observation have zero smoothed weight: no data, so no rate (NaN,
        # not 0, so a gap is not mistaken for an observed zero rate).
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_c = np.where(den_s > 0.0, num_s / den_s, np.nan)
        rate = np.interp(grid_logit, centers, rate_c)
        density = np.interp(grid_logit, centers, den_s)
    else:
        diff = (grid_logit[:, None] - t[None, :]) / sigma
        taps = np.exp(-0.5 * diff**2) / (sigma * math.sqrt(2.0 * math.pi))
        num = taps @ (w * y)
        den = taps @ w
        # Same gap rule as the lattice path: no observation within 5 sigma.
        ts = np.sort(t)
        j = np.clip(np.searchsorted(ts, grid_logit), 1, len(ts) - 1) if len(ts) > 1 else None
        if j is None:
            near = np.abs(grid_logit - ts[0])
        else:
            near = np.minimum(np.abs(grid_logit - ts[j - 1]), np.abs(grid_logit - ts[j]))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where((near <= 5.0 * sigma) & (den > 0.0), num / den, np.nan)
        density = den
    density = np.clip(density, 0.0, None)
    density = density / density.sum()
    return rate, density


def reliability_smooth(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    grid_size: int = 200,
    n_boot: int = 100,
    confidence: float = 0.9,
    random_state: int = 42,
    bins: int | None = 8192,
) -> KernelReliabilityCurve:
    """smECE-consistent kernel reliability curve (Blasiok-Nakkiran).

    Shares its bandwidth and lattice with ``metrics.smooth_ece``: both solve
    the same fixed point ``sigma_star`` on the same equal-width logit
    lattice (``metrics.smooth._smece_solve``), so
    ``curve.smooth_ece`` reproduces ``metrics.smooth_ece(y, p, bins=bins)``
    exactly instead of merely agreeing with it. The event rate and
    prediction density are then Nadaraya-Watson kernel estimates at that one
    fixed ``sigma_star`` — ``rate = K*bincount(w*y) / K*bincount(w)`` on the
    lattice, interpolated onto ``grid_logit`` — using the same truncated
    Gaussian kernel ``smooth_ece`` used to reach ``sigma_star``
    (``metrics.smooth._lattice_kernel_smooth``). When ``smooth_ece``'s path
    selection falls back to its exact (non-lattice) computation — degenerate
    logit range, ``bins=None``, or an infeasible/under-resolved refinement —
    the curve falls back the same way, to direct O(n * grid_size) Gaussian
    smoothing on ``logit(p)`` at ``sigma_star``.

    The confidence ribbon bootstraps ``(y, p, sample_weight)`` triples
    (``numpy.random.default_rng(random_state)``, resampling with
    replacement) and recomputes the rate at the point estimate's *fixed*
    ``sigma_star`` — the ribbon conditions on the bandwidth, it does not
    reflect uncertainty in choosing it. The ribbon is clamped to contain the
    point estimate (``ci_low <= event_rate <= ci_high``), so a bootstrap
    quantile falling on the wrong side of it is pulled back to it.
    ``n_boot=0`` disables the ribbon (``ci_low`` and ``ci_high`` both equal
    ``event_rate``).

    Grid points with no observation within the kernel's ``+-5 sigma_star``
    reach (data gaps, on the logit scale) have no estimate: ``event_rate``,
    ``ci_low`` and ``ci_high`` are ``nan`` there. Bootstrap resamples that leave a grid point
    uncovered are skipped in that point's quantiles.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
    grid_size : int, keyword-only
        Number of evaluation points, spanning the 0.5th to 99.5th percentile
        of ``p`` (``curves._grid``).
    n_boot : int, keyword-only
        Number of bootstrap resamples for the confidence ribbon; ``0``
        disables it.
    confidence : float, keyword-only
        Nominal coverage level of the ribbon; must satisfy
        ``0 < confidence < 1``.
    random_state : int, keyword-only
        Seed for ``numpy.random.default_rng``, used by the bootstrap.
    bins : int or None, keyword-only
        Lattice bin count passed through to the shared smECE solve; see
        ``metrics.smooth_ece``. ``None`` forces the exact path.

    Returns
    -------
    KernelReliabilityCurve
        Grid coordinates, kernel-smoothed event rate and density, the
        bootstrap ribbon, and the shared ``sigma_star`` / ``smooth_ece``.

    Raises
    ------
    ValueError
        If ``confidence`` is not in ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal import make_pd_portfolio
    >>> from probcal.curves import reliability_smooth
    >>> d = make_pd_portfolio(n=2000, random_state=0)
    >>> curve = reliability_smooth(d.y, d.scores, n_boot=0)
    >>> len(curve.grid_p) == 200
    True
    >>> abs(float(curve.density.sum()) - 1.0) < 1e-10
    True
    """
    conf = validate_level(confidence, "confidence")
    n_boot = validate_nonneg_int(n_boot, "n_boot")
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    grid_p = _grid(p_arr, grid_size)
    grid_logit = logit(grid_p)
    t = logit(p_arr)
    mass = (w / w.sum()) * (y_arr - p_arr)
    smooth_ece_value, sigma_star, lattice = _smece_solve(t, mass, bins)

    event_rate, density = _kernel_rate_density(t, y_arr, w, sigma_star, grid_logit, lattice)

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        n = len(y_arr)
        boot_rate = np.empty((n_boot, grid_size))
        for i in range(n_boot):
            idx_b = rng.integers(0, n, n)
            # `lattice` is the point estimate's fixed (width, t_lo, b) — never
            # rederived here. The ribbon is meant to reflect uncertainty in the
            # rate given sigma_star, not uncertainty in sigma_star or its
            # lattice; re-solving the smECE fixed point per resample would also
            # make each resample's rate estimate use a different bandwidth and
            # bin grid, so resamples would stop being comparable pointwise.
            boot_rate[i], _ = _kernel_rate_density(
                t[idx_b], y_arr[idx_b], w[idx_b], sigma_star, grid_logit, lattice
            )
        a = (1.0 - conf) / 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            q_lo = np.nanquantile(boot_rate, a, axis=0)
            q_hi = np.nanquantile(boot_rate, 1.0 - a, axis=0)
        gap = np.isnan(event_rate)
        ci_low = np.where(gap, np.nan, np.fmin(q_lo, event_rate))
        ci_high = np.where(gap, np.nan, np.fmax(q_hi, event_rate))
    else:
        ci_low = event_rate.copy()
        ci_high = event_rate.copy()

    return KernelReliabilityCurve(
        grid_p=grid_p,
        grid_logit=grid_logit,
        event_rate=event_rate,
        density=density,
        ci_low=ci_low,
        ci_high=ci_high,
        sigma_star=sigma_star,
        smooth_ece=smooth_ece_value,
    )


_BANDS = ("consistency", "confidence", None)


def corp_reliability(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    bands: str | None = "consistency",
    confidence: float = 0.9,
    n_resamples: int = 200,
    random_state: int = 42,
) -> CorpResult:
    """CORP reliability diagram with the Brier/log-loss MCB-DSC-UNC decomposition.

    Fits the isotonic (PAV) recalibration map of ``y`` on ``p`` — the unique
    "consistent, optimally binned, reproducible" reliability diagram of
    Dimitriadis, Gneiting & Jordan (2021) — and decomposes both the Brier
    score and log loss into miscalibration (MCB), discrimination (DSC), and
    uncertainty (UNC) terms, with ``score == mcb - dsc + unc`` holding
    exactly. Log loss clips PAV levels and predictions to
    ``[1e-12, 1 - 1e-12]`` before taking logarithms, so degenerate blocks
    (exact 0 or 1 event rate) stay finite.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
    bands : {"consistency", "confidence", None}, keyword-only
        Band type to compute around the PAV fit. ``"consistency"`` resamples
        ``y ~ Bernoulli(p)`` under the null that ``p`` is calibrated;
        ``"confidence"`` bootstraps ``(y, p, sample_weight)`` triples. Both
        give pointwise, not simultaneous, bands (see Notes).
    confidence : float, keyword-only
        Nominal coverage level of the bands; must satisfy
        ``0 < confidence < 1`` (stored as ``CorpResult.level``).
    n_resamples : int, keyword-only
        Number of resamples used to build the bands; ``0`` skips them (as
        ``bands=None``).
    random_state : int, keyword-only
        Seed for ``numpy.random.default_rng``, used by the band resampling.

    Returns
    -------
    CorpResult
        PAV block structure, the pointwise fit, the Brier/log-loss
        decomposition, and the (possibly empty) bands.

    Raises
    ------
    ValueError
        If ``bands`` is not one of ``"consistency"``, ``"confidence"``, or
        ``None``, or if ``confidence`` is not in ``(0, 1)``.

    Notes
    -----
    Bands are pointwise: at each grid point, ``confidence`` of resamples fall
    inside, not that the whole curve does so simultaneously (the
    ``docs/scripts/corp_sim.py`` coverage simulation reports the gap between
    pointwise and uniform coverage). ``corp_reliability`` with
    ``n=10_000, n_resamples=200`` takes about 3.5 s (measured once on the
    development machine) — the PAV step is a Python loop over unique scores
    (``_math.pava``), and the bands refit PAV ``n_resamples`` times.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal import corp_reliability
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.1, 0.9, 200)
    >>> y = (rng.random(200) < p).astype(float)
    >>> r = corp_reliability(y, p, bands=None)
    >>> abs(r.brier - (r.brier_mcb - r.brier_dsc + r.brier_unc)) < 1e-12
    True
    """
    if bands not in _BANDS:
        raise ValueError('bands must be "consistency", "confidence", or None')
    conf = validate_level(confidence, "confidence")
    n_resamples = validate_nonneg_int(n_resamples, "n_resamples")
    if n_resamples == 0:
        bands = None
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    lo, hi, level_b, w_b, pav = corp_fit(y_arr, p_arr, w)
    b = decompose(y_arr, p_arr, pav, w, "brier")
    ll = decompose(y_arr, p_arr, pav, w, "log_loss")
    grid, low, high = corp_bands(y_arr, p_arr, w, bands, conf, n_resamples, random_state)
    return CorpResult(
        block_lo=lo,
        block_hi=hi,
        block_level=level_b,
        block_weight=w_b,
        pav=pav,
        brier=b[0],
        brier_mcb=b[1],
        brier_dsc=b[2],
        brier_unc=b[3],
        log_loss=ll[0],
        log_loss_mcb=ll[1],
        log_loss_dsc=ll[2],
        log_loss_unc=ll[3],
        bands=bands,
        level=conf,
        band_grid=grid,
        band_low=low,
        band_high=high,
        n=int(len(y_arr)),
        events=int(y_arr.sum()),
    )


@dataclass(frozen=True, eq=False)
class EcceCurve(_ResultBase):
    """Cumulative-deviation walk over predictions sorted ascending (ECCE).

    Compares by value (arrays element-wise); unhashable.

    Attributes
    ----------
    frac : numpy.ndarray
        Cumulative weight fraction at the end of each block of tied
        predictions (``1/n .. 1`` for distinct scores and unit weights).
    cumdev : numpy.ndarray
        Cumulative-deviation walk value at each ``frac``.
    sd_null : numpy.ndarray
        Pointwise standard deviation of the walk under calibration.
    stat_max : float
        Maximum absolute value of ``cumdev`` (agrees with ``metrics.ecce``'s
        ``stat_max``).
    argmax_frac : float
        ``frac`` at which the maximum is attained.
    """

    frac: np.ndarray
    cumdev: np.ndarray
    sd_null: np.ndarray
    stat_max: float
    argmax_frac: float


def ecce_curve(y: object, p: object, *, sample_weight: object = None) -> EcceCurve:
    """Cumulative-deviation walk for the ECCE plot (Arrieta-Ibarra et al., 2022).

    Sorts by prediction and accumulates weighted residuals with the same
    helper as ``metrics.ecce``, so ``stat_max`` agrees with the metric. The
    walk is reported only at the end of each block of tied predictions (the
    order of tied rows is arbitrary, so intermediate points carry no
    information); ``frac`` is the cumulative weight fraction. ``sd_null`` is
    the pointwise standard deviation of the walk under calibration — an
    envelope for reading, not a simultaneous band — the exact null SD
    ``sqrt(cumsum(w**2 p (1 - p))) / sum(w)`` of the weighted walk
    (``sqrt(cumsum(p (1 - p))) / n`` for unit weights; invariant to
    rescaling the weights).

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    EcceCurve
        Cumulative walk, null-envelope SD, and the max-deviation summary.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    walk = _ecce_walk(y_arr, p_arr, w, sd=True)
    assert walk.sd_null is not None
    k = int(np.argmax(np.abs(walk.cumdev)))
    return EcceCurve(
        frac=walk.frac,
        cumdev=walk.cumdev,
        sd_null=walk.sd_null,
        stat_max=float(np.abs(walk.cumdev[k])),
        argmax_frac=float(walk.frac[k]),
    )


def calibration_belt(
    y: object,
    p: object,
    *,
    confidence: tuple[float, float] = (0.8, 0.95),
    grid_size: int = 100,
    sample_weight: object = None,
) -> BeltResult:
    """GiViTI-style calibration belt (Nattino et al., 2014, 2017).

    Fits a polynomial logistic recalibration of the outcome on
    ``logit(p)``, selecting the degree by forward likelihood-ratio testing
    (p < 0.05 to add a term, capped at degree 4). The polynomial is fitted on
    the centred and scaled logit (same model, better-conditioned design).

    Both outputs are approximations of the published construction:

    * **Belt.** At each confidence level the band is the image of the Wald
      (information-matrix) ellipsoid with radius ``chi2_ppf(level, degree +
      1)`` — a quadratic approximation of the likelihood-ratio confidence
      region the papers invert. Using the ``degree + 1``-df radius makes it
      a *simultaneous* (Scheffé-type) band over the grid for the selected
      polynomial, not a pointwise one; it conditions on the selected degree.
    * **p-value.** The likelihood-ratio test of the selected polynomial
      against the identity, referred to ``chi2(degree + 1)``; it ignores the
      forward selection of the degree, whose exact null distribution
      (Nattino et al., 2014) puts more mass in the upper tail, so the
      p-value is anti-conservative.

    Where the band excludes the diagonal, the data reject calibration in
    that region. With non-uniform weights (package convention,
    :mod:`probcal.metrics._common`) the fits maximise the weighted
    likelihood, the band uses the sandwich covariance ``I**-1 J I**-1``
    with ``J = X' diag(w**2 mu (1 - mu)) X`` (the exact variance of the
    weighted score under the fitted model), the forward-selection steps
    divide each 1-df LR statistic by the added coefficient's design effect
    (sandwich over model-based variance), and the p-value is second-order
    Rao-Scott corrected (:func:`probcal.metrics._common.rao_scott`, its
    reference distribution ``chi2`` with the corrected, possibly fractional,
    degrees of freedom). All of it is invariant to rescaling the weights;
    unit weights give the unweighted belt unchanged.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    confidence : tuple of float, keyword-only
        The two (low, high) confidence levels for the bands, e.g. ``(0.8,
        0.95)``.
    grid_size : int, keyword-only
        Number of evaluation points, spanning the 0.5th to 99.5th percentile
        of ``p``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    BeltResult
        Grid coordinates, the band ``bands[level] = (lower, upper)`` for
        each of the two ``levels``, the selected polynomial degree, and the
        associated calibration-test p-value.
    """
    levels = tuple(float(c) for c in confidence)
    if len(levels) != 2 or not all(0.0 < c < 1.0 for c in levels):
        raise ValueError("confidence must be two levels in (0, 1)")
    grid_size = validate_positive_int(grid_size, "grid_size")
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    # Mean-one weights (unit weights unchanged) keep the ridge and the fit
    # tolerances on the same scale whatever the weights' scale.
    w = w / float(np.mean(w))
    uniform = is_uniform(w)
    z = logit(p_arr)
    centre = float(np.average(z, weights=w))
    spread = float(np.sqrt(np.average((z - centre) ** 2, weights=w)))
    if not spread > 0.0:
        spread = 1.0

    def design(deg: int, zz: np.ndarray) -> np.ndarray:
        t = (zz - centre) / spread
        return np.column_stack([t**k for k in range(deg + 1)])

    def information_inv(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Inverse of the (ridged) model-based information."""
        return np.linalg.inv(gram(X, w * mu * (1.0 - mu)) + 1e-10 * np.eye(X.shape[1]))

    def covariance(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Coefficient covariance: ``I**-1`` scaled to unit weights, or the sandwich."""
        info_inv = information_inv(X, mu)
        if uniform:
            return info_inv * float(w[0])
        return info_inv @ gram(X, w * w * mu * (1.0 - mu)) @ info_inv

    def step_stat(lr: float, X: np.ndarray, mu: np.ndarray) -> float:
        """1-df LR statistic for the last column, divided by its design effect."""
        if uniform:
            return lr / float(w[0])
        deff = covariance(X, mu)[-1, -1] / information_inv(X, mu)[-1, -1]
        return lr / deff if deff > 0.0 else lr

    # Forward LR selection of the polynomial degree. A separated fit's
    # coefficients come from the ridge fallback: usable as a terminal fit,
    # never a basis for extension.
    degree = 1
    fit = irls_logistic(design(1, z), y_arr, w=w)
    ll = bern_loglik(y_arr, expit(design(1, z) @ fit.beta), w)
    while degree < 4 and not fit.separation:
        X_cand = design(degree + 1, z)
        cand = irls_logistic(X_cand, y_arr, w=w)
        if cand.separation:
            break
        mu_cand = expit(X_cand @ cand.beta)
        ll_cand = bern_loglik(y_arr, mu_cand, w)
        if chi2_sf(step_stat(max(2.0 * (ll_cand - ll), 0.0), X_cand, mu_cand), 1.0) >= 0.05:
            break
        degree += 1
        fit, ll = cand, ll_cand

    # Associated calibration test: fitted polynomial vs the identity map.
    df = degree + 1
    X = design(degree, z)
    stat, df_rs = rao_scott(max(2.0 * (ll - bern_loglik(y_arr, p_arr, w)), 0.0), X, p_arr, w)
    p_value = chi2_sf(stat, df_rs)

    # Bands from the (sandwich) covariance ellipsoid.
    mu = expit(np.clip(X @ fit.beta, -30.0, 30.0))
    info_inv = covariance(X, mu)
    grid_p = _grid(p_arr, grid_size)
    grid_z = logit(grid_p)
    Xg = design(degree, grid_z)
    eta = Xg @ fit.beta
    se_sq = np.einsum("ij,jk,ik->i", Xg, info_inv, Xg)
    bands: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for conf in levels:
        radius = np.sqrt(chi2_ppf(conf, float(df)) * se_sq)
        bands[conf] = (expit(eta - radius), expit(eta + radius))
    return BeltResult(
        grid_p=grid_p,
        grid_logit=grid_z,
        levels=(levels[0], levels[1]),
        bands=bands,
        degree=degree,
        p_value=p_value,
    )
