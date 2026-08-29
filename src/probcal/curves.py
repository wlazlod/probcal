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
from dataclasses import dataclass

import numpy as np

from ._corp import corp_bands, corp_fit, decompose
from ._math import chi2_ppf, expit, gammainc_lower, irls_logistic, loess, logit
from ._results import (
    BeltResult,
    CorpResult,
    KernelReliabilityCurve,
    ReliabilityCurve,
    SmoothReliabilityCurve,
)
from .metrics.binned import _bin_index
from .metrics.scores import _prep
from .metrics.smooth import _lattice_kernel_smooth, _smece_solve

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
    idx, m = _bin_index(p_arr, n_bins, strategy)
    w_sum = np.bincount(idx, weights=w, minlength=m)
    keep = w_sum > 0
    wy = np.bincount(idx, weights=w * y_arr, minlength=m)[keep]
    wp = np.bincount(idx, weights=w * p_arr, minlength=m)[keep]
    counts = np.bincount(idx, minlength=m)[keep]
    w_kept = w_sum[keep]
    pred_mean = wp / w_kept
    event_rate = wy / w_kept
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
        Validated (must match ``y`` in length) but not used: the LOESS fit
        itself is unweighted.

    Returns
    -------
    SmoothReliabilityCurve
        Grid coordinates (probability and logit scale) and the smoothed
        event rate at each point.
    """
    y_arr, p_arr, _ = _prep(y, p, sample_weight)
    grid = _grid(p_arr, grid_size)
    rate = np.clip(loess(p_arr, y_arr, frac=frac, xeval=grid), 0.0, 1.0)
    return SmoothReliabilityCurve(grid_p=grid, grid_logit=logit(grid), event_rate=rate)


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
        Optional non-negative weights passed to the spline fit.

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
    lattice: tuple[float, float, int] | None,
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
        idx = np.clip(((t - t_lo) / width).astype(np.int64), 0, b - 1)
        num = np.bincount(idx, weights=w * y, minlength=b)
        den = np.bincount(idx, weights=w, minlength=b)
        centers, num_s = _lattice_kernel_smooth(num, width, sigma, t_lo)
        _, den_s = _lattice_kernel_smooth(den, width, sigma, t_lo)
        # An empty lattice cell has no data to average: its smoothed
        # numerator is 0 too, so dividing by 1 reports a rate of 0 there
        # rather than a NaN that np.interp would spread to its neighbours.
        den_safe = np.where(den_s > 0.0, den_s, 1.0)
        rate = np.interp(grid_logit, centers, num_s / den_safe)
        density = np.interp(grid_logit, centers, den_s)
    else:
        diff = (grid_logit[:, None] - t[None, :]) / sigma
        taps = np.exp(-0.5 * diff**2) / (sigma * math.sqrt(2.0 * math.pi))
        num = taps @ (w * y)
        den = taps @ w
        rate = num / den
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
    level: float = 0.9,
    random_state: int = 42,
    bins: int | None = 8192,
) -> KernelReliabilityCurve:
    """smECE-consistent kernel reliability curve (Blasiok-Nakkiran).

    Shares its bandwidth and lattice with ``metrics.smooth_ece``: both solve
    the same fixed point ``sigma_star`` on the same equal-width logit
    lattice (``metrics.smooth._lattice`` / ``_smece_solve``), so
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

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.
    grid_size : int, keyword-only
        Number of evaluation points, spanning the 0.5th to 99.5th percentile
        of ``p`` (``curves._grid``).
    n_boot : int, keyword-only
        Number of bootstrap resamples for the confidence ribbon; ``0``
        disables it.
    level : float, keyword-only
        Nominal coverage level of the ribbon; must satisfy ``0 < level < 1``.
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
        If ``level`` is not in ``(0, 1)``.

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
    if not (0.0 < level < 1.0):
        raise ValueError("level must satisfy 0 < level < 1")
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    grid_p = _grid(p_arr, grid_size)
    grid_logit = logit(grid_p)
    t = logit(p_arr)
    mass = (w / w.sum()) * (y_arr - p_arr)
    smooth_ece_value, sigma_star, m, width, t_lo, b = _smece_solve(t, mass, bins)
    lattice = None if m is None or b is None else (width, t_lo, b)

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
        a = (1.0 - level) / 2.0
        ci_low = np.minimum(np.quantile(boot_rate, a, axis=0), event_rate)
        ci_high = np.maximum(np.quantile(boot_rate, 1.0 - a, axis=0), event_rate)
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
    level: float = 0.9,
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
        Optional non-negative weights, same length as ``y``.
    bands : {"consistency", "confidence", None}, keyword-only
        Band type to compute around the PAV fit. ``"consistency"`` resamples
        ``y ~ Bernoulli(p)`` under the null that ``p`` is calibrated;
        ``"confidence"`` bootstraps ``(y, p, sample_weight)`` triples. Both
        give pointwise, not simultaneous, bands (see Notes).
    level : float, keyword-only
        Nominal coverage level of the bands; must satisfy ``0 < level < 1``.
    n_resamples : int, keyword-only
        Number of resamples used to build the bands.
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
        ``None``, or if ``level`` is not in ``(0, 1)``.

    Notes
    -----
    Bands are pointwise: at each grid point, ``level`` of resamples fall
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
    if not (0.0 < level < 1.0):
        raise ValueError("level must satisfy 0 < level < 1")
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    lo, hi, level_b, w_b, pav = corp_fit(y_arr, p_arr, w)
    b = decompose(y_arr, p_arr, pav, w, "brier")
    ll = decompose(y_arr, p_arr, pav, w, "log_loss")
    grid, low, high = corp_bands(y_arr, p_arr, w, bands, level, n_resamples, random_state)
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
        level=level,
        band_grid=grid,
        band_low=low,
        band_high=high,
        n=int(len(y_arr)),
        events=int(y_arr.sum()),
    )


@dataclass(frozen=True)
class EcceCurve:
    """Cumulative-deviation walk over predictions sorted ascending (ECCE).

    Attributes
    ----------
    frac : numpy.ndarray
        Cumulative fraction of observations, ``1/n .. 1``.
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

    Sorts by prediction and accumulates weighted residuals, mirroring
    ``metrics.ecce`` exactly so ``stat_max`` agrees with the metric.
    ``sd_null`` is the pointwise standard deviation of the walk under
    calibration — an envelope for reading, not a simultaneous band.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    EcceCurve
        Cumulative walk, null-envelope SD, and the max-deviation summary.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    order = np.argsort(p_arr, kind="stable")
    n = len(p_arr)
    wsum = w.sum()
    cumdev = np.cumsum(w[order] * (y_arr[order] - p_arr[order])) / wsum
    # Pointwise H0 SD; reduces to sqrt(cumsum(p(1-p)))/n for unit weights.
    sd_null = np.sqrt(np.cumsum(w[order] ** 2 * p_arr[order] * (1.0 - p_arr[order]))) / wsum
    frac = np.arange(1, n + 1) / n
    k = int(np.argmax(np.abs(cumdev)))
    return EcceCurve(
        frac=frac,
        cumdev=cumdev,
        sd_null=sd_null,
        stat_max=float(np.abs(cumdev[k])),
        argmax_frac=float(frac[k]),
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
    (p < 0.05 to add a term, capped at degree 4), then draws pointwise
    confidence bands from the information-matrix ellipsoid — a Wald
    approximation of the LR-region inversion. The
    associated p-value tests the fitted polynomial against the identity.
    Where the band excludes the diagonal, the data reject calibration in
    that region.

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
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    BeltResult
        Grid coordinates, both confidence bands, selected polynomial degree,
        and the associated calibration-test p-value.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    z = logit(p_arr)

    def design(deg: int, t: np.ndarray) -> np.ndarray:
        return np.column_stack([t**k for k in range(deg + 1)])

    def loglik(beta: np.ndarray, deg: int) -> float:
        prob = np.clip(expit(design(deg, z) @ beta), 1e-12, 1.0 - 1e-12)
        return float(np.sum(w * (y_arr * np.log(prob) + (1.0 - y_arr) * np.log1p(-prob))))

    # Forward LR selection of the polynomial degree. A separated fit's
    # coefficients come from the ridge fallback: usable as a terminal fit,
    # never a basis for extension (IRLS_SPEC W3.3).
    degree = 1
    fit = irls_logistic(design(1, z), y_arr, w=w)
    ll = loglik(fit.beta, 1)
    while degree < 4 and not fit.separation:
        cand = irls_logistic(design(degree + 1, z), y_arr, w=w)
        if cand.separation:
            break
        ll_cand = loglik(cand.beta, degree + 1)
        lr = max(2.0 * (ll_cand - ll), 0.0)
        p_add = 1.0 - float(gammainc_lower(0.5, lr / 2.0))  # chi-square df=1
        if p_add >= 0.05:
            break
        degree += 1
        fit, ll = cand, ll_cand

    # Associated calibration test: fitted polynomial vs the identity map.
    ll_null = float(
        np.sum(
            w
            * (
                y_arr * np.log(np.clip(p_arr, 1e-12, 1))
                + (1.0 - y_arr) * np.log(np.clip(1.0 - p_arr, 1e-12, 1))
            )
        )
    )
    lr_cal = max(2.0 * (ll - ll_null), 0.0)
    df = degree + 1
    p_value = 1.0 - float(gammainc_lower(df / 2.0, lr_cal / 2.0))

    # Pointwise bands from the information-matrix ellipsoid.
    X = design(degree, z)
    mu = expit(np.clip(X @ fit.beta, -30.0, 30.0))
    info = (X * (w * mu * (1.0 - mu))[:, None]).T @ X
    info_inv = np.linalg.inv(info + 1e-10 * np.eye(df))
    grid_p = _grid(p_arr, grid_size)
    grid_z = logit(grid_p)
    Xg = design(degree, grid_z)
    eta = Xg @ fit.beta
    se_sq = np.einsum("ij,jk,ik->i", Xg, info_inv, Xg)
    bands = {}
    for conf in confidence:
        radius = np.sqrt(chi2_ppf(conf, float(df)) * se_sq)
        bands[conf] = (expit(eta - radius), expit(eta + radius))
    lo_80, hi_80 = bands[confidence[0]]
    lo_95, hi_95 = bands[confidence[1]]
    return BeltResult(
        grid_p=grid_p,
        grid_logit=grid_z,
        lower_80=lo_80,
        upper_80=hi_80,
        lower_95=lo_95,
        upper_95=hi_95,
        degree=degree,
        p_value=p_value,
    )
