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

from dataclasses import dataclass

import numpy as np

from ._math import chi2_ppf, expit, gammainc_lower, irls_logistic, loess, logit
from ._results import BeltResult, ReliabilityCurve, SmoothReliabilityCurve
from .metrics.binned import _bin_index
from .metrics.scores import _prep

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
    approximation of the LR-region inversion (DECISIONS entry). The
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
    # never a basis for extension (IRLS_SPEC W3.3 / DECISIONS 57).
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
