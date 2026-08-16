"""Binned calibration-error estimators: ECE family, MCE, Hosmer–Lemeshow.

Pathologies (binning sensitivity, finite-sample bias, HL power issues) are
documented in ``docs/concepts/metrics.md``. None of these are selection
criteria; the Hosmer–Lemeshow test is report-only.
"""

from dataclasses import dataclass

import numpy as np

from .._math import gammainc_lower
from .scores import _prep


def _bin_index(p: np.ndarray, n_bins: int, strategy: str) -> tuple[np.ndarray, int]:
    """Bin assignment and effective bin count."""
    if strategy == "mass":
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(p, qs))
    elif strategy == "width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    else:
        raise ValueError(f"strategy must be 'mass' or 'width', got {strategy!r}")
    return np.searchsorted(edges, p, side="right"), len(edges) + 1


def _bin_gaps(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, n_bins: int, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (weight share, |gap|, event rate, raw count) for non-empty bins."""
    idx, m = _bin_index(p, n_bins, strategy)
    w_sum = np.bincount(idx, weights=w, minlength=m)
    wy_sum = np.bincount(idx, weights=w * y, minlength=m)
    wp_sum = np.bincount(idx, weights=w * p, minlength=m)
    counts = np.bincount(idx, minlength=m)
    keep = w_sum > 0
    w_kept = w_sum[keep]
    rates = wy_sum[keep] / w_kept
    p_means = wp_sum[keep] / w_kept
    return (
        w_kept / float(w.sum()),
        np.abs(p_means - rates),
        rates,
        counts[keep].astype(np.int64),
    )


def ece(
    y: object,
    p: object,
    *,
    n_bins: int = 15,
    strategy: str = "mass",
    norm: str = "l1",
    sample_weight: object = None,
) -> float:
    """Expected calibration error; ``norm="max"`` gives the MCE.

    Binning-sensitive and upward-biased in finite samples — report, never
    select on it (see the metrics chapter's table).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    n_bins : int, keyword-only
        Requested number of bins.
    strategy : {"mass", "width"}, keyword-only
        ``"mass"`` (equal-count, default) or ``"width"`` (equal-width over [0, 1]).
    norm : {"l1", "l2", "max"}, keyword-only
        ``"l1"`` (default, the usual ECE), ``"l2"``, or ``"max"`` (the MCE).
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    float
        Weighted binned calibration error under the chosen norm.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    shares, gaps, _, _ = _bin_gaps(y_arr, p_arr, w, n_bins, strategy)
    if norm == "l1":
        return float(np.sum(shares * gaps))
    if norm == "l2":
        return float(np.sqrt(np.sum(shares * gaps**2)))
    if norm == "max":
        return float(gaps.max())
    raise ValueError(f"norm must be 'l1', 'l2', or 'max', got {norm!r}")


def ece_debiased(
    y: object,
    p: object,
    *,
    n_bins: int = 15,
    strategy: str = "mass",
    sample_weight: object = None,
) -> float:
    """Bias-corrected ECE, floored at zero.

    Per-bin squared gaps minus the within-bin variance of the event rate
    (correction in the spirit of Bröcker 2009 / Ferro & Fricker 2012; exact
    estimator in the DECISIONS log).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    n_bins : int, keyword-only
        Requested number of bins.
    strategy : {"mass", "width"}, keyword-only
        ``"mass"`` (equal-count, default) or ``"width"`` (equal-width over [0, 1]).
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    float
        Bias-corrected calibration error.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    shares, gaps, rates, counts = _bin_gaps(y_arr, p_arr, w, n_bins, strategy)
    corrected = np.empty_like(gaps)
    for i in range(len(gaps)):
        if counts[i] > 1:
            var_b = rates[i] * (1.0 - rates[i]) / (counts[i] - 1)
            corrected[i] = np.sqrt(max(gaps[i] ** 2 - var_b, 0.0))
        else:
            corrected[i] = gaps[i]
    return float(np.sum(shares * corrected))


def ece_sweep(
    y: object,
    p: object,
    *,
    norm: str = "l1",
    sample_weight: object = None,
) -> float:
    """Monotonic-sweep calibration error (Roelofs et al., 2022).

    Uses equal-mass bins with the largest ``B`` whose bin event rates remain
    monotone non-decreasing (scan 2..min(n, 100); DECISIONS entry).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    norm : {"l1", "l2", "max"}, keyword-only
        Norm passed to the final :func:`ece` call at the selected bin count.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    float
        Calibration error at the largest monotone bin count.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    best_b = 1
    for b in range(2, min(len(p_arr), 100) + 1):
        _, _, rates, _ = _bin_gaps(y_arr, p_arr, w, b, "mass")
        if np.all(np.diff(rates) >= 0.0):
            best_b = b
    if best_b == 1:
        pb = float(np.average(p_arr, weights=w))
        yb = float(np.average(y_arr, weights=w))
        return abs(pb - yb)
    return ece(y_arr, p_arr, n_bins=best_b, strategy="mass", norm=norm, sample_weight=w)


def adaptive_ece(
    y: object,
    p: object,
    *,
    n_bins: int = 15,
    norm: str = "l1",
    sample_weight: object = None,
) -> float:
    """Adaptive ECE: an explicit alias for equal-mass ``ece``.

    The literature uses both names for the same estimator.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    n_bins : int, keyword-only
        Requested number of bins.
    norm : {"l1", "l2", "max"}, keyword-only
        Norm passed through to :func:`ece`.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    float
        Equal-mass binned calibration error under the chosen norm.
    """
    return ece(y, p, n_bins=n_bins, strategy="mass", norm=norm, sample_weight=sample_weight)


@dataclass(frozen=True)
class HosmerLemeshowResult:
    """Hosmer–Lemeshow chi-square test (report-only; never a selection criterion).

    Attributes
    ----------
    statistic : float
        Chi-square test statistic.
    df : int
        Degrees of freedom (used groups minus 2, floored at 1).
    p_value : float
        Upper-tail p-value of the chi-square statistic.
    """

    statistic: float
    df: int
    p_value: float


def hosmer_lemeshow(
    y: object,
    p: object,
    *,
    g: int = 10,
    sample_weight: object = None,
) -> HosmerLemeshowResult:
    """Hosmer–Lemeshow goodness-of-fit test on ``g`` equal-mass risk groups.

    The statistic depends on an essentially arbitrary grouping and its power
    scales with n — see the metrics chapter for why this is report-only.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    g : int, keyword-only
        Requested number of equal-mass risk groups.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    HosmerLemeshowResult
        Chi-square statistic, degrees of freedom, and p-value.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    idx, m = _bin_index(p_arr, g, "mass")
    stat = 0.0
    used = 0
    for b in range(m):
        mask = idx == b
        if not np.any(mask):
            continue
        nb = float(w[mask].sum())
        obs = float(np.sum(w[mask] * y_arr[mask]))
        exp = float(np.sum(w[mask] * p_arr[mask]))
        denom = exp * (1.0 - exp / nb)
        if denom > 0:
            stat += (obs - exp) ** 2 / denom
        used += 1
    df = max(used - 2, 1)
    p_value = 1.0 - float(gammainc_lower(df / 2.0, stat / 2.0))
    return HosmerLemeshowResult(statistic=float(stat), df=df, p_value=p_value)
