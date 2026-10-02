"""Binned calibration-error estimators: ECE family, MCE, Hosmer–Lemeshow.

Pathologies (binning sensitivity, finite-sample bias, HL power issues) are
documented in ``docs/concepts/metrics.md``. None of these are selection
criteria; the Hosmer–Lemeshow test is report-only. Sample weights follow the
package convention in :mod:`probcal.metrics._common` (relative weights;
Kish effective sample size wherever a sample size enters).
"""

import math
from dataclasses import dataclass

import numpy as np

from .._math import chi2_sf
from .._validation import validate_positive_int
from ._binstats import BinStats, bin_stats
from ._common import _prep, effective_weights
from ._deprecation import UNSET, deprecated, renamed_kwarg


def _shares_gaps(bs: BinStats, w_total: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin weight share and absolute calibration gap."""
    return bs.w_sum / w_total, np.abs(bs.p_mean - bs.rate)


def _reduce(shares: np.ndarray, gaps: np.ndarray, norm: str) -> float:
    """Reduce per-bin (weight share, |gap|) under ``norm``."""
    if norm == "l1":
        return float(np.sum(shares * gaps))
    if norm == "l2":
        return float(np.sqrt(np.sum(shares * gaps**2)))
    if norm == "max":
        return float(gaps.max())
    raise ValueError(f"norm must be 'l1', 'l2', or 'max', got {norm!r}")


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
        Requested number of bins (positive integer).
    strategy : {"mass", "width"}, keyword-only
        ``"mass"`` (equal-count, default; the "adaptive ECE" of the
        literature) or ``"width"`` (equal-width over [0, 1]).
    norm : {"l1", "l2", "max"}, keyword-only
        ``"l1"`` (default, the usual ECE), ``"l2"``, or ``"max"`` (the MCE).
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.

    Returns
    -------
    float
        Weighted binned calibration error under the chosen norm.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    bs = bin_stats(y_arr, p_arr, w, n_bins, strategy)
    return _reduce(*_shares_gaps(bs, float(w.sum())), norm)


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
    (correction in the spirit of Bröcker 2009 / Ferro & Fricker 2012):
    ``sqrt(max(gap_b**2 - rate_b * (1 - rate_b) / (n_b - 1), 0))`` for bins
    with ``n_b > 1``. ``n_b`` is the bin's effective count under the package
    weight convention (the sum of the Kish-rescaled weights in the bin; the
    raw count for unit weights).

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
        Optional positive weights, same length as ``y``.

    Returns
    -------
    float
        Bias-corrected calibration error.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    return _debiased(bin_stats(y_arr, p_arr, w, n_bins, strategy), w)


def _debiased(bs: BinStats, w: np.ndarray) -> float:
    """Variance-corrected ECE from one binning pass (Kish effective bin counts)."""
    shares, gaps = _shares_gaps(bs, float(w.sum()))
    n_eff = bs.w_sum * (float(w.sum()) / float(np.dot(w, w)))
    rate = bs.rate
    multi = n_eff > 1.0
    var_b = rate * (1.0 - rate) / np.where(multi, n_eff - 1.0, 1.0)
    corrected = np.where(multi, np.sqrt(np.maximum(gaps**2 - var_b, 0.0)), gaps)
    return float(np.sum(shares * corrected))


def _ece_family(y: np.ndarray, p: np.ndarray, w: np.ndarray, names: set[str]) -> dict[str, float]:
    """``ece``/``ece_debiased``/``mce`` at their defaults (15 equal-mass bins)
    off one shared binning pass; equal to the public calls bit for bit."""
    bs = bin_stats(y, p, w, 15, "mass")
    shares, gaps = _shares_gaps(bs, float(w.sum()))
    out: dict[str, float] = {}
    if "ece" in names:
        out["ece"] = _reduce(shares, gaps, "l1")
    if "ece_debiased" in names:
        out["ece_debiased"] = _debiased(bs, w)
    if "mce" in names:
        out["mce"] = _reduce(shares, gaps, "max")
    return out


_SWEEP_RULES = ("first_violation", "largest")


def _sweep_best_b(y: np.ndarray, p: np.ndarray, w: np.ndarray, max_bins: int, rule: str) -> int:
    """Bin count chosen by the monotone sweep (see :func:`ece_sweep`)."""
    best_b = 1
    for b in range(2, min(len(p), max_bins) + 1):
        if np.all(np.diff(bin_stats(y, p, w, b, "mass").rate) >= 0.0):
            best_b = b
        elif rule == "first_violation":
            break
    return best_b


def ece_sweep(
    y: object,
    p: object,
    *,
    norm: str = "l1",
    sample_weight: object = None,
    max_bins: int = 100,
    rule: str = "first_violation",
) -> float:
    """Monotonic-sweep calibration error (Roelofs et al., 2022).

    Equal-mass bins; the bin count is the largest ``b`` such that ``b`` *and
    every smaller bin count* give non-decreasing bin event rates — the
    paper's ``b* = max{b : for all b' <= b, ybar_1 <= ... <= ybar_b'}``
    (its Algorithm 1: sweep ``b = 2, 3, ...`` and stop at the first
    non-monotone binning). The paper sweeps up to ``b = n``; the sweep here
    stops at ``max_bins`` (default 100) as a cost cap.

    0.3.x took the largest monotone ``b`` anywhere in ``2..100`` instead,
    skipping over non-monotone counts; ``rule="largest"`` recovers it.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    norm : {"l1", "l2", "max"}, keyword-only
        Norm of the final binned error at the selected bin count.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
    max_bins : int, keyword-only
        Upper end of the sweep (capped at ``n``).
    rule : {"first_violation", "largest"}, keyword-only
        ``"first_violation"`` (default, the paper's rule) or ``"largest"``
        (the 0.3.x rule).

    Returns
    -------
    float
        Calibration error at the selected bin count.
    """
    if rule not in _SWEEP_RULES:
        raise ValueError(f"rule must be one of {_SWEEP_RULES}, got {rule!r}")
    cap = validate_positive_int(max_bins, "max_bins")
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    best_b = _sweep_best_b(y_arr, p_arr, w, cap, rule)
    if best_b == 1:
        _reduce(np.ones(1), np.zeros(1), norm)  # validate norm
        return abs(float(np.average(p_arr, weights=w)) - float(np.average(y_arr, weights=w)))
    bs = bin_stats(y_arr, p_arr, w, best_b, "mass")
    return _reduce(*_shares_gaps(bs, float(w.sum())), norm)


def adaptive_ece(
    y: object,
    p: object,
    *,
    n_bins: int = 15,
    norm: str = "l1",
    sample_weight: object = None,
) -> float:
    """Deprecated alias of ``ece(strategy="mass")`` (removed in 0.4.0).

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
        Optional positive weights, same length as ``y``.

    Returns
    -------
    float
        Equal-mass binned calibration error under the chosen norm.
    """
    deprecated('adaptive_ece() is an alias of ece(strategy="mass"); call ece directly')
    return ece(y, p, n_bins=n_bins, strategy="mass", norm=norm, sample_weight=sample_weight)


@dataclass(frozen=True)
class HosmerLemeshowResult:
    """Hosmer–Lemeshow chi-square test (report-only; never a selection criterion).

    Attributes
    ----------
    statistic : float
        Chi-square test statistic.
    df : int
        Degrees of freedom: non-empty groups minus 2 (``0`` when fewer than
        three groups are non-empty).
    p_value : float
        Upper-tail p-value of the statistic; ``nan`` when ``df == 0``.
    """

    statistic: float
    df: int
    p_value: float


def hosmer_lemeshow(
    y: object,
    p: object,
    *,
    n_bins: int = 10,
    sample_weight: object = None,
    g: object = UNSET,
) -> HosmerLemeshowResult:
    """Hosmer–Lemeshow goodness-of-fit test on ``n_bins`` equal-mass risk groups.

    The statistic depends on an essentially arbitrary grouping and its power
    scales with n — see the metrics chapter for why this is report-only.
    Weighted counts use the Kish-rescaled weights (package convention), so
    the statistic is invariant to rescaling the weights and unit weights give
    the unweighted test.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    n_bins : int, keyword-only
        Requested number of equal-mass risk groups. Tied scores can collapse
        groups; with fewer than three non-empty groups the test has no
        degrees of freedom and returns ``df=0``, ``p_value=nan``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
    g : int, keyword-only
        Deprecated spelling of ``n_bins`` (removed in 0.4.0).

    Returns
    -------
    HosmerLemeshowResult
        Chi-square statistic, degrees of freedom, and p-value.
    """
    n_groups = renamed_kwarg("hosmer_lemeshow", "g", "n_bins", g, n_bins, 10)
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    bs = bin_stats(y_arr, p_arr, effective_weights(w), n_groups, "mass")
    obs, exp, nb = bs.wy_sum, bs.wp_sum, bs.w_sum
    denom = exp * (1.0 - exp / nb)
    ok = denom > 0
    stat = float(np.sum((obs[ok] - exp[ok]) ** 2 / denom[ok]))
    df = max(len(nb) - 2, 0)
    p_value = chi2_sf(stat, df) if df > 0 else math.nan
    return HosmerLemeshowResult(statistic=stat, df=df, p_value=p_value)
