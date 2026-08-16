"""Proper scoring rules: log loss, Brier score, and their decompositions.

Theory, formulas, and pathologies: ``docs/concepts/metrics.md``.
"""

from dataclasses import dataclass

import numpy as np

from .._math import loess
from .._validation import validate_binary_y, validate_scores, validate_weights


def _prep(y: object, p: object, sample_weight: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = validate_binary_y(y)
    p_arr = validate_scores(p, name="p")
    w_arr = validate_weights(sample_weight, len(p_arr))
    if len(y_arr) != len(p_arr):
        raise ValueError("y and p must have equal length")
    return y_arr, p_arr, w_arr


def log_loss(y: object, p: object, *, sample_weight: object = None) -> float:
    """Weighted mean negative log-likelihood.

    Strictly proper; the default selection criterion.

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
    float
        Weighted mean negative log-likelihood.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    ll = y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log1p(-p_arr)
    return float(-np.average(ll, weights=w))


def brier_score(y: object, p: object, *, sample_weight: object = None) -> float:
    """Weighted mean squared error of the probability forecast (strictly proper).

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
    float
        Weighted mean squared error.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    return float(np.average((p_arr - y_arr) ** 2, weights=w))


def brier_skill_score(y: object, p: object, *, sample_weight: object = None) -> float:
    """Brier skill score vs the climatology forecast ``p = mean(y)``.

    Positive values beat the base rate; 0 equals it.

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
    float
        Skill score relative to the weighted base rate.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    base = float(np.average(y_arr, weights=w))
    bs_ref = float(np.average((base - y_arr) ** 2, weights=w))
    bs = float(np.average((p_arr - y_arr) ** 2, weights=w))
    return 1.0 - bs / bs_ref


@dataclass(frozen=True)
class MurphyDecomposition:
    """Binned Murphy (1973) partition of the Brier score.

    ``reliability - resolution + uncertainty`` equals the Brier score exactly
    when predictions are constant within bins; otherwise the identity holds
    up to the within-bin variance of ``p`` (documented binning bias).

    Attributes
    ----------
    reliability : float
        Mean squared gap between within-bin predicted and observed rates.
    resolution : float
        Mean squared gap between within-bin observed rate and the overall base rate.
    uncertainty : float
        Base-rate variance ``y_bar * (1 - y_bar)``.
    """

    reliability: float
    resolution: float
    uncertainty: float


def murphy_decomposition(
    y: object,
    p: object,
    *,
    n_bins: int = 10,
    strategy: str = "mass",
    bias_corrected: bool = False,
    sample_weight: object = None,
) -> MurphyDecomposition:
    """Binned reliability/resolution/uncertainty split of the Brier score.

    ``bias_corrected=True`` subtracts the within-bin variance of the event
    rate from the squared-gap terms (within-bin variance corrections in the
    manner of Ferro & Fricker, 2012); the naive plug-in otherwise. The
    decomposition inherits the binning choice — see the metrics chapter.

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
    bias_corrected : bool, keyword-only
        If ``True`` (default ``False``), apply the Ferro & Fricker (2012)
        within-bin variance correction to the reliability and resolution terms.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    MurphyDecomposition
        Reliability, resolution, and uncertainty terms.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    from .binned import _bin_index

    idx, m = _bin_index(p_arr, n_bins, strategy)
    w_tot = float(w.sum())
    y_bar = float(np.average(y_arr, weights=w))
    rel = res = 0.0
    for b in range(m):
        mask = idx == b
        if not np.any(mask):
            continue
        wb = float(w[mask].sum())
        pb = float(np.average(p_arr[mask], weights=w[mask]))
        yb = float(np.average(y_arr[mask], weights=w[mask]))
        nb = int(np.sum(mask))
        rel_term = (pb - yb) ** 2
        res_term = (yb - y_bar) ** 2
        if bias_corrected and nb > 1:
            var_yb = yb * (1.0 - yb) / (nb - 1)
            rel_term = max(rel_term - var_yb, 0.0)
            res_term = max(res_term - var_yb, 0.0)
        rel += (wb / w_tot) * rel_term
        res += (wb / w_tot) * res_term
    unc = y_bar * (1.0 - y_bar)
    return MurphyDecomposition(reliability=rel, resolution=res, uncertainty=unc)


@dataclass(frozen=True)
class LogLossDecomposition:
    """Calibration/refinement split of the log loss via a plug-in
    recalibration curve (LOESS; DECISIONS entry).

    Attributes
    ----------
    calibration : float
        Mean KL divergence between the plug-in and predicted Bernoullis.
    refinement : float
        Mean entropy of the plug-in Bernoulli.
    """

    calibration: float
    refinement: float


def logloss_calibration_refinement(
    y: object,
    p: object,
    *,
    frac: float = 0.75,
    sample_weight: object = None,
) -> LogLossDecomposition:
    """Split the log loss into a calibration (KL) and refinement (entropy) part.

    The conditional event rate ``c(p)`` is estimated by a LOESS smoother of
    the outcome on the prediction; calibration is the mean
    ``KL(Bernoulli(c) || Bernoulli(p))`` and refinement the mean entropy of
    ``Bernoulli(c)``. Only as good as the plug-in estimate of ``c``.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    frac : float, keyword-only
        LOESS smoothing fraction passed through to the recalibration curve.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    LogLossDecomposition
        Calibration and refinement terms.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    c = np.clip(loess(p_arr, y_arr, frac=frac), 1e-12, 1.0 - 1e-12)
    kl = c * (np.log(c) - np.log(p_arr)) + (1.0 - c) * (np.log1p(-c) - np.log1p(-p_arr))
    ent = -(c * np.log(c) + (1.0 - c) * np.log1p(-c))
    return LogLossDecomposition(
        calibration=float(np.average(kl, weights=w)),
        refinement=float(np.average(ent, weights=w)),
    )
