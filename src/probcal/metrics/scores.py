"""Proper scoring rules: log loss, Brier score, and their decompositions.

Theory, formulas, and pathologies: ``docs/concepts/metrics.md``.
"""

from dataclasses import dataclass

import numpy as np

from .._math import loess
from .._results import _ResultBase
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
class MurphyCurve(_ResultBase):
    """Murphy diagram: mean elementary score of the binary mean functional across thresholds.

    Attributes
    ----------
    thresholds : numpy.ndarray
        Threshold grid :math:`\\theta \\in [0, 1]`.
    score : numpy.ndarray
        Weighted mean elementary score :math:`S_\\theta` at each threshold.
    n : int
        Number of observations.
    """

    thresholds: np.ndarray
    score: np.ndarray
    n: int

    def __repr__(self) -> str:
        return f"MurphyCurve (n={self.n}, {len(self.thresholds)} thresholds)"


def murphy_curve(
    y: object,
    p: object,
    *,
    thresholds: object = 513,
    sample_weight: object = None,
) -> MurphyCurve:
    """Murphy diagram data: mean elementary score of the mean functional across a threshold grid.

    Uses the Ehm, Gneiting, Jordan & Krüger (2016) elementary score for the
    mean functional of a binary outcome,

    ``S_theta(p, y) = theta * 1{p > theta, y = 0} + (1 - theta) * 1{p <= theta, y = 1}``,

    whose weighted mean at each ``theta`` is this curve's ``score``.
    ``2 * integral(S_theta, theta in [0, 1])`` equals the Brier score
    exactly (a per-observation calculation: integrating a single
    observation's elementary score over ``theta in [0, 1]`` gives
    ``p**2 / 2`` when ``y=0`` and ``(1 - p)**2 / 2`` when ``y=1``, whose
    doubled weighted mean is exactly ``E[(1-y)*p**2 + y*(1-p)**2] ==
    E[(p - y)**2]``, the Brier score). ``S_theta`` is piecewise linear in
    ``theta`` between consecutive unique ``p`` values but jumps exactly at
    each one (the observation crossing sides), so a threshold grid built
    from those breakpoints reproduces the Brier identity to a numerical
    error that shrinks with the sample size — exactly at the continuum
    limit, and already far inside the default grid's 1e-3 budget at
    realistic ``n`` — rather than to machine precision at any finite
    ``thresholds`` array (the discretized trapezoid necessarily samples one
    side of each jump). Isotonic (PAV) recalibration of ``p`` never
    increases the score at any threshold (Ehm et al., 2016), so the two
    curves' relative position diagnoses the value of recalibration without
    collapsing to one scalar. Computed by sorting ``p`` once and
    accumulating weighted class-conditional sums via ``searchsorted`` —
    ``O(n log n + T log n)`` for ``T`` thresholds, never the naive
    ``O(n * T)`` mask.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    thresholds : int or array_like, keyword-only
        Either the number of equally spaced points in ``[0, 1]``
        (``numpy.linspace(0, 1, thresholds)``; default 513, the same
        default resolution used elsewhere in the package for dense grids)
        or an explicit 1-D array of thresholds in ``[0, 1]`` (sorted
        internally).
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    MurphyCurve
        Threshold grid, weighted mean elementary score, and observation
        count.

    Raises
    ------
    ValueError
        If ``thresholds`` is not 1-D, or contains values outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import murphy_curve
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p).astype(float)
    >>> curve = murphy_curve(y, p, thresholds=101)
    >>> curve.score.shape
    (101,)
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    if isinstance(thresholds, (int, np.integer)):
        theta = np.linspace(0.0, 1.0, int(thresholds))
    else:
        theta = np.sort(np.asarray(thresholds, dtype=np.float64))
        if theta.ndim != 1:
            raise ValueError(f"thresholds must be a 1-D array, got shape {theta.shape}")
        if np.any(theta < 0.0) or np.any(theta > 1.0):
            raise ValueError("thresholds must lie in [0, 1]")

    # Sort p once; cumulative weighted sums over the y==0/y==1 subsets let
    # searchsorted read off, for every threshold at once, how much weight
    # lies on each side — avoids an O(n * T) mask per threshold.
    order = np.argsort(p_arr, kind="stable")
    p_sorted = p_arr[order]
    is_event = y_arr[order] == 1.0
    w_sorted = w[order]
    cum_w0 = np.concatenate(([0.0], np.cumsum(np.where(is_event, 0.0, w_sorted))))
    cum_w1 = np.concatenate(([0.0], np.cumsum(np.where(is_event, w_sorted, 0.0))))

    idx = np.searchsorted(p_sorted, theta, side="right")
    w0_above = cum_w0[-1] - cum_w0[idx]  # y=0, p > theta
    w1_at_or_below = cum_w1[idx]  # y=1, p <= theta

    score = (theta * w0_above + (1.0 - theta) * w1_at_or_below) / w.sum()
    return MurphyCurve(thresholds=theta, score=score, n=len(y_arr))


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
