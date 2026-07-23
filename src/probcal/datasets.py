"""Synthetic dataset generators (make_pd_portfolio)."""

from dataclasses import dataclass

import numpy as np

from ._math import bisect, expit


@dataclass(frozen=True)
class PdPortfolio:
    """Synthetic PD portfolio: model scores, outcomes, and the true probabilities.

    Attributes
    ----------
    scores : numpy.ndarray
        The model's reported PDs — miscalibrated unless generated with
        ``slope=1, asymmetry=0, intercept=0``.
    y : numpy.ndarray
        Bernoulli outcomes drawn from ``p_true``.
    p_true : numpy.ndarray
        True conditional probabilities (mean anchored at ``event_rate``).
    """

    scores: np.ndarray
    y: np.ndarray
    p_true: np.ndarray


def make_pd_portfolio(
    n: int = 5000,
    *,
    event_rate: float = 0.03,
    slope: float = 0.7,
    intercept: float = 0.0,
    asymmetry: float = 0.4,
    score_location: float = -3.2,
    score_scale: float = 1.1,
    random_state: int = 42,
) -> PdPortfolio:
    """Generate a synthetic, controllably miscalibrated PD portfolio.

    The model's scores are drawn as ``s = sigma(N(score_location,
    score_scale))``; the true probability follows the beta-calibration family

    ``logit p_true = a_lo * ln(s) - a_hi * ln(1 - s) + c``

    with ``a_lo = slope * (1 + asymmetry)`` (low-PD tail) and ``a_hi = slope``
    (high tail), so ``asymmetry != 0`` produces exactly the one-sided tail
    distortion low-event-rate portfolios exhibit, and `BetaCalibrator` can
    recover the generative exponents. ``c`` absorbs ``intercept`` plus a
    portfolio-level anchor solved so that ``mean(p_true) == event_rate``
    (unique by monotonicity, via bisection). With ``slope=1, asymmetry=0,
    intercept=0`` the scores are exactly calibrated.

    Parameters
    ----------
    n : int
        Portfolio size.
    event_rate : float
        Target mean of ``p_true`` (the central tendency), ~3% by default.
    slope : float
        Base exponent of the distortion; ``< 1`` means the model's scores are
        too spread out (overconfident).
    intercept : float
        Additional log-odds shift applied before the mean anchor is solved.
    asymmetry : float
        Relative extra distortion of the low-PD tail (``a_lo/a_hi - 1``).
    score_location, score_scale : float
        Parameters of the normal generating the score logits.
    random_state : int
        Seed.

    Returns
    -------
    PdPortfolio
    """
    if not 0.0 < event_rate < 1.0:
        raise ValueError("event_rate must lie in (0, 1)")
    rng = np.random.default_rng(random_state)
    s = expit(rng.normal(score_location, score_scale, n))
    a_lo = slope * (1.0 + asymmetry)
    a_hi = slope
    core = a_lo * np.log(s) - a_hi * np.log1p(-s) + intercept

    identity_case = slope == 1.0 and asymmetry == 0.0 and intercept == 0.0
    if identity_case:
        p_true = s.copy()
    else:

        def gap(c: float) -> float:
            return float(np.mean(expit(core + c))) - event_rate

        c_anchor = bisect(gap, -60.0, 60.0, tol=1e-14)
        p_true = expit(core + c_anchor)
    y = (rng.random(n) < p_true).astype(np.float64)
    return PdPortfolio(scores=s, y=y, p_true=p_true)
