"""CORP reliability: consistent, optimally binned, reproducible.

Pure numpy; no dataclass rendering (that lives in ``curves.corp_reliability``).

References
----------
Dimitriadis, Gneiting & Jordan (2021), "Stable reliability diagrams for
probabilistic classifiers", PNAS.
"""

from __future__ import annotations

import numpy as np

from ._math import pava
from .isotonic import _aggregate_ties

_CLIP = 1e-12
"""Log-loss clip for degenerate PAV levels (exact 0 or 1 blocks)."""


def corp_fit(
    y: np.ndarray, p: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PAV recalibration of ``p``; returns block edges/levels/weights and the fit in input order.

    Parameters
    ----------
    y : numpy.ndarray
        Binary outcomes, already validated.
    p : numpy.ndarray
        Predicted probabilities, already validated.
    w : numpy.ndarray
        Non-negative weights, already validated.

    Returns
    -------
    block_lo, block_hi : numpy.ndarray
        Left and right edge (min/max ``p``) of each PAV block.
    block_level : numpy.ndarray
        PAV fitted event rate per block.
    block_weight : numpy.ndarray
        Pooled weight per block.
    pav : numpy.ndarray
        PAV fit expanded to observations, in the original input order.
    """
    order = np.argsort(p, kind="stable")
    s_u, ybar_u, w_u = _aggregate_ties(p[order], y[order], w[order])
    res = pava(ybar_u, w_u)
    starts = res.block_start
    ends = np.append(starts[1:], len(s_u))
    lo = s_u[starts]
    hi = s_u[ends - 1]
    # expand block level back to unique scores, then to observations
    level_u = res.fitted
    idx_u = np.searchsorted(s_u, p)  # p values are exactly in s_u
    pav = level_u[idx_u]
    return lo, hi, res.block_mean, res.block_weight, pav


def _mean_score(y: np.ndarray, q: np.ndarray, w: np.ndarray, score: str) -> float:
    if score == "brier":
        s = (y - q) ** 2
    else:
        qc = np.clip(q, _CLIP, 1.0 - _CLIP)
        s = -(y * np.log(qc) + (1.0 - y) * np.log1p(-qc))
    return float(np.sum(w * s) / np.sum(w))


def decompose(
    y: np.ndarray, p: np.ndarray, pav: np.ndarray, w: np.ndarray, score: str
) -> tuple[float, float, float, float]:
    """Return ``(mean, MCB, DSC, UNC)`` with ``mean == MCB - DSC + UNC`` exactly.

    Parameters
    ----------
    y : numpy.ndarray
        Binary outcomes.
    p : numpy.ndarray
        Predicted probabilities.
    pav : numpy.ndarray
        PAV-recalibrated fit of ``p``, in the same order as ``y``/``p``.
    w : numpy.ndarray
        Non-negative weights.
    score : {"brier", "log_loss"}
        Scoring rule to decompose.

    Returns
    -------
    tuple of float
        ``(mean_score, mcb, dsc, unc)``.
    """
    ybar = float(np.sum(w * y) / np.sum(w))
    s_p = _mean_score(y, p, w, score)
    s_pav = _mean_score(y, pav, w, score)
    unc = _mean_score(y, np.full_like(p, ybar), w, score)
    return s_p, s_p - s_pav, unc - s_pav, unc


def eval_step(lo: np.ndarray, hi: np.ndarray, level: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Value of a PAV block fit at each grid point.

    Right-continuous step function: a grid point below the first block's
    left edge takes the first block's level.

    Parameters
    ----------
    lo : numpy.ndarray
        Left edge of each block (``corp_fit``'s ``block_lo``).
    hi : numpy.ndarray
        Right edge of each block (``corp_fit``'s ``block_hi``); unused, kept
        for symmetry with ``block_lo``/``block_hi`` call sites.
    level : numpy.ndarray
        Fitted level of each block (``corp_fit``'s ``block_level``).
    grid : numpy.ndarray
        Points at which to evaluate the step function.

    Returns
    -------
    numpy.ndarray
        ``level`` indexed by the block containing (or preceding) each grid
        point, same shape as ``grid``.

    Examples
    --------
    >>> import numpy as np
    >>> lo = np.array([0.1, 0.5])
    >>> hi = np.array([0.4, 0.9])
    >>> level = np.array([0.2, 0.7])
    >>> eval_step(lo, hi, level, np.array([0.0, 0.3, 0.6]))
    array([0.2, 0.2, 0.7])
    """
    idx = np.searchsorted(lo, grid, side="right") - 1
    idx = np.clip(idx, 0, len(level) - 1)
    return level[idx]


def corp_bands(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    bands: str | None,
    level: float,
    n_resamples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consistency/confidence bands for the CORP fit, by resampling.

    Consistency bands simulate the PAV fit under the null that ``p`` is
    already calibrated (``y_b ~ Bernoulli(p)``); confidence bands bootstrap
    ``(y, p, w)`` triples. Both evaluate each resampled fit on a shared grid
    of 201 points between the 0.5% and 99.5% quantiles of ``p`` (via
    ``eval_step``) and take the ``level`` central quantile interval across
    resamples at each grid point.

    Parameters
    ----------
    y : numpy.ndarray
        Binary outcomes.
    p : numpy.ndarray
        Predicted probabilities.
    w : numpy.ndarray
        Non-negative weights.
    bands : {"consistency", "confidence", None}
        Band type; already validated by the caller.
    level : float
        Nominal coverage level; must satisfy ``0 < level < 1``.
    n_resamples : int
        Number of resamples to draw.
    random_state : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    band_grid, band_low, band_high : numpy.ndarray
        Empty arrays when ``bands`` is ``None``; otherwise 201-point grid and
        pointwise band bounds.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.1, 0.9, 500)
    >>> y = (rng.random(500) < p).astype(float)
    >>> grid, lo, hi = corp_bands(y, p, np.ones(500), "consistency", 0.9, 50, 0)
    >>> len(grid) == len(lo) == len(hi) == 201
    True
    """
    if bands is None:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    rng = np.random.default_rng(random_state)
    grid = np.linspace(np.quantile(p, 0.005), np.quantile(p, 0.995), 201)
    sims = np.empty((n_resamples, grid.size))
    for b in range(n_resamples):
        if bands == "consistency":
            y_b = (rng.random(p.size) < p).astype(float)
            lo, hi, lev, _, _ = corp_fit(y_b, p, w)
        else:  # "confidence"
            idx = rng.integers(0, p.size, p.size)
            lo, hi, lev, _, _ = corp_fit(y[idx], p[idx], w[idx])
        sims[b] = eval_step(lo, hi, lev, grid)
    a = (1.0 - level) / 2.0
    return grid, np.quantile(sims, a, axis=0), np.quantile(sims, 1.0 - a, axis=0)
