"""Shared helpers for the step-function and binning calibrators.

One implementation each of: tie aggregation, tie-pooled PAVA blocks,
equal-mass edges (weighted),
per-bin weighted sums, right-continuous step evaluation, the generalized
inverse of a non-decreasing step map, and legacy state-name migration.
"""

import math
from collections.abc import Callable, Mapping

import numpy as np

from ._math import PavaResult, pava, weighted_quantile


def aggregate_ties(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by score and pool tied scores.

    Rows are ordered by ``(s, y, w)`` before summing, so tied groups are summed
    in a canonical order and the result is bit-for-bit independent of row
    order.

    Returns
    -------
    s_unique, y_mean, w_sum : numpy.ndarray
        Sorted distinct scores, the weighted mean of ``y`` at each, and the
        pooled weight.
    """
    order = np.lexsort((w, y, s))
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    s_unique, start = np.unique(s_sorted, return_index=True)
    w_sum = np.add.reduceat(w_sorted, start)
    wy_sum = np.add.reduceat(w_sorted * y_sorted, start)
    return s_unique, wy_sum / w_sum, w_sum


def isotonic_blocks(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PavaResult, np.ndarray, np.ndarray]:
    """PAVA fit of ``y`` on tie-pooled scores, with each block's score range.

    Returns
    -------
    s_unique, w_sum : numpy.ndarray
        Output of :func:`aggregate_ties` (the PAVA input grid and its weights).
    res : PavaResult
        The fit on that grid.
    block_first, block_last : numpy.ndarray
        Smallest and largest distinct score of each block.
    """
    s_u, y_u, w_u = aggregate_ties(s, y, w)
    res = pava(y_u, w_u)
    ends = np.append(res.block_start[1:], len(s_u)) - 1
    return s_u, w_u, res, s_u[res.block_start], s_u[ends]


def equal_mass_edges(values: np.ndarray, n_bins: int, w: np.ndarray | None = None) -> np.ndarray:
    """Interior equal-mass edges, deduplicated, with no empty bin.

    With ``w`` given, the edges are weighted quantiles
    (:func:`probcal._math.weighted_quantile`), so the bins carry equal
    *weight*; unit or constant weights give ``np.quantile`` exactly.

    Ties can make quantile edges coincide (deduplicated) or leave a bin of
    the ``side="right"`` rule without any value — most commonly bin 0 when
    more than ``1/n_bins`` of the mass is tied at the minimum, so the first
    edge equals ``min(values)``. Each empty bin is merged into its upper
    neighbour by dropping its lower edge (the first edge, for bin 0), so
    every bin of the returned edges holds at least one value. When no bin is
    empty the quantile edges are returned unchanged.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs) if w is None else weighted_quantile(values, qs, w))
    occupied = np.unique(np.searchsorted(edges, values, side="right"))
    if len(occupied) == len(edges) + 1:
        return edges
    # Bin k is [edges[k-1], edges[k]); keep the lower edge of every occupied
    # bin except the lowest one, which becomes the new (non-empty) bin 0.
    return edges[occupied[1:] - 1]


def refine_lower(x: float, reaches: Callable[[float], bool], max_ulps: int = 64) -> float:
    """Smallest float near ``x`` in ``[0, 1]`` at which ``reaches`` holds.

    ``reaches`` must be monotone (false below some point, true above it), as
    ``lambda v: g(v) >= t`` is for a non-decreasing ``g``. A closed-form
    estimate ``x`` of the crossing is off by a few ulps either way; it is
    stepped down while its predecessor still reaches, then up until it
    reaches, so that ``g(x) >= t > g(prev(x))`` holds in floating point —
    the rule :meth:`probcal.Masterscale.assign` applies to ``g(x)``.
    """
    x = float(x)
    for _ in range(max_ulps):
        if x <= 0.0:
            break
        prev = math.nextafter(x, 0.0)
        if not reaches(prev):
            break
        x = prev
    for _ in range(max_ulps):
        if x >= 1.0 or reaches(x):
            break
        x = math.nextafter(x, 1.0)
    return x


def refine_upper(x: float, within: Callable[[float], bool], max_ulps: int = 64) -> float:
    """Largest float near ``x`` in ``[0, 1]`` at which ``within`` holds.

    Mirror of :func:`refine_lower` for a predicate that is true below some
    point and false above it, such as ``lambda v: g(v) <= t``: stepped up
    while the successor stays within, then down until ``x`` is within.
    """
    x = float(x)
    for _ in range(max_ulps):
        if x >= 1.0:
            break
        nxt = math.nextafter(x, 1.0)
        if not within(nxt):
            break
        x = nxt
    for _ in range(max_ulps):
        if x <= 0.0 or within(x):
            break
        x = math.nextafter(x, 0.0)
    return x


def bin_sums(
    idx: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin weighted sums ``(sum w*y, sum w)`` for bin indices ``idx``."""
    return (
        np.bincount(idx, weights=w * y, minlength=n_bins),
        np.bincount(idx, weights=w, minlength=n_bins),
    )


def eval_step(starts: np.ndarray, levels: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Right-continuous step function: ``levels[j]`` on ``[starts[j], starts[j+1])``.

    Points below ``starts[0]`` take ``levels[0]``.
    """
    idx = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, len(levels) - 1)
    return levels[idx]


def step_inverse_left(
    levels: np.ndarray,
    starts: np.ndarray,
    t: float,
    forward: Callable[[float], float] | None = None,
    pullback: Callable[[float], float] | None = None,
) -> float:
    """``inf{x : g(x) >= t}`` for the non-decreasing step map ``levels``/``starts``.

    Callers guarantee ``t > levels[0]``. With ``pullback`` (the inverse of an
    increasing inner map ``forward``, e.g. the Platt stage of scaling-binning)
    the edge is mapped back to the raw scale and refined by ulps in either
    direction (:func:`refine_lower`) to the smallest ``x`` with
    ``forward(x) >= edge``.
    """
    j = int(np.searchsorted(levels, t, side="left"))
    edge = float(starts[j])
    if pullback is None or forward is None:
        return edge
    return refine_lower(pullback(edge), lambda v: forward(v) >= edge)


def step_inverse_right(
    levels: np.ndarray,
    starts: np.ndarray,
    t: float,
    forward: Callable[[float], float] | None = None,
    pullback: Callable[[float], float] | None = None,
) -> float:
    """``sup{x : g(x) <= t}``, returned one float below the next level's edge.

    The bound is itself in the preimage, so consumers may treat both bounds
    as closed. ``1.0`` when ``t`` reaches the top level. With ``pullback``
    the edge is mapped back and refined (:func:`refine_upper`) to the largest
    ``x`` with ``forward(x) < edge``.
    """
    j = int(np.searchsorted(levels, t, side="right")) - 1
    if j >= len(levels) - 1:
        return 1.0
    edge = float(starts[j + 1])
    if pullback is None or forward is None:
        return float(np.nextafter(edge, 0.0))
    return refine_upper(math.nextafter(pullback(edge), 0.0), lambda v: forward(v) < edge)


def migrate_state(
    state: Mapping[str, object],
    renames: Mapping[str, str],
    drop: tuple[str, ...] = (),
) -> dict[str, object]:
    """Map the private state keys of payloads written by probcal < 0.3.4 onto their public names."""
    out: dict[str, object] = {}
    for key, value in state.items():
        if key in drop:
            continue
        out[renames.get(key, key)] = value
    return out


def linear_inverse_left(
    xs: np.ndarray, ms: np.ndarray, t: float, forward: Callable[[float], float]
) -> float:
    """``inf{x : g(x) >= t}`` for ``g = np.interp(x, xs, ms)`` with ``ms`` non-decreasing.

    Callers guarantee ``ms[0] < t <= ms[-1]``. The closed-form crossing on the
    bracketing segment is refined by ulps in either direction
    (:func:`refine_lower`) to the smallest ``x`` with ``forward(x) >= t``.
    """
    j = int(np.searchsorted(ms, t, side="left"))
    frac = (t - ms[j - 1]) / (ms[j] - ms[j - 1])
    x = float(xs[j - 1] + frac * (xs[j] - xs[j - 1]))
    x = min(max(x, float(xs[j - 1])), float(xs[j]))
    return refine_lower(x, lambda v: forward(v) >= t)


def linear_inverse_right(
    xs: np.ndarray, ms: np.ndarray, t: float, forward: Callable[[float], float]
) -> float:
    """``sup{x : g(x) <= t}`` for ``g = np.interp(x, xs, ms)``; ``ms[0] <= t < ms[-1]``.

    Refined by ulps in either direction (:func:`refine_upper`) to the
    largest ``x`` with ``forward(x) <= t``.
    """
    j = int(np.searchsorted(ms, t, side="right")) - 1
    frac = (t - ms[j]) / (ms[j + 1] - ms[j])
    x = float(xs[j] + frac * (xs[j + 1] - xs[j]))
    x = min(max(x, float(xs[j])), float(xs[j + 1]))
    return refine_upper(x, lambda v: forward(v) <= t)
