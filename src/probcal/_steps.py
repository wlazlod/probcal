"""Shared helpers for the step-function and binning calibrators.

One implementation each of: tie aggregation, equal-mass edges (weighted),
per-bin weighted sums, right-continuous step evaluation, the generalized
inverse of a non-decreasing step map, and legacy state-name migration.
"""

from collections.abc import Callable, Mapping

import numpy as np

from ._math import weighted_quantile


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


def equal_mass_edges(values: np.ndarray, n_bins: int, w: np.ndarray | None = None) -> np.ndarray:
    """Interior equal-mass edges, deduplicated (ties can collapse bins).

    With ``w`` given and not constant, the edges are weighted quantiles
    (:func:`probcal._math.weighted_quantile`), so the bins carry equal
    *weight*; unit or constant weights keep ``np.quantile`` exactly.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if w is None or bool(np.all(w == w[0])):
        return np.unique(np.quantile(values, qs))
    return np.unique(weighted_quantile(values, qs, w))


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
    the edge is mapped back to the raw scale and nudged by ulps until
    ``forward(x) >= edge``, so the returned bound is inside the preimage.
    """
    j = int(np.searchsorted(levels, t, side="left"))
    edge = float(starts[j])
    if pullback is None or forward is None:
        return edge
    x = pullback(edge)
    for _ in range(64):
        if forward(x) >= edge or x >= 1.0:
            break
        x = float(np.nextafter(x, 1.0))
    return x


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
    the edge is mapped back and nudged down until ``forward(x) < edge``.
    """
    j = int(np.searchsorted(levels, t, side="right")) - 1
    if j >= len(levels) - 1:
        return 1.0
    edge = float(starts[j + 1])
    if pullback is None or forward is None:
        return float(np.nextafter(edge, 0.0))
    x = float(np.nextafter(pullback(edge), 0.0))
    for _ in range(64):
        if forward(x) < edge or x <= 0.0:
            break
        x = float(np.nextafter(x, 0.0))
    return x


def migrate_state(
    state: Mapping[str, object],
    renames: Mapping[str, str],
    drop: tuple[str, ...] = (),
) -> dict[str, object]:
    """Map pre-0.4 private state keys onto their public names (schema-1 reads)."""
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
    bracketing segment is nudged up by ulps until ``forward(x) >= t``.
    """
    j = int(np.searchsorted(ms, t, side="left"))
    frac = (t - ms[j - 1]) / (ms[j] - ms[j - 1])
    x = float(xs[j - 1] + frac * (xs[j] - xs[j - 1]))
    x = min(max(x, float(xs[j - 1])), float(xs[j]))
    for _ in range(64):
        if forward(x) >= t or x >= 1.0:
            break
        x = float(np.nextafter(x, 1.0))
    return x


def linear_inverse_right(
    xs: np.ndarray, ms: np.ndarray, t: float, forward: Callable[[float], float]
) -> float:
    """``sup{x : g(x) <= t}`` for ``g = np.interp(x, xs, ms)``; ``ms[0] <= t < ms[-1]``.

    Nudged down by ulps until ``forward(x) <= t``.
    """
    j = int(np.searchsorted(ms, t, side="right")) - 1
    frac = (t - ms[j]) / (ms[j + 1] - ms[j])
    x = float(xs[j] + frac * (xs[j + 1] - xs[j]))
    x = min(max(x, float(xs[j])), float(xs[j + 1]))
    for _ in range(64):
        if forward(x) <= t or x <= 0.0:
            break
        x = float(np.nextafter(x, 0.0))
    return x
