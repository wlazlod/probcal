"""Shared plotting internals for ``probcal.plots`` and ``probcal._plots_diag``.

Import guard, style constants, and small axis/data helpers used by every plot
function. Split out of ``plots.py`` so the diagnostic module can share them
without a circular import.
"""

import math
from typing import Any

import numpy as np

from ._math import logit

try:
    import matplotlib.pyplot as _plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _plt = None  # type: ignore[assignment]
    _HAS_MPL = False

_TICK_PROBS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99)

_STYLE: Any = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
}
_BLUE, _ORANGE = "#2f5f8a", "#d97b29"  # primary data; smooth overlays
_GREEN, _RED = "#3a8a4d", "#b23a3a"  # pass/chosen/after; fail/events/before
_AMBER, _GREY = "#d9a521", "#9a9a9a"  # warnings; identity/reference lines
_BOX = {"boxstyle": "round", "fc": "#f7f7f5", "ec": "#cccccc"}
_RUG_MAX = 1000

# Probability clip applied before a logit transform: curves/edges use the tight
# 1e-12 (only exact 0/1 move), confidence bounds the looser 1e-6 so that a CI
# touching 0 or 1 draws a long-but-finite whisker instead of rescaling the axes
# to +-27 log-odds.
_CLIP_EPS = 1e-12
_CI_CLIP_EPS = 1e-6
# Logit-axis fallback span when a curve has no finite point to anchor the identity.
_DEFAULT_LOGIT_SPAN = (float(logit(np.array([0.001]))[0]), float(logit(np.array([0.5]))[0]))


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for probcal.plots — install the viz extra: "
            "pip install probcal[viz]"
        )


def _style_axes(ax: Any) -> None:
    """Apply the axes-level part of ``_STYLE`` explicitly.

    ``rc_context`` only affects axes *created* inside it, so a caller-supplied
    ``ax`` would otherwise keep its own spines and grid; applying them here
    makes the output independent of whether ``ax`` was passed. On axes created
    under ``_STYLE`` this is a no-op (same values).
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=_STYLE["grid.alpha"], linewidth=_STYLE["grid.linewidth"])


def _get_axes(ax: Any, figsize: tuple[float, float]) -> Any:
    """Return ``ax`` (or a fresh one of ``figsize``) with the house style applied.

    Must be called inside ``_plt.rc_context(_STYLE)``.
    """
    if ax is None:
        _, ax = _plt.subplots(figsize=figsize)
    _style_axes(ax)
    return ax


def _tr(x: Any, scale: str, eps: float = _CLIP_EPS) -> np.ndarray:
    """Map probabilities to plot coordinates: identity, or clipped logit."""
    arr = np.asarray(x, dtype=np.float64)
    if scale == "logit":
        return logit(np.clip(arr, eps, 1.0 - eps))
    return arr


def _scale_axis_labels(ax: Any, scale: str, xlabel: str, ylabel: str) -> None:
    """Set axis labels, plus probability-labelled logit ticks on ``scale="logit"``."""
    if scale == "logit":
        _logit_axis(ax)
        ax.set_xlabel(f"{xlabel} (logit scale)")
        ax.set_ylabel(f"{ylabel} (logit scale)")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


def _pav_step_xy(
    lo: np.ndarray, hi: np.ndarray, level: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """PAV blocks ``[lo, hi]`` at ``level`` as ``step(where="post")`` coordinates."""
    x_edges = np.empty(2 * len(lo))
    x_edges[0::2] = lo
    x_edges[1::2] = hi
    return x_edges, np.repeat(level, 2)


def _text_box(ax: Any, txt: str) -> None:
    """The top-left annotation box shared by every plot that prints numbers."""
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9, bbox=_BOX)


def _logit_axis(ax: Any, axis: str = "both") -> None:
    """Label logit-positioned ticks in probabilities."""
    ticks = logit(np.array(_TICK_PROBS))
    labels = [f"{q:g}" for q in _TICK_PROBS]
    if axis in ("x", "both"):
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    if axis in ("y", "both"):
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def _rug_subsample(values: np.ndarray) -> np.ndarray:
    """Deterministic thinning: sort, then take an evenly strided subset (no RNG)."""
    v = np.sort(values)
    if len(v) > _RUG_MAX:
        v = v[:: math.ceil(len(v) / _RUG_MAX)]
    return v
