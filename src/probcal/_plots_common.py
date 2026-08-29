"""Shared plotting internals for ``probcal.plots`` and ``probcal._plots_diag``.

Import guard, style constants, and small axis helpers used by every plot
function. Split out of ``plots.py`` so the diagram modules can share them
without a circular import (``plots.py`` re-exports the same names).
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


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for probcal.plots — install the viz extra: "
            "pip install probcal[viz]"
        )


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
