"""The one binning pass behind every binned metric and curve.

``ece``/``ece_debiased``/``ece_sweep``, ``hosmer_lemeshow``,
``murphy_decomposition`` and ``curves.reliability_binned`` all bin through
:func:`bin_stats`. Only numpy is imported, so a calibrator module can reuse it
without pulling in the metrics package.
"""

from typing import NamedTuple

import numpy as np

from .._validation import validate_positive_int


class BinStats(NamedTuple):
    """Per-bin weighted sums over the *non-empty* bins, in ascending-``p`` order."""

    w_sum: np.ndarray
    """Sum of weights."""
    wy_sum: np.ndarray
    """Sum of ``w * y``."""
    wp_sum: np.ndarray
    """Sum of ``w * p``."""
    n_obs: np.ndarray
    """Raw observation count (int64)."""

    @property
    def rate(self) -> np.ndarray:
        """Weighted event rate per bin."""
        return self.wy_sum / self.w_sum

    @property
    def p_mean(self) -> np.ndarray:
        """Weighted mean prediction per bin."""
        return self.wp_sum / self.w_sum


def bin_index(p: np.ndarray, n_bins: object, strategy: str) -> tuple[np.ndarray, int]:
    """Bin label per observation and the effective bin count.

    ``"mass"`` cuts at the ``1/n_bins`` quantiles of ``p`` (duplicate edges
    collapse, so heavily tied scores give fewer bins); ``"width"`` cuts
    ``[0, 1]`` into equal-width bins. A score equal to an edge goes to the
    upper bin (``searchsorted(edges, p, side="right")``).
    """
    b = validate_positive_int(n_bins, "n_bins")
    if strategy == "mass":
        qs = np.linspace(0.0, 1.0, b + 1)[1:-1]
        edges = np.unique(np.quantile(p, qs))
    elif strategy == "width":
        edges = np.linspace(0.0, 1.0, b + 1)[1:-1]
    else:
        raise ValueError(f"strategy must be 'mass' or 'width', got {strategy!r}")
    return np.searchsorted(edges, p, side="right"), len(edges) + 1


def bin_stats(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, n_bins: object, strategy: str
) -> BinStats:
    """Weighted per-bin sums for the non-empty bins of :func:`bin_index`.

    Weights are strictly positive (validated upstream), so a bin is non-empty
    exactly when its raw count is positive.
    """
    idx, m = bin_index(p, n_bins, strategy)
    count = np.bincount(idx, minlength=m)
    keep = count > 0
    return BinStats(
        w_sum=np.bincount(idx, weights=w, minlength=m)[keep],
        wy_sum=np.bincount(idx, weights=w * y, minlength=m)[keep],
        wp_sum=np.bincount(idx, weights=w * p, minlength=m)[keep],
        n_obs=count[keep].astype(np.int64),
    )
