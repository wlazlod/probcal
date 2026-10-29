"""The one binning pass behind every binned metric and curve.

``ece``/``ece_debiased``/``ece_sweep``, ``hosmer_lemeshow``,
``murphy_decomposition`` and ``curves.reliability_binned`` all bin through
:func:`bin_stats`. Only numpy is imported, so a calibrator module can reuse it
without pulling in the metrics package.
"""

from typing import NamedTuple

import numpy as np

from .._steps import bin_lookup, bin_sums, equal_mass_edges, equal_width_edges
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
    w2_sum: np.ndarray
    """Sum of ``w**2`` (equals ``n_obs`` for unit weights)."""
    w2p_sum: np.ndarray | None = None
    """Sum of ``w**2 * p`` (only when requested, ``None`` otherwise)."""

    @property
    def n_eff(self) -> np.ndarray:
        """Per-bin effective count ``w_sum**2 / w2_sum`` (``n_obs`` for unit weights)."""
        return self.w_sum * self.w_sum / self.w2_sum

    def debias(self, sq: np.ndarray) -> np.ndarray:
        """Per-bin squared gaps ``sq`` minus the event rate's within-bin variance.

        The variance is ``rate * (1 - rate) / (n_eff - 1)``, floored at zero
        after subtraction; bins with ``n_eff <= 1`` are returned unchanged.
        """
        n_eff = self.n_eff
        multi = n_eff > 1.0
        var = self.rate * (1.0 - self.rate) / np.where(multi, n_eff - 1.0, 1.0)
        return np.where(multi, np.maximum(sq - var, 0.0), sq)

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

    ``"mass"`` cuts at the ``1/n_bins`` quantiles of ``p``
    (:func:`probcal._steps.equal_mass_edges`: duplicate edges collapse and
    an empty bin merges into its upper neighbour, so heavily tied scores give
    fewer bins); ``"width"`` cuts ``[0, 1]`` into equal-width bins. A score
    equal to an edge goes to the upper bin (``searchsorted(edges, p,
    side="right")``).
    """
    b = validate_positive_int(n_bins, "n_bins")
    if strategy == "mass":
        edges = equal_mass_edges(p, b)
    elif strategy == "width":
        edges = equal_width_edges(b)
    else:
        raise ValueError(f"strategy must be 'mass' or 'width', got {strategy!r}")
    return bin_lookup(edges, p), len(edges) + 1


def bin_stats(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    n_bins: object,
    strategy: str,
    *,
    w2p: bool = False,
) -> BinStats:
    """Weighted per-bin sums for the non-empty bins of :func:`bin_index`.

    Weights are strictly positive (validated upstream), so a bin is non-empty
    exactly when its raw count is positive. ``w2p=True`` also fills
    ``w2p_sum`` (the Hosmer-Lemeshow variance term).
    """
    idx, m = bin_index(p, n_bins, strategy)
    count = np.bincount(idx, minlength=m)
    keep = count > 0
    wy_sum, w_sum = bin_sums(idx, y, w, m)
    w2 = w * w
    return BinStats(
        w_sum=w_sum[keep],
        wy_sum=wy_sum[keep],
        wp_sum=np.bincount(idx, weights=w * p, minlength=m)[keep],
        n_obs=count[keep].astype(np.int64),
        w2_sum=np.bincount(idx, weights=w2, minlength=m)[keep],
        w2p_sum=np.bincount(idx, weights=w2 * p, minlength=m)[keep] if w2p else None,
    )
