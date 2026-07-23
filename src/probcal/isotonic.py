"""Isotonic calibrators: PAVA-based isotonic and centered isotonic regression (CIR).

Theory and worked example: ``docs/concepts/methods-nonparametric.md``.

References
----------
Barlow, Bartholomew, Bremner & Brunk (1972); Zadrozny & Elkan (2002);
Oron & Flournoy (2017) — full records in the documentation.
"""

import numpy as np

from ._math import pava
from ._results import Interpretation
from .base import BaseCalibrator


def _aggregate_ties(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by score and pool tied scores into weighted means."""
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    s_unique, start = np.unique(s_sorted, return_index=True)
    w_sum = np.add.reduceat(w_sorted, start)
    wy_sum = np.add.reduceat(w_sorted * y_sorted, start)
    return s_unique, wy_sum / w_sum, w_sum


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic calibration: the PAVA step function.

    Fits the least-squares non-decreasing map of outcomes on scores. The
    fitted map is a right-continuous step function with one level per pooled
    block; scores outside the calibration range clamp to the first/last
    level. ``interpolation="linear"`` instead joins block midpoints, removing
    the discontinuities.

    Attributes
    ----------
    n_blocks_ : int
        Number of pooled blocks — the effective complexity estimated from
        the data.
    block_mean_ : numpy.ndarray
        Event rate of each pooled block (the step levels).
    block_first_s_, block_last_s_ : numpy.ndarray
        Score range covered by each block.
    block_center_s_ : numpy.ndarray
        Weight-centered score coordinate of each block (used by CIR).

    References
    ----------
    Barlow et al. (1972) for PAVA; Zadrozny & Elkan (2002) for its use in
    classifier calibration.
    """

    def __init__(self, interpolation: str = "none") -> None:
        self.interpolation = interpolation

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.interpolation not in ("none", "linear"):
            raise ValueError(
                f"interpolation must be 'none' or 'linear', got {self.interpolation!r}"
            )
        s_u, y_u, w_u = _aggregate_ties(s, y, w)
        res = pava(y_u, w_u)
        starts = res.block_start
        ends = np.append(starts[1:], len(s_u)) - 1
        self.block_mean_ = res.block_mean
        self.block_first_s_ = s_u[starts]
        self.block_last_s_ = s_u[ends]
        centers = np.empty(len(starts))
        for j, (a, b) in enumerate(zip(starts, ends + 1, strict=True)):
            centers[j] = float(np.average(s_u[a:b], weights=w_u[a:b]))
        self.block_center_s_ = centers
        self.n_blocks_ = int(len(starts))

    def _predict(self, s: np.ndarray) -> np.ndarray:
        if self.interpolation == "linear":
            mid = 0.5 * (self.block_first_s_ + self.block_last_s_)
            return np.interp(s, mid, self.block_mean_)
        idx = np.searchsorted(self.block_first_s_, s, side="right") - 1
        idx = np.clip(idx, 0, self.n_blocks_ - 1)
        return self.block_mean_[idx]

    def _output_range(self) -> tuple[float, float]:
        return float(self.block_mean_[0]), float(self.block_mean_[-1])

    def _inverse_left(self, t: float) -> float:
        # Left edge of the first block whose level reaches t (spec block-edge semantics).
        j = int(np.searchsorted(self.block_mean_, t, side="left"))
        return float(self.block_first_s_[j])

    def _inverse_right(self, t: float) -> float:
        # Boundary after the last block whose level stays within t.
        j = int(np.searchsorted(self.block_mean_, t, side="right")) - 1
        if j >= self.n_blocks_ - 1:
            return 1.0
        return float(self.block_first_s_[j + 1])

    def interpret(self) -> Interpretation:
        """Read the block structure as effective complexity and local event rates."""
        self._check_fitted()
        flats = int(np.sum(np.isclose(np.diff(self.block_mean_), 0.0)))
        messages = [
            (
                f"{self.n_blocks_} pooled blocks: each step level is the empirical event "
                "rate of a score region the data could not subdivide further"
            ),
            (
                f"block count = effective complexity actually estimated from the data "
                f"(range of levels: {self.block_mean_[0]:.4g} to {self.block_mean_[-1]:.4g})"
            ),
        ]
        if flats:
            messages.append(
                f"{flats} adjacent block pairs share a level: expect tied predictions there"
            )
        messages.append(
            "output range is limited to the span of block levels; targets outside it are "
            "unattainable (relevant for interval_inverse)"
        )
        return Interpretation(
            method=type(self).__name__,
            param_names=("n_blocks",),
            param_values=(float(self.n_blocks_),),
            messages=tuple(messages),
        )


class CenteredIsotonicCalibrator(IsotonicCalibrator):
    """Centered isotonic regression (CIR): strictly increasing where data permit.

    Post-processes the PAVA solution by collapsing each block to its
    weight-centered score coordinate and interpolating linearly through the
    points (Oron & Flournoy, 2017). Removes the step function's tied
    predictions — preferred when downstream ranking must be strict.

    References
    ----------
    Oron & Flournoy (2017).
    """

    def __init__(self) -> None:
        super().__init__(interpolation="none")

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.block_center_s_, self.block_mean_)

    def _inverse_left(self, t: float) -> float:
        m, c = self.block_mean_, self.block_center_s_
        j = int(np.searchsorted(m, t, side="left"))
        if j == 0:
            return 0.0
        frac = (t - m[j - 1]) / (m[j] - m[j - 1])
        return float(c[j - 1] + frac * (c[j] - c[j - 1]))

    def _inverse_right(self, t: float) -> float:
        m, c = self.block_mean_, self.block_center_s_
        j = int(np.searchsorted(m, t, side="right")) - 1
        if j >= len(m) - 1:
            return 1.0
        frac = (t - m[j]) / (m[j + 1] - m[j])
        return float(c[j] + frac * (c[j + 1] - c[j]))

    def interpret(self) -> Interpretation:
        """Isotonic reading plus the strictness property CIR adds."""
        base = super().interpret()
        return Interpretation(
            method=type(self).__name__,
            param_names=base.param_names,
            param_values=base.param_values,
            messages=base.messages
            + (
                "centered isotonic interpolation is strictly increasing wherever block "
                "levels differ: distinct scores keep distinct predictions (no ties)",
            ),
        )
