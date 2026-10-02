"""Isotonic calibrators: PAVA-based isotonic and centered isotonic regression (CIR).

Theory and worked example: ``docs/concepts/methods-nonparametric.md``.

References
----------
Barlow, Bartholomew, Bremner & Brunk (1972); Zadrozny & Elkan (2002);
Oron & Flournoy (2017) — full records in the documentation.
"""

import numpy as np

from ._math import pava
from ._registry import register
from ._results import Interpretation
from ._steps import (
    aggregate_ties,
    eval_step,
    linear_inverse_left,
    linear_inverse_right,
    step_inverse_left,
    step_inverse_right,
)
from .base import BaseCalibrator


@register
class IsotonicCalibrator(BaseCalibrator):
    """Isotonic calibration: the PAVA step function.

    Fits the least-squares non-decreasing map of outcomes on scores. The
    fitted map is a right-continuous step function with one level per pooled
    block; scores outside the calibration range clamp to the first/last
    level. ``interpolation="linear"`` instead joins block midpoints, removing
    the discontinuities.

    Parameters
    ----------
    interpolation : {"none", "linear"}
        ``"none"`` (default) keeps the raw step function; ``"linear"`` joins
        the block midpoints, removing the tied-prediction plateaus.

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

    _STATE_ATTRS = (
        "block_mean_",
        "block_first_s_",
        "block_last_s_",
        "block_center_s_",
        "n_blocks_",
    )

    def __init__(self, interpolation: str = "none") -> None:
        self.interpolation = interpolation

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.interpolation not in ("none", "linear"):
            raise ValueError(
                f"interpolation must be 'none' or 'linear', got {self.interpolation!r}"
            )
        s_u, y_u, w_u = aggregate_ties(s, y, w)
        res = pava(y_u, w_u)
        starts = res.block_start
        ends = np.append(starts[1:], len(s_u)) - 1
        self.block_mean_ = res.block_mean
        self.block_first_s_ = s_u[starts]
        self.block_last_s_ = s_u[ends]
        self.block_center_s_ = np.add.reduceat(w_u * s_u, starts) / np.add.reduceat(w_u, starts)
        self.n_blocks_ = int(len(starts))

    def _block_mid(self) -> np.ndarray:
        return 0.5 * (self.block_first_s_ + self.block_last_s_)

    def _predict(self, s: np.ndarray) -> np.ndarray:
        if self.interpolation == "linear":
            return np.interp(s, self._block_mid(), self.block_mean_)
        return eval_step(self.block_first_s_, self.block_mean_, s)

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 50.0: nonparametric, data-driven block count."""
        return 50.0

    def _output_range(self) -> tuple[float, float]:
        return float(self.block_mean_[0]), float(self.block_mean_[-1])

    def _inverse_left(self, t: float) -> float:
        if self.interpolation == "linear":
            # The interpolated map is continuous: invert it on the bracketing
            # segment between block midpoints, not through the step blocks.
            return linear_inverse_left(self._block_mid(), self.block_mean_, t, self._predict_scalar)
        # Left edge of the first block whose level reaches t (spec block-edge semantics).
        return step_inverse_left(self.block_mean_, self.block_first_s_, t)

    def _inverse_right(self, t: float) -> float:
        if self.interpolation == "linear":
            return linear_inverse_right(
                self._block_mid(), self.block_mean_, t, self._predict_scalar
            )
        # Largest raw score whose level stays within t: one float below the
        # next block's left edge, so the returned bound is itself in the
        # preimage — consumers may treat both bounds as closed.
        return step_inverse_right(self.block_mean_, self.block_first_s_, t)

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


@register
class CenteredIsotonicCalibrator(IsotonicCalibrator):
    """Centered isotonic regression (CIR): strictly increasing where data permit.

    Post-processes the PAVA solution by collapsing each block to its
    weight-centered score coordinate and interpolating linearly through the
    points (Oron & Flournoy, 2017). Removes the step function's tied
    predictions — preferred when downstream ranking must be strict.

    Attributes
    ----------
    n_blocks_ : int
        Number of pooled blocks — the effective complexity estimated from
        the data. Inherited from the PAVA fit.
    block_mean_ : numpy.ndarray
        Event rate of each pooled block (the interpolation y-values).
    block_first_s_, block_last_s_ : numpy.ndarray
        Score range covered by each block (inherited; not used for
        prediction, which interpolates through ``block_center_s_`` instead).
    block_center_s_ : numpy.ndarray
        Weight-centered score coordinate of each block — the interpolation
        x-values that make CIR strictly increasing.

    References
    ----------
    Oron & Flournoy (2017).
    """

    def __init__(self) -> None:
        super().__init__(interpolation="none")

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.block_center_s_, self.block_mean_)

    def _inverse_left(self, t: float) -> float:
        return linear_inverse_left(self.block_center_s_, self.block_mean_, t, self._predict_scalar)

    def _inverse_right(self, t: float) -> float:
        return linear_inverse_right(self.block_center_s_, self.block_mean_, t, self._predict_scalar)

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
