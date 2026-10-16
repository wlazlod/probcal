"""Binning calibrators: histogram binning and scaling-binning.

Theory: ``docs/concepts/methods-nonparametric.md``.

References
----------
Zadrozny & Elkan (2001); Kumar, Liang & Ma (2019) — full records in the
documentation.
"""

from collections.abc import Callable

import numpy as np

from ._registry import register
from ._results import Interpretation
from ._steps import bin_sums, equal_mass_edges, step_inverse_left, step_inverse_right
from ._validation import validate_positive_int
from .base import BaseCalibrator
from .parametric import PlattCalibrator


def _bin_starts(edges: np.ndarray) -> np.ndarray:
    """Left end of every bin: ``0`` followed by the interior edges."""
    return np.concatenate([[0.0], edges])


@register
class HistogramBinningCalibrator(BaseCalibrator):
    """Histogram binning: per-bin event rates with optional Jeffreys shrinkage.

    Parameters
    ----------
    n_bins : int
        Requested number of bins ``B`` — the bias–variance dial.
    strategy : {"mass", "width"}
        ``"mass"`` (equal-mass, recommended default: lower estimator bias,
        no empty bins; with ``sample_weight`` the bins carry equal weight)
        or ``"width"`` (equal-width over [0, 1]).
    shrinkage : {"jeffreys", None}
        ``"jeffreys"`` replaces the raw rate ``k/n`` with ``(k + 1/2)/(n + 1)``
        — the posterior mean under the Beta(1/2, 1/2) prior — keeping small
        bins away from 0 and 1.

    Attributes
    ----------
    bin_rate_ : numpy.ndarray
        Calibrated value per (non-degenerate) bin.
    is_monotone_ : bool
        Computed after fitting: binning does not assume monotonicity, so the
        flag reports whether the fitted rates happen to be non-decreasing.

    References
    ----------
    Zadrozny & Elkan (2001).
    """

    _STATE_ATTRS = ("edges_", "bin_rate_", "bin_weight_", "is_monotone_")

    def __init__(
        self, n_bins: int = 10, strategy: str = "mass", shrinkage: str | None = "jeffreys"
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.shrinkage = shrinkage

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        n_bins = validate_positive_int(self.n_bins, "n_bins")
        if self.strategy not in ("mass", "width"):
            raise ValueError(f"strategy must be 'mass' or 'width', got {self.strategy!r}")
        if self.shrinkage not in ("jeffreys", None):
            raise ValueError(f"shrinkage must be 'jeffreys' or None, got {self.shrinkage!r}")
        if self.strategy == "mass":
            edges = equal_mass_edges(s, n_bins, w)
        else:
            edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        idx = np.searchsorted(edges, s, side="right")
        k, n = bin_sums(idx, y, w, len(edges) + 1)
        prior = 0.5 if self.shrinkage == "jeffreys" else 0.0
        # Equal-mass edges leave no bin empty. Under "width" a bin can be
        # empty; it takes its neighbours' rate (interpolated by bin index, the
        # nearest filled bin at either end), so it never breaks the ordering
        # of the filled bins around it.
        filled = n > 0
        filled_rate = (k[filled] + prior) / (n[filled] + 2.0 * prior)
        rate = np.interp(np.arange(len(n)), np.flatnonzero(filled), filled_rate)
        self.edges_ = edges
        self.bin_rate_ = rate
        self.bin_weight_ = n
        self.is_monotone_ = bool(np.all(np.diff(rate) >= 0.0))

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return self.bin_rate_[np.searchsorted(self.edges_, s, side="right")]

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 10.0, for either strategy ("mass" or "width")."""
        return 10.0

    def _output_range(self) -> tuple[float, float]:
        return float(self.bin_rate_[0]), float(self.bin_rate_[-1])

    def _inverse_left(self, t: float) -> float:
        return step_inverse_left(self.bin_rate_, _bin_starts(self.edges_), t)

    def _inverse_right(self, t: float) -> float:
        # One float below the next bin's edge: the bound is in the preimage,
        # so closed-bound consumers cannot overshoot a plateau.
        return step_inverse_right(self.bin_rate_, _bin_starts(self.edges_), t)

    def interpret(self) -> Interpretation:
        """Read bin rates as local event frequencies and B as the complexity dial."""
        self._check_fitted()
        messages = [
            (
                f"{len(self.bin_rate_)} bins ({self.strategy} strategy): each calibrated "
                "value is the (shrunken) empirical event rate of its score bin"
            ),
            "B controls bias-variance: few bins are stable but coarse, many are sharp but noisy",
        ]
        if self.shrinkage == "jeffreys":
            messages.append("Jeffreys shrinkage (k+1/2)/(n+1) keeps sparse bins away from 0 and 1")
        if not self.is_monotone_:
            messages.append(
                "fitted bin rates are not monotone: binning does not enforce ranking "
                "preservation — read inversions as noise, not signal"
            )
        return Interpretation(
            method=type(self).__name__,
            param_names=("n_bins",),
            param_values=(float(len(self.bin_rate_)),),
            messages=tuple(messages),
        )


@register
class ScalingBinningCalibrator(BaseCalibrator):
    """Scaling-binning (Kumar–Liang–Ma): Platt stage, then bin the fitted values.

    Fits Platt scaling first, then forms equal-mass bins of the *fitted
    function values* and outputs the mean of the fitted values within each
    bin. Achieves measurable calibration error with O(1/eps^2 + B) samples
    versus O(B/eps^2) for histogram binning.

    Parameters
    ----------
    n_bins : int
        Requested number of equal-mass bins of the Platt-fitted values (equal
        weight under ``sample_weight``).

    Attributes
    ----------
    platt_ : PlattCalibrator
        The fitted first-stage Platt calibrator.
    edges_ : numpy.ndarray
        Interior quantile edges of the Platt-fitted values.
    bin_value_ : numpy.ndarray
        Mean Platt-fitted value per bin (the calibrated output for that bin).
    is_monotone_ : bool
        Derived from the fitted state: ``True`` when the Platt stage is
        increasing (``a > 0``) — the bin means are then non-decreasing by
        construction. A decreasing Platt stage makes the composite map
        decreasing, and ``interval_inverse`` refuses it.

    References
    ----------
    Kumar, Liang & Ma (2019).
    """

    _STATE_ATTRS = ("platt_", "edges_", "bin_value_")

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        n_bins = validate_positive_int(self.n_bins, "n_bins")
        self.platt_ = PlattCalibrator()
        self.platt_.fit(s, y, sample_weight=w)
        g = self.platt_.predict_proba(s)
        edges = equal_mass_edges(g, n_bins, w)
        idx = np.searchsorted(edges, g, side="right")
        sums, cnts = bin_sums(idx, g, w, len(edges) + 1)
        self.edges_ = edges
        self.bin_value_ = sums / cnts  # equal_mass_edges leaves no bin empty

    @property  # type: ignore[override]
    def is_monotone_(self) -> bool:  # type: ignore[override]
        """``True`` iff the Platt stage is increasing (see class docstring)."""
        platt = self.__dict__.get("platt_")
        if platt is None:
            return True
        return bool(platt.is_monotone_) and bool(np.all(np.diff(self.bin_value_) >= 0.0))

    def _predict(self, s: np.ndarray) -> np.ndarray:
        g = self.platt_.predict_proba(s)
        return self.bin_value_[np.searchsorted(self.edges_, g, side="right")]

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 4.0: a Platt stage plus a bin count, still lightweight."""
        return 4.0

    def _output_range(self) -> tuple[float, float]:
        return float(self.bin_value_[0]), float(self.bin_value_[-1])

    def _inverse_left(self, t: float) -> float:
        return self._step_inverse(step_inverse_left, t)

    def _inverse_right(self, t: float) -> float:
        # The pulled-back edge is nudged below the bin boundary, so the bound
        # is itself in the preimage (closed-bound consumers cannot overshoot).
        return self._step_inverse(step_inverse_right, t)

    def _step_inverse(self, inverse: Callable[..., float], t: float) -> float:
        # Bin edges live on the Platt scale; the Platt stage's closed-form
        # inverse pulls them back to raw scores. Only monotone fits reach here,
        # so the Platt slope is positive and that inverse exists.
        return inverse(
            self.bin_value_,
            _bin_starts(self.edges_),
            t,
            forward=self.platt_._predict_scalar,
            pullback=self.platt_._affine_inverse,
        )

    def interpret(self) -> Interpretation:
        """Two-stage reading: Platt map, then the error-measurability discretization."""
        self._check_fitted()
        platt_interp = self.platt_.interpret()
        return Interpretation(
            method=type(self).__name__,
            param_names=platt_interp.param_names + ("n_bins",),
            param_values=platt_interp.param_values + (float(len(self.bin_value_)),),
            messages=platt_interp.messages
            + (
                (
                    f"binning stage: {len(self.bin_value_)} equal-mass bins of the fitted "
                    "Platt values; outputs are bin means, which makes the residual "
                    "calibration error estimable with O(1/eps^2 + B) samples "
                    "(vs O(B/eps^2) for histogram binning)"
                ),
            ),
        )
