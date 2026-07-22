"""Binning calibrators: histogram binning and scaling-binning.

Theory: ``docs/concepts/methods-nonparametric.md``.

References
----------
Zadrozny & Elkan (2001); Kumar, Liang & Ma (2019) — full records in the
documentation.
"""

import numpy as np

from ._results import Interpretation
from .base import BaseCalibrator
from .parametric import PlattCalibrator


def _equal_mass_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantile edges, deduplicated (ties can collapse bins)."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, qs))


class HistogramBinningCalibrator(BaseCalibrator):
    """Histogram binning: per-bin event rates with optional Jeffreys shrinkage.

    Parameters
    ----------
    n_bins : int
        Requested number of bins ``B`` — the bias–variance dial.
    strategy : {"mass", "width"}
        ``"mass"`` (equal-count, recommended default: lower estimator bias,
        no empty bins) or ``"width"`` (equal-width over [0, 1]).
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

    def __init__(
        self, n_bins: int = 10, strategy: str = "mass", shrinkage: str | None = "jeffreys"
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.shrinkage = shrinkage

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.strategy not in ("mass", "width"):
            raise ValueError(f"strategy must be 'mass' or 'width', got {self.strategy!r}")
        if self.shrinkage not in ("jeffreys", None):
            raise ValueError(f"shrinkage must be 'jeffreys' or None, got {self.shrinkage!r}")
        if self.strategy == "mass":
            edges = _equal_mass_edges(s, self.n_bins)
        else:
            edges = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        idx = np.searchsorted(edges, s, side="right")
        n_bins_eff = len(edges) + 1
        k = np.bincount(idx, weights=w * y, minlength=n_bins_eff)
        n = np.bincount(idx, weights=w, minlength=n_bins_eff)
        global_rate = float(np.average(y, weights=w))
        if self.shrinkage == "jeffreys":
            rate = (k + 0.5) / (n + 1.0)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(n > 0, k / np.maximum(n, 1e-300), np.nan)
        # Empty bins (possible under "width"): fall back to the global rate.
        empty = n == 0
        if self.shrinkage == "jeffreys":
            rate = np.where(empty, global_rate, rate)
        else:
            rate = np.where(empty | ~np.isfinite(rate), global_rate, rate)
        self.edges_ = edges
        self.bin_rate_ = rate
        self.bin_weight_ = n
        self.is_monotone_ = bool(np.all(np.diff(rate) >= 0.0))

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return self.bin_rate_[np.searchsorted(self.edges_, s, side="right")]

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


class ScalingBinningCalibrator(BaseCalibrator):
    """Scaling-binning (Kumar–Liang–Ma): Platt stage, then bin the fitted values.

    Fits Platt scaling first, then forms equal-mass bins of the *fitted
    function values* and outputs the mean of the fitted values within each
    bin. Achieves measurable calibration error with O(1/eps^2 + B) samples
    versus O(B/eps^2) for histogram binning.

    References
    ----------
    Kumar, Liang & Ma (2019).
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.platt_ = PlattCalibrator()
        self.platt_.fit(s, y, sample_weight=w)
        g = self.platt_.predict_proba(s)
        edges = _equal_mass_edges(g, self.n_bins)
        idx = np.searchsorted(edges, g, side="right")
        n_bins_eff = len(edges) + 1
        sums = np.bincount(idx, weights=w * g, minlength=n_bins_eff)
        cnts = np.bincount(idx, weights=w, minlength=n_bins_eff)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(cnts > 0, sums / np.maximum(cnts, 1e-300), np.nan)
        # Empty bins cannot arise under equal-mass edges built from g itself,
        # except through extreme ties; fall back to interpolation between neighbors.
        if np.any(~np.isfinite(means)):
            valid = np.isfinite(means)
            means = np.interp(np.arange(n_bins_eff), np.flatnonzero(valid), means[valid])
        self.edges_ = edges
        self.bin_value_ = means

    def _predict(self, s: np.ndarray) -> np.ndarray:
        g = self.platt_.predict_proba(s)
        return self.bin_value_[np.searchsorted(self.edges_, g, side="right")]

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
