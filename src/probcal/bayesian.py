"""Bayesian-ensemble calibrators: BBQ and ENIR.

Theory: ``docs/concepts/methods-nonparametric.md``.

References
----------
Naeini, Cooper & Hauskrecht (2015); Naeini & Cooper (2016); Tibshirani,
Hoefling & Tibshirani (2011) — full records in the documentation.
"""

import numpy as np

from ._math import lgamma_vec
from ._results import Interpretation
from ._validation import EPS
from .base import BaseCalibrator
from .binning import _equal_mass_edges

_JEFFREYS = 0.5


class BBQCalibrator(BaseCalibrator):
    """Bayesian Binning into Quantiles: model averaging over equal-mass binnings.

    Considers equal-mass binning models over a range of bin counts, scores
    each by its Beta-Binomial log marginal likelihood under a per-bin
    Jeffreys Beta(1/2, 1/2) prior, and predicts with the posterior-weighted
    average of the models' (posterior-mean) bin rates.

    Parameters
    ----------
    min_bins, max_bins : int or None
        Range of candidate bin counts; defaults to ``[2, ceil(sqrt(n))]``
        capped at 50 (DECISIONS entry).

    Attributes
    ----------
    bins_grid_ : numpy.ndarray
        Candidate bin counts.
    weights_ : numpy.ndarray
        Posterior weights over the candidates (sum to 1).

    References
    ----------
    Naeini, Cooper & Hauskrecht (2015).
    """

    def __init__(self, min_bins: int | None = None, max_bins: int | None = None) -> None:
        self.min_bins = min_bins
        self.max_bins = max_bins

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        n = len(s)
        lo = 2 if self.min_bins is None else self.min_bins
        hi = min(50, int(np.ceil(np.sqrt(n)))) if self.max_bins is None else self.max_bins
        hi = max(hi, lo)
        self.bins_grid_ = np.arange(lo, hi + 1)

        models = []
        log_marg = np.empty(len(self.bins_grid_))
        for i, b in enumerate(self.bins_grid_):
            edges = _equal_mass_edges(s, int(b))
            idx = np.searchsorted(edges, s, side="right")
            m = len(edges) + 1
            k = np.bincount(idx, weights=w * y, minlength=m)
            tot = np.bincount(idx, weights=w, minlength=m)
            # Beta-Binomial log marginal likelihood, Jeffreys prior per bin.
            a0 = b0 = _JEFFREYS
            log_marg[i] = float(
                np.sum(
                    lgamma_vec(k + a0)
                    + lgamma_vec(tot - k + b0)
                    - lgamma_vec(tot + a0 + b0)
                    - (
                        lgamma_vec(np.full(m, a0))
                        + lgamma_vec(np.full(m, b0))
                        - lgamma_vec(np.full(m, a0 + b0))
                    )
                )
            )
            rate = (k + a0) / (tot + a0 + b0)
            models.append((edges, rate))
        shifted = log_marg - log_marg.max()
        wgt = np.exp(shifted)
        self.weights_ = wgt / wgt.sum()
        self._models = models
        probe = np.linspace(0.01, 0.99, 199)
        self.is_monotone_ = bool(np.all(np.diff(self._predict(probe)) >= -1e-12))

    def _predict(self, s: np.ndarray) -> np.ndarray:
        out = np.zeros(len(s))
        for weight, (edges, rate) in zip(self.weights_, self._models, strict=True):
            out += weight * rate[np.searchsorted(edges, s, side="right")]
        return out

    def interpret(self) -> Interpretation:
        """Read the posterior weights as uncertainty about the data's resolution."""
        self._check_fitted()
        top = np.argsort(self.weights_)[::-1][:3]
        top_txt = ", ".join(
            f"B={int(self.bins_grid_[i])} (weight {self.weights_[i]:.3f})" for i in top
        )
        return Interpretation(
            method=type(self).__name__,
            param_names=("n_models",),
            param_values=(float(len(self.bins_grid_)),),
            messages=(
                f"top-3 binning models by posterior weight: {top_txt}",
                "concentrated weight = the data speak clearly about their own resolution; "
                "diffuse weight = the averaging is doing real work",
            ),
        )


class ENIRCalibrator(BaseCalibrator):
    """Ensemble of near-isotonic regressions (ENIR).

    Computes the full nearly-isotonic solution path (modified PAVA of
    Tibshirani, Hoefling & Tibshirani, 2011) from the raw data (lambda = 0)
    to the fully isotonic fit, then averages the breakpoint solutions with
    BIC weights. The combined map may be non-monotone: ``is_monotone_`` is
    ``False`` and consumers requiring order preservation should prefer a
    monotone calibrator.

    Attributes
    ----------
    path_lambdas_ : numpy.ndarray
        Breakpoints of the penalty parameter, starting at 0.
    path_solutions_ : numpy.ndarray of shape (T, m)
        Fitted values on the tie-aggregated score grid at each breakpoint.
    weights_ : numpy.ndarray
        BIC weights over the path solutions (sum to 1).

    References
    ----------
    Naeini & Cooper (2016); Tibshirani, Hoefling & Tibshirani (2011).
    """

    is_monotone_: bool = False

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        order = np.argsort(s, kind="stable")
        s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
        s_u, start = np.unique(s_sorted, return_index=True)
        w_u = np.add.reduceat(w_sorted, start)
        y_u = np.add.reduceat(w_sorted * y_sorted, start) / w_u
        self._x = s_u

        lambdas, solutions = self._nearly_isotonic_path(y_u, w_u)
        self.path_lambdas_ = np.asarray(lambdas)
        self.path_solutions_ = np.asarray(solutions)

        # BIC weights: binomial log-likelihood with clipped probabilities,
        # k = number of distinct fitted levels (group count).
        n_tot = float(w_u.sum())
        bics = np.empty(len(solutions))
        for t, sol in enumerate(solutions):
            p = np.clip(sol, EPS, 1.0 - EPS)
            loglik = float(np.sum(w_u * (y_u * np.log(p) + (1.0 - y_u) * np.log(1.0 - p))))
            k_groups = 1 + int(np.sum(np.abs(np.diff(sol)) > 1e-12))
            bics[t] = -2.0 * loglik + k_groups * np.log(n_tot)
        shifted = -0.5 * (bics - bics.min())
        wgt = np.exp(shifted)
        self.weights_ = wgt / wgt.sum()

    @staticmethod
    def _nearly_isotonic_path(y: np.ndarray, w: np.ndarray) -> tuple[list, list]:
        """Modified PAVA: breakpoints and solutions of the nearly-isotonic path."""
        # Groups: values beta_g(lam) = mean_g - (lam/weight_g) * slope_g with
        # slope_g = 1{beta_g > beta_{g+1}} - 1{beta_{g-1} > beta_g}.
        means = list(y.astype(float))
        weights = list(w.astype(float))
        members = [[i] for i in range(len(y))]
        lam = 0.0
        lambdas = [0.0]
        solutions = [np.array(y, dtype=float)]

        def expand(vals: list) -> np.ndarray:
            out = np.empty(len(y))
            for g, mem in enumerate(members):
                out[mem] = vals[g]
            return out

        while True:
            g_count = len(means)
            if g_count == 1:
                break
            # `means` holds the group values AT the current lam. Stationarity of
            # 1/2 sum w (y - m)^2 + lam * sum (m_i - m_{i+1})_+ gives, while the
            # active violation set is fixed, d beta_g / d lam = -slope_g with
            # slope_g = (1{beta_g > beta_{g+1}} - 1{beta_{g-1} > beta_g}) / W_g.
            viol = [means[g] > means[g + 1] + 1e-15 for g in range(g_count - 1)]
            if not any(viol):
                break
            slopes = []
            for g in range(g_count):
                s_g = 0.0
                if g < g_count - 1 and viol[g]:
                    s_g += 1.0
                if g > 0 and viol[g - 1]:
                    s_g -= 1.0
                slopes.append(s_g / weights[g])
            # Earliest collision among adjacent groups. gap(t) = gap - t * dv,
            # so a pair meets at t = gap / dv when gap and dv share a sign:
            # violating pairs (gap > 0) close with dv > 0; non-violating pairs
            # (gap < 0) can be driven together by outer violations (dv < 0).
            t_best, g_best = np.inf, -1
            for g in range(g_count - 1):
                dv = slopes[g] - slopes[g + 1]
                gap = means[g] - means[g + 1]
                if (gap > 0 and dv > 1e-300) or (gap < 0 and dv < -1e-300):
                    t = gap / dv
                    if 0.0 <= t < t_best:
                        t_best, g_best = t, g
            if not np.isfinite(t_best):
                break
            for g in range(g_count):
                means[g] -= slopes[g] * t_best
            lam += t_best
            g = g_best
            new_w = weights[g] + weights[g + 1]
            new_mean = (weights[g] * means[g] + weights[g + 1] * means[g + 1]) / new_w
            means[g : g + 2] = [new_mean]
            weights[g : g + 2] = [new_w]
            members[g : g + 2] = [members[g] + members[g + 1]]
            lambdas.append(lam)
            solutions.append(expand(means))
        return lambdas, solutions

    def _predict(self, s: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self._x, s, side="right") - 1, 0, len(self._x) - 1)
        out = np.zeros(len(s))
        for weight, sol in zip(self.weights_, self.path_solutions_, strict=True):
            out += weight * sol[idx]
        return np.clip(out, EPS, 1.0 - EPS)

    def interpret(self) -> Interpretation:
        """Read the path length and BIC weights; warn about non-monotonicity."""
        self._check_fitted()
        top = np.argsort(self.weights_)[::-1][:3]
        top_txt = ", ".join(
            f"lambda={self.path_lambdas_[i]:.4g} (weight {self.weights_[i]:.3f})" for i in top
        )
        return Interpretation(
            method=type(self).__name__,
            param_names=("n_path_solutions",),
            param_values=(float(len(self.path_lambdas_)),),
            messages=(
                f"top-3 path solutions by BIC weight: {top_txt}",
                "lambda trades monotonicity strictness against fit; BIC weights are model "
                "plausibility along the path",
                "the ensemble output may be non-monotone (is_monotone_ = False): consumers "
                "requiring order preservation should use a monotone calibrator",
            ),
        )
