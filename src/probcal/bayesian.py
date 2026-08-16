"""Bayesian-ensemble calibrators: BBQ and ENIR.

Theory: ``docs/concepts/methods-nonparametric.md``.

References
----------
Naeini, Cooper & Hauskrecht (2015); Naeini & Cooper (2016); Tibshirani,
Hoefling & Tibshirani (2011) — full records in the documentation.
"""

import heapq
import math
import warnings

import numpy as np

from ._math import lgamma_vec, pava
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

    Parameters
    ----------
    max_solutions : int or None
        Number of path solutions to keep for the ensemble, chosen by best
        (lowest) BIC; ``None`` keeps every breakpoint. Retention is what
        bounds memory: the path has up to ``m`` breakpoints, so keeping all
        of them costs ``O(m^2)``.

    Attributes
    ----------
    path_lambdas_ : numpy.ndarray of shape (T,)
        Breakpoints of the penalty parameter, starting at 0. All breakpoints
        are recorded, whether or not their solution is retained.
    path_solutions_ : numpy.ndarray of shape (K, m)
        Fitted values on the tie-aggregated score grid at the retained
        breakpoints, in breakpoint order. ``K`` is the number of retained
        breakpoints: at most ``max_solutions``, and fewer when breakpoints are
        pruned (a breakpoint whose BIC weight is provably below 1e-15 relative
        is skipped without being scored, and is retained only when
        ``max_solutions`` is ``None``).
    kept_breakpoints_ : numpy.ndarray of shape (K,)
        Indices into ``path_lambdas_`` of the retained breakpoints.
    weights_ : numpy.ndarray of shape (K,)
        BIC weights over the retained solutions, renormalized to sum to 1.
    dropped_weight_ : float
        BIC weight lost to retention — the weight of scored solutions that the
        ``max_solutions`` cap evicted, before renormalization; a
        ``UserWarning`` is raised above 1e-6. Pruned breakpoints do not count
        towards it: their weight is exactly 0 in double precision.

    References
    ----------
    Naeini & Cooper (2016); Tibshirani, Hoefling & Tibshirani (2011).
    """

    is_monotone_: bool = False

    def __init__(self, max_solutions: int | None = 256) -> None:
        self.max_solutions = max_solutions

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.max_solutions is not None and self.max_solutions < 1:
            raise ValueError("max_solutions must be a positive integer or None")
        order = np.argsort(s, kind="stable")
        s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
        s_u, start = np.unique(s_sorted, return_index=True)
        w_u = np.add.reduceat(w_sorted, start)
        y_u = np.add.reduceat(w_sorted * y_sorted, start) / w_u
        self._x = s_u
        self._path_fit(y_u, w_u)
        self._mixed = self.weights_ @ self.path_solutions_

    def _path_fit(self, y: np.ndarray, w: np.ndarray) -> None:
        """Nearly-isotonic path (modified PAVA) with BIC weights, in one sweep.

        The path state is kept as compacted per-group arrays in grid order; every
        breakpoint recomputes all pair violation flags from the current group means,
        so exact ties that separate under lambda are picked up. Solutions are
        expanded and scored as they are produced and only the best ``max_solutions``
        by BIC are retained, which bounds memory at ``O(m * max_solutions)``.
        """
        cap = self.max_solutions
        n_tot = float(w.sum())
        log_n = math.log(n_tot)

        def loglik(p: np.ndarray, s1: np.ndarray, s0: np.ndarray) -> float:
            q = np.clip(p, EPS, 1.0 - EPS)
            return float(np.sum(s1 * np.log(q) + s0 * np.log(1.0 - q)))

        S1_init = w * y
        S0_init = w * (1.0 - y)
        # Pruning anchors: ll_sat is the path's maximum log-likelihood and the
        # isotonic fit bounds the minimum BIC, so a solution with more than
        # k_prune levels has relative BIC weight below 1e-15 (exactly 0 in
        # double precision) and never needs to be scored or stored.
        ll_sat = loglik(y, S1_init, S0_init)
        iso = pava(y, w)
        ll_iso = loglik(iso.fitted, S1_init, S0_init)
        k_iso = 1 + int(np.sum(np.abs(np.diff(iso.fitted)) > 1e-12))
        if log_n > 0.0:
            k_prune = k_iso + (2.0 * (ll_sat - ll_iso) + 2.0 * math.log(1e15)) / log_n
        else:
            # The bound assumes n_tot > 1: with log_n == 0 it is undefined and with
            # log_n < 0 the BIC penalty rewards extra levels instead of charging for
            # them, so no solution can be ruled out. Weights below unity are legal
            # (validate_weights only requires positivity), so score everything.
            k_prune = math.inf

        # Compacted group state in grid order; sizes[g] is the group's run length.
        means = y.astype(float).copy()
        weights = w.astype(float).copy()
        S1 = S1_init.copy()
        S0 = S0_init.copy()
        sizes = np.ones(len(y), dtype=np.int64)
        lam = 0.0
        lambdas = [0.0]
        bics: list[float] = []
        kept: list[tuple[float, int, np.ndarray]] = []  # min-heap on -bic: root = worst kept
        t_index = 0

        def record() -> None:
            nonlocal t_index
            k_t = 1 + int(np.sum(np.abs(np.diff(means)) > 1e-12))
            if k_t > k_prune:
                bics.append(math.inf)
                if cap is None:
                    heapq.heappush(kept, (-math.inf, t_index, np.repeat(means, sizes)))
                t_index += 1
                return
            bic = -2.0 * loglik(means, S1, S0) + k_t * log_n
            bics.append(bic)
            if cap is None or len(kept) < cap:
                heapq.heappush(kept, (-bic, t_index, np.repeat(means, sizes)))
            elif bic < -kept[0][0]:
                heapq.heapreplace(kept, (-bic, t_index, np.repeat(means, sizes)))
            t_index += 1

        record()  # breakpoint 0: the raw solution at lambda = 0
        while len(means) > 1:
            # `means` holds the group values AT the current lam. Stationarity of
            # 1/2 sum w (y - m)^2 + lam * sum (m_i - m_{i+1})_+ gives, while the
            # active violation set is fixed, d beta_g / d lam = -slope_g with
            # slope_g = (1{beta_g > beta_{g+1}} - 1{beta_{g-1} > beta_g}) / W_g.
            viol = means[:-1] > means[1:] + 1e-15
            if not viol.any():
                break
            vr = viol.astype(float)
            slopes = (np.concatenate((vr, [0.0])) - np.concatenate(([0.0], vr))) / weights
            # Earliest collision among adjacent groups. gap(t) = gap - t * dv,
            # so a pair meets at t = gap / dv when gap and dv share a sign:
            # violating pairs (gap > 0) close with dv > 0; non-violating pairs
            # (gap < 0) can be driven together by outer violations (dv < 0).
            gaps = means[:-1] - means[1:]
            dv = slopes[:-1] - slopes[1:]
            cand = ((gaps > 0) & (dv > 1e-300)) | ((gaps < 0) & (dv < -1e-300))
            if not cand.any():
                break
            t = np.full(len(gaps), np.inf)
            np.divide(gaps, dv, out=t, where=cand)
            t[t < 0.0] = np.inf  # candidates always give t > 0; guard the 0 <= t rule
            g = int(np.argmin(t))  # first minimum == the leftmost strict-< winner
            t_best = float(t[g])
            if not np.isfinite(t_best):
                break
            means = means - slopes * t_best
            lam += t_best
            new_w = weights[g] + weights[g + 1]
            means[g] = (weights[g] * means[g] + weights[g + 1] * means[g + 1]) / new_w
            weights[g] = new_w
            S1[g] += S1[g + 1]
            S0[g] += S0[g + 1]
            sizes[g] += sizes[g + 1]
            keep = np.arange(len(means)) != g + 1
            means = means[keep]
            weights = weights[keep]
            S1 = S1[keep]
            S0 = S0[keep]
            sizes = sizes[keep]
            lambdas.append(lam)
            record()

        bic_arr = np.asarray(bics)
        finite = np.isfinite(bic_arr)
        wgt = np.zeros(len(bic_arr))
        if finite.any():
            wgt[finite] = np.exp(-0.5 * (bic_arr[finite] - bic_arr[finite].min()))
            wgt /= wgt.sum()
        else:  # nothing scored: fall back to the last breakpoint, the isotonic fit
            wgt[-1] = 1.0
        if not kept:  # ditto, so that the ensemble always has a solution to average
            heapq.heappush(kept, (0.0, len(bic_arr) - 1, np.repeat(means, sizes)))
        kept_t = np.array(sorted(e[1] for e in kept), dtype=np.int64)
        sols = {e[1]: e[2] for e in kept}
        kept_w = wgt[kept_t]
        self.path_lambdas_ = np.asarray(lambdas)
        self.path_solutions_ = np.stack([sols[t_i] for t_i in kept_t])
        self.kept_breakpoints_ = kept_t
        self.dropped_weight_ = float(max(0.0, 1.0 - kept_w.sum()))
        self.weights_ = kept_w / kept_w.sum()
        if self.dropped_weight_ > 1e-6:
            warnings.warn(
                f"ENIR dropped {self.dropped_weight_:.2e} of the BIC ensemble weight; "
                "consider a larger max_solutions",
                UserWarning,
                stacklevel=2,
            )

    def _predict(self, s: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self._x, s, side="right") - 1, 0, len(self._x) - 1)
        return np.clip(self._mixed[idx], EPS, 1.0 - EPS)

    def interpret(self) -> Interpretation:
        """Read the path length and BIC weights; warn about non-monotonicity."""
        self._check_fitted()
        top = np.argsort(self.weights_)[::-1][:3]
        top_txt = ", ".join(
            f"lambda={self.path_lambdas_[self.kept_breakpoints_[i]]:.4g} "
            f"(weight {self.weights_[i]:.3f})"
            for i in top
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
