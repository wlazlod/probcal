"""Frozen v0.1.2 ENIR path/BIC reference, kept verbatim as the equivalence gate's
ground truth for the heap-scheduled rewrite. Do not "fix" or
optimize this file — its only job is to keep reproducing the pre-rewrite behavior.
"""

import numpy as np

from probcal._validation import EPS


def reference_aggregate(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verbatim copy of ``ENIRCalibrator._fit``'s tie-aggregation preamble."""
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    s_u, start = np.unique(s_sorted, return_index=True)
    w_u = np.add.reduceat(w_sorted, start)
    y_u = np.add.reduceat(w_sorted * y_sorted, start) / w_u
    return s_u, y_u, w_u


def reference_path(y: np.ndarray, w: np.ndarray) -> tuple[list, list]:
    """Verbatim copy of the v0.1.2 ``ENIRCalibrator._nearly_isotonic_path``."""
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


def reference_weights(y_u: np.ndarray, w_u: np.ndarray, solutions: list) -> np.ndarray:
    """Verbatim copy of the v0.1.2 BIC-weight block from ``ENIRCalibrator._fit``."""
    n_tot = float(w_u.sum())
    bics = np.empty(len(solutions))
    for t, sol in enumerate(solutions):
        p = np.clip(sol, EPS, 1.0 - EPS)
        loglik = float(np.sum(w_u * (y_u * np.log(p) + (1.0 - y_u) * np.log(1.0 - p))))
        k_groups = 1 + int(np.sum(np.abs(np.diff(sol)) > 1e-12))
        bics[t] = -2.0 * loglik + k_groups * np.log(n_tot)
    shifted = -0.5 * (bics - bics.min())
    wgt = np.exp(shifted)
    return wgt / wgt.sum()
