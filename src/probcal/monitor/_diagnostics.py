"""Post-alarm diagnostics behind ``CalibrationMonitor.report()``'s recommendation.

These are *diagnostics with no error guarantee* (the e-processes are the
evidence): a trailing-window offset, a bootstrap CI for the Cox slope, and
the Cox-vs-offset residual likelihood ratio. See the "recommendation rule"
section of ``docs/concepts/monitoring.md``.
"""

import math

import numpy as np

from .._math import expit
from .._validation import EPS
from ..metrics.regression import calibration_slope
from ._processes import plug_in_delta, plug_in_shape

CHI2_1_95 = 3.841
"""chi-square(1) upper 5% point, the residual-LR bound."""


def residual_shape_lr(z: np.ndarray, y: np.ndarray, w: np.ndarray, delta: float) -> float:
    """``2 * (loglik of the Cox fit - loglik of the offset-only fit)`` on the window."""
    if z.size == 0 or np.unique(y).size < 2:
        return 0.0
    c, a = plug_in_shape(z, y, w)

    def ll(q: np.ndarray) -> float:
        qc = np.clip(q, EPS, 1.0 - EPS)
        return float(np.sum(w * (y * np.log(qc) + (1.0 - y) * np.log1p(-qc))))

    return max(0.0, 2.0 * (ll(expit(c + a * z)) - ll(expit(z + delta))))


def slope_ci(z: np.ndarray, y: np.ndarray, w: np.ndarray, n_boot: int = 200) -> tuple[float, float]:
    """Percentile bootstrap CI of the Cox slope on the trailing window (seeded)."""
    if z.size < 10 or np.unique(y).size < 2:
        return -np.inf, np.inf
    rng = np.random.default_rng(0)
    p = expit(z)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(z), len(z))
        if np.unique(y[idx]).size < 2:
            continue
        slopes.append(calibration_slope(y[idx], p[idx], sample_weight=w[idx]))
    if len(slopes) < 20:
        return -np.inf, np.inf
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def recommend(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    alarm_at: str,
    e_shape: float,
    alpha: float,
    onset_label: str | None,
) -> tuple[str, tuple[str, ...]]:
    """The re-offset/re-fit rule on the diagnostic window, plus its reasoning.

    Re-offset when the Cox slope bootstrap CI contains 1 *and* the
    residual LR stays within the chi-square(1) 5% bound; otherwise re-fit.
    The shape e-process is reported, not used: its alternative family
    contains the intercept, so it also fires under pure level drift.
    """
    delta_now = plug_in_delta(z, y, w)
    lo, hi = slope_ci(z, y, w)
    slope_ok = lo <= 1.0 <= hi
    resid_lr = residual_shape_lr(z, y, w, delta_now)
    shape_needed = resid_lr > CHI2_1_95
    shape_line = (
        f"shape e-process {e_shape:.3g} vs 1/alpha = {1.0 / alpha:.1f} "
        "(reported; fires under level drift too, so not decisive alone)"
        if not math.isnan(e_shape)
        else "shape e-process not monitored (not in components)"
    )
    reasoning = (
        f"alarm at {alarm_at!r}; trailing-window offset {delta_now:+.3f} log-odds",
        shape_line,
        f"trailing-window Cox slope 95% bootstrap CI [{lo:.3f}, {hi:.3f}] "
        + ("contains" if slope_ok else "excludes")
        + " 1",
        f"Cox-vs-offset residual LR on the trailing window {resid_lr:.2f} "
        + ("exceeds" if shape_needed else "is within")
        + " the chi-square(1) 5% bound 3.84",
        (
            f"estimated drift onset at {onset_label} (backward-CUSUM argmax of the "
            "plug-in log-LR increments — an estimate, not a test)"
            if onset_label is not None
            else "drift onset unavailable: steps recorded before 0.3.0 carry no log-e "
            "increments (trailing window used)"
        ),
        "the recommendation is a diagnostic, not a test — see the monitoring chapter",
    )
    return ("re-offset" if (slope_ok and not shape_needed) else "re-fit"), reasoning


__all__ = ["recommend", "residual_shape_lr", "slope_ci"]
