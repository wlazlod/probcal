"""Post-alarm diagnostics behind ``CalibrationMonitor.report()``'s recommendation.

These are *diagnostics with no error guarantee* (the e-processes are the
evidence): a trailing-window offset, a bootstrap CI for the Cox slope, and
the Cox-vs-offset residual likelihood ratio. See the "recommendation rule"
section of ``docs/concepts/monitoring.md``.
"""

import math

import numpy as np

from .._math import bern_loglik, bootstrap_indices, chi2_ppf, expit, percentile_band
from .._validation import EPS
from ..metrics._common import rao_scott
from ..metrics.regression import _slope_fit
from ._processes import plug_in_delta, plug_in_shape

CHI2_1_95 = chi2_ppf(0.95, 1.0)
"""chi-square(1) upper 5% point (3.8415), the residual-LR bound."""


def residual_shape_lr(z: np.ndarray, y: np.ndarray, w: np.ndarray, delta: float) -> float:
    """``2 * (loglik of the Cox fit - loglik of the offset-only fit)`` on the window.

    With non-uniform weights the weighted LR is a scaled ``chi2_1`` under
    the offset-only model, not ``chi2_1``; it is divided by its design
    effect (first-order Rao-Scott on the slope score, the intercept
    projected out: :func:`probcal.metrics._common.rao_scott` on the centred
    logits), so it compares to the chi-square(1) bound on the right scale
    and is invariant to rescaling the weights. Unit weights give the plain
    LR, unchanged.
    """
    if z.size == 0 or np.unique(y).size < 2:
        return 0.0
    c, a = plug_in_shape(z, y, w)
    mu = expit(z + delta)
    lr = max(0.0, 2.0 * (bern_loglik(y, expit(c + a * z), w) - bern_loglik(y, mu, w)))
    # Slope score with the (free) intercept projected out under the null info.
    v = w * mu * (1.0 - mu)
    z_c = z - float(np.sum(v * z)) / float(np.sum(v))
    stat, _ = rao_scott(lr, z_c[:, None], mu, w)
    return stat


def slope_ci(z: np.ndarray, y: np.ndarray, w: np.ndarray, n_boot: int = 200) -> tuple[float, float]:
    """Percentile bootstrap CI of the Cox slope on the trailing window (seeded)."""
    if z.size < 10 or np.unique(y).size < 2:
        return -np.inf, np.inf
    p = np.clip(expit(z), EPS, 1.0 - EPS)  # the validated p calibration_slope would fit
    slopes = []
    for idx in bootstrap_indices(np.random.default_rng(0), len(z), n_boot):
        # One-class or constant-logit resamples identify no slope: skip them
        # (the fit would return nan and warn).
        if np.unique(y[idx]).size < 2 or np.unique(z[idx]).size < 2:
            continue
        slope = float(_slope_fit(y[idx], p[idx], w[idx])[1][1])
        if np.isfinite(slope):
            slopes.append(slope)
    if len(slopes) < 20:
        return -np.inf, np.inf
    lo, hi = percentile_band(slopes, 0.025)
    return float(lo), float(hi)


def recommend(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    alarm_at: str,
    e_shape: float,
    alpha: float,
    onset_label: str,
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
        f"estimated drift onset at {onset_label} (backward-CUSUM argmax of the "
        "plug-in log-LR increments — an estimate, not a test)",
        "the recommendation is a diagnostic, not a test — see the monitoring chapter",
    )
    return ("re-offset" if (slope_ok and not shape_needed) else "re-fit"), reasoning
