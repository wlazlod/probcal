"""Matplotlib plotting helpers (requires the [viz] extra; import-guarded).

All computation lives in ``probcal.curves``; this module only renders. The
logit-scale views are the flagship for low-PD portfolios: axis ticks sit at
logit positions but are labeled in probabilities, so the low-probability
region stays readable. Theory: ``docs/concepts/visualization.md``.
"""

from typing import Any

import numpy as np

from ._math import logit
from ._results import BeltResult, ReliabilityCurve, SelectionReport, SmoothReliabilityCurve

try:
    import matplotlib.pyplot as _plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _plt = None  # type: ignore[assignment]
    _HAS_MPL = False

_TICK_PROBS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99)


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for probcal.plots — install the viz extra: "
            "pip install probcal[viz]"
        )


def _logit_axis(ax: Any, axis: str = "both") -> None:
    """Label logit-positioned ticks in probabilities."""
    ticks = logit(np.array(_TICK_PROBS))
    labels = [f"{q:g}" for q in _TICK_PROBS]
    if axis in ("x", "both"):
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    if axis in ("y", "both"):
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def plot_reliability(
    curve: ReliabilityCurve,
    *,
    smooth: SmoothReliabilityCurve | None = None,
    scale: str = "probability",
    ax: Any = None,
) -> Any:
    """Reliability diagram: binned points with Wilson CIs, optional smooth overlay,
    and a count histogram margin.

    ``scale="logit"`` stretches the low-probability region — the recommended
    view for PD portfolios. Bins whose event rate is exactly 0 or 1 have no
    finite logit and are omitted from the logit-scale point layer; they remain
    visible in the count margin.
    """
    _require_mpl()
    if ax is None:
        _, ax = _plt.subplots(figsize=(6.5, 6))
    if scale == "logit":
        keep = (curve.event_rate > 0.0) & (curve.event_rate < 1.0)
        x, ylo = curve.pred_mean_logit[keep], logit(curve.ci_low[keep])
        yv, yhi = logit(curve.event_rate[keep]), logit(curve.ci_high[keep])
        diag = np.linspace(x.min() - 0.5, x.max() + 0.5, 50)
        ax.plot(diag, diag, ls="--", c="grey", lw=1, label="identity")
        ax.errorbar(
            x,
            yv,
            yerr=[np.maximum(yv - ylo, 0.0), np.maximum(yhi - yv, 0.0)],
            fmt="o",
            ms=4,
            capsize=2,
            label="binned",
        )
        if smooth is not None:
            ax.plot(smooth.grid_logit, logit(smooth.event_rate), lw=1.5, label="smoothed")
        _logit_axis(ax)
        ax.set_xlabel("predicted probability (logit scale)")
        ax.set_ylabel("event rate (logit scale)")
    else:
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="identity")
        ax.errorbar(
            curve.pred_mean,
            curve.event_rate,
            yerr=[curve.event_rate - curve.ci_low, curve.ci_high - curve.event_rate],
            fmt="o",
            ms=4,
            capsize=2,
            label="binned",
        )
        if smooth is not None:
            ax.plot(smooth.grid_p, smooth.event_rate, lw=1.5, label="smoothed")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("event rate")
    # Count margin as a twin bar strip along the x-axis.
    ax2 = ax.twinx()
    xs = curve.pred_mean_logit if scale == "logit" else curve.pred_mean
    ax2.bar(xs, curve.count, width=np.ptp(xs) / (3 * len(xs) + 1), alpha=0.15, color="grey")
    ax2.set_yticks([])
    ax.legend(loc="upper left")
    return ax


def plot_belt(belt: BeltResult, *, scale: str = "probability", ax: Any = None) -> Any:
    """GiViTI-style calibration belt with 80/95% bands and the test p-value."""
    _require_mpl()
    if ax is None:
        _, ax = _plt.subplots(figsize=(6.5, 6))
    if scale == "logit":
        x = belt.grid_logit
        ax.plot(x, x, ls="--", c="grey", lw=1)
        ax.fill_between(x, logit(belt.lower_95), logit(belt.upper_95), alpha=0.2, label="95%")
        ax.fill_between(x, logit(belt.lower_80), logit(belt.upper_80), alpha=0.35, label="80%")
        _logit_axis(ax)
    else:
        x = belt.grid_p
        ax.plot(x, x, ls="--", c="grey", lw=1)
        ax.fill_between(x, belt.lower_95, belt.upper_95, alpha=0.2, label="95%")
        ax.fill_between(x, belt.lower_80, belt.upper_80, alpha=0.35, label="80%")
    ax.set_title(f"calibration belt (degree {belt.degree}, p = {belt.p_value:.3g})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("event rate")
    ax.legend(loc="upper left")
    return ax


def plot_comparison(
    before: ReliabilityCurve,
    after: ReliabilityCurve,
    *,
    scale: str = "probability",
    labels: tuple[str, str] = ("before", "after"),
) -> Any:
    """Side-by-side reliability diagrams (pre/post calibration or offset)."""
    _require_mpl()
    fig, axes = _plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, curve, label in zip(axes, (before, after), labels, strict=True):
        plot_reliability(curve, scale=scale, ax=ax)
        ax.set_title(label)
    return fig


def plot_interval(intervals: np.ndarray, s: np.ndarray, *, ax: Any = None) -> Any:
    """Venn–Abers interval widths against the score: where is calibration uncertain?"""
    _require_mpl()
    if ax is None:
        _, ax = _plt.subplots(figsize=(6.5, 4.5))
    p0, p1 = intervals[:, 0], intervals[:, 1]
    ax.fill_between(s, p0, p1, alpha=0.3, label="Venn–Abers interval")
    ax.plot(s, p1 / (1.0 - p0 + p1), lw=1.2, label="scalarized")
    ax.set_xlabel("score")
    ax.set_ylabel("calibrated probability")
    ax.legend(loc="upper left")
    return ax


def plot_selection(report: SelectionReport, *, ax: Any = None) -> Any:
    """SelectionReport as a ranked dot plot with fold-spread whiskers."""
    _require_mpl()
    if ax is None:
        _, ax = _plt.subplots(figsize=(6.5, 0.6 * len(report.methods) + 1.5))
    order = np.argsort(report.score_mean)
    ys = np.arange(len(order))
    for rank, i in enumerate(order):
        marker = "o" if report.guardrails_ok[i] else "x"
        color = "tab:green" if report.chosen[i] else "tab:blue"
        ax.errorbar(
            report.score_mean[i],
            rank,
            xerr=report.score_sd[i],
            fmt=marker,
            color=color,
            capsize=3,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([report.methods[i] for i in order])
    ax.set_xlabel(report.criterion)
    ax.set_title("calibrator selection (chosen in green; x = guardrail flag)")
    return ax
