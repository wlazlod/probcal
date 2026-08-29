"""Matplotlib plotting helpers (requires the [viz] extra; import-guarded).

All computation lives in ``probcal.curves`` and ``probcal.metrics``; this
module only renders. The logit-scale views are the flagship for low-PD
portfolios: axis ticks sit at logit positions but are labeled in
probabilities, so the low-probability region stays readable. Styling is
applied per call via ``rc_context`` — global ``rcParams`` are never touched.
Theory: ``docs/concepts/visualization.md``.
"""

import math
from typing import Any

import numpy as np

from ._math import logit

# _HAS_MPL, _RUG_MAX, _TICK_PROBS: unused directly here, re-exported so
# `probcal.plots._STYLE`-style attribute access keeps working post-split.
from ._plots_common import (
    _AMBER,
    _BLUE,
    _BOX,
    _GREEN,
    _GREY,
    _HAS_MPL,  # noqa: F401
    _ORANGE,
    _RED,
    _RUG_MAX,  # noqa: F401
    _STYLE,
    _TICK_PROBS,  # noqa: F401
    _logit_axis,
    _plt,
    _require_mpl,
    _rug_subsample,
)
from ._results import (
    BeltResult,
    KernelReliabilityCurve,
    MetricReport,
    ReliabilityCurve,
    SelectionReport,
    SmoothReliabilityCurve,
)
from .curves import EcceCurve
from .metrics import brier_score, reliability_summary, smooth_ece

# Names shown, in order, by `stats=<MetricReport>` when present in the report.
_STATS_REPORT_NAMES = ("intercept", "slope", "ici", "smooth_ece", "brier")


_SPLIT_BINS = 30
_SPLIT_BASELINE = 0.12


def _draw_kernel_curve(ax: Any, curve: KernelReliabilityCurve, scale: str) -> None:
    """Render a ``KernelReliabilityCurve``: a density-weighted line
    (``LineCollection``, wide where predictions are dense), the
    miscalibration area between the curve and the identity, the bootstrap
    ribbon, and the ``smECE`` readout.

    Points whose event rate is exactly 0 or 1 have no finite logit and are
    dropped on ``scale="logit"`` (mirrors the binned-point layer above).
    """
    from matplotlib.collections import LineCollection

    if scale == "logit":
        keep = (curve.event_rate > 0.0) & (curve.event_rate < 1.0)
        grid = curve.grid_logit[keep]
        rate = logit(curve.event_rate[keep])
        ci_low = logit(curve.ci_low[keep])
        ci_high = logit(curve.ci_high[keep])
        density = curve.density[keep]
    else:
        grid = curve.grid_p
        rate = curve.event_rate
        ci_low = curve.ci_low
        ci_high = curve.ci_high
        density = curve.density

    points = np.column_stack([grid, rate])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    linewidths = 0.5 + 4.0 * density[:-1] / density.max()
    lc = LineCollection(list(segments), linewidths=linewidths, colors=_ORANGE, label="smoothed")
    ax.add_collection(lc)
    ax.fill_between(grid, grid, rate, alpha=0.12, color=_ORANGE)
    ax.fill_between(grid, ci_low, ci_high, alpha=0.15, color=_ORANGE)
    ax.text(
        0.97,
        0.03,
        f"smECE = {curve.smooth_ece:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def _draw_split_risk_dist(ax: Any, y_arr: np.ndarray, p_arr: np.ndarray, scale: str) -> None:
    """Spike-histogram risk distribution, replacing the rug: 30 equal-mass
    bins of ``p``, events up / non-events down from a ``y=0.12`` baseline in
    axis-fraction coordinates (axis coords cannot go below 0), heights
    scaled so whichever class peaks higher reaches the full 0.12.
    """
    edges = np.unique(np.quantile(p_arr, np.linspace(0.0, 1.0, _SPLIT_BINS + 1)))
    if len(edges) < 2:
        return
    n_bins = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, p_arr, side="right") - 1, 0, n_bins - 1)
    ev_counts = np.bincount(idx[y_arr == 1.0], minlength=n_bins).astype(np.float64)
    ne_counts = np.bincount(idx[y_arr == 0.0], minlength=n_bins).astype(np.float64)
    peak = max(float(ev_counts.max()), float(ne_counts.max()), 1.0)
    ev_heights = ev_counts / peak * _SPLIT_BASELINE
    ne_heights = ne_counts / peak * _SPLIT_BASELINE

    x_edges = logit(edges) if scale == "logit" else edges
    widths = np.diff(x_edges)
    tf = ax.get_xaxis_transform()
    ax.bar(
        x_edges[:-1], ev_heights, width=widths, align="edge",
        bottom=_SPLIT_BASELINE, transform=tf, color=_RED, alpha=0.35, linewidth=0,
    )  # fmt: skip
    ax.bar(
        x_edges[:-1], -ne_heights, width=widths, align="edge",
        bottom=_SPLIT_BASELINE, transform=tf, color=_GREY, alpha=0.35, linewidth=0,
    )  # fmt: skip


def _stats_box_text(stats: bool | MetricReport, y_arr: np.ndarray, p_arr: np.ndarray) -> str:
    """Build the ``stats`` box text: fixed n/events/intercept/slope/ICI/smECE/Brier
    for ``stats=True``, or ``name = value [ci_low, ci_high]`` for the
    ``_STATS_REPORT_NAMES`` present in a given ``MetricReport`` (plus n/events
    from ``y``).
    """
    if not isinstance(stats, MetricReport):
        s = reliability_summary(y_arr, p_arr)
        sece = smooth_ece(y_arr, p_arr)
        brier = brier_score(y_arr, p_arr)
        return (
            f"n = {s.n:,}\n"
            f"events = {s.events:,}\n"
            f"intercept = {s.intercept:+.3f}\n"
            f"slope = {s.slope:.3f}\n"
            f"ICI = {s.ici:.3f}\n"
            f"smECE = {sece:.3f}\n"
            f"Brier = {brier:.3f}"
        )
    lines = [f"n = {len(y_arr):,}", f"events = {int(y_arr.sum()):,}"]
    present = set(stats.names)
    for name in _STATS_REPORT_NAMES:
        if name not in present:
            continue
        i = stats.names.index(name)
        lines.append(
            f"{name} = {stats.values[i]:.3f} [{stats.ci_low[i]:.3f}, {stats.ci_high[i]:.3f}]"
        )
    return "\n".join(lines)


def plot_reliability(
    curve: ReliabilityCurve,
    *,
    smooth: SmoothReliabilityCurve | KernelReliabilityCurve | None = None,
    scale: str = "probability",
    y: object = None,
    p: object = None,
    annotate: bool = True,
    rug: bool = True,
    counts: bool = False,
    ax: Any = None,
    stats: bool | MetricReport = False,
    risk_dist: str | None = "rug",
) -> Any:
    """Annotated reliability diagram.

    Binned points with Wilson CIs, optional smooth overlay, stats box, and
    event/non-event risk distribution.

    ``scale="logit"`` stretches the low-probability region — the recommended
    view for PD portfolios. Bins whose event rate is exactly 0 or 1 have no
    finite logit and are omitted from the logit-scale point layer; they remain
    visible in the risk distribution (or the ``counts=True`` margin).

    Passing a :class:`probcal.curves.KernelReliabilityCurve` (from
    :func:`probcal.curves.reliability_smooth`) as ``smooth`` renders the
    density-weighted variable-width curve instead of a plain line: a
    ``LineCollection`` whose width tracks the local prediction density, the
    shaded miscalibration area between the curve and the identity, the
    bootstrap ribbon, and an ``smECE = ...`` readout in the bottom-right
    corner.

    Passing the raw ``y``/``p`` enables the stats box and the risk
    distribution; both are silently skipped when ``y``/``p`` are absent.
    ``annotate=True`` (default) draws the classic stats box, computed by
    :func:`probcal.metrics.reliability_summary`. ``stats=True`` replaces it
    with a box reporting ``n, events, intercept, slope, ICI, smECE, Brier``
    instead (``annotate`` is then ignored); ``stats=<MetricReport>`` instead
    reports ``name = value [ci_low, ci_high]`` for whichever of
    ``{"intercept", "slope", "ici", "smooth_ece", "brier"}`` the report
    carries, plus ``n``/``events`` computed from ``y``.

    ``risk_dist`` selects the density layer: ``"rug"`` (default) draws the
    0.2.0 event/non-event tick marks along the top/bottom edges,
    deterministically thinned to at most 1000 marks per class; ``"split"``
    replaces it with a 30-equal-mass-bin spike histogram of ``p`` (events
    up, non-events down, from a ``y=0.12`` baseline in axis-fraction
    coordinates, heights scaled so the taller class reaches the full 0.12 —
    axis coordinates cannot go below 0, so both classes share the one
    baseline); ``None`` draws no density layer. ``rug=False`` disables the
    density layer regardless of ``risk_dist`` (equivalent to
    ``risk_dist=None``). ``counts=True`` restores the twin-axis count-bar
    margin, independent of ``risk_dist``.

    Parameters
    ----------
    curve : ReliabilityCurve
        Binned curve, e.g. from :func:`probcal.curves.reliability_binned`.
    smooth : SmoothReliabilityCurve, KernelReliabilityCurve, or None, keyword-only
        Optional smooth overlay, e.g. from
        :func:`probcal.curves.reliability_loess` or
        :func:`probcal.curves.reliability_smooth`.
    scale : {"probability", "logit"}, keyword-only
        Axis scale; ``"logit"`` stretches the low-probability region.
    y, p : array_like or None, keyword-only
        Raw outcomes and predictions; must be given together (or not at all).
        Enables the stats box and risk distribution.
    annotate : bool, keyword-only
        If ``True`` (default) and ``y``/``p`` are given, draw the classic
        stats box; ignored when ``stats`` is truthy.
    rug : bool, keyword-only
        If ``True`` (default) and ``y``/``p`` are given, draw the density
        layer selected by ``risk_dist``.
    counts : bool, keyword-only
        If ``True``, add a twin-axis bar strip of per-bin counts.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.
    stats : bool or MetricReport, keyword-only
        If truthy and ``y``/``p`` are given, draw the ``n, events,
        intercept, slope, ICI, smECE, Brier`` stats box (``True``) or a
        ``MetricReport``-driven box, replacing ``annotate``'s box.
    risk_dist : {"rug", "split"} or None, keyword-only
        Density-layer style; see above. Anything else raises ``ValueError``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn on.

    Raises
    ------
    ValueError
        If ``y``/``p`` are not given together, or ``risk_dist`` is not one
        of ``"rug"``, ``"split"``, ``None``.
    """
    _require_mpl()
    if (y is None) != (p is None):
        raise ValueError("y and p must be given together")
    if risk_dist not in ("rug", "split", None):
        raise ValueError('risk_dist must be one of "rug", "split", None')
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))
        if scale == "logit":
            keep = (curve.event_rate > 0.0) & (curve.event_rate < 1.0)
            x, ylo = curve.pred_mean_logit[keep], logit(curve.ci_low[keep])
            yv, yhi = logit(curve.event_rate[keep]), logit(curve.ci_high[keep])
            diag = np.linspace(x.min() - 0.5, x.max() + 0.5, 50)
            ax.plot(diag, diag, ls="--", c=_GREY, lw=1, label="identity")
            ax.errorbar(
                x,
                yv,
                yerr=[np.maximum(yv - ylo, 0.0), np.maximum(yhi - yv, 0.0)],
                fmt="o",
                ms=4,
                capsize=2,
                color=_BLUE,
                label="binned",
            )
            if isinstance(smooth, KernelReliabilityCurve):
                _draw_kernel_curve(ax, smooth, "logit")
            elif smooth is not None:
                ax.plot(
                    smooth.grid_logit, logit(smooth.event_rate), lw=1.5, c=_ORANGE, label="smoothed"
                )
            _logit_axis(ax)
            ax.set_xlabel("predicted probability (logit scale)")
            ax.set_ylabel("event rate (logit scale)")
        else:
            ax.plot([0, 1], [0, 1], ls="--", c=_GREY, lw=1, label="identity")
            ax.errorbar(
                curve.pred_mean,
                curve.event_rate,
                yerr=[curve.event_rate - curve.ci_low, curve.ci_high - curve.event_rate],
                fmt="o",
                ms=4,
                capsize=2,
                color=_BLUE,
                label="binned",
            )
            if isinstance(smooth, KernelReliabilityCurve):
                _draw_kernel_curve(ax, smooth, "probability")
            elif smooth is not None:
                ax.plot(smooth.grid_p, smooth.event_rate, lw=1.5, c=_ORANGE, label="smoothed")
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("event rate")

        boxed = False
        if y is not None and p is not None:
            y_arr = np.asarray(y, dtype=np.float64)
            p_arr = np.asarray(p, dtype=np.float64)
            show_density = rug and risk_dist is not None
            if show_density and risk_dist == "rug":
                ev = _rug_subsample(p_arr[y_arr == 1.0])
                ne = _rug_subsample(p_arr[y_arr == 0.0])
                if scale == "logit":
                    ev, ne = logit(ev), logit(ne)
                tf = ax.get_xaxis_transform()
                ax.plot(
                    ev, np.full(len(ev), 0.99), transform=tf,
                    ls="none", marker="|", ms=7, c=_RED, alpha=0.25,
                )  # fmt: skip
                ax.plot(
                    ne, np.full(len(ne), 0.01), transform=tf,
                    ls="none", marker="|", ms=7, c="#777777", alpha=0.18,
                )  # fmt: skip
            elif show_density and risk_dist == "split":
                _draw_split_risk_dist(ax, y_arr, p_arr, scale)
            if stats:
                txt = _stats_box_text(stats, y_arr, p_arr)
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9, bbox=_BOX)
                boxed = True
            elif annotate:
                s = reliability_summary(y_arr, p_arr)
                txt = (
                    f"n = {s.n:,}\n"
                    f"events = {s.events:,}\n"
                    f"intercept = {s.intercept:+.3f}\n"
                    f"slope = {s.slope:.3f}\n"
                    f"ICI = {s.ici:.4f}\n"
                    f"E90 = {s.e90:.4f}\n"
                    f"Spiegelhalter p = {s.spiegelhalter_p:.3f}"
                )
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9, bbox=_BOX)
                boxed = True
        if counts:
            # Count margin as a twin bar strip along the x-axis.
            ax2 = ax.twinx()
            xs = curve.pred_mean_logit if scale == "logit" else curve.pred_mean
            ax2.bar(xs, curve.count, width=np.ptp(xs) / (3 * len(xs) + 1), alpha=0.15, color=_GREY)
            ax2.set_yticks([])
        ax.legend(loc="lower right" if boxed else "upper left")
        return ax


def plot_belt(belt: BeltResult, *, scale: str = "probability", ax: Any = None) -> Any:
    """GiViTI-style calibration belt with 80/95% bands and the test p-value.

    Parameters
    ----------
    belt : BeltResult
        Result of :func:`probcal.curves.calibration_belt`.
    scale : {"probability", "logit"}, keyword-only
        Axis scale; ``"logit"`` stretches the low-probability region.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the belt was drawn on.
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))
        if scale == "logit":
            x = belt.grid_logit
            ax.plot(x, x, ls="--", c=_GREY, lw=1)
            ax.fill_between(
                x, logit(belt.lower_95), logit(belt.upper_95), color=_BLUE, alpha=0.2, label="95%"
            )
            ax.fill_between(
                x, logit(belt.lower_80), logit(belt.upper_80), color=_BLUE, alpha=0.35, label="80%"
            )
            _logit_axis(ax)
        else:
            x = belt.grid_p
            ax.plot(x, x, ls="--", c=_GREY, lw=1)
            ax.fill_between(x, belt.lower_95, belt.upper_95, color=_BLUE, alpha=0.2, label="95%")
            ax.fill_between(x, belt.lower_80, belt.upper_80, color=_BLUE, alpha=0.35, label="80%")
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
    """Side-by-side reliability diagrams (pre/post calibration or offset).

    Parameters
    ----------
    before, after : ReliabilityCurve
        Binned curves to compare, e.g. raw vs calibrated.
    scale : {"probability", "logit"}, keyword-only
        Axis scale; ``"logit"`` stretches the low-probability region.
    labels : tuple of str, keyword-only
        Panel titles for ``(before, after)``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing both panels.
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        fig, axes = _plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
        panel_colors = (_RED, _GREEN)
        for ax, curve, label, color in zip(
            axes, (before, after), labels, panel_colors, strict=True
        ):
            plot_reliability(curve, scale=scale, ax=ax)
            # Recolor the binned series to the panel's before/after semantics.
            for line in ax.lines:
                if line.get_label() == "binned":
                    line.set_color(color)
            for container in ax.containers:
                for artist in container.get_children():
                    artist.set_color(color)
            ax.set_title(label)
        return fig


def plot_interval(intervals: np.ndarray, s: np.ndarray, *, ax: Any = None) -> Any:
    """Venn–Abers interval widths against the score: where is calibration uncertain?

    Parameters
    ----------
    intervals : numpy.ndarray of shape (n, 2)
        ``(p0, p1)`` Venn–Abers interval bounds per score, e.g. from
        :meth:`probcal.vennabers.CrossVennAbersCalibrator.predict_interval`.
    s : numpy.ndarray of shape (n,)
        Scores the intervals are plotted against.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the intervals were drawn on.
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 4.5))
        p0, p1 = intervals[:, 0], intervals[:, 1]
        ax.fill_between(s, p0, p1, color=_BLUE, alpha=0.3, label="Venn–Abers interval")
        ax.plot(s, p1 / (1.0 - p0 + p1), lw=1.2, c=_ORANGE, label="scalarized")
        ax.set_xlabel("score")
        ax.set_ylabel("calibrated probability")
        ax.legend(loc="upper left")
        return ax


def plot_selection(report: SelectionReport, *, ax: Any = None) -> Any:
    """SelectionReport as a ranked dot plot with fold-spread whiskers.

    Parameters
    ----------
    report : SelectionReport
        Result of :meth:`probcal.selection.CalibratorSelector.fit`, read
        from its ``report_`` attribute.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the dot plot was drawn on.
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 0.6 * len(report.methods) + 1.5))
        order = np.argsort(report.score_mean)
        ys = np.arange(len(order))
        for rank, i in enumerate(order):
            ok = report.guardrails_ok[i]
            marker = "o" if ok else "x"
            color = _GREEN if report.chosen[i] else (_BLUE if ok else _RED)
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


def plot_ecce(
    curves: Any,
    *,
    labels: Any = None,
    show_band: bool = True,
    ax: Any = None,
) -> Any:
    """ECCE cumulative-drift walk(s) from :func:`probcal.curves.ecce_curve`.

    Accepts a single ``EcceCurve`` or a sequence (e.g. raw vs calibrated).
    The grey envelope (``show_band=True``, from the first curve) is ±2
    *pointwise* standard deviations under calibration — an aid for reading
    the walk, NOT a simultaneous confidence band; the formal max-statistic
    test of Arrieta-Ibarra et al. (2022) is out of scope for this release.

    Parameters
    ----------
    curves : EcceCurve or sequence of EcceCurve
        One or more cumulative-drift walks to overlay.
    labels : sequence of str or None, keyword-only
        Legend labels, aligned with ``curves``; ``None`` uses
        ``"curve 1", "curve 2", ...``.
    show_band : bool, keyword-only
        If ``True`` (default), draw the ±2 SD envelope from the first curve.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the walk(s) were drawn on.
    """
    _require_mpl()
    if isinstance(curves, EcceCurve):
        curves = [curves]
    curves = list(curves)
    if labels is None:
        labels = [f"curve {i + 1}" for i in range(len(curves))]
    palette = [_RED, _GREEN, _BLUE, _ORANGE]
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(7.5, 4.8))
        if show_band:
            c0 = curves[0]
            ax.fill_between(
                c0.frac,
                -2.0 * c0.sd_null,
                2.0 * c0.sd_null,
                color=_GREY,
                alpha=0.3,
                label="±2 SD under calibration (pointwise)",
            )
        ax.axhline(0.0, ls="--", c=_GREY, lw=1)
        for i, (c, label) in enumerate(zip(curves, labels, strict=True)):
            color = palette[i % len(palette)]
            ax.plot(
                c.frac, c.cumdev, lw=1.6, c=color, label=f"{label} (max drift {c.stat_max:.4f})"
            )
            ax.axvline(c.argmax_frac, ls=":", c=color, lw=1, alpha=0.7)
        ax.set_xlabel("cumulative share of portfolio (sorted by prediction)")
        ax.set_ylabel("cumulative deviation")
        ax.legend(loc="best")
        return ax


def plot_grade_backtest(result: Any, *, log_scale: bool = True, ax: Any = None) -> Any:
    """Per-grade traffic-light backtest chart (Jeffreys or exact binomial).

    Observed default rates as circles colored by the grade's traffic light,
    grey 90% display intervals (``ci_low``/``ci_high``), and the assigned PDs
    as wide blue dashes. The intervals are display companions only — the
    verdict is carried by the lights from the unchanged one-sided tests, so
    no p-values are printed on the canvas. ``log_scale=True`` is the right
    default for PD grades spanning orders of magnitude.

    Parameters
    ----------
    result : BinomialGradeResult or JeffreysGradeResult
        Per-grade backtest result, from
        :func:`probcal.metrics.binomial_grade_test` or
        :func:`probcal.metrics.jeffreys_grade_test`.
    log_scale : bool, keyword-only
        If ``True`` (default), use a log-scale y-axis — the right default
        for PD grades spanning orders of magnitude.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the backtest chart was drawn on.
    """
    _require_mpl()
    light_color = {"green": _GREEN, "yellow": _AMBER, "amber": _AMBER, "red": _RED}
    name = "Jeffreys" if hasattr(result, "p_value") else "exact binomial"
    x = np.arange(len(result.grades))
    rate = result.k / result.n
    colors = [light_color.get(li, _GREY) for li in result.light]
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(1.1 * len(x) + 3.0, 4.8))
        ax.scatter(x, result.pd, marker="_", s=500, c=_BLUE, zorder=2, label="assigned PD")
        ax.errorbar(
            x,
            rate,
            yerr=[np.maximum(rate - result.ci_low, 0.0), np.maximum(result.ci_high - rate, 0.0)],
            fmt="none",
            ecolor=_GREY,
            capsize=4,
            zorder=2,
        )
        ax.scatter(x, rate, s=90, c=colors, edgecolors="white", zorder=3, label="observed rate")
        for i in range(len(x)):
            ax.annotate(
                f"n={int(result.n[i]):,}\nk={int(result.k[i])}",
                xy=(float(x[i]), float(result.ci_high[i])),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#666666",
                clip_on=True,
            )
        if log_scale:
            ax.set_yscale("log")
        # Headroom so the n/k labels never collide with the title.
        lo, hi = ax.get_ylim()
        if log_scale:
            ax.set_ylim(lo, hi * (hi / lo) ** 0.12)
        else:
            ax.set_ylim(lo, hi + 0.12 * (hi - lo))
        ax.set_xticks(x)
        ax.set_xticklabels(result.grades)
        ax.set_xlabel("grade")
        ax.set_ylabel("default rate")
        ax.set_title(f"per-grade backtest ({name}, 90% display intervals)")
        ax.legend(loc="upper left")
        return ax


def plot_offset_audit(offset: Any, *, ax: Any = None) -> Any:
    """Audit chart for a fitted :class:`probcal.offset.LogitOffset` stage.

    Draws the offset map ``t -> t + delta`` on the logit scale against the
    identity, marks the pre- and post-adjustment central tendencies, and
    prints the audit numbers read directly from the fitted attributes. This
    chart audits the *stage*, not the outcomes — for the before/after
    guardrail comparison use ``LogitOffset.audit_report(y, p)``.

    Parameters
    ----------
    offset : LogitOffset
        A fitted :class:`probcal.offset.LogitOffset` instance.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the audit chart was drawn on.
    """
    _require_mpl()
    if not getattr(offset, "fitted_", False):
        raise RuntimeError("LogitOffset is not fitted; call fit() first")
    lo_g = float(logit(np.array([0.001]))[0])
    hi_g = float(logit(np.array([0.5]))[0])
    t = np.linspace(lo_g, hi_g, 200)
    lp = float(logit(np.array([offset.pre_mean_]))[0])
    lq = float(logit(np.array([offset.post_mean_]))[0])
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))
        ax.plot(t, t, ls="--", c=_GREY, lw=1, label="identity")
        ax.plot(t, t + offset.delta_, c=_BLUE, lw=1.6, label="offset map")
        if offset.target_mean is not None:
            ax.axhline(float(logit(np.array([offset.target_mean]))[0]), c=_GREY, lw=0.8, alpha=0.7)
        pre_xy = (lp, lp)
        post_xy = (lq - offset.delta_, lq)
        ax.scatter(*pre_xy, c=_RED, s=60, zorder=3, label="pre mean")
        ax.scatter(*post_xy, c=_GREEN, s=60, zorder=3, label="post mean")
        ax.annotate(
            "", xy=post_xy, xytext=pre_xy, arrowprops={"arrowstyle": "->", "color": "#555555"}
        )
        ax.annotate(
            f"δ = {offset.delta_:+.3f}",
            xy=((pre_xy[0] + post_xy[0]) / 2.0, (pre_xy[1] + post_xy[1]) / 2.0),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color="#555555",
        )
        txt = (
            f"delta = {offset.delta_:+.4f} log-odds\n"
            f"odds factor = {math.exp(offset.delta_):.3f}\n"
            f"pre mean = {offset.pre_mean_:.4%}\n"
            f"post mean = {offset.post_mean_:.4%}\n"
            f"fitted {offset.timestamp_}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9, bbox=_BOX)
        _logit_axis(ax)
        ax.set_xlabel("input probability (logit scale)")
        ax.set_ylabel("shifted probability (logit scale)")
        ax.set_title("logit offset audit")
        ax.legend(loc="lower right")
        return ax


def plot_e_process(report: Any, *, ax: Any = None) -> Any:
    """Monitoring wealth per component on a log scale, with the 1/alpha line.

    Parameters
    ----------
    report : MonitorReport
        Result of :meth:`probcal.monitor.CalibrationMonitor.report`.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the e-processes were drawn on.
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(7.5, 4.2))
        steps = report.steps
        x = np.arange(1, len(steps) + 1)
        series = [
            ("global", [s.e_global for s in steps], "black", 2.0),
            ("offset", [s.e_offset for s in steps], _BLUE, 1.4),
            ("shape", [s.e_shape for s in steps], _ORANGE, 1.4),
        ]
        grades = sorted({g for s in steps for g in s.e_grades})
        for g in grades:
            series.append((f"grade {g}", [s.e_grades.get(g, np.nan) for s in steps], _GREY, 1.0))
        for name, values, color, lw in series:
            vals = np.asarray(values, dtype=np.float64)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(x, vals, label=name, color=color, linewidth=lw, marker=".")
        ax.set_yscale("log")
        ax.axhline(1.0 / report.alpha, color=_RED, linestyle="--", linewidth=1.2, label="1/alpha")
        alarm_x = next((i + 1 for i, s in enumerate(steps) if s.alarm), None)
        if alarm_x is not None:
            ax.axvline(alarm_x, color=_RED, linestyle=":", linewidth=1.0)
            ax.annotate(
                f"alarm: {report.alarm_at}",
                xy=(alarm_x, 1.0),
                xytext=(4, 6),
                textcoords="offset points",
                color=_RED,
                fontsize=9,
            )
        ax.axhline(1.0, color=_GREY, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.label for s in steps], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("e-process wealth (log scale)")
        ax.set_title("anytime-valid calibration monitoring")
        ax.legend(loc="upper left", fontsize=9)
        return ax


from ._plots_diag import plot_attributes, plot_corp, plot_mcb_dsc, plot_murphy  # noqa: E402

__all__ = [
    "plot_reliability",
    "plot_belt",
    "plot_comparison",
    "plot_interval",
    "plot_selection",
    "plot_ecce",
    "plot_grade_backtest",
    "plot_offset_audit",
    "plot_e_process",
    "plot_corp",
    "plot_mcb_dsc",
    "plot_attributes",
    "plot_murphy",
]
