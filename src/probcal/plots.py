"""Matplotlib plotting helpers (requires the [viz] extra; import-guarded).

All computation lives in ``probcal.curves`` and ``probcal.metrics``; this
module only renders. The logit-scale views are the flagship for low-PD
portfolios: axis ticks sit at logit positions but are labeled in
probabilities, so the low-probability region stays readable. Styling is
applied per call via ``rc_context`` — global ``rcParams`` are never touched —
and the axes-level part of it (spines, grid) is applied explicitly as well, so
a caller-supplied ``ax`` is styled the same as one created here. Figures this
module creates are made through ``pyplot`` (so ``plt.show()`` and notebook
display work) and are therefore registered in pyplot's figure manager; close
them with ``plt.close(fig)`` when rendering many in a loop.
Theory: ``docs/concepts/visualization.md``.
"""

import math
from typing import Any

import numpy as np

from ._math import logit
from ._plots_common import (
    _AMBER,
    _BLUE,
    _CI_CLIP_EPS,
    _DEFAULT_LOGIT_SPAN,
    _GREEN,
    _GREY,
    _ORANGE,
    _RED,
    _STYLE,
    _get_axes,
    _plt,
    _require_mpl,
    _rug_subsample,
    _scale_axis_labels,
    _style_axes,
    _text_box,
    _tr,
)
from ._plots_diag import plot_attributes, plot_corp, plot_mcb_dsc, plot_murphy
from ._results import (
    BeltResult,
    KernelReliabilityCurve,
    MetricReport,
    ReliabilityCurve,
    SelectionReport,
    SmoothReliabilityCurve,
)
from .curves import EcceCurve, reliability_binned
from .metrics import brier_score, reliability_summary, smooth_ece
from .metrics.grade import JeffreysGradeResult

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
    dropped on ``scale="logit"`` (mirrors the binned-point layer); bootstrap
    bounds at 0/1 are clipped to ``[1e-6, 1 - 1e-6]`` first. With fewer than
    two drawable points only the readout is drawn.
    """
    from matplotlib.collections import LineCollection

    if scale == "logit":
        keep = (curve.event_rate > 0.0) & (curve.event_rate < 1.0)
        grid = curve.grid_logit[keep]
    else:
        keep = np.ones(len(curve.event_rate), dtype=bool)
        grid = curve.grid_p
    rate = _tr(curve.event_rate[keep], scale)
    ci_low = _tr(curve.ci_low[keep], scale, _CI_CLIP_EPS)
    ci_high = _tr(curve.ci_high[keep], scale, _CI_CLIP_EPS)
    density = curve.density[keep]

    if len(grid) >= 2:
        points = np.column_stack([grid, rate])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        peak = float(density.max()) or 1.0
        linewidths = 0.5 + 4.0 * density[:-1] / peak
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

    x_edges = _tr(edges, scale)
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


def _stats_box_text(
    stats: bool | MetricReport, annotate: bool, y_arr: np.ndarray, p_arr: np.ndarray
) -> str | None:
    """Text of the top-left box, or ``None`` for no box.

    ``stats=True``: fixed n/events/intercept/slope/ICI/smECE/Brier.
    ``stats=<MetricReport>``: ``name = value [ci_low, ci_high]`` for the
    ``_STATS_REPORT_NAMES`` present in the report (plus n/events from ``y``).
    Otherwise, with ``annotate=True``: the classic
    :func:`probcal.metrics.reliability_summary` box.
    """
    if isinstance(stats, MetricReport):
        lines = [f"n = {len(y_arr):,}", f"events = {int(y_arr.sum()):,}"]
        for name in _STATS_REPORT_NAMES:
            if name in stats.names:
                i = stats.names.index(name)
                lines.append(
                    f"{name} = {stats.values[i]:.3f} "
                    f"[{stats.ci_low[i]:.3f}, {stats.ci_high[i]:.3f}]"
                )
        return "\n".join(lines)
    if not stats and not annotate:
        return None
    s = reliability_summary(y_arr, p_arr)
    head = (
        f"n = {s.n:,}\n"
        f"events = {s.events:,}\n"
        f"intercept = {s.intercept:+.3f}\n"
        f"slope = {s.slope:.3f}\n"
    )
    if stats:
        sece = smooth_ece(y_arr, p_arr)
        brier = brier_score(y_arr, p_arr)
        return head + f"ICI = {s.ici:.3f}\nsmECE = {sece:.3f}\nBrier = {brier:.3f}"
    return head + (
        f"ICI = {s.ici:.4f}\nE90 = {s.e90:.4f}\nSpiegelhalter p = {s.spiegelhalter_p:.3f}"
    )


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
    by: object = None,
) -> Any:
    """Annotated reliability diagram.

    Binned points with Wilson CIs, optional smooth overlay, stats box, and
    event/non-event risk distribution.

    ``scale="logit"`` stretches the low-probability region — the recommended
    view for PD portfolios. Bins whose event rate is exactly 0 or 1 have no
    finite logit and are omitted from the logit-scale point layer; they remain
    visible in the risk distribution (or the ``counts=True`` margin).
    Confidence bounds at exactly 0 or 1 are clipped to ``[1e-6, 1 - 1e-6]``
    before the logit transform.

    Passing a :class:`probcal.curves.KernelReliabilityCurve` (from
    :func:`probcal.curves.reliability_smooth`) as ``smooth`` renders the
    density-weighted variable-width curve instead of a plain line: a
    ``LineCollection`` whose width tracks the local prediction density (one
    width per segment, ``density[:-1]`` — the density at the *left* endpoint
    of each ``[grid[i], grid[i+1]]`` segment, since a ``LineCollection`` of
    ``len(grid) - 1`` segments needs exactly that many widths), the shaded
    miscalibration area between the curve and the identity, the bootstrap
    ribbon, and an ``smECE = ...`` readout in the bottom-right corner.

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

    Passing ``by`` switches to a faceted grid: one panel per sorted,
    stringified group in ``by`` (matching :func:`probcal.metrics.evaluate`'s
    ``by=`` convention) plus a leading "pooled" panel, each a fresh
    :func:`probcal.curves.reliability_binned` panel built from that group's
    slice of ``y``/``p`` — the given ``curve`` is ignored for the panels
    (it would otherwise be ambiguous which group it represents). ``y`` and
    ``p`` are required in this mode. Every panel honours ``scale``,
    ``annotate``, ``stats``, ``rug``, ``risk_dist`` and ``counts`` exactly as
    the single-panel diagram does (pass ``annotate=False, rug=False`` for
    light panels), sharing x/y limits across the grid; the function then
    returns the **Figure**, not an ``Axes`` (unlike the ``by=None`` default,
    matching :func:`plot_comparison`). ``ax`` and ``smooth`` are not
    supported with ``by`` and raise ``ValueError``.
    ``"pooled"`` is a reserved panel title: a group of your own by that name
    is indistinguishable from the pooled panel. Group-conditional
    statistical *testing* is out of scope here — see
    ``docs/guide/groups.md``.

    Parameters
    ----------
    curve : ReliabilityCurve
        Binned curve, e.g. from :func:`probcal.curves.reliability_binned`.
        Ignored when ``by`` is given.
    smooth : SmoothReliabilityCurve, KernelReliabilityCurve, or None, keyword-only
        Optional smooth overlay, e.g. from
        :func:`probcal.curves.reliability_loess` or
        :func:`probcal.curves.reliability_smooth`.
    scale : {"probability", "logit"}, keyword-only
        Axis scale; ``"logit"`` stretches the low-probability region.
    y, p : array_like or None, keyword-only
        Raw outcomes and predictions; must be given together (or not at all).
        Enables the stats box and risk distribution; required when ``by``
        is given.
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
        Must be ``None`` when ``by`` is given (a new figure of panels is
        always created).
    stats : bool or MetricReport, keyword-only
        If truthy and ``y``/``p`` are given, draw the ``n, events,
        intercept, slope, ICI, smECE, Brier`` stats box (``True``) or a
        ``MetricReport``-driven box, replacing ``annotate``'s box.
    risk_dist : {"rug", "split"} or None, keyword-only
        Density-layer style; see above. Anything else raises ``ValueError``.
    by : array_like or None, keyword-only
        Optional group labels, one per observation (same length as ``y``);
        see above. ``None`` (default) is the single-panel diagram above,
        unchanged.

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.figure.Figure
        The axes the diagram was drawn on (``by=None``, the default), or
        the figure of faceted panels (``by`` given).

    Raises
    ------
    ValueError
        If ``y``/``p`` are not given together, or ``risk_dist`` is not one
        of ``"rug"``, ``"split"``, ``None``; or if ``by`` is given without
        both ``y`` and ``p``, with a length that does not match ``y``, or
        together with ``ax`` or ``smooth``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.curves import reliability_binned
    >>> from probcal.plots import plot_reliability
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p).astype(float)
    >>> curve = reliability_binned(y, p, n_bins=10)
    >>> ax = plot_reliability(curve, scale="logit", y=y, p=p)  # doctest: +SKIP
    >>> segment = np.where(p < 0.2, "low", "high")
    >>> fig = plot_reliability(curve, y=y, p=p, by=segment)  # doctest: +SKIP
    """
    _require_mpl()
    if (y is None) != (p is None):
        raise ValueError("y and p must be given together")
    if risk_dist not in ("rug", "split", None):
        raise ValueError('risk_dist must be one of "rug", "split", None')
    if by is not None:
        if y is None or p is None:
            raise ValueError("by requires y and p")
        if ax is not None:
            raise ValueError(
                "ax is not supported with by= (a new figure of panels is always created); "
                "draw each group on your own axes with by=None instead"
            )
        if smooth is not None:
            raise ValueError(
                "smooth is not supported with by= (one smooth curve cannot describe every "
                "group); draw each group with by=None and its own smooth curve instead"
            )
        return _plot_reliability_faceted(
            y,
            p,
            by,
            scale=scale,
            annotate=annotate,
            rug=rug,
            counts=counts,
            stats=stats,
            risk_dist=risk_dist,
        )
    y_arr = None if y is None else np.asarray(y, dtype=np.float64)
    p_arr = None if p is None else np.asarray(p, dtype=np.float64)
    with _plt.rc_context(_STYLE):
        ax = _get_axes(ax, (6.5, 6))
        _draw_reliability(
            ax,
            curve,
            smooth=smooth,
            scale=scale,
            y_arr=y_arr,
            p_arr=p_arr,
            annotate=annotate,
            rug=rug,
            counts=counts,
            stats=stats,
            risk_dist=risk_dist,
        )
        return ax


def _draw_reliability(
    ax: Any,
    curve: ReliabilityCurve,
    *,
    smooth: SmoothReliabilityCurve | KernelReliabilityCurve | None = None,
    scale: str = "probability",
    y_arr: np.ndarray | None = None,
    p_arr: np.ndarray | None = None,
    annotate: bool = True,
    rug: bool = True,
    counts: bool = False,
    stats: bool | MetricReport = False,
    risk_dist: str | None = "rug",
    color: str = _BLUE,
) -> None:
    """Body of :func:`plot_reliability` on a given (styled) ``ax``; ``color``
    is the binned series' color (``plot_comparison`` uses before/after colors,
    so the legend matches the points)."""
    if scale == "logit":
        keep = (curve.event_rate > 0.0) & (curve.event_rate < 1.0)
        x = curve.pred_mean_logit[keep]
        anchor = x if len(x) else curve.pred_mean_logit[np.isfinite(curve.pred_mean_logit)]
        lo, hi = (anchor.min(), anchor.max()) if len(anchor) else _DEFAULT_LOGIT_SPAN
        diag: Any = np.linspace(lo - 0.5, hi + 0.5, 50)
    else:
        keep = np.ones(len(curve.event_rate), dtype=bool)
        x = curve.pred_mean
        diag = [0, 1]
    yv = _tr(curve.event_rate[keep], scale)
    ylo = _tr(curve.ci_low[keep], scale, _CI_CLIP_EPS)
    yhi = _tr(curve.ci_high[keep], scale, _CI_CLIP_EPS)
    ax.plot(diag, diag, ls="--", c=_GREY, lw=1, label="identity")
    ax.errorbar(
        x,
        yv,
        yerr=[np.maximum(yv - ylo, 0.0), np.maximum(yhi - yv, 0.0)],
        fmt="o",
        ms=4,
        capsize=2,
        color=color,
        label="binned",
    )
    if isinstance(smooth, KernelReliabilityCurve):
        _draw_kernel_curve(ax, smooth, scale)
    elif smooth is not None:
        grid = smooth.grid_logit if scale == "logit" else smooth.grid_p
        ax.plot(grid, _tr(smooth.event_rate, scale), lw=1.5, c=_ORANGE, label="smoothed")
    _scale_axis_labels(ax, scale, "predicted probability", "event rate")

    boxed = False
    if y_arr is not None and p_arr is not None:
        show_density = rug and risk_dist is not None
        if show_density and risk_dist == "rug":
            ev = _tr(_rug_subsample(p_arr[y_arr == 1.0]), scale)
            ne = _tr(_rug_subsample(p_arr[y_arr == 0.0]), scale)
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
        txt = _stats_box_text(stats, annotate, y_arr, p_arr)
        if txt is not None:
            _text_box(ax, txt)
            boxed = True
    if counts and len(curve.count):
        # Count margin as a twin bar strip along the x-axis.
        ax2 = ax.twinx()
        ax2.grid(False)
        xs = curve.pred_mean_logit if scale == "logit" else curve.pred_mean
        span = float(np.ptp(xs))
        width = span / (3 * len(xs) + 1) if span > 0 else (0.2 if scale == "logit" else 0.02)
        ax2.bar(xs, curve.count, width=width, alpha=0.15, color=_GREY)
        ax2.set_yticks([])
    ax.legend(loc="lower right" if boxed else "upper left")


def _plot_reliability_faceted(
    y: object,
    p: object,
    by: object,
    *,
    scale: str,
    annotate: bool,
    rug: bool,
    counts: bool,
    stats: bool | MetricReport,
    risk_dist: str | None,
) -> Any:
    """``plot_reliability(..., by=...)``: a pooled panel plus one panel per
    sorted group, each built fresh from that group's slice of ``y``/``p``
    (``curve`` is not used here — see the docstring above)."""
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    by_arr = np.asarray(by)
    if len(by_arr) != len(y_arr):
        raise ValueError("by must have the same length as y")
    labels = np.array([str(g) for g in by_arr])
    groups = tuple(sorted(set(labels.tolist())))
    panels = [("pooled", y_arr, p_arr)]
    panels += [(g, y_arr[labels == g], p_arr[labels == g]) for g in groups]

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)
    with _plt.rc_context(_STYLE):
        fig, axes = _plt.subplots(
            nrows, ncols, figsize=(6.5 * ncols, 6 * nrows), sharex=True, sharey=True, squeeze=False
        )
        axes_flat = axes.ravel()
        for ax, (label, yy, pp) in zip(axes_flat, panels, strict=False):
            _style_axes(ax)
            _draw_reliability(
                ax,
                reliability_binned(yy, pp),
                scale=scale,
                y_arr=yy,
                p_arr=pp,
                annotate=annotate,
                rug=rug,
                counts=counts,
                stats=stats,
                risk_dist=risk_dist,
            )
            ax.set_title(label)
        for ax in axes_flat[n_panels:]:
            fig.delaxes(ax)
        return fig


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

    def _band(bound: np.ndarray) -> np.ndarray:
        return _tr(bound, scale, _CI_CLIP_EPS)

    with _plt.rc_context(_STYLE):
        ax = _get_axes(ax, (6.5, 6))
        x = belt.grid_logit if scale == "logit" else belt.grid_p
        ax.plot(x, x, ls="--", c=_GREY, lw=1)
        ax.fill_between(
            x, _band(belt.lower_95), _band(belt.upper_95), color=_BLUE, alpha=0.2, label="95%"
        )
        ax.fill_between(
            x, _band(belt.lower_80), _band(belt.upper_80), color=_BLUE, alpha=0.35, label="80%"
        )
        ax.set_title(f"calibration belt (degree {belt.degree}, p = {belt.p_value:.3g})")
        _scale_axis_labels(ax, scale, "predicted probability", "event rate")
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
            # The binned series takes the panel's before/after color at draw
            # time, so the legend handle matches the points.
            _style_axes(ax)
            _draw_reliability(ax, curve, scale=scale, color=color)
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
        Scores the intervals are plotted against; need not be sorted (the
        pairs are drawn in ascending order of ``s``).
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the intervals were drawn on.
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        ax = _get_axes(ax, (6.5, 4.5))
        s_arr = np.asarray(s, dtype=np.float64)
        order = np.argsort(s_arr, kind="stable")
        s_arr = s_arr[order]
        p0 = np.asarray(intervals, dtype=np.float64)[order, 0]
        p1 = np.asarray(intervals, dtype=np.float64)[order, 1]
        ax.fill_between(s_arr, p0, p1, color=_BLUE, alpha=0.3, label="Venn–Abers interval")
        ax.plot(s_arr, p1 / (1.0 - p0 + p1), lw=1.2, c=_ORANGE, label="scalarized")
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
        ax = _get_axes(ax, (6.5, 0.6 * len(report.methods) + 1.5))
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
        ax = _get_axes(ax, (7.5, 4.8))
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
    name = "Jeffreys" if isinstance(result, JeffreysGradeResult) else "exact binomial"
    x = np.arange(len(result.grades))
    rate = result.k / result.n
    colors = [light_color.get(li, _GREY) for li in result.light]
    with _plt.rc_context(_STYLE):
        ax = _get_axes(ax, (1.1 * len(x) + 3.0, 4.8))
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
        ax = _get_axes(ax, (6.5, 6))
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
        _text_box(ax, txt)
        _scale_axis_labels(ax, "logit", "input probability", "shifted probability")
        ax.set_title("logit offset audit")
        ax.legend(loc="lower right")
        return ax


def plot_e_process(report: Any, *, grades_panel: bool = False, ax: Any = None) -> Any:
    """Monitoring wealth per component on a log scale, with the 1/alpha line.

    Parameters
    ----------
    report : MonitorReport
        Result of :meth:`probcal.monitor.CalibrationMonitor.report`.
    grades_panel : bool, default False
        Add a second, shorter axes below the main plot showing each grade's
        offset confidence-sequence band (``MonitorStep.grade_delta_ci``)
        across steps. The panel is carved out of the main axes' own area
        (``mpl_toolkits.axes_grid1`` divider, shared x-axis), so it stays
        inside the figure and inside a caller-supplied ``ax``'s slot; the
        step labels then move to the panel. When this function creates the
        figure, it is made taller (7.5 x 6.5 in) to fit the panel. Additive:
        the default (``False``) output is pinned by
        ``tests/test_plots_regression.py``.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The main axes the e-processes were drawn on (unchanged even when
        ``grades_panel=True`` adds a second axes to the same figure).
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        ax = _get_axes(ax, (7.5, 6.5) if grades_panel else (7.5, 4.2))
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
        if grades_panel:
            _draw_grades_panel(ax, steps, x)
        return ax


def _draw_grades_panel(ax: Any, steps: Any, x: np.ndarray) -> None:
    """Second axes: each grade's offset confidence-sequence band across steps."""
    grade_names = sorted({g for s in steps for g in s.grade_delta_ci})
    if not grade_names:
        return
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    gax = make_axes_locatable(ax).append_axes("bottom", size="55%", pad=0.55, sharex=ax)
    _style_axes(gax)
    ax.tick_params(axis="x", labelbottom=False)
    palette = (_BLUE, _ORANGE, _GREEN, _RED, _AMBER)
    for i, g in enumerate(grade_names):
        lo = np.array(
            [s.grade_delta_ci[g][0] if s.grade_delta_ci.get(g) else np.nan for s in steps]
        )
        hi = np.array(
            [s.grade_delta_ci[g][1] if s.grade_delta_ci.get(g) else np.nan for s in steps]
        )
        color = palette[i % len(palette)]
        gax.fill_between(x, lo, hi, color=color, alpha=0.25, label=f"grade {g}")
    gax.axhline(0.0, color=_GREY, linewidth=0.8)
    gax.set_xticks(x)
    gax.set_xticklabels([s.label for s in steps], rotation=45, ha="right", fontsize=8)
    gax.set_ylabel("offset CS (log-odds)")
    gax.set_title("per-grade confidence sequences")
    gax.legend(loc="upper left", fontsize=8)


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
