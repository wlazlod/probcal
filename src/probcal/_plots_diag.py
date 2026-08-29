"""Diagnostic plots split out of ``probcal.plots`` (requires the [viz] extra).

Holds :func:`plot_corp`, the CORP reliability diagram, and
:func:`plot_mcb_dsc`, the MCB-DSC plane. Split into its own module to keep
files small; ``plots.py`` imports them back so ``probcal.plots.plot_corp``/
``plot_mcb_dsc`` remain the public path. Theory:
``docs/concepts/visualization.md``.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._math import logit
from ._plots_common import _BLUE, _BOX, _GREEN, _GREY, _STYLE, _logit_axis, _plt, _require_mpl
from ._results import CorpResult, SelectionReport
from .curves import corp_reliability, reliability_binned
from .metrics.scores import MurphyCurve, _prep, murphy_curve


def plot_corp(
    result: CorpResult,
    *,
    scale: str = "probability",
    show_decomposition: bool = True,
    ax: Any = None,
) -> Any:
    """CORP reliability diagram: PAV step fit, resampled bands, and the Brier decomposition.

    Draws the PAV-recalibrated step function (each block's ``[block_lo,
    block_hi]`` at its ``block_level``, joined vertically between
    consecutive blocks) against the identity, with the resampled bands from
    :func:`probcal.curves.corp_reliability` shaded around it. Grey tick
    marks along the x-axis show each PAV block's centre, scaled to its
    weight share of the portfolio. ``scale="logit"`` clips edges to
    ``[1e-12, 1 - 1e-12]`` before the logit transform and stretches the
    low-probability region — the recommended view for PD portfolios.

    Parameters
    ----------
    result : CorpResult
        Result of :func:`probcal.curves.corp_reliability`.
    scale : {"probability", "logit"}, keyword-only
        Axis scale; ``"logit"`` stretches the low-probability region.
    show_decomposition : bool, keyword-only
        If ``True`` (default), draw the Brier/MCB/DSC/UNC decomposition box.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.curves import corp_reliability
    >>> from probcal.plots import plot_corp
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p).astype(float)
    >>> ax = plot_corp(corp_reliability(y, p, n_resamples=20))  # doctest: +SKIP
    """
    _require_mpl()

    def _tr(x: np.ndarray) -> np.ndarray:
        if scale == "logit":
            return logit(np.clip(x, 1e-12, 1.0 - 1e-12))
        return np.asarray(x, dtype=np.float64)

    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))
        lo, hi, level, weight = (
            result.block_lo,
            result.block_hi,
            result.block_level,
            result.block_weight,
        )
        domain = _tr(np.array([lo[0], hi[-1]]))
        ax.plot(domain, domain, ls="--", c=_GREY, lw=1, label="identity")

        if len(result.band_grid) > 0:
            ax.fill_between(
                _tr(result.band_grid),
                _tr(result.band_low),
                _tr(result.band_high),
                color=_BLUE,
                alpha=0.15,
                label=f"{result.level:.0%} {result.bands} band",
            )

        # Each block contributes [lo, hi] at its level; steps-post joins
        # consecutive blocks with a vertical segment at the right edge.
        x_edges = np.empty(2 * len(lo))
        x_edges[0::2] = lo
        x_edges[1::2] = hi
        y_levels = np.repeat(level, 2)
        ax.step(_tr(x_edges), _tr(y_levels), where="post", color=_BLUE, lw=2, label="PAV fit")

        centres = _tr((lo + hi) / 2.0)
        heights = 0.08 * weight / weight.max()
        ax.vlines(centres, 0.0, heights, color=_GREY, alpha=0.6, transform=ax.get_xaxis_transform())

        if scale == "logit":
            _logit_axis(ax)
            ax.set_xlabel("predicted probability (logit scale)")
            ax.set_ylabel("PAV-recalibrated probability (logit scale)")
        else:
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("PAV-recalibrated probability")

        if show_decomposition:
            txt = (
                f"Brier {result.brier:.4f}\n"
                f"MCB {result.brier_mcb:.4f}\n"
                f"DSC {result.brier_dsc:.4f}\n"
                f"UNC {result.brier_unc:.4f}"
            )
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9, bbox=_BOX)

        ax.set_title("CORP reliability diagram")
        ax.legend(loc="lower right")
        return ax


def plot_mcb_dsc(
    candidates: "Mapping[str, tuple[Any, Any]] | SelectionReport",
    *,
    score: str = "brier",
    ax: Any = None,
) -> Any:
    """MCB-DSC plane: CORP miscalibration vs. discrimination, one point per candidate.

    Each candidate is a point at ``(DSC, MCB)`` from its CORP decomposition
    (:func:`probcal.curves.corp_reliability`). Dashed grey iso-score
    diagonals trace ``MCB = DSC + (S̄ - UNC)`` for five values of the mean
    score S̄ spaced between the candidates' min and max — candidates on the
    same diagonal tie on ``score`` despite different miscalibration/
    discrimination splits, so the plane separates "worse calibrated" from
    "less discriminating" for two methods that score the same. Lower-right
    is better: more discrimination (DSC) for no more miscalibration (MCB).

    Parameters
    ----------
    candidates : mapping of str to (y, p), or SelectionReport
        Either a ``{name: (y, p)}`` mapping — each entry's CORP
        decomposition is computed fresh via ``corp_reliability(y, p,
        bands=None)``, and every entry's ``y`` must share the same weighted
        mean to ``1e-12`` (UNC, and therefore the iso-score diagonals, is
        only shared across candidates when it is) — or a fitted
        :class:`probcal.selection.CalibratorSelector`'s ``report_``, whose
        ``mcb``/``dsc``/``unc`` columns (probcal >= 0.3) are plotted
        directly; ``score`` is then informational only, since the report
        already fixed Brier vs. log loss at selection time.
    score : {"brier", "log_loss"}, keyword-only
        Which CORP decomposition to plot for a mapping input.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plane was drawn on.

    Raises
    ------
    ValueError
        If ``score`` is not ``"brier"`` or ``"log_loss"``; if a
        ``SelectionReport`` is given without ``mcb``/``dsc`` columns
        (fitted before probcal 0.3); if a mapping's candidates do not share
        the same weighted mean ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.plots import plot_mcb_dsc
    >>> rng = np.random.default_rng(0)
    >>> p_a = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p_a).astype(float)
    >>> p_b = np.clip(p_a * 0.9, 1e-6, 1 - 1e-6)
    >>> ax = plot_mcb_dsc({"a": (y, p_a), "b": (y, p_b)})  # doctest: +SKIP
    """
    _require_mpl()
    if score not in ("brier", "log_loss"):
        raise ValueError('score must be "brier" or "log_loss"')

    if isinstance(candidates, SelectionReport):
        if candidates.mcb is None or candidates.dsc is None or candidates.unc is None:
            raise ValueError("report has no mcb/dsc columns; refit with probcal>=0.3")
        names = list(candidates.methods)
        mcb = np.asarray(candidates.mcb, dtype=np.float64)
        dsc = np.asarray(candidates.dsc, dtype=np.float64)
        unc = float(candidates.unc)
    else:
        names = list(candidates)
        mcb = np.empty(len(names))
        dsc = np.empty(len(names))
        unc = 0.0
        ybar0: float | None = None
        for i, name in enumerate(names):
            y_i, p_i = candidates[name]
            y_arr, p_arr, w_arr = _prep(y_i, p_i, None)
            ybar = float(np.average(y_arr, weights=w_arr))
            if ybar0 is None:
                ybar0 = ybar
            elif abs(ybar - ybar0) > 1e-12:
                raise ValueError(
                    "candidates must share the same weighted mean y "
                    "(UNC would differ across candidates)"
                )
            r = corp_reliability(y_arr, p_arr, sample_weight=w_arr, bands=None)
            mcb[i] = r.brier_mcb if score == "brier" else r.log_loss_mcb
            dsc[i] = r.brier_dsc if score == "brier" else r.log_loss_dsc
            unc = r.brier_unc if score == "brier" else r.log_loss_unc

    mean_score = mcb - dsc + unc

    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))

        spread = float(dsc.max() - dsc.min())
        pad = spread * 0.15 if spread > 0 else max(0.05, 0.1 * float(dsc.max()))
        x0, x1 = max(0.0, float(dsc.min()) - pad), float(dsc.max()) + pad

        for s_bar in np.linspace(mean_score.min(), mean_score.max(), 5):
            y0, y1 = x0 + (s_bar - unc), x1 + (s_bar - unc)
            ax.plot([x0, x1], [y0, y1], color=_GREY, lw=0.8, ls=":", zorder=1)
            ax.annotate(
                f"S̄={s_bar:.4g}",
                xy=(x1, y1),
                fontsize=8,
                color=_GREY,
                ha="right",
                va="bottom",
            )

        ax.scatter(dsc, mcb, color=_BLUE, zorder=3)
        for name, d, m in zip(names, dsc, mcb, strict=True):
            ax.annotate(name, (d, m), textcoords="offset points", xytext=(5, 5), fontsize=9)

        ax.set_xlim(x0, x1)
        ax.set_xlabel("DSC (discrimination)")
        ax.set_ylabel("MCB (miscalibration)")
        ax.set_title("MCB-DSC plane")
        return ax


def plot_attributes(
    y: object,
    p: object,
    *,
    method: str = "binned",
    n_bins: int = 10,
    scale: str = "probability",
    sample_weight: object = None,
    ax: Any = None,
) -> Any:
    """Attributes diagram: reliability curve against climatology and no-skill references.

    Draws the classic Hsu & Murphy (1986) attributes diagram: the identity
    (perfect calibration), the horizontal and vertical climatology
    references at the weighted base rate :math:`\\bar y`, the no-skill line
    :math:`y = (x + \\bar y) / 2` (equidistant between the climatology
    level and the identity), and a light shading of the region where a
    point beats climatology, :math:`(y - x)^2 \\le (x - \\bar y)^2` — i.e.
    where the point sits closer to the identity than the horizontal
    no-resolution line, the geometric criterion for positive Brier skill.
    The reliability curve is drawn on top: ``method="binned"`` overlays
    :func:`probcal.curves.reliability_binned` as markers sized by bin
    count; ``method="corp"`` overlays the PAV step fit from
    :func:`probcal.curves.corp_reliability` (``bands=None``), the same
    convention as :func:`probcal.plots.plot_corp`. ``scale="logit"``
    transforms every drawn quantity (clipped to ``[1e-12, 1 - 1e-12]``)
    through :func:`probcal._math.logit` and relabels the axes in
    probabilities.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    method : {"binned", "corp"}, keyword-only
        Reliability construction to overlay.
    n_bins : int, keyword-only
        Bin count passed to :func:`probcal.curves.reliability_binned`
        (``method="binned"`` only).
    scale : {"probability", "logit"}, keyword-only
        Axis scale; ``"logit"`` stretches the low-probability region.
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn on.

    Raises
    ------
    ValueError
        If ``method`` is not ``"binned"`` or ``"corp"``.

    Notes
    -----
    The identity, climatology, no-skill, and shading layers are evaluated on
    a fixed 400-point grid over ``[0, 1]`` — a plotting-fidelity choice
    (dense enough to render as smooth curves at the figure's default
    ``figsize=(6.5, 6)``), not a statistical one; it does not depend on
    ``n_bins`` or the data.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.plots import plot_attributes
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p).astype(float)
    >>> ax = plot_attributes(y, p, method="corp")  # doctest: +SKIP
    """
    _require_mpl()
    if method not in ("binned", "corp"):
        raise ValueError('method must be "binned" or "corp"')
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    ybar = float(np.average(y_arr, weights=w))

    def _tr(x: np.ndarray) -> np.ndarray:
        if scale == "logit":
            return logit(np.clip(x, 1e-12, 1.0 - 1e-12))
        return np.asarray(x, dtype=np.float64)

    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))

        grid = np.linspace(0.0, 1.0, 400)
        ybar_t = float(_tr(np.array([ybar]))[0])
        ax.plot(_tr(grid), _tr(grid), ls="--", color=_GREY, lw=1, label="identity")
        ax.axhline(ybar_t, color=_GREY, ls=":", lw=1, label="climatology / no resolution")
        ax.axvline(ybar_t, color=_GREY, ls=":", lw=1, label="climatology")
        no_skill = (grid + ybar) / 2.0
        ax.plot(_tr(grid), _tr(no_skill), color=_GREY, ls="-.", lw=1, label="no skill")

        half_width = np.abs(grid - ybar)
        y_upper = np.clip(grid + half_width, 0.0, 1.0)
        y_lower = np.clip(grid - half_width, 0.0, 1.0)
        ax.fill_between(_tr(grid), _tr(y_lower), _tr(y_upper), color=_GREEN, alpha=0.08)

        if method == "binned":
            curve = reliability_binned(y_arr, p_arr, n_bins=n_bins, sample_weight=w)
            sizes = 15.0 + 200.0 * curve.count / curve.count.max()
            ax.scatter(
                _tr(curve.pred_mean),
                _tr(curve.event_rate),
                s=sizes,
                color=_BLUE,
                zorder=5,
                label="binned",
            )
        else:
            result = corp_reliability(y_arr, p_arr, sample_weight=w, bands=None)
            lo, hi, level = result.block_lo, result.block_hi, result.block_level
            x_edges = np.empty(2 * len(lo))
            x_edges[0::2] = lo
            x_edges[1::2] = hi
            y_levels = np.repeat(level, 2)
            ax.step(_tr(x_edges), _tr(y_levels), where="post", color=_BLUE, lw=2, label="PAV fit")

        if scale == "logit":
            _logit_axis(ax)
            ax.set_xlabel("predicted probability (logit scale)")
            ax.set_ylabel("observed event rate (logit scale)")
        else:
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("observed event rate")

        ax.set_title("attributes diagram")
        ax.legend(loc="lower right")
        return ax


def plot_murphy(
    curves: "MurphyCurve | Mapping[str, MurphyCurve] | Mapping[str, tuple[Any, Any]]",
    *,
    diff: bool = False,
    n_boot: int = 200,
    random_state: int = 42,
    ax: Any = None,
) -> Any:
    """Murphy diagram: mean elementary score across a threshold grid, or a paired difference.

    ``diff=False`` (default) draws one line per curve: a single
    :class:`probcal.metrics.MurphyCurve`, or a ``{name: MurphyCurve}``
    mapping with a legend. ``diff=True`` instead requires a mapping of
    exactly two ``{name: (y, p)}`` raw-data entries — a ``MurphyCurve``
    does not retain ``y``/``p``, so the pointwise difference and its
    bootstrap band are recomputed from the paired data — and draws
    ``S_theta(A) - S_theta(B)`` (on ``A``'s default 513-point threshold
    grid) with a seeded pointwise bootstrap band (paired-index resampling,
    5th/95th percentile) and a zero reference line.

    Parameters
    ----------
    curves : MurphyCurve, mapping of str to MurphyCurve, or mapping of str to (y, p)
        See above; the last form only when ``diff=True``.
    diff : bool, keyword-only
        If ``True``, draw the paired difference of the two named curves'
        elementary scores instead of the curves themselves.
    n_boot : int, keyword-only
        Bootstrap resamples for the difference band (``diff=True`` only).
    random_state : int, keyword-only
        Seed for ``numpy.random.default_rng``, used by the bootstrap band.
    ax : matplotlib.axes.Axes or None, keyword-only
        Axes to draw on; a new figure and axes are created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn on.

    Raises
    ------
    ValueError
        If ``diff=True`` and ``curves`` is not a mapping of exactly two
        raw ``(y, p)`` pairs of equal length.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import murphy_curve
    >>> from probcal.plots import plot_murphy
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.5, 300)
    >>> y = (rng.random(300) < p).astype(float)
    >>> ax = plot_murphy({"model": murphy_curve(y, p)})  # doctest: +SKIP
    """
    _require_mpl()
    with _plt.rc_context(_STYLE):
        if ax is None:
            _, ax = _plt.subplots(figsize=(6.5, 6))

        if diff:
            if not isinstance(curves, Mapping):
                raise ValueError(
                    "diff=True needs a mapping of exactly two {name: (y, p)} raw-data pairs"
                )
            pairs = list(curves.items())
            if len(pairs) != 2:
                raise ValueError(
                    "diff=True needs a mapping of exactly two {name: (y, p)} raw-data pairs"
                )
            name_a, pair_a = pairs[0]
            name_b, pair_b = pairs[1]
            if (
                not isinstance(pair_a, tuple)
                or not isinstance(pair_b, tuple)
                or len(pair_a) != 2
                or len(pair_b) != 2
            ):
                raise ValueError(
                    "diff=True needs a mapping of exactly two {name: (y, p)} raw-data pairs"
                )
            y_a, p_a = pair_a
            y_b, p_b = pair_b
            y_a, p_a, _w_a = _prep(y_a, p_a, None)
            y_b, p_b, _w_b = _prep(y_b, p_b, None)
            if len(y_a) != len(y_b):
                raise ValueError("diff=True needs paired (y, p) of equal length")

            theta = murphy_curve(y_a, p_a).thresholds
            score_a = murphy_curve(y_a, p_a, thresholds=theta).score
            score_b = murphy_curve(y_b, p_b, thresholds=theta).score
            delta = score_a - score_b

            rng = np.random.default_rng(random_state)
            n = len(y_a)
            boot = np.empty((n_boot, len(theta)))
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                ca = murphy_curve(y_a[idx], p_a[idx], thresholds=theta)
                cb = murphy_curve(y_b[idx], p_b[idx], thresholds=theta)
                boot[b] = ca.score - cb.score
            lo = np.percentile(boot, 5, axis=0)
            hi = np.percentile(boot, 95, axis=0)

            ax.fill_between(theta, lo, hi, color=_BLUE, alpha=0.15, label="90% bootstrap band")
            ax.plot(theta, delta, color=_BLUE, lw=2, label=f"{name_a} - {name_b}")
            ax.axhline(0.0, color=_GREY, lw=1, ls="--", label="zero")
            ax.legend(loc="best")
        elif isinstance(curves, MurphyCurve):
            ax.plot(curves.thresholds, curves.score, color=_BLUE, lw=2)
        else:
            for name, curve in curves.items():
                if not isinstance(curve, MurphyCurve):
                    raise ValueError(
                        "plot_murphy: mapping values must be MurphyCurve objects; "
                        "pass diff=True with (y, p) pairs for a difference plot"
                    )
                ax.plot(curve.thresholds, curve.score, lw=2, label=name)
            ax.legend(loc="best")

        ax.set_xlabel("threshold θ")
        ax.set_ylabel("mean elementary score")
        ax.set_title("Murphy diagram")
        return ax
