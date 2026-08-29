"""Diagnostic plots split out of ``probcal.plots`` (requires the [viz] extra).

Currently holds :func:`plot_corp`, the CORP reliability diagram. Split into
its own module to keep files small; ``plots.py`` imports it back so
``probcal.plots.plot_corp`` remains the public path. Theory:
``docs/concepts/visualization.md``.
"""

from typing import Any

import numpy as np

from ._math import logit
from ._plots_common import _BLUE, _BOX, _GREY, _STYLE, _logit_axis, _plt, _require_mpl
from ._results import CorpResult


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
