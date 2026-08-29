"""Self-contained HTML/markdown validation report (spec E2).

:func:`validation_report` assembles a single document — one HTML file with
base64-embedded PNG figures, or a markdown file plus a sibling directory of
PNGs — out of the diagnostics already computed elsewhere in the package:
the reliability diagrams (``curves``/``plots``), the metric catalog
(``metrics.evaluate``), the CORP score decomposition, the per-grade
Jeffreys/Pluto-Tasche backtests, grouped evaluation, and monitor
trajectories. Nothing here computes new statistics; every number and figure
is produced by the existing public API and merely rendered into one
document for handoff (a model-risk file, an audit trail, a stakeholder
readout).

Import cost: this module is stdlib + numpy + probcal at import time —
matplotlib is only ever imported lazily, inside the figure-rendering path,
so ``import probcal.report`` never pulls in the ``[viz]`` extra even when
it is installed. Calling :func:`validation_report` does require it (every
section renders at least one figure from ``y``/``p`` alone); the
``ImportError`` it raises without the extra names ``probcal[viz]``.

Determinism: every resampling site (``metrics.evaluate``,
``curves.corp_reliability``, ``curves.reliability_smooth``) is driven by
the single ``seed`` keyword, and ``n_boot`` sizes all of them at once — the
report is bit-reproducible given the same inputs and ``seed``, apart from
the one ``Generated ... UTC`` timestamp line.
"""

import base64
import io
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from . import __version__
from ._results import GroupedMetricReport, MetricReport, _format_cell
from ._serialize import data_fingerprint
from .curves import corp_reliability, reliability_binned, reliability_smooth
from .metrics import (
    evaluate,
    jeffreys_grade_test,
    jeffreys_upper_bands,
    pluto_tasche_from_arrays,
)

# Availability check only (no import): keeps this module matplotlib-free at
# import time even when the [viz] extra is installed. `_figure_sink` below
# is the only place that actually imports matplotlib (via `_plots_common`).
_HAS_MPL = find_spec("matplotlib") is not None


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for probcal.report figures — install the "
            "viz extra: pip install probcal[viz]"
        )


# --------------------------------------------------------------------- render

_HTML_STYLE = """
body { font-family: -apple-system, sans-serif; max-width: 960px; margin: 2rem auto;
  padding: 0 1rem; color: #222; }
h1 { border-bottom: 2px solid #2f5f8a; padding-bottom: .3rem; }
h2 { color: #2f5f8a; margin-top: 2.5rem; }
h3 { margin-top: 1.5rem; }
table { border-collapse: collapse; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: .3rem .6rem; text-align: right;
  font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { background: #f0f0ee; }
.meta { color: #666; font-size: .9rem; }
img { max-width: 100%; margin: .5rem 0; }
details { margin: 1rem 0; }
pre { background: #f7f7f5; padding: .75rem; overflow-x: auto; }
"""

_HTML_TEMPLATE = Template(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$title</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<h1>$title</h1>
<p class="meta">$timestamp (probcal $version)</p>
<section class="fingerprints">$fingerprints</section>
$sections
</body>
</html>
""")

_MD_TEMPLATE = Template("""# $title

$timestamp (probcal $version)

$fingerprints

$sections""")


class _FigureSink:
    """Renders one figure per call, embedded (html) or written to disk (markdown)."""

    def __init__(self, fmt: str, path: "str | os.PathLike[str] | None") -> None:
        self.fmt = fmt
        self.fig_dir: Path | None = None
        self.fig_dir_name: str | None = None
        if fmt == "markdown":
            p = Path(path)  # type: ignore[arg-type]  # validated non-None by the caller
            self.fig_dir_name = f"{p.stem}_figures"
            self.fig_dir = p.parent / self.fig_dir_name

    def figure(self, draw: Any, name: str) -> str:
        """``draw()`` returns an Axes or Figure; render it to PNG and reference it."""
        _require_mpl()
        from ._plots_common import _plt  # local: the only matplotlib import site

        artist = draw()
        fig = artist if hasattr(artist, "savefig") else artist.figure
        if self.fmt == "html":
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            _plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f'<img src="data:image/png;base64,{b64}" alt="{name}">'
        assert self.fig_dir is not None
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.fig_dir / f"{name}.png", format="png", dpi=110)
        _plt.close(fig)
        return f"![{name}]({self.fig_dir_name}/{name}.png)"


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table(fmt: str, headers: "tuple[str, ...]", rows: "list[tuple[Any, ...]]") -> str:
    cells = [tuple(_format_cell(v) for v in row) for row in rows]
    if fmt == "html":
        thead = "".join(f"<th>{h}</th>" for h in headers)
        tbody = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)
        return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [("| " + " | ".join(row) + " |") for row in cells]
    return "\n".join([header_line, sep_line, *body_lines]) + "\n"


def _kv(fmt: str, pairs: "Sequence[tuple[str, object]]") -> str:
    if fmt == "html":
        items = "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in pairs)
        return f"<ul>{items}</ul>"
    return "\n".join(f"- **{k}:** {v}" for k, v in pairs) + "\n"


def _bullets(fmt: str, items: "tuple[str, ...]") -> str:
    if not items:
        return ""
    if fmt == "html":
        return "<ul>" + "".join(f"<li>{_escape_html(m)}</li>" for m in items) + "</ul>"
    return "\n".join(f"- {m}" for m in items) + "\n"


def _subheading(fmt: str, text: str) -> str:
    return f"<h3>{text}</h3>\n" if fmt == "html" else f"### {text}\n\n"


def _section(fmt: str, title: str, body: str) -> str:
    if not body:
        return ""
    if fmt == "html":
        return f"<section>\n<h2>{title}</h2>\n{body}\n</section>\n"
    return f"## {title}\n\n{body}\n\n"


# --------------------------------------------------------------------- sections


def _section_header(fmt: str, y_arr: np.ndarray, p_arr: np.ndarray) -> str:
    rows = [
        ("n", f"{len(y_arr):,}"),
        ("events", f"{int(np.sum(y_arr)):,}"),
        ("event rate", f"{float(np.mean(y_arr)):.4%}"),
        ("mean predicted probability", f"{float(np.mean(p_arr)):.4%}"),
    ]
    return _section(fmt, "Portfolio summary", _kv(fmt, rows))


def _section_reliability(
    fmt: str,
    y_arr: np.ndarray,
    p_arr: np.ndarray,
    corp: Any,
    *,
    n_boot: int,
    seed: int,
    sink: _FigureSink,
) -> str:
    kernel = reliability_smooth(y_arr, p_arr, n_boot=n_boot, random_state=seed)
    binned = reliability_binned(y_arr, p_arr)

    def draw_corp() -> Any:
        from .plots import plot_corp

        return plot_corp(corp)

    def draw_kernel() -> Any:
        from .plots import plot_reliability

        return plot_reliability(binned, smooth=kernel, y=y_arr, p=p_arr, stats=True)

    body = (
        _subheading(fmt, "CORP (PAV) reliability diagram")
        + sink.figure(draw_corp, "reliability_corp")
        + _subheading(fmt, "smECE-consistent kernel reliability curve")
        + sink.figure(draw_kernel, "reliability_kernel")
    )
    return _section(fmt, "Reliability", body)


def _section_evaluate(
    fmt: str, y_arr: np.ndarray, p_arr: np.ndarray, *, n_boot: int, seed: int
) -> str:
    report = evaluate(y_arr, p_arr, n_boot=n_boot, seed=seed)
    assert isinstance(report, MetricReport)  # by=None (unset here) always returns MetricReport
    headers = ("metric", "value", "ci_low", "ci_high")
    rows = list(zip(report.names, report.values, report.ci_low, report.ci_high, strict=True))
    return _section(fmt, "Metric report", _table(fmt, headers, rows))


def _section_corp_decomposition(fmt: str, corp: Any) -> str:
    headers = ("score", "value", "mcb", "dsc", "unc")
    rows = [
        ("brier", corp.brier, corp.brier_mcb, corp.brier_dsc, corp.brier_unc),
        ("log_loss", corp.log_loss, corp.log_loss_mcb, corp.log_loss_dsc, corp.log_loss_unc),
    ]
    note = "score == mcb - dsc + unc (Dimitriadis, Gneiting & Jordan 2021)."
    note_html = f"<p>{note}</p>" if fmt == "html" else f"\n{note}\n"
    return _section(fmt, "CORP decomposition", _table(fmt, headers, rows) + note_html)


def _grade_order(p_arr: np.ndarray, grades: object) -> "tuple[str, ...]":
    """Grade labels sorted best to worst by mean predicted probability (ascending)."""
    g_arr = np.array([str(x) for x in np.asarray(grades)])
    labels = sorted(set(g_arr.tolist()))
    mean_p = {lab: float(np.mean(p_arr[g_arr == lab])) for lab in labels}
    return tuple(sorted(labels, key=lambda lab: mean_p[lab]))


def _section_grades(
    fmt: str, y_arr: np.ndarray, p_arr: np.ndarray, grades: object, *, sink: _FigureSink
) -> str:
    order = _grade_order(p_arr, grades)
    backtest = jeffreys_grade_test(y_arr, p_arr, grades)

    def draw() -> Any:
        from .plots import plot_grade_backtest

        return plot_grade_backtest(backtest)

    headers = ("grade", "n", "k", "pd", "p_value", "light", "ci_low", "ci_high")
    rows = list(
        zip(
            backtest.grades,
            backtest.n,
            backtest.k,
            backtest.pd,
            backtest.p_value,
            backtest.light,
            backtest.ci_low,
            backtest.ci_high,
            strict=True,
        )
    )

    pt = pluto_tasche_from_arrays(grades, y_arr, order=order)
    pt_headers = ("grade", "n", "d", "n_pooled", "d_pooled", "pd_upper")
    pt_rows = list(zip(pt.grades, pt.n, pt.d, pt.n_pooled, pt.d_pooled, pt.pd_upper, strict=True))

    bands = jeffreys_upper_bands(y_arr, p_arr, grades, order=order)
    band_headers = ("grade", "lo", "hi")
    band_rows = [(g, lo, hi) for g, (lo, hi) in bands.items()]

    order_note = f"Grades ordered best to worst by mean predicted probability: {', '.join(order)}."
    order_note_block = f"<p>{order_note}</p>" if fmt == "html" else f"\n{order_note}\n"

    body = (
        order_note_block
        + _subheading(fmt, "Per-grade backtest (Jeffreys)")
        + sink.figure(draw, "grade_backtest")
        + _table(fmt, headers, rows)
        + _subheading(fmt, "Pluto-Tasche most-prudent PD")
        + _table(fmt, pt_headers, pt_rows)
        + _subheading(fmt, "Jeffreys upper masterscale bands")
        + _table(fmt, band_headers, band_rows)
    )
    return _section(fmt, "Rating grades", body)


def _section_groups(
    fmt: str,
    y_arr: np.ndarray,
    p_arr: np.ndarray,
    by: object,
    *,
    n_boot: int,
    seed: int,
    sink: _FigureSink,
) -> str:
    grouped = evaluate(y_arr, p_arr, n_boot=n_boot, seed=seed, by=by)
    assert isinstance(grouped, GroupedMetricReport)  # by is not None here

    def draw() -> Any:
        from .plots import plot_reliability

        return plot_reliability(reliability_binned(y_arr, p_arr), y=y_arr, p=p_arr, by=by)

    headers = ("group", "metric", "value", "ci_low", "ci_high")
    panels = [("pooled", grouped.pooled), *zip(grouped.groups, grouped.reports, strict=True)]
    rows = [
        (label, n, v, lo, hi)
        for label, rep in panels
        for n, v, lo, hi in zip(rep.names, rep.values, rep.ci_low, rep.ci_high, strict=True)
    ]
    body = sink.figure(draw, "groups_panel") + _table(fmt, headers, rows)
    return _section(fmt, "Grouped evaluation", body)


def _section_monitor(fmt: str, monitor: Any, *, sink: _FigureSink) -> str:
    report = monitor.report()

    def draw() -> Any:
        from .plots import plot_e_process

        return plot_e_process(report)

    summary = _kv(
        fmt,
        [
            ("recommendation", report.recommendation),
            ("alarm_at", report.alarm_at if report.alarm_at is not None else "none"),
            ("onset_label", report.onset_label if report.onset_label is not None else "n/a"),
        ],
    )
    body = sink.figure(draw, "monitor_trajectory") + summary + _bullets(fmt, report.reasoning)
    return _section(fmt, "Calibration monitoring", body)


def _section_appendix(fmt: str, calibrator: Any) -> str:
    messages = calibrator.interpret().messages
    json_text = calibrator.to_json()
    if fmt == "html":
        body = (
            f"<details><summary>Fitted state (JSON)</summary>"
            f"<pre>{_escape_html(json_text)}</pre></details>" + _bullets(fmt, messages)
        )
    else:
        body = "```json\n" + json_text + "\n```\n\n" + _bullets(fmt, messages)
    return _section(fmt, "Appendix: calibrator", body)


# --------------------------------------------------------------------- public API


def validation_report(
    y: object,
    p: object,
    *,
    calibrator: Any = None,
    monitor: Any = None,
    grades: object = None,
    by: object = None,
    title: "str | None" = None,
    path: "str | os.PathLike[str] | None" = None,
    format: str = "html",
    n_boot: int = 200,
    seed: int = 42,
) -> str:
    """Self-contained validation report: reliability, metrics, grades, monitoring.

    One document — HTML with base64-embedded PNG figures, or markdown with a
    sibling directory of PNGs — built entirely from the existing public API
    (``curves``, ``metrics``, ``plots``, ``monitor``): nothing here computes
    a new statistic. Sections are omitted, not left blank, when their input
    is absent: reliability, the metric report, and the CORP decomposition
    always render (they need only ``y``/``p``); the rating-grade backtests
    render only when ``grades`` is given, the grouped-evaluation panel only
    when ``by`` is given, the monitoring trajectory only when ``monitor`` is
    given, and the calibrator appendix only when ``calibrator`` is given.

    Parameters
    ----------
    y, p : array_like
        Outcomes and predicted probabilities.
    calibrator : BaseCalibrator or None, keyword-only
        A fitted calibrator; adds its fingerprint to the header and an
        appendix with its serialized state (:meth:`BaseCalibrator.to_json`)
        and :meth:`BaseCalibrator.interpret` messages.
    monitor : CalibrationMonitor or None, keyword-only
        Adds its fingerprint to the header and a monitoring section with the
        e-process trajectory (:func:`probcal.plots.plot_e_process`) and its
        report's ``recommendation``/``reasoning``/``onset_label``.
    grades : array_like or None, keyword-only
        Rating grade label per observation. Adds a rating-grades section:
        the Jeffreys per-grade backtest and chart, the Pluto-Tasche
        most-prudent PD table, and the Jeffreys upper masterscale bands —
        all three ordered best to worst by mean predicted probability
        (ascending), stated explicitly in the section.
    by : array_like or None, keyword-only
        Group labels, one per observation. Adds a grouped-evaluation
        section: the faceted reliability panel
        (:func:`probcal.plots.plot_reliability` with ``by=``) plus the
        pooled-and-per-group metric table.
    title : str or None, keyword-only
        Document title; ``None`` uses ``"probcal validation report"``.
    path : path-like or None, keyword-only
        When given, the rendered document is written here in addition to
        being returned. Required when ``format="markdown"`` (figures are
        written to ``<path stem>_figures/`` next to it).
    format : {"html", "markdown"}, keyword-only
        Output format. ``"html"`` embeds every figure as a base64 PNG data
        URI; ``"markdown"`` writes GFM tables and references PNG files
        written alongside ``path``.
    n_boot : int, keyword-only
        Bootstrap/resample count shared by every resampling site in the
        report (``metrics.evaluate``, ``curves.corp_reliability``,
        ``curves.reliability_smooth``) — one knob to keep the whole report
        fast (small ``n_boot``) or tighter (large ``n_boot``).
    seed : int, keyword-only
        RNG seed shared by the same resampling sites; the report is
        bit-reproducible given the same inputs and ``seed`` apart from its
        one ``Generated ... UTC`` timestamp line.

    Returns
    -------
    str
        The rendered document text (also written to ``path`` when given).

    Raises
    ------
    ValueError
        If ``format`` is not ``"html"`` or ``"markdown"``, or if
        ``format="markdown"`` is given without ``path``.
    ImportError
        If matplotlib is not installed (every section renders at least one
        figure); names the ``probcal[viz]`` extra.

    Examples
    --------
    >>> from probcal import make_pd_portfolio
    >>> from probcal.report import validation_report
    >>> d = make_pd_portfolio(n=500, random_state=0)
    >>> html = validation_report(d.y, d.scores, n_boot=20)  # doctest: +SKIP
    >>> "Generated" in html  # doctest: +SKIP
    True
    """
    if format not in ("html", "markdown"):
        raise ValueError(f'format must be "html" or "markdown", got {format!r}')
    if format == "markdown" and path is None:
        raise ValueError('format="markdown" requires path (figures are written next to it)')

    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    sink = _FigureSink(format, path)

    page_title = title if title is not None else "probcal validation report"
    if format == "html":
        page_title = _escape_html(page_title)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    timestamp = f"Generated {ts} UTC"

    fp_pairs: list[tuple[str, object]] = [("data", data_fingerprint(y_arr, p_arr))]
    if calibrator is not None:
        fp_pairs.append(("calibrator", calibrator.fingerprint()))
    if monitor is not None:
        fp_pairs.append(("monitor", monitor.fingerprint()))
    fingerprints = _kv(format, fp_pairs)

    corp = corp_reliability(y_arr, p_arr, n_resamples=n_boot, random_state=seed)
    sections = [
        _section_header(format, y_arr, p_arr),
        _section_reliability(format, y_arr, p_arr, corp, n_boot=n_boot, seed=seed, sink=sink),
        _section_evaluate(format, y_arr, p_arr, n_boot=n_boot, seed=seed),
        _section_corp_decomposition(format, corp),
    ]
    if grades is not None:
        sections.append(_section_grades(format, y_arr, p_arr, grades, sink=sink))
    if by is not None:
        sections.append(
            _section_groups(format, y_arr, p_arr, by, n_boot=n_boot, seed=seed, sink=sink)
        )
    if monitor is not None:
        sections.append(_section_monitor(format, monitor, sink=sink))
    if calibrator is not None:
        sections.append(_section_appendix(format, calibrator))

    template = _HTML_TEMPLATE if format == "html" else _MD_TEMPLATE
    text = template.substitute(
        title=page_title,
        version=__version__,
        timestamp=timestamp,
        fingerprints=fingerprints,
        sections="".join(sections),
    )

    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text
