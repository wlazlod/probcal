"""Behavioural tests for probcal.plots ([viz]-guarded).

Default output of every public plot is pinned structurally by
test_plots_regression.py; this module covers options, faceting, bug fixes and
argument validation.
"""

import dataclasses

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from probcal import calibration_belt, make_pd_portfolio  # noqa: E402
from probcal._math import expit, logit  # noqa: E402
from probcal.curves import (  # noqa: E402
    corp_reliability,
    ecce_curve,
    reliability_binned,
    reliability_loess,
    reliability_smooth,
)
from probcal.metrics import binomial_grade_test, evaluate, jeffreys_grade_test  # noqa: E402
from probcal.plots import (  # noqa: E402
    plot_attributes,
    plot_belt,
    plot_comparison,
    plot_corp,
    plot_ecce,
    plot_grade_backtest,
    plot_mcb_dsc,
    plot_reliability,
)

RNG = np.random.default_rng(89)


def _calibrated(n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


@pytest.fixture(scope="module")
def portfolio():
    return make_pd_portfolio(n=3000, random_state=0)


def _rug(ax) -> list:
    return [ln for ln in ax.lines if ln.get_marker() == "|"]


def _box(ax) -> str:
    return ax.texts[0].get_text()


# ---------------------------------------------------------------- plot_reliability options


def test_plot_reliability_annotation_and_rug() -> None:
    y, p = _calibrated()
    curve = reliability_binned(y, p)

    # With y/p: stats box present, rug lines present.
    ax = plot_reliability(curve, y=y, p=p)
    assert any("slope" in t.get_text() for t in ax.texts)
    rug_lines = _rug(ax)
    assert len(rug_lines) == 2
    # Deterministic rug: a second identical call yields identical marker data.
    for a, b in zip(rug_lines, _rug(plot_reliability(curve, y=y, p=p)), strict=True):
        np.testing.assert_array_equal(a.get_xdata(), b.get_xdata())

    # Without y/p: no box, no rug, no error; count margin absent by default.
    ax3 = plot_reliability(curve)
    assert not any("slope" in t.get_text() for t in ax3.texts)
    assert not _rug(ax3)
    assert len(ax3.figure.axes) == 1


def test_plot_reliability_counts_margin() -> None:
    y, p = _calibrated()
    ax = plot_reliability(reliability_binned(y, p), counts=True)
    assert len(ax.figure.axes) == 2  # counts=True adds the twin-axis bar margin
    # The 0.1.0 logit-scale fix stays intact: a zero-event bin renders without error.
    port = make_pd_portfolio(n=6000, random_state=42)
    curve0 = reliability_binned(port.y, port.scores)
    assert float(curve0.event_rate[0]) == 0.0
    plot_reliability(curve0, scale="logit", y=port.y, p=port.scores, counts=True)
    # A single bin still gets a positive-width bar, and the twin axis has no grid.
    ax = plot_reliability(reliability_binned(y, p, n_bins=1), counts=True)
    twin = ax.figure.axes[1]
    assert twin.patches and twin.patches[0].get_width() > 0
    assert not any(gl.get_visible() for gl in twin.yaxis.get_gridlines())


def test_plot_reliability_stats_box() -> None:
    from probcal._results import MetricReport

    y, p = _calibrated()
    curve = reliability_binned(y, p)
    box_text = "\n".join(t.get_text() for t in plot_reliability(curve, y=y, p=p, stats=True).texts)
    for label in ("n", "events", "intercept", "slope", "ICI", "smECE", "Brier"):
        assert f"{label} =" in box_text
    # stats replaces the default annotate box, not both.
    assert "Spiegelhalter" not in box_text
    assert "E90" not in box_text

    report = MetricReport(
        names=("intercept", "slope", "log_loss"),
        values=np.array([0.01, 0.98, 0.42]),
        ci_low=np.array([-0.05, 0.9, 0.4]),
        ci_high=np.array([0.07, 1.05, 0.44]),
    )
    box_text = "\n".join(
        t.get_text() for t in plot_reliability(curve, y=y, p=p, stats=report).texts
    )
    for label in ("intercept", "slope", "n", "events"):
        assert f"{label} = " in box_text
    assert "log_loss" not in box_text  # not in the stats-box name set


def test_plot_reliability_risk_dist_split() -> None:
    y, p = _calibrated()
    curve = reliability_binned(y, p)
    ax = plot_reliability(curve, y=y, p=p, risk_dist="split")
    heights = [patch.get_height() for patch in ax.patches]
    assert any(h > 0 for h in heights)
    assert any(h < 0 for h in heights)
    assert not _rug(ax)
    off = plot_reliability(curve, y=y, p=p, rug=False, risk_dist="split")  # rug=False wins
    assert not off.patches
    assert not _rug(off)


def test_plot_reliability_kernel_smooth_renders_variable_width_curve() -> None:
    from matplotlib.collections import LineCollection

    y, p = _calibrated(800)
    curve = reliability_binned(y, p)
    smooth = reliability_smooth(y, p, grid_size=60, n_boot=0)

    for scale in ("probability", "logit"):
        ax = plot_reliability(curve, smooth=smooth, scale=scale)
        # `errorbar` also emits a (constant-width) LineCollection for its bars;
        # the kernel curve's is the one with more than one distinct width.
        variable_width = [
            c
            for c in ax.collections
            if isinstance(c, LineCollection) and len(set(c.get_linewidths())) > 1
        ]
        assert len(variable_width) == 1
        # Miscalibration-area shading and the CI ribbon are both fill_betweens
        # (PolyCollections), in addition to the two LineCollections above.
        assert len(ax.collections) >= 4
        assert any("smECE" in t.get_text() for t in ax.texts)


def test_kernel_curve_ci_at_bounds_and_empty_logit_keep() -> None:
    y, p = _calibrated(800)
    curve = reliability_binned(y, p)
    smooth = reliability_smooth(y, p, grid_size=40, n_boot=0)
    touching = dataclasses.replace(
        smooth, ci_low=np.zeros_like(smooth.ci_low), ci_high=np.ones_like(smooth.ci_high)
    )
    ax = plot_reliability(curve, smooth=touching, scale="logit")
    for coll in ax.collections:
        for path in coll.get_paths():
            assert np.all(np.isfinite(path.vertices))
    degenerate = dataclasses.replace(smooth, event_rate=np.zeros_like(smooth.event_rate))
    ax = plot_reliability(curve, smooth=degenerate, scale="logit")
    assert any("smECE" in t.get_text() for t in ax.texts)


def test_plot_reliability_logit_with_no_finite_bin() -> None:
    _, p = _calibrated(300)
    # Perfectly separated: every bin's event rate is 0 or 1, nothing finite on logit.
    y = (p > np.median(p)).astype(float)
    curve = reliability_binned(y, p)
    assert np.all((curve.event_rate == 0.0) | (curve.event_rate == 1.0))
    ax = plot_reliability(curve, scale="logit")
    identity = next(ln for ln in ax.lines if ln.get_label() == "identity")
    assert np.all(np.isfinite(identity.get_xdata()))


# ---------------------------------------------------------------- faceting (by=)


def test_plot_reliability_by_forwards_panel_options() -> None:
    y, p = _calibrated(900)
    by = np.where(p < 0.3, "low", "high")
    curve = reliability_binned(y, p)
    fig = plot_reliability(curve, y=y, p=p, by=by, stats=True, counts=True)
    panels = [ax for ax in fig.axes if ax.get_title()]
    assert len(panels) == 3
    for ax in panels:
        assert any("Brier" in t.get_text() for t in ax.texts)
        assert _rug(ax)  # rug forwarded
    assert len(fig.axes) == 6  # one twin count axis per panel
    light = plot_reliability(curve, y=y, p=p, by=by, annotate=False, rug=False)
    for ax in light.axes:
        assert not ax.texts
        assert not _rug(ax)


def test_faceted_stats_are_computed_per_group(portfolio) -> None:
    y, p = portfolio.y, portfolio.scores
    seg = np.where(p < 0.05, "lo", "hi")
    fig = plot_reliability(None, y=y, p=p, by=seg, stats=True)
    assert len({_box(ax) for ax in fig.axes}) == 3

    grouped = evaluate(y, p, n_boot=0, by=seg)
    fig = plot_reliability(None, y=y, p=p, by=seg, stats=grouped)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["pooled", "hi", "lo"]
    reports = dict(zip(titles, (grouped.pooled, *grouped.reports), strict=True))
    for ax in fig.axes:
        rep = reports[ax.get_title()]
        slope = rep.values[rep.names.index("slope")]
        assert f"slope = {slope:.3f}" in _box(ax)


def test_faceted_groups_by_raw_value(portfolio) -> None:
    by = np.arange(len(portfolio.y)) % 3
    fig = plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=by, annotate=False)
    # four panels on a wider grid: the unused slots are removed
    assert [ax.get_title() for ax in fig.axes] == ["pooled", "0", "1", "2"]
    alike = np.empty(len(portfolio.y), dtype=object)
    alike[:] = [1 if i % 2 else "1" for i in range(len(portfolio.y))]
    with pytest.raises(ValueError, match="both display as"):
        evaluate(portfolio.y, portfolio.scores, n_boot=0, by=alike)


# ---------------------------------------------------------------- other plots


def test_plot_comparison_legend_matches_panel_colors() -> None:
    from matplotlib.colors import same_color

    from probcal._plots_common import _GREEN, _RED

    y, p = _calibrated()
    fig = plot_comparison(reliability_binned(y, p), reliability_binned(y, p * 0.9))
    for ax, color in zip(fig.axes, (_RED, _GREEN), strict=True):
        legend = ax.get_legend()
        handles = dict(
            zip([t.get_text() for t in legend.get_texts()], legend.legend_handles, strict=True)
        )
        assert same_color(np.atleast_2d(handles["binned"].get_color())[0], color)


def test_plot_ecce_band_and_multiple_curves() -> None:
    y, p = _calibrated()
    c1 = ecce_curve(y, p)
    c2 = ecce_curve(y, np.clip(p * 0.8, 1e-6, 1 - 1e-6))
    ax = plot_ecce(c1)
    legend_texts = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "pointwise" in legend_texts
    assert "max drift" in legend_texts
    assert len(ax.collections) >= 1  # the band
    ax2 = plot_ecce([c1, c2], labels=["raw", "shrunk"], show_band=False)
    assert len(ax2.collections) == 0
    assert "pointwise" not in " ".join(t.get_text() for t in ax2.get_legend().get_texts())


def test_plot_grade_backtest() -> None:
    y, p = _calibrated(1200)
    grades = np.array(["G1", "G2", "G3"])[np.searchsorted([0.1, 0.4], p)]
    for res, name in (
        (jeffreys_grade_test(y, p, grades), "Jeffreys"),
        (binomial_grade_test(y, p, grades), "exact binomial"),
    ):
        ax = plot_grade_backtest(res)
        assert name in ax.get_title()
        # The observed-rate scatter carries one point per grade.
        assert len(res.grades) in [len(c.get_offsets()) for c in ax.collections]
        assert ax.get_yscale() == "log"
        # Annotations sit inside the final y-limits (headroom applied).
        lo, hi = ax.get_ylim()
        for ann in [a for a in ax.texts if hasattr(a, "xy")]:
            assert lo <= ann.xy[1] <= hi
    ax = plot_grade_backtest(jeffreys_grade_test(y, p, grades), log_scale=False)
    assert ax.get_yscale() == "linear"


def test_plot_offset_audit_both_modes() -> None:
    from probcal.offset import LogitOffset
    from probcal.plots import plot_offset_audit

    _, p = _calibrated(2000)
    for off in (LogitOffset(delta=-0.3).fit(p), LogitOffset(target_mean=0.2).fit(p)):
        ax = plot_offset_audit(off)
        assert any("odds factor" in t.get_text() for t in ax.texts)
    with pytest.raises(RuntimeError, match="not fitted"):
        plot_offset_audit(LogitOffset(delta=0.1))


def test_plot_interval_sorts_by_score() -> None:
    from probcal.plots import plot_interval
    from probcal.vennabers import VennAbersCalibrator

    y, p = _calibrated(400)
    cal = VennAbersCalibrator().fit(p, y)
    grid = np.linspace(0.05, 0.95, 30)
    shuffled = np.random.default_rng(0).permutation(grid)
    ref = plot_interval(cal.predict_interval(grid), grid)
    got = plot_interval(cal.predict_interval(shuffled), shuffled)
    np.testing.assert_allclose(got.lines[0].get_xydata(), ref.lines[0].get_xydata())
    np.testing.assert_allclose(
        got.collections[0].get_paths()[0].vertices, ref.collections[0].get_paths()[0].vertices
    )


def test_plot_e_process_grades_panel_stays_inside_figure_and_ax() -> None:
    from probcal.monitor import CalibrationMonitor
    from probcal.plots import plot_e_process

    mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 21), grades=("A", "B"))
    rng = np.random.default_rng(4)
    for k in range(3):
        d = make_pd_portfolio(n=400, random_state=700 + k)
        y = (rng.random(400) < expit(logit(d.scores) + 0.9)).astype(float)
        mon.update(y, d.scores, grade=np.array(["A", "B"] * 200), label=f"m{k}")
    report = mon.report()

    ax = plot_e_process(report, grades_panel=True)
    ax.figure.canvas.draw()
    assert len(ax.figure.axes) == 2
    for a in ax.figure.axes:
        x0, y0, w, h = a.get_position().bounds
        assert x0 >= 0 and y0 >= 0 and x0 + w <= 1 and y0 + h <= 1

    # A caller-supplied axes: the panel is carved out of that axes' own slot.
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))
    slot = right.get_position()
    plot_e_process(report, grades_panel=True, ax=right)
    fig.canvas.draw()
    new = [a for a in fig.axes if a not in (left, right)]
    assert len(new) == 1
    pos = new[0].get_position()
    assert pos.x0 >= slot.x0 - 1e-9 and pos.x1 <= slot.x1 + 1e-9
    assert pos.y0 >= slot.y0 - 1e-9 and pos.y1 <= slot.y1 + 1e-9
    assert left.get_position().bounds == pytest.approx(
        plt.subplots(1, 2, figsize=(12, 6))[1][0].get_position().bounds
    )


# ---------------------------------------------------------------- figure hygiene


def test_user_supplied_ax_gets_house_style() -> None:
    y, p = _calibrated(600)
    with plt.rc_context({"axes.grid": False, "axes.spines.top": True}):
        _, ax = plt.subplots()
        _, ax_e = plt.subplots()
        _, ax_c = plt.subplots()
    own = plot_reliability(reliability_binned(y, p))
    plot_reliability(reliability_binned(y, p), ax=ax)
    plot_ecce(ecce_curve(y, p), ax=ax_e)
    plot_corp(corp_reliability(y, p, n_resamples=10), scale="logit", ax=ax_c)
    assert ax_c.get_xlabel().endswith("(logit scale)")
    for a in (own, ax, ax_e, ax_c):
        assert not a.spines["top"].get_visible()
        assert not a.spines["right"].get_visible()
        assert all(gl.get_visible() for gl in a.xaxis.get_gridlines())
    assert ax.xaxis.get_gridlines()[0].get_alpha() == own.xaxis.get_gridlines()[0].get_alpha()


def test_plots_do_not_mutate_global_rcparams() -> None:
    y, p = _calibrated(600)
    before = dict(plt.rcParams)
    plot_reliability(reliability_binned(y, p), y=y, p=p)
    plot_ecce(ecce_curve(y, p))
    assert dict(plt.rcParams) == before


def test_late_error_closes_only_the_figure_it_created(portfolio) -> None:
    curve = reliability_binned(portfolio.y, portfolio.scores)
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_reliability(curve, y=portfolio.y * 2, p=portfolio.scores, stats=True)
    assert plt.get_fignums() == before
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_reliability(curve, y=portfolio.y * 2, p=portfolio.scores, stats=True, ax=ax)
    assert plt.fignum_exists(fig.number)


def _validation_cases():
    port = make_pd_portfolio(n=3000, random_state=0)
    y, p = port.y, port.scores
    curve = reliability_binned(y, p)
    seg = np.where(p < 0.05, "lo", "hi")
    other = np.where(p < 0.05, "a", "b")
    alike = np.empty(len(y), dtype=object)
    alike[:] = [1 if i % 2 else "1" for i in range(len(y))]
    y_b = make_pd_portfolio(n=3000, random_state=19).y  # a different weighted mean y
    return [
        ("risk_dist", lambda: plot_reliability(curve, risk_dist="bogus"), "risk_dist"),
        ("y_without_p", lambda: plot_reliability(curve, y=y), "together"),
        ("by_without_yp", lambda: plot_reliability(curve, by=seg), "by requires y and p"),
        (
            "by_length",
            lambda: plot_reliability(curve, y=y, p=p, by=np.array(["lo", "hi"])),
            "by must have the same length as y",
        ),
        (
            "by_smooth",
            lambda: plot_reliability(curve, y=y, p=p, by=seg, smooth=reliability_loess(y, p)),
            "smooth is not supported with by=",
        ),
        (
            "by_pooled_report",
            lambda: plot_reliability(None, y=y, p=p, by=seg, stats=evaluate(y, p, n_boot=0)),
            "GroupedMetricReport",
        ),
        (
            "by_report_groups",
            lambda: plot_reliability(
                None, y=y, p=p, by=seg, stats=evaluate(y, p, n_boot=0, by=other)
            ),
            "groups",
        ),
        (
            "grouped_report_without_by",
            lambda: plot_reliability(curve, y=y, p=p, stats=evaluate(y, p, n_boot=0, by=seg)),
            "by=",
        ),
        ("by_alike", lambda: plot_reliability(None, y=y, p=p, by=alike), "both display as"),
        ("scale", lambda: plot_reliability(curve, scale="logits"), "scale must be"),
        (
            "by_scale",
            lambda: plot_reliability(None, y=y, p=p, by=p < 0.05, scale="Logit"),
            "scale must be",
        ),
        ("belt_scale", lambda: plot_belt(calibration_belt(y, p), scale="log"), "scale must be"),
        ("cmp_scale", lambda: plot_comparison(curve, curve, scale="prob"), "scale must be"),
        (
            "corp_scale",
            lambda: plot_corp(corp_reliability(y, p, n_resamples=20), scale="logits"),
            "scale must be",
        ),
        ("attr_scale", lambda: plot_attributes(y, p, scale="logits"), "scale must be"),
        ("attr_method", lambda: plot_attributes(y, p, method="bogus"), "method"),
        (
            "mcb_dsc_mean",
            lambda: plot_mcb_dsc({"a": (y, p), "b": (y_b, p)}),
            "same weighted mean",
        ),
    ]


@pytest.mark.parametrize(
    ("call", "match"),
    [pytest.param(c, m, id=i) for i, c, m in _validation_cases()],
)
def test_plot_argument_validation_leaves_no_figure(call, match) -> None:
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=match):
        call()
    assert plt.get_fignums() == before


def test_by_rejects_a_caller_axes() -> None:
    y, p = _calibrated(300)
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="ax is not supported with by="):
        plot_reliability(reliability_binned(y, p), y=y, p=p, by=p < 0.3, ax=ax)
