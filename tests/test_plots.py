"""Tests for probcal.plots ([viz]-guarded)."""

import importlib.util

import numpy as np
import pytest

from probcal._math import expit

HAS_MPL = importlib.util.find_spec("matplotlib") is not None

RNG = np.random.default_rng(89)


def _calibrated(n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


@pytest.mark.skipif(HAS_MPL, reason="matplotlib installed; guard path unreachable")
def test_missing_matplotlib_raises_helpful_error() -> None:
    from probcal.curves import reliability_binned
    from probcal.plots import plot_reliability

    y, p = _calibrated(300)
    curve = reliability_binned(y, p)
    with pytest.raises(ImportError, match=r"probcal\[viz\]"):
        plot_reliability(curve)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_reliability_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from probcal.curves import reliability_binned, reliability_loess
    from probcal.plots import plot_reliability

    y, p = _calibrated()
    ax = plot_reliability(reliability_binned(y, p), smooth=reliability_loess(y, p), scale="logit")
    assert ax is not None


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_belt_and_comparison_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from probcal.curves import calibration_belt, reliability_binned
    from probcal.plots import plot_belt, plot_comparison

    y, p = _calibrated()
    assert plot_belt(calibration_belt(y, p)) is not None
    before = reliability_binned(y, p)
    after = reliability_binned(y, np.clip(p * 0.9, 1e-6, 1 - 1e-6))
    assert plot_comparison(before, after) is not None


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_interval_and_selection_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from probcal._results import SelectionReport
    from probcal.plots import plot_interval, plot_selection
    from probcal.vennabers import VennAbersCalibrator

    y, p = _calibrated(400)
    cal = VennAbersCalibrator().fit(p, y)
    grid = np.linspace(0.05, 0.95, 30)
    assert plot_interval(cal.predict_interval(grid), grid) is not None
    rep = SelectionReport(
        methods=("platt", "beta"),
        score_mean=np.array([0.4, 0.39]),
        score_sd=np.array([0.01, 0.02]),
        guardrails_ok=np.array([True, True]),
        chosen=np.array([False, True]),
        criterion="log_loss",
    )
    assert plot_selection(rep) is not None


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_reliability_annotation_and_rug() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.curves import reliability_binned
    from probcal.plots import plot_reliability

    y, p = _calibrated()
    curve = reliability_binned(y, p)

    # With y/p: stats box present, rug lines present.
    ax = plot_reliability(curve, y=y, p=p)
    assert any("slope" in t.get_text() for t in ax.texts)
    rug_lines = [ln for ln in ax.lines if ln.get_marker() == "|"]
    assert len(rug_lines) == 2
    # Deterministic rug: a second identical call yields identical marker data.
    ax2 = plot_reliability(curve, y=y, p=p)
    rug_lines2 = [ln for ln in ax2.lines if ln.get_marker() == "|"]
    for a, b in zip(rug_lines, rug_lines2, strict=True):
        np.testing.assert_array_equal(a.get_xdata(), b.get_xdata())

    # Without y/p: no box, no rug, no error; count margin absent by default.
    ax3 = plot_reliability(curve)
    assert not any("slope" in t.get_text() for t in ax3.texts)
    assert not [ln for ln in ax3.lines if ln.get_marker() == "|"]
    assert len(ax3.figure.axes) == 1
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_reliability_counts_and_partial_args() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal import make_pd_portfolio
    from probcal.curves import reliability_binned
    from probcal.plots import plot_reliability

    y, p = _calibrated()
    curve = reliability_binned(y, p)
    # counts=True restores the twin-axis bar margin.
    ax = plot_reliability(curve, counts=True)
    assert len(ax.figure.axes) == 2
    # The 0.1.0 logit-scale fix stays intact: a zero-event bin renders without error.
    port = make_pd_portfolio(n=6000, random_state=42)
    curve0 = reliability_binned(port.y, port.scores)
    assert float(curve0.event_rate[0]) == 0.0
    ax2 = plot_reliability(curve0, scale="logit", y=port.y, p=port.scores, counts=True)
    assert ax2 is not None
    with pytest.raises(ValueError, match="together"):
        plot_reliability(curve, y=y)
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_ecce_smoke_and_band() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.curves import ecce_curve
    from probcal.plots import plot_ecce

    y, p = _calibrated()
    c1 = ecce_curve(y, p)
    c2 = ecce_curve(y, np.clip(p * 0.8, 1e-6, 1 - 1e-6))
    ax = plot_ecce(c1)
    assert ax is not None
    legend_texts = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "pointwise" in legend_texts
    assert "max drift" in legend_texts
    assert len(ax.collections) >= 1  # the band
    ax2 = plot_ecce([c1, c2], labels=["raw", "shrunk"], show_band=False)
    assert len(ax2.collections) == 0
    legend_texts2 = " ".join(t.get_text() for t in ax2.get_legend().get_texts())
    assert "pointwise" not in legend_texts2
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_grade_backtest_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.metrics import binomial_grade_test, jeffreys_grade_test
    from probcal.plots import plot_grade_backtest

    y, p = _calibrated(1200)
    grades = np.array(["G1", "G2", "G3"])[np.searchsorted([0.1, 0.4], p)]
    for res in (jeffreys_grade_test(y, p, grades), binomial_grade_test(y, p, grades)):
        ax = plot_grade_backtest(res)
        # The observed-rate scatter carries one point per grade.
        sizes = [len(c.get_offsets()) for c in ax.collections]
        assert len(res.grades) in sizes
        assert ax.get_yscale() == "log"
        # Annotations sit inside the final y-limits (headroom applied).
        lo, hi = ax.get_ylim()
        for ann in [a for a in ax.texts if hasattr(a, "xy")]:
            assert lo <= ann.xy[1] <= hi
        plt.close("all")
    ax = plot_grade_backtest(jeffreys_grade_test(y, p, grades), log_scale=False)
    assert ax.get_yscale() == "linear"
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_offset_audit_smoke() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.offset import LogitOffset
    from probcal.plots import plot_offset_audit

    _, p = _calibrated(2000)
    for off in (LogitOffset(delta=-0.3).fit(p), LogitOffset(target_mean=0.2).fit(p)):
        ax = plot_offset_audit(off)
        assert any("odds factor" in t.get_text() for t in ax.texts)
        plt.close("all")
    with pytest.raises(RuntimeError, match="not fitted"):
        plot_offset_audit(LogitOffset(delta=0.1))


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plots_do_not_mutate_global_rcparams() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.curves import ecce_curve, reliability_binned
    from probcal.plots import plot_ecce, plot_reliability

    y, p = _calibrated(600)
    before = dict(plt.rcParams)
    plot_reliability(reliability_binned(y, p), y=y, p=p)
    plot_ecce(ecce_curve(y, p))
    assert dict(plt.rcParams) == before
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_e_process_smoke() -> None:
    from probcal import make_pd_portfolio
    from probcal._math import expit, logit
    from probcal.monitor import CalibrationMonitor
    from probcal.plots import plot_e_process

    mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 21))
    rng = np.random.default_rng(4)
    for k in range(4):
        d = make_pd_portfolio(n=400, random_state=700 + k)
        y = (rng.random(400) < expit(logit(d.scores) + 0.9)).astype(float)
        mon.update(y, d.scores, grade=np.array(["A", "B"] * 200), label=f"m{k}")
    ax = plot_e_process(mon.report())
    assert ax.get_yscale() == "log"
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [f"m{k}" for k in range(4)]
