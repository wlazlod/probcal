"""Default-output pixel regression for probcal.plots.

Hashes the Agg buffer of every public plot called with 0.2.0 defaults. The
expected hashes are computed on the 0.2.0 code path and must not change in
0.3.0 unless a new option is passed. Regenerate ONLY on a deliberate
default change recorded in CHANGELOG.
"""

import hashlib
import importlib.util

import numpy as np
import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")

# Version pinned in uv.lock at the time these hashes were recorded. Agg output
# can differ across matplotlib/freetype versions, so the pinned-hash equality
# checks below are skipped (not failed) off this version; the determinism
# check (render twice, compare) always runs regardless of version.
PINNED_MPL = "3.11.1"


def _render_hash(artist) -> str:
    import matplotlib.pyplot as plt

    fig = artist if hasattr(artist, "savefig") else artist.figure
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).tobytes()
    plt.close(fig)
    return hashlib.sha256(buf).hexdigest()


def _assert_pinned(h: str, key: str) -> None:
    import matplotlib

    if matplotlib.__version__ != PINNED_MPL:
        reason = f"hash baseline pinned to matplotlib=={PINNED_MPL}, found {matplotlib.__version__}"
        pytest.skip(reason)
    assert h == EXPECTED[key]


def _data():
    from probcal import make_pd_portfolio

    d = make_pd_portfolio(n=3000, random_state=11)
    return d.y, d.scores


def test_plot_reliability_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.curves import reliability_binned, reliability_loess
    from probcal.plots import plot_reliability

    y, p = _data()

    def _artist():
        return plot_reliability(reliability_binned(y, p), smooth=reliability_loess(y, p), y=y, p=p)

    h1 = _render_hash(_artist())
    h2 = _render_hash(_artist())
    assert h1 == h2  # deterministic on this host
    _assert_pinned(h1, "plot_reliability")


def test_plot_belt_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.curves import calibration_belt
    from probcal.plots import plot_belt

    y, p = _data()
    h1 = _render_hash(plot_belt(calibration_belt(y, p)))
    h2 = _render_hash(plot_belt(calibration_belt(y, p)))
    assert h1 == h2
    _assert_pinned(h1, "plot_belt")


def test_plot_selection_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.plots import plot_selection
    from probcal.selection import CalibratorSelector

    y, p = _data()
    report = CalibratorSelector(cv=3).fit(p[:2000], y[:2000]).report_
    h1 = _render_hash(plot_selection(report))
    h2 = _render_hash(plot_selection(report))
    assert h1 == h2
    _assert_pinned(h1, "plot_selection")


def test_plot_ecce_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.curves import ecce_curve
    from probcal.plots import plot_ecce

    y, p = _data()
    h1 = _render_hash(plot_ecce([ecce_curve(y, p)]))
    h2 = _render_hash(plot_ecce([ecce_curve(y, p)]))
    assert h1 == h2
    _assert_pinned(h1, "plot_ecce")


def test_plot_grade_backtest_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.metrics import jeffreys_grade_test
    from probcal.plots import plot_grade_backtest

    y, p = _data()
    grades = np.array(["G1", "G2", "G3"])[np.searchsorted([0.01, 0.05], p)]
    h1 = _render_hash(plot_grade_backtest(jeffreys_grade_test(y, p, grades)))
    h2 = _render_hash(plot_grade_backtest(jeffreys_grade_test(y, p, grades)))
    assert h1 == h2
    _assert_pinned(h1, "plot_grade_backtest")


def test_plot_offset_audit_default_is_stable(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from datetime import UTC, datetime

    import probcal.offset as _offset_module
    from probcal.offset import LogitOffset
    from probcal.plots import plot_offset_audit

    # plot_offset_audit prints offset.timestamp_ (wall-clock fit time) on the chart;
    # freeze probcal.offset's clock so the rendered hash is reproducible.
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(_offset_module, "datetime", _FrozenDatetime)

    y, p = _data()

    def _fit():
        return LogitOffset(target_mean=float(y.mean())).fit(p)

    h1 = _render_hash(plot_offset_audit(_fit()))
    h2 = _render_hash(plot_offset_audit(_fit()))
    assert h1 == h2
    _assert_pinned(h1, "plot_offset_audit")


def test_plot_e_process_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.monitor import CalibrationMonitor
    from probcal.plots import plot_e_process

    y, p = _data()
    y_batches = np.array_split(y, 3)
    p_batches = np.array_split(p, 3)

    def _report():
        mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 41))
        for i, (yb, pb) in enumerate(zip(y_batches, p_batches, strict=True)):
            mon.update(yb, pb, label=f"batch{i}")
        return mon.report()

    h1 = _render_hash(plot_e_process(_report()))
    h2 = _render_hash(plot_e_process(_report()))
    assert h1 == h2
    _assert_pinned(h1, "plot_e_process")


def test_plot_comparison_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.curves import reliability_binned
    from probcal.plots import plot_comparison

    y, p = _data()
    before = reliability_binned(y, p)
    after = reliability_binned(y, np.clip(p * 0.9, 1e-6, 1 - 1e-6))
    h1 = _render_hash(plot_comparison(before, after))
    h2 = _render_hash(plot_comparison(before, after))
    assert h1 == h2
    _assert_pinned(h1, "plot_comparison")


def test_plot_interval_default_is_stable():
    import matplotlib

    matplotlib.use("Agg")
    from probcal.plots import plot_interval
    from probcal.vennabers import VennAbersCalibrator

    y, p = _data()
    grid = np.linspace(0.005, 0.5, 60)
    cal = VennAbersCalibrator().fit(p[:1500], y[:1500])
    h1 = _render_hash(plot_interval(cal.predict_interval(grid), grid))
    h2 = _render_hash(plot_interval(cal.predict_interval(grid), grid))
    assert h1 == h2
    _assert_pinned(h1, "plot_interval")


EXPECTED = {
    "plot_reliability": "f90d3dd4f83717e8f009691d0b519f7b478735329b304b9bf95e5b2865642cc8",
    "plot_belt": "0463cf5dece134f654d3860e9a655c3442c132681110c981c96a4a74c8e5ae76",
    "plot_selection": "cd7deda1e739a0511b26ce8b510a2bcb0f620a2a9680f6e5995bb3c34e794493",
    "plot_ecce": "f399114a8499dc7fbbc0b8af70d4069e04feeb0f338e20c5cb9146cd6f5ed0d1",
    "plot_grade_backtest": "3f32278c362688bc5c47288cb5d7e75c0f3d64dd785109f6d1456d588d45d26b",
    "plot_offset_audit": "866a7f28f5211a578c6f7dc999a60f5cde5de92557292b298567510e83faff78",
    "plot_e_process": "8e0d67cfe0fef1aba1a914da8e64aa60f28ca4b0f56857c578e9743b047502ce",
    "plot_comparison": "e149955a561acac0d46adc34b1648063633642b3368b102337fdffeac62193bf",
    "plot_interval": "4086410f26af83c2a49b1d4a5dc01fea1bfaa83685ce8ddbc489f2b48b5068ce",
}
