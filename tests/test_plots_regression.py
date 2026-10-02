"""Default-output regression for probcal.plots.

Every public plot is called with its defaults and the resulting figure is reduced
to a structural snapshot — figure size, axes layout/limits/scales/ticks/labels,
and every line, collection, patch and text with its data and style — which is
compared against ``tests/baseline/plot_snapshots.json`` with a small numeric
tolerance. Unlike a hash of the Agg pixel buffer, the snapshot does not depend
on the platform's font rasterisation or the last ulp of a BLAS call, yet it
still catches any change to what a default call draws.

Each test also renders the figure twice and compares the pixel buffers on this
host (determinism; host-local, so platform-independent).

Regenerate the baseline ONLY on a deliberate default change recorded in
CHANGELOG::

    PROBCAL_REGEN_PLOT_BASELINE=1 pytest tests/test_plots_regression.py
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import re
from typing import Any

import numpy as np
import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")

_BASELINE = pathlib.Path(__file__).parent / "baseline" / "plot_snapshots.json"
_REGEN = os.environ.get("PROBCAL_REGEN_PLOT_BASELINE") == "1"
_RTOL, _ATOL = 1e-3, 1e-6
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------- snapshot


_FULL_MAX = 48  # longer arrays are stored as a fixed-index sample plus moments


def _round(v: float) -> float | None:
    return float(f"{v:.6g}") if np.isfinite(v) else None


def _num(values: Any) -> Any:
    arr = np.ravel(np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan))
    if len(arr) <= _FULL_MAX:
        return [_round(v) for v in arr]
    finite = arr[np.isfinite(arr)]
    idx = np.linspace(0, len(arr) - 1, _FULL_MAX // 2).round().astype(int)
    return {
        "n": len(arr),
        "n_finite": len(finite),
        "sample": [_round(v) for v in arr[idx]],
        "mean": _round(float(finite.mean())) if len(finite) else None,
        "mean_abs": _round(float(np.abs(finite).mean())) if len(finite) else None,
    }


def _color(c: Any) -> Any:
    from matplotlib.colors import to_rgba_array

    if c is None or (isinstance(c, str) and c == "none"):
        return None
    arr = to_rgba_array(c)
    return _num(arr)


def _artist_common(a: Any) -> dict[str, Any]:
    return {"type": type(a).__name__, "visible": a.get_visible(), "alpha": a.get_alpha()}


def _line(ln: Any) -> dict[str, Any]:
    return {
        **_artist_common(ln),
        "label": ln.get_label() if not ln.get_label().startswith("_") else None,
        "xy": _num(ln.get_xydata()),
        "color": _color(ln.get_color()),
        "ls": str(ln.get_linestyle()),
        "lw": float(ln.get_linewidth()),
        "marker": str(ln.get_marker()),
        "ms": float(ln.get_markersize()),
    }


def _collection(c: Any) -> dict[str, Any]:
    paths = [_num(p.vertices) for p in c.get_paths()]
    return {
        **_artist_common(c),
        "label": c.get_label() if not c.get_label().startswith("_") else None,
        "paths": paths,
        "offsets": _num(c.get_offsets()),
        "face": _color(c.get_facecolor()),
        "edge": _color(c.get_edgecolor()),
        "lw": _num(c.get_linewidths()),
        "sizes": _num(c.get_sizes()) if hasattr(c, "get_sizes") else None,
    }


def _patch(p: Any) -> dict[str, Any]:
    return {
        **_artist_common(p),
        "verts": _num(p.get_verts()),
        "face": _color(p.get_facecolor()),
    }


def _text(t: Any) -> dict[str, Any]:
    return {
        "text": t.get_text(),
        "pos": _num(t.get_position()),
        "size": float(t.get_fontsize()),
        "color": _color(t.get_color()),
    }


def _handle_color(h: Any) -> Any:
    getter = getattr(h, "get_color", None) or getattr(h, "get_facecolor", None)
    return None if getter is None else _color(np.atleast_2d(getter())[0])


def _axes(ax: Any) -> dict[str, Any]:
    legend = ax.get_legend()
    return {
        "bounds": _num(ax.get_position().bounds),
        "xlim": _num(ax.get_xlim()),
        "ylim": _num(ax.get_ylim()),
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks": _num(ax.get_xticks()),
        "yticks": _num(ax.get_yticks()),
        "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        "spines": {k: s.get_visible() for k, s in sorted(ax.spines.items())},
        "facecolor": _color(ax.get_facecolor()),
        "lines": [_line(ln) for ln in ax.lines],
        "collections": [_collection(c) for c in ax.collections],
        "patches": [_patch(p) for p in ax.patches],
        "texts": [_text(t) for t in ax.texts],
        "legend": None if legend is None else [t.get_text() for t in legend.get_texts()],
        "legend_colors": None
        if legend is None
        else [_handle_color(h) for h in legend.legend_handles],
    }


def _snapshot(artist: Any) -> dict[str, Any]:
    fig = artist if hasattr(artist, "savefig") else artist.figure
    fig.canvas.draw()  # resolve autoscaled limits and ticks
    return {
        "figsize": _num(fig.get_size_inches()),
        "facecolor": _color(fig.get_facecolor()),
        "axes": [_axes(ax) for ax in fig.axes],
    }


def _assert_close(got: Any, want: Any, path: str = "$") -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        assert sorted(got) == sorted(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            _assert_close(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list), path
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_close(g, w, f"{path}[{i}]")
    elif isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        assert np.isclose(got, want, rtol=_RTOL, atol=_ATOL), f"{path}: {got} != {want}"
    elif isinstance(want, int) and not isinstance(want, bool):
        assert got == want, f"{path}: {got!r} != {want!r}"
    elif isinstance(want, str) and isinstance(got, str):
        # Numbers printed on the canvas (p-values, metrics) are compared with a
        # loose tolerance: they are formatted to 3-4 significant digits, and a
        # numerically-zero tail probability may print as 0 or 1e-17 by platform.
        got_parts, want_parts = _NUMBER.split(got), _NUMBER.split(want)
        assert got_parts == want_parts, f"{path}: {got!r} != {want!r}"
        for g, w in zip(_NUMBER.findall(got), _NUMBER.findall(want), strict=True):
            assert np.isclose(float(g), float(w), rtol=1e-2, atol=1e-6), (
                f"{path}: {got!r} != {want!r}"
            )
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _pixels(artist: Any) -> str:
    fig = artist if hasattr(artist, "savefig") else artist.figure
    fig.canvas.draw()
    return hashlib.sha256(np.asarray(fig.canvas.buffer_rgba()).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return {}


@pytest.fixture(scope="module", autouse=True)
def _write_baseline(baseline: dict[str, Any]):
    yield
    if _REGEN:
        _BASELINE.parent.mkdir(exist_ok=True)
        _BASELINE.write_text(json.dumps(baseline, sort_keys=True, separators=(",", ":")) + "\n")


def _check(baseline: dict[str, Any], key: str, make: Any) -> None:
    import matplotlib.pyplot as plt

    first = make()
    snap = _snapshot(first)
    h1 = _pixels(first)
    plt.close("all")
    h2 = _pixels(make())
    assert h1 == h2  # deterministic on this host
    snap = json.loads(json.dumps(snap))  # normalise tuples etc. as the baseline file would
    if _REGEN:
        baseline[key] = snap
        return
    assert key in baseline, f"no baseline for {key}; regenerate with PROBCAL_REGEN_PLOT_BASELINE=1"
    _assert_close(snap, baseline[key], key)


# ----------------------------------------------------------------- data


def _data():
    from probcal import make_pd_portfolio

    d = make_pd_portfolio(n=3000, random_state=11)
    return d.y, d.scores


# ----------------------------------------------------------------- tests


def test_plot_reliability_default_is_stable(baseline):
    from probcal.curves import reliability_binned, reliability_loess
    from probcal.plots import plot_reliability

    y, p = _data()
    _check(
        baseline,
        "plot_reliability",
        lambda: plot_reliability(
            reliability_binned(y, p), smooth=reliability_loess(y, p), y=y, p=p
        ),
    )


def test_plot_belt_default_is_stable(baseline):
    from probcal.curves import calibration_belt
    from probcal.plots import plot_belt

    y, p = _data()
    belt = calibration_belt(y, p)
    _check(baseline, "plot_belt", lambda: plot_belt(belt))


def test_plot_selection_default_is_stable(baseline):
    from probcal.plots import plot_selection
    from probcal.selection import CalibratorSelector

    y, p = _data()
    report = CalibratorSelector(cv=3).fit(p[:2000], y[:2000]).report_
    _check(baseline, "plot_selection", lambda: plot_selection(report))


def test_plot_ecce_default_is_stable(baseline):
    from probcal.curves import ecce_curve
    from probcal.plots import plot_ecce

    y, p = _data()
    _check(baseline, "plot_ecce", lambda: plot_ecce([ecce_curve(y, p)]))


def test_plot_grade_backtest_default_is_stable(baseline):
    from probcal.metrics import jeffreys_grade_test
    from probcal.plots import plot_grade_backtest

    y, p = _data()
    grades = np.array(["G1", "G2", "G3"])[np.searchsorted([0.01, 0.05], p)]
    _check(
        baseline,
        "plot_grade_backtest",
        lambda: plot_grade_backtest(jeffreys_grade_test(y, p, grades)),
    )


def test_plot_offset_audit_default_is_stable(baseline, monkeypatch):
    from datetime import UTC, datetime

    import probcal.offset as _offset_module
    from probcal.offset import LogitOffset
    from probcal.plots import plot_offset_audit

    # plot_offset_audit prints offset.timestamp_ (wall-clock fit time) on the chart;
    # freeze probcal.offset's clock so the rendered output is reproducible.
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(_offset_module, "datetime", _FrozenDatetime)

    y, p = _data()
    _check(
        baseline,
        "plot_offset_audit",
        lambda: plot_offset_audit(LogitOffset(target_mean=float(y.mean())).fit(p)),
    )


def _monitor_report():
    from probcal.monitor import CalibrationMonitor

    y, p = _data()
    mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 41))
    for i, (yb, pb) in enumerate(zip(np.array_split(y, 3), np.array_split(p, 3), strict=True)):
        mon.update(yb, pb, label=f"batch{i}")
    return mon.report()


def test_plot_e_process_default_is_stable(baseline):
    from probcal.plots import plot_e_process

    report = _monitor_report()
    _check(baseline, "plot_e_process", lambda: plot_e_process(report))


def test_plot_comparison_default_is_stable(baseline):
    from probcal.curves import reliability_binned
    from probcal.plots import plot_comparison

    y, p = _data()
    before = reliability_binned(y, p)
    after = reliability_binned(y, np.clip(p * 0.9, 1e-6, 1 - 1e-6))
    _check(baseline, "plot_comparison", lambda: plot_comparison(before, after))


def test_plot_interval_default_is_stable(baseline):
    from probcal.plots import plot_interval
    from probcal.vennabers import VennAbersCalibrator

    y, p = _data()
    grid = np.linspace(0.005, 0.5, 60)
    cal = VennAbersCalibrator().fit(p[:1500], y[:1500])
    _check(baseline, "plot_interval", lambda: plot_interval(cal.predict_interval(grid), grid))


def test_snapshot_detects_a_change(baseline):
    """Guard against a vacuous comparison: a recoloured line must fail."""
    if _REGEN or "plot_interval" not in baseline:
        pytest.skip("baseline being regenerated")
    from probcal.plots import plot_interval
    from probcal.vennabers import VennAbersCalibrator

    y, p = _data()
    grid = np.linspace(0.005, 0.5, 60)
    cal = VennAbersCalibrator().fit(p[:1500], y[:1500])
    ax = plot_interval(cal.predict_interval(grid), grid)
    ax.lines[0].set_color("black")
    with pytest.raises(AssertionError):
        _assert_close(json.loads(json.dumps(_snapshot(ax))), baseline["plot_interval"])
