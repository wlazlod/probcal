"""Tests for probcal.plots.plot_corp and plot_mcb_dsc ([viz]-guarded)."""

import importlib.util

import numpy as np
import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")


def _data(n=2000, seed=17):
    from probcal import make_pd_portfolio

    d = make_pd_portfolio(n=n, random_state=seed)
    return d.y, d.scores


def test_plot_corp_draws_step_bands_and_box():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.curves import corp_reliability
    from probcal.plots import plot_corp

    y, p = _data()
    r = corp_reliability(y, p, n_resamples=20)
    ax = plot_corp(r)
    assert any(ln.get_drawstyle() == "steps-post" for ln in ax.lines)
    assert len(ax.collections) >= 1  # band fill
    assert any("MCB" in t.get_text() for t in ax.texts)
    ax2 = plot_corp(r, scale="logit", show_decomposition=False)
    assert not any("MCB" in t.get_text() for t in ax2.texts)
    plt.close("all")


def test_plot_mcb_dsc_places_one_point_per_candidate():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.plots import plot_mcb_dsc

    y, p_a = _data(seed=17)
    p_b = np.clip(p_a * 0.9, 1e-6, 1 - 1e-6)
    candidates = {"a": (y, p_a), "b": (y, p_b)}

    ax = plot_mcb_dsc(candidates)
    assert len(ax.collections) >= 1
    assert ax.collections[0].get_offsets().shape[0] == 2
    name_labels = [t.get_text() for t in ax.texts if t.get_text() in candidates]
    assert sorted(name_labels) == ["a", "b"]

    diag_slopes = []
    for ln in ax.lines:
        (x0, x1), (y0, y1) = ln.get_xdata(), ln.get_ydata()
        if x1 != x0:
            diag_slopes.append((y1 - y0) / (x1 - x0))
    assert sum(abs(s - 1.0) < 1e-9 for s in diag_slopes) >= 3
    assert any("S̄=" in t.get_text() for t in ax.texts)
    assert ax.get_xlabel() == "DSC (discrimination)"
    assert ax.get_ylabel() == "MCB (miscalibration)"
    plt.close("all")

    # SelectionReport input works too.
    from probcal import CalibratorSelector, make_pd_portfolio

    d = make_pd_portfolio(n=1500, random_state=2)
    sel = CalibratorSelector(cv=3).fit(d.scores, d.y)
    ax2 = plot_mcb_dsc(sel.report_)
    assert ax2.collections[0].get_offsets().shape[0] == len(sel.report_.methods)
    plt.close("all")

    # SelectionReport without mcb/dsc (pre-0.3 report) raises.
    import dataclasses

    stale = dataclasses.replace(sel.report_, mcb=None, dsc=None, unc=None)
    with pytest.raises(ValueError, match="report has no mcb/dsc columns"):
        plot_mcb_dsc(stale)


def test_plot_mcb_dsc_rejects_mismatched_weighted_mean_y():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.plots import plot_mcb_dsc

    y_a, p_a = _data(seed=17)
    y_b, p_b = _data(seed=19)  # different portfolio -> different weighted mean y
    with pytest.raises(ValueError, match="same weighted mean"):
        plot_mcb_dsc({"a": (y_a, p_a), "b": (y_b, p_b)})
    plt.close("all")
