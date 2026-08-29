"""Tests for probcal.plots.plot_corp ([viz]-guarded)."""

import importlib.util

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
