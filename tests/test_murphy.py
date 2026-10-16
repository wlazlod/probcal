"""Tests for probcal.metrics.murphy_curve and probcal.plots.plot_murphy."""

import importlib.util

import numpy as np
import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.7, n)
    y = (rng.random(n) < p).astype(float)
    return y, p


def test_murphy_curve_exact_midpoint_identity():
    """2 * sum(w * S_theta(mid)) == brier_score(y, p) to 1e-10, via the midpoint rule.

    ``S_theta`` is piecewise linear in ``theta`` on each OPEN interval
    between consecutive breakpoints ``u = sorted(unique(p) | {0, 1})``
    (it only jumps exactly at the breakpoints themselves), and
    ``integral_0^1 S_theta dtheta == (y - p)**2 / 2`` exactly per
    observation. The midpoint rule is exact for a linear function on an
    interval (the value at the midpoint equals the interval's average), so
    evaluating at ``mid = (u[1:] + u[:-1]) / 2`` — interior points, never a
    breakpoint itself — sidesteps the jump-discontinuity ambiguity of
    sampling exactly at a breakpoint entirely, unlike the raw-breakpoint
    trapezoid rule checked below.
    """
    from probcal.metrics import brier_score, murphy_curve

    for seed in range(2):
        y, p = _data(n=250, seed=seed)
        u = np.union1d(np.array([0.0, 1.0]), np.unique(p))
        mid = (u[1:] + u[:-1]) / 2.0
        w = np.diff(u)
        curve = murphy_curve(y, p, thresholds=mid)
        lhs = 2.0 * np.sum(w * curve.score)
        rhs = brier_score(y, p)
        assert abs(lhs - rhs) < 1e-10


def test_murphy_curve_dominance_after_pav_recalibration():
    from probcal._corp import corp_fit
    from probcal.metrics import murphy_curve
    from probcal.metrics._common import _prep

    y, p = _data(n=400, seed=2)
    y_arr, p_arr, w = _prep(y, p, None)
    _, _, _, _, pav = corp_fit(y_arr, p_arr, w)

    curve = murphy_curve(y_arr, p_arr, thresholds=201)
    curve_pav = murphy_curve(y_arr, pav, thresholds=curve.thresholds)
    assert np.all(curve_pav.score <= curve.score + 1e-12)


def test_murphy_curve_weighting_equals_row_duplication():
    from probcal.metrics import murphy_curve

    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.6, 0.3, 0.8])
    w = np.array([2.0, 1.0, 3.0, 1.0])
    curve_w = murphy_curve(y, p, sample_weight=w, thresholds=101)

    reps = w.astype(int)
    y_dup = np.repeat(y, reps)
    p_dup = np.repeat(p, reps)
    curve_dup = murphy_curve(y_dup, p_dup, thresholds=101)

    assert np.allclose(curve_w.score, curve_dup.score, atol=1e-12)


def test_murphy_curve_rejects_thresholds_outside_unit_interval():
    from probcal.metrics import murphy_curve

    y, p = _data()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        murphy_curve(y, p, thresholds=np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        murphy_curve(y, p, thresholds=np.array([0.5, 1.2]))


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_murphy_diff_draws_band_and_zero_line():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.plots import plot_murphy

    y, p_a = _data(seed=5)
    p_b = np.clip(p_a * 0.9, 1e-6, 1 - 1e-6)
    ax = plot_murphy({"a": (y, p_a), "b": (y, p_b)}, diff=True, n_boot=25)
    labels = {ln.get_label() for ln in ax.lines}
    assert "a - b" in labels
    assert "zero" in labels
    assert len(ax.collections) >= 1  # bootstrap band
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_murphy_diff_rejects_non_pair_mapping():
    import matplotlib

    matplotlib.use("Agg")

    from probcal.metrics import murphy_curve
    from probcal.plots import plot_murphy

    y, p = _data()

    # Wrong entry count.
    with pytest.raises(ValueError, match="raw"):
        plot_murphy({"a": (y, p)}, diff=True)

    # Values are MurphyCurve, not raw (y, p) pairs.
    curves = {"a": murphy_curve(y, p), "b": murphy_curve(y, p)}
    with pytest.raises(ValueError, match="raw"):
        plot_murphy(curves, diff=True)
