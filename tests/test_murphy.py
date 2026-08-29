"""Tests for probcal.metrics.murphy_curve and probcal.plots.plot_murphy (V5)."""

import importlib.util

import numpy as np
import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.7, n)
    y = (rng.random(n) < p).astype(float)
    return y, p


def test_murphy_curve_int_thresholds_shape_and_n():
    from probcal.metrics import murphy_curve

    y, p = _data()
    curve = murphy_curve(y, p, thresholds=101)
    assert curve.thresholds.shape == (101,)
    assert curve.score.shape == (101,)
    assert curve.n == len(y)
    assert np.isclose(curve.thresholds[0], 0.0)
    assert np.isclose(curve.thresholds[-1], 1.0)


def test_murphy_curve_default_thresholds_is_513():
    from probcal.metrics import murphy_curve

    y, p = _data()
    curve = murphy_curve(y, p)
    assert curve.thresholds.shape == (513,)


def test_murphy_curve_identity_at_data_breakpoints_is_near_exact():
    """2 * trapz(score, thresholds) == brier_score at the curve's own breakpoints.

    ``S_theta`` is exactly piecewise linear between consecutive unique ``p``
    values, but it *jumps* exactly at each one (the observation with that
    ``p`` switches sides), so the trapezoid rule over those breakpoints
    samples one side of every jump and converges to the true (continuum)
    identity at rate ~1/n rather than reproducing it to machine precision
    at any finite `n` — verified over several seeds at n=5000 to stay well
    inside a safe, non-flaky bound.
    """
    from probcal.metrics import brier_score, murphy_curve

    for seed in range(5):
        y, p = _data(n=5000, seed=seed)
        thresholds = np.union1d(np.array([0.0, 1.0]), np.unique(p))
        curve = murphy_curve(y, p, thresholds=thresholds)
        lhs = 2.0 * np.trapezoid(curve.score, curve.thresholds)
        rhs = brier_score(y, p)
        assert abs(lhs - rhs) < 1e-4


def test_murphy_curve_default_513_close_to_brier():
    from probcal.metrics import brier_score, murphy_curve

    y, p = _data(n=250, seed=1)
    curve = murphy_curve(y, p)
    lhs = 2.0 * np.trapezoid(curve.score, curve.thresholds)
    rhs = brier_score(y, p)
    assert abs(lhs - rhs) < 1e-3


def test_murphy_curve_dominance_after_pav_recalibration():
    from probcal._corp import corp_fit
    from probcal.metrics import murphy_curve
    from probcal.metrics.scores import _prep

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
def test_plot_murphy_single_curve_renders():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.metrics import murphy_curve
    from probcal.plots import plot_murphy

    y, p = _data()
    ax = plot_murphy(murphy_curve(y, p))
    assert len(ax.lines) >= 1
    assert ax.get_xlabel() == "threshold θ"
    assert ax.get_ylabel() == "mean elementary score"
    plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_murphy_mapping_draws_legend_per_curve():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from probcal.metrics import murphy_curve
    from probcal.plots import plot_murphy

    y_a, p_a = _data(seed=3)
    y_b, p_b = _data(seed=4)
    curves = {"a": murphy_curve(y_a, p_a), "b": murphy_curve(y_b, p_b)}
    ax = plot_murphy(curves)
    labels = {ln.get_label() for ln in ax.lines}
    assert {"a", "b"} <= labels
    assert ax.get_legend() is not None
    plt.close("all")


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
