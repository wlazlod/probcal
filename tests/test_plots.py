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
