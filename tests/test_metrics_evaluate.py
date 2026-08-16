"""Tests for probcal.metrics.evaluate."""

import numpy as np
import pytest

from probcal._math import expit
from probcal._results import MetricReport
from probcal.datasets import make_pd_portfolio
from probcal.metrics import (
    ReliabilitySummary,
    calibration_intercept,
    calibration_slope,
    e90,
    evaluate,
    ici,
    reliability_summary,
    spiegelhalter_z,
)

RNG = np.random.default_rng(73)


def _calibrated(n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_evaluate_returns_metric_report_with_cis() -> None:
    y, p = _calibrated()
    rep = evaluate(y, p, n_boot=30, seed=1)
    assert isinstance(rep, MetricReport)
    for name in ("log_loss", "brier", "ece", "ece_debiased", "mce", "ici", "smooth_ece"):
        assert name in rep.names
    # CIs bracket the point estimates for the well-behaved metrics.
    for i, name in enumerate(rep.names):
        if name in ("log_loss", "brier"):
            assert rep.ci_low[i] <= rep.values[i] <= rep.ci_high[i]


def test_evaluate_seeded_reproducible() -> None:
    y, p = _calibrated(400)
    a = evaluate(y, p, n_boot=20, seed=7)
    b = evaluate(y, p, n_boot=20, seed=7)
    np.testing.assert_allclose(a.ci_low, b.ci_low)
    np.testing.assert_allclose(a.ci_high, b.ci_high)


def test_reliability_summary_matches_metric_calls() -> None:
    y, p = _calibrated(1500)
    s = reliability_summary(y, p)
    assert isinstance(s, ReliabilitySummary)
    assert s.n == 1500 and s.events == int(y.sum())
    assert s.intercept == calibration_intercept(y, p)
    assert s.slope == calibration_slope(y, p)
    assert s.ici == ici(y, p)
    assert s.e90 == e90(y, p)
    assert s.spiegelhalter_p == spiegelhalter_z(y, p).p_value


def test_evaluate_metrics_subset_matches_full_run() -> None:
    d = make_pd_portfolio(n=400)
    full = evaluate(d.y, d.scores, n_boot=25, seed=3)
    sub = evaluate(d.y, d.scores, n_boot=25, seed=3, metrics=["e90", "ici"])
    assert sub.names == ("ici", "e90")  # catalog order, not argument order
    for name in sub.names:
        i, j = full.names.index(name), sub.names.index(name)
        assert (full.values[i], full.ci_low[i], full.ci_high[i]) == (
            sub.values[j],
            sub.ci_low[j],
            sub.ci_high[j],
        )


def test_evaluate_unknown_metric_raises_listing_valid_names() -> None:
    d = make_pd_portfolio(n=100)
    with pytest.raises(ValueError, match="log_loss"):
        evaluate(d.y, d.scores, n_boot=5, metrics=["nope"])


@pytest.mark.slow
def test_evaluate_default_boot() -> None:
    y, p = _calibrated(300)
    rep = evaluate(y, p)  # n_boot=1000 default
    assert len(rep.names) == len(rep.values)
