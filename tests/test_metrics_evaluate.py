"""Tests for probcal.metrics.evaluate."""

import numpy as np
import pytest

from probcal._math import expit
from probcal._results import MetricReport
from probcal.metrics import evaluate

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


@pytest.mark.slow
def test_evaluate_default_boot() -> None:
    y, p = _calibrated(300)
    rep = evaluate(y, p)  # n_boot=1000 default
    assert len(rep.names) == len(rep.values)
