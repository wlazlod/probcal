"""Tests for probcal.datasets.make_pd_portfolio."""

import numpy as np

from probcal._math import logit
from probcal.datasets import make_pd_portfolio
from probcal.metrics.regression import calibration_test
from probcal.parametric import BetaCalibrator


def test_event_rate_anchor_and_reproducibility() -> None:
    port = make_pd_portfolio(n=20000, event_rate=0.03, random_state=1)
    assert abs(port.p_true.mean() - 0.03) < 1e-6  # anchored exactly by construction
    assert abs(port.y.mean() - 0.03) < 0.005  # sampling noise only
    assert make_pd_portfolio(n=100, random_state=1) == make_pd_portfolio(n=100, random_state=1)
    assert make_pd_portfolio(n=100, random_state=1) != make_pd_portfolio(n=100, random_state=2)


def test_default_portfolio_is_miscalibrated() -> None:
    port = make_pd_portfolio(n=20000, random_state=3)
    res = calibration_test(port.y, port.scores)
    assert res.p_value < 1e-4


def test_asymmetry_recoverable_by_beta() -> None:
    # Recovery of BOTH exponents needs data in both tails, so this test widens
    # the score distribution; on the 3% default portfolio the high-tail
    # exponent is unidentifiable (no data there).
    port = make_pd_portfolio(
        n=40000,
        slope=0.8,
        asymmetry=0.5,
        event_rate=0.25,
        score_location=-0.8,
        score_scale=1.4,
        random_state=11,
    )
    cal = BetaCalibrator().fit(port.scores, port.y)
    # Generative exponents: a_lo = slope*(1+asymmetry) = 1.2, a_hi = slope = 0.8.
    assert cal.a_ > cal.b_  # asymmetry direction recovered
    assert abs(cal.a_ - 1.2) < 0.2
    assert abs(cal.b_ - 0.8) < 0.2


def test_intercept_and_event_rate_are_honoured() -> None:
    # OFF-4: the intercept was absorbed by the mean anchor (no effect at all),
    # and slope=1, asymmetry=0 ignored event_rate entirely.
    base = make_pd_portfolio(n=5000, random_state=2)
    shifted = make_pd_portfolio(n=5000, intercept=0.5, random_state=2)
    np.testing.assert_array_equal(base.scores, shifted.scores)
    np.testing.assert_allclose(logit(shifted.p_true) - logit(base.p_true), 0.5, atol=1e-9)
    assert shifted.p_true.mean() > 0.03

    port = make_pd_portfolio(n=20000, slope=1.0, asymmetry=0.0, event_rate=0.1, random_state=5)
    assert abs(port.scores.mean() - 0.1) < 1e-9
    np.testing.assert_array_equal(port.scores, port.p_true)
