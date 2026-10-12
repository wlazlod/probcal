"""Tests for probcal.base.BaseCalibrator contract and the shared argument validation."""

import numpy as np
import pytest

from probcal import (
    BBQCalibrator,
    BetaCalibrator,
    CenteredIsotonicCalibrator,
    Chain,
    CrossVennAbersCalibrator,
    ENIRCalibrator,
    HistogramBinningCalibrator,
    IsotonicCalibrator,
    LogitOffset,
    PlattCalibrator,
    ScalingBinningCalibrator,
    SplineCalibrator,
    TemperatureCalibrator,
    VennAbersCalibrator,
)
from probcal._results import Interpretation

RNG = np.random.default_rng(7)


def _calibrated_sample(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.02, 0.98, n)
    y = (RNG.random(n) < s).astype(float)
    return s, y


def test_predict_proba_2d() -> None:
    s, y = _calibrated_sample()
    cal = PlattCalibrator().fit(s, y)
    p2 = cal.predict_proba_2d(s[:5])
    assert p2.shape == (5, 2)
    np.testing.assert_allclose(p2.sum(axis=1), 1.0)
    np.testing.assert_allclose(p2[:, 1], cal.predict_proba(s[:5]))


@pytest.mark.parametrize(
    ("call", "exc", "match"),
    [
        (lambda: PlattCalibrator().set_params(nope=1), ValueError, "unknown parameter"),
        (lambda: PlattCalibrator().fit([0.1, 0.2], [0, 1, 1]), ValueError, "length"),
        (lambda: PlattCalibrator().interval_inverse(0.0, 0.02), RuntimeError, "not fitted"),
    ],
    ids=["unknown-param", "length-mismatch", "not-fitted"],
)
def test_invalid_calls_raise(call, exc: type, match: str) -> None:
    with pytest.raises(exc, match=match):
        call()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: BetaCalibrator(variant="xyz"), "variant"),
        (lambda: HistogramBinningCalibrator(n_bins=True), "n_bins"),
        (lambda: HistogramBinningCalibrator(strategy="nope"), "strategy"),
        (lambda: HistogramBinningCalibrator(shrinkage="nope"), "shrinkage"),
        (lambda: ScalingBinningCalibrator(n_bins=0), "n_bins"),
        (lambda: BBQCalibrator(min_bins=2.5), "bins"),
        (lambda: BBQCalibrator(max_bins=-1), "bins"),
        (lambda: SplineCalibrator(cv=1), "cv"),
        (lambda: SplineCalibrator(lambdas=[-1.0, 1.0]), "lambdas"),
        (lambda: SplineCalibrator(n_knots=0), "n_knots"),
        (lambda: CrossVennAbersCalibrator(cv=True), "cv"),
    ],
    ids=[
        "beta-variant",
        "hist-n_bins",
        "hist-strategy",
        "hist-shrinkage",
        "scaling-n_bins",
        "bbq-min_bins",
        "bbq-max_bins",
        "spline-cv",
        "spline-lambdas",
        "spline-n_knots",
        "cvap-cv",
    ],
)
def test_bad_constructor_arguments_raise_at_fit(make, match: str) -> None:
    """CAL-14/CAL-25: sklearn convention, construction never raises; fit does."""
    cal = make()
    with pytest.raises(ValueError, match=match):
        cal.fit(*_calibrated_sample(300))


@pytest.mark.parametrize(
    "make",
    [
        PlattCalibrator,
        TemperatureCalibrator,
        BetaCalibrator,
        HistogramBinningCalibrator,
        ScalingBinningCalibrator,
        IsotonicCalibrator,
        CenteredIsotonicCalibrator,
        SplineCalibrator,
        VennAbersCalibrator,
        BBQCalibrator,
        ENIRCalibrator,
        lambda: Chain([BetaCalibrator(), LogitOffset(delta=0.2)]),
    ],
    ids=lambda m: "Chain" if m.__name__ == "<lambda>" else m.__name__,
)
def test_interpret_pairs_names_with_values(make) -> None:
    """One check for the whole Interpretation family: names and values align."""
    interp = make().fit(*_calibrated_sample(600)).interpret()
    assert isinstance(interp, Interpretation)
    assert len(interp.param_names) == len(interp.param_values) >= 1
    assert interp.messages
