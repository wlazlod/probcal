"""Tests for probcal.parametric: Platt, temperature, and beta calibrators."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal._results import Interpretation
from probcal.parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator

RNG = np.random.default_rng(123)
GRID = np.linspace(0.001, 0.999, 400)


def _sample(a: float, b: float, n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Scores s and outcomes drawn from Bern(sigma(a * logit(s) + b))."""
    s = expit(RNG.normal(loc=-1.0, scale=1.5, size=n))
    p_true = expit(a * logit(s) + b)
    y = (RNG.random(n) < p_true).astype(float)
    return s, y


# ---------------------------------------------------------------- identity recovery


def test_platt_identity_recovery() -> None:
    s, y = _sample(1.0, 0.0)
    cal = PlattCalibrator().fit(s, y)
    assert abs(cal.a_ - 1.0) < 0.1
    assert abs(cal.b_) < 0.1


def test_temperature_identity_recovery() -> None:
    s, y = _sample(1.0, 0.0)
    cal = TemperatureCalibrator().fit(s, y)
    assert abs(cal.T_ - 1.0) < 0.05


def test_beta_identity_recovery() -> None:
    s, y = _sample(1.0, 0.0)
    cal = BetaCalibrator().fit(s, y)
    assert abs(cal.a_ - 1.0) < 0.15
    assert abs(cal.b_ - 1.0) < 0.15
    assert abs(cal.c_) < 0.15


# ---------------------------------------------------------------- distortion recovery


def test_platt_distortion_recovery() -> None:
    s, y = _sample(0.7, -0.5)
    cal = PlattCalibrator().fit(s, y)
    assert abs(cal.a_ - 0.7) < 0.08
    assert abs(cal.b_ - (-0.5)) < 0.08


def test_temperature_recovers_pure_scaling() -> None:
    s, y = _sample(0.5, 0.0)  # T = 2
    cal = TemperatureCalibrator().fit(s, y)
    assert abs(cal.T_ - 2.0) < 0.15


def test_temperature_cannot_shift_base_rate() -> None:
    s, y = _sample(1.0, -0.8)  # pure intercept error
    cal = TemperatureCalibrator().fit(s, y)
    # sigma(logit(0.5)/T) == 0.5 for every T: the midpoint is a fixed point.
    np.testing.assert_allclose(cal.predict_proba(np.array([0.5])), [0.5], atol=1e-12)


# ---------------------------------------------------------------- beta specifics


def test_beta_ab_matches_platt() -> None:
    s, y = _sample(0.8, 0.3)
    beta = BetaCalibrator(variant="ab").fit(s, y)
    platt = PlattCalibrator().fit(s, y)
    # "ab" ties the exponents: logit g = a*logit(s) + c, the same family Platt fits
    # (up to Platt's target smoothing).
    assert abs(beta.a_ - platt.a_) < 0.05
    assert abs(beta.c_ - platt.b_) < 0.05


def test_beta_a_variant_single_parameter() -> None:
    s, y = _sample(0.5, 0.0)
    cal = BetaCalibrator(variant="a").fit(s, y)
    assert cal.a_ == cal.b_
    assert cal.c_ == 0.0


def test_beta_negative_a_triggers_refit() -> None:
    # True map decreases in ln s: unconstrained a would go negative.
    n = 8000
    s = expit(RNG.normal(-1.0, 1.2, n))
    p_true = expit(-0.6 * np.log(s) - 1.5)  # increasing in -ln s => decreasing in s region
    y = (RNG.random(n) < p_true).astype(float)
    cal = BetaCalibrator().fit(s, y)
    assert cal.a_ == 0.0
    assert cal.constraint_active_
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-12)


def test_beta_invalid_variant_raises() -> None:
    with pytest.raises(ValueError, match="variant"):
        BetaCalibrator(variant="xyz").fit(*_sample(1.0, 0.0, n=500))


# ---------------------------------------------------------------- shared properties


@pytest.mark.parametrize(
    "cal_factory",
    [PlattCalibrator, TemperatureCalibrator, BetaCalibrator],
    ids=["platt", "temperature", "beta"],
)
def test_monotone_prediction(cal_factory) -> None:
    s, y = _sample(0.7, -0.3, n=5000)
    cal = cal_factory().fit(s, y)
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-12)


@pytest.mark.parametrize(
    "cal_factory",
    [PlattCalibrator, TemperatureCalibrator, BetaCalibrator],
    ids=["platt", "temperature", "beta"],
)
def test_small_sample_stability(cal_factory) -> None:
    # n=200 with ~20 events (spec §13).
    n = 200
    s = expit(RNG.normal(-2.2, 1.0, n))
    y = np.zeros(n)
    y[RNG.choice(n, 20, replace=False, p=s / s.sum())] = 1.0
    cal = cal_factory().fit(s, y)
    p = cal.predict_proba(GRID)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))


def test_sample_weight_changes_fit() -> None:
    s, y = _sample(0.7, -0.3, n=3000)
    w = np.where(s > 0.5, 5.0, 1.0)
    cal_unw = PlattCalibrator().fit(s, y)
    cal_w = PlattCalibrator().fit(s, y, sample_weight=w)
    assert cal_unw.a_ != cal_w.a_


# ---------------------------------------------------------------- interpret / affine


def test_interpret_returns_interpretation() -> None:
    s, y = _sample(0.7, -0.3, n=5000)
    for cal, names in [
        (PlattCalibrator().fit(s, y), ("a", "b")),
        (TemperatureCalibrator().fit(s, y), ("T",)),
        (BetaCalibrator().fit(s, y), ("a", "b", "c")),
    ]:
        interp = cal.interpret()
        assert isinstance(interp, Interpretation)
        assert interp.param_names == names
        assert len(interp.messages) >= 1


def test_affine_logit_coeffs() -> None:
    s, y = _sample(0.7, -0.3, n=5000)
    platt = PlattCalibrator().fit(s, y)
    a, b = platt.affine_logit_coeffs_
    np.testing.assert_allclose((a, b), (platt.a_, platt.b_))

    temp = TemperatureCalibrator().fit(s, y)
    a, b = temp.affine_logit_coeffs_
    np.testing.assert_allclose((a, b), (1.0 / temp.T_, 0.0))

    assert BetaCalibrator(variant="abm").fit(s, y).affine_logit_coeffs_ is None
    ab = BetaCalibrator(variant="ab").fit(s, y)
    a, b = ab.affine_logit_coeffs_
    np.testing.assert_allclose((a, b), (ab.a_, ab.c_))


def test_affine_coeffs_reproduce_prediction() -> None:
    s, y = _sample(0.7, -0.3, n=5000)
    cal = PlattCalibrator().fit(s, y)
    a, b = cal.affine_logit_coeffs_
    np.testing.assert_allclose(cal.predict_proba(GRID), expit(a * logit(GRID) + b), atol=1e-12)


# ---------------------------------------------------------------- exports


def test_public_exports() -> None:
    import probcal

    for name in (
        "BaseCalibrator",
        "PlattCalibrator",
        "TemperatureCalibrator",
        "BetaCalibrator",
        "logit",
        "expit",
    ):
        assert name in probcal.__all__
        assert getattr(probcal, name) is not None
