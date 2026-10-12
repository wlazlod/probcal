"""Tests for probcal.parametric: Platt, temperature, and beta calibrators."""

import warnings

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator

RNG = np.random.default_rng(123)
GRID = np.linspace(0.001, 0.999, 400)


def _sample(a: float, b: float, n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Scores s and outcomes drawn from Bern(sigma(a * logit(s) + b))."""
    s = expit(RNG.normal(loc=-1.0, scale=1.5, size=n))
    p_true = expit(a * logit(s) + b)
    y = (RNG.random(n) < p_true).astype(float)
    return s, y


# ---------------------------------------------------------------- parameter recovery


def test_beta_identity_recovery() -> None:
    s, y = _sample(1.0, 0.0)
    cal = BetaCalibrator().fit(s, y)
    assert abs(cal.a_ - 1.0) < 0.15
    assert abs(cal.b_ - 1.0) < 0.15
    assert abs(cal.c_) < 0.15


def test_platt_distortion_recovery() -> None:
    s, y = _sample(0.7, -0.5)
    cal = PlattCalibrator().fit(s, y)
    assert abs(cal.a_ - 0.7) < 0.08
    assert abs(cal.b_ - (-0.5)) < 0.08


def test_temperature_recovers_pure_scaling() -> None:
    s, y = _sample(0.5, 0.0)  # T = 2
    cal = TemperatureCalibrator().fit(s, y)
    assert abs(cal.T_ - 2.0) < 0.15


# ---------------------------------------------------------------- beta specifics


def test_beta_ab_matches_platt() -> None:
    s, y = _sample(0.8, 0.3)
    beta = BetaCalibrator(variant="ab").fit(s, y)
    platt = PlattCalibrator().fit(s, y)
    # "ab" ties the exponents: logit g = a*logit(s) + c, the same family Platt fits
    # (up to Platt's target smoothing).
    assert abs(beta.a_ - platt.a_) < 0.05
    assert abs(beta.c_ - platt.b_) < 0.05


@pytest.mark.parametrize(
    ("b_ln1ms", "seed", "b_survives"),
    [(0.0, 0, False), (1.0, 1, True)],
    ids=["intercept-only", "b-refit"],
)
def test_beta_negative_a_triggers_refit(b_ln1ms: float, seed: int, b_survives: bool) -> None:
    # True map decreases in ln s: unconstrained a would go negative. The refit on
    # ln(1 - s) alone keeps b when it is positive, else falls back to the intercept.
    rng = np.random.default_rng(seed)
    n = 8000
    s = expit(rng.normal(-1.0, 1.2, n))
    p_true = expit(-0.6 * np.log(s) - b_ln1ms * np.log1p(-s) - 1.5)
    y = (rng.random(n) < p_true).astype(float)
    cal = BetaCalibrator().fit(s, y)
    assert cal.a_ == 0.0
    assert cal.constraint_active_
    assert (cal.b_ > 0.0) is b_survives
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-12)


# ---------------------------------------------------------------- sample weights


def test_platt_integer_weights_equal_row_duplication() -> None:
    """Weighted class totals in the smoothing prior make weighting = duplication."""
    s, y = _sample(0.7, -0.3, n=1500)
    repeats = RNG.integers(1, 4, size=s.size)
    weighted = PlattCalibrator().fit(s, y, sample_weight=repeats.astype(float))
    duplicated = PlattCalibrator().fit(np.repeat(s, repeats), np.repeat(y, repeats))
    np.testing.assert_allclose(
        weighted.predict_proba(GRID), duplicated.predict_proba(GRID), atol=1e-12, rtol=0.0
    )


# ---------------------------------------------------------------- interpret / affine


@pytest.mark.parametrize(
    "factory",
    [PlattCalibrator, TemperatureCalibrator, lambda: BetaCalibrator(variant="ab")],
    ids=["platt", "temperature", "beta-ab"],
)
def test_affine_coeffs_reproduce_prediction(factory) -> None:
    s, y = _sample(0.7, -0.3, n=5000)
    cal = factory().fit(s, y)
    a, b = cal.affine_logit_coeffs_
    np.testing.assert_allclose(cal.predict_proba(GRID), expit(a * logit(GRID) + b), atol=1e-12)


# ---------------------------------------------------------------- separation / convergence


def _wide_sample(seed: int, n: int = 20_000) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Wide-score design: z ~ N(0, 8), true relation logit p = 1.5 z + 0.5."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 8.0, n)
    y = (rng.random(n) < expit(1.5 * z + 0.5)).astype(float)
    return expit(z), y, rng


def test_platt_steep_map_unbiased() -> None:
    # Regression for the v0.1.1 false-separation abort (true a = 1.5 -> 1.18).
    s, y, _ = _wide_sample(31)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        cal = PlattCalibrator().fit(s, y)
    assert 1.45 <= cal.a_ <= 1.55
    assert cal.converged_ is True


def test_interpret_reports_unconverged() -> None:
    s, y, _ = _wide_sample(34, n=2000)
    for cal in (PlattCalibrator().fit(s, y), BetaCalibrator(variant="ab").fit(s, y)):
        cal.converged_ = False
        assert any("did not converge" in m for m in cal.interpret().messages)


def test_beta_separation_fallback_surfaces() -> None:
    s = np.concatenate([np.linspace(0.05, 0.3, 10), np.linspace(0.7, 0.95, 10)])
    y = np.concatenate([np.zeros(10), np.ones(10)])  # perfectly separated
    with pytest.warns(UserWarning, match="[Ss]eparation"):
        cal = BetaCalibrator(variant="ab").fit(s, y)
    assert cal.separation_fallback_ is True
    assert cal.converged_ is True  # the ridge fallback must converge
    assert np.isfinite(cal.a_) and np.isfinite(cal.c_)
    assert any("separation" in m for m in cal.interpret().messages)


# ---------------------------------------------------------------- 0.3.4 regressions


def test_beta_a_boundary_clamp_warns_like_temperature() -> None:
    """CAL-13: separated data push the "a" exponent to the 1e6 cap; 0.3.x clamped
    there silently while TemperatureCalibrator warned for the same NLL."""
    s = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    with pytest.warns(UserWarning, match="no interior minimum"):
        cal = BetaCalibrator(variant="a").fit(s, y)
    assert cal.a_ == 1e6
    with pytest.warns(UserWarning, match="no interior minimum"):
        TemperatureCalibrator().fit(s, y)


def test_beta_a_and_temperature_share_the_estimate() -> None:
    """Beta "a" is temperature scaling in another parameterization: a = 1/T."""
    s, y = _sample(0.7, 0.0, n=5000)
    beta = BetaCalibrator(variant="a").fit(s, y)
    temp = TemperatureCalibrator().fit(s, y)
    np.testing.assert_allclose(beta.a_, 1.0 / temp.T_, rtol=1e-9)


@pytest.mark.parametrize("variant", ["abm", "ab", "a"])
def test_beta_point_inverse_round_trip_through_base_hook(variant: str) -> None:
    cal = BetaCalibrator(variant=variant).fit(*_sample(0.8, -0.3, n=5000))
    p = np.linspace(0.02, 0.9, 25)
    np.testing.assert_allclose(cal.predict_proba(cal.point_inverse(p)), p, atol=1e-12)
    z = cal.point_inverse(p, space="logit")
    np.testing.assert_allclose(expit(z), cal.point_inverse(p), atol=1e-15)


@pytest.mark.parametrize("cls", [PlattCalibrator, TemperatureCalibrator])
def test_affine_interval_inverse_is_closed_form(cls: type) -> None:
    """CAL-17: Platt/Temperature invert through the base affine-logit closed form."""
    cal = cls().fit(*_sample(0.8, -0.3, n=5000))
    a, b = cal.affine_logit_coeffs_
    lo, hi = cal.interval_inverse(0.1, 0.4, space="logit")
    np.testing.assert_allclose([lo, hi], (logit(np.array([0.1, 0.4])) - b) / a, rtol=1e-12)
