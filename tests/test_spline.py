"""Tests for probcal.spline.SplineCalibrator."""

import numpy as np

from probcal._math import expit, logit
from probcal.parametric import PlattCalibrator
from probcal.spline import SplineCalibrator

RNG = np.random.default_rng(41)
GRID = np.linspace(0.005, 0.995, 300)


def _identity_sample(n: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    s = expit(RNG.normal(-0.5, 1.4, n))
    y = (RNG.random(n) < s).astype(float)
    return s, y


def _curved_sample(n: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    # Monotone, non-affine logit distortion: z_true = z + 0.8 sin(z).
    s = expit(RNG.normal(0.0, 1.6, n))
    z = logit(s)
    p_true = expit(z + 0.8 * np.sin(z))
    y = (RNG.random(n) < p_true).astype(float)
    return s, y


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_spline_tracks_identity_data() -> None:
    cal = SplineCalibrator().fit(*_identity_sample())
    p = cal.predict_proba(GRID)
    assert np.max(np.abs(p - GRID)) < 0.08


def test_spline_repairs_curvature_better_than_platt() -> None:
    s_train, y_train = _curved_sample(6000)
    s_test, y_test = _curved_sample(6000)
    spline = SplineCalibrator(random_state=0).fit(s_train, y_train)
    platt = PlattCalibrator().fit(s_train, y_train)
    ll_spline = _log_loss(y_test, spline.predict_proba(s_test))
    ll_platt = _log_loss(y_test, platt.predict_proba(s_test))
    assert ll_spline < ll_platt


def test_spline_edof_and_lambda_attrs() -> None:
    cal = SplineCalibrator().fit(*_identity_sample(2000))
    assert 1.0 < cal.edof_ <= cal.n_knots_
    assert cal.lambda_ in cal.lambdas_grid_


def test_spline_cv_reproducible() -> None:
    s, y = _identity_sample(1500)
    a = SplineCalibrator(random_state=5).fit(s, y)
    b = SplineCalibrator(random_state=5).fit(s, y)
    assert a.lambda_ == b.lambda_
    np.testing.assert_allclose(a.predict_proba(GRID), b.predict_proba(GRID))


def test_spline_monotone_on_wellbehaved_data() -> None:
    cal = SplineCalibrator().fit(*_identity_sample(4000))
    p = cal.predict_proba(GRID)
    assert np.all(np.diff(p) >= -1e-10)
    assert cal.is_monotone_ is True


def test_spline_small_sample_stability() -> None:
    n = 200
    s = expit(RNG.normal(-2.2, 1.0, n))
    y = np.zeros(n)
    y[RNG.choice(n, 20, replace=False)] = 1.0
    cal = SplineCalibrator().fit(s, y)
    p = cal.predict_proba(GRID)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))


def test_spline_interpret_reports_edof() -> None:
    cal = SplineCalibrator().fit(*_identity_sample(2000))
    interp = cal.interpret()
    assert "edof" in interp.param_names
    assert any("degrees of freedom" in m for m in interp.messages)


def test_export() -> None:
    import probcal

    assert "SplineCalibrator" in probcal.__all__
