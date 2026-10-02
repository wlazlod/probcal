"""Tests for probcal.spline.SplineCalibrator."""

import numpy as np
import pytest

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


# ---------------------------------------------------------------- 0.4.0 regressions


def _far_tail_inversion() -> tuple[np.ndarray, np.ndarray]:
    """Event rate 0.3 at logit(s) in [-14, -11], 0 at [-11, -7], then rising."""
    rng = np.random.default_rng(0)
    z = np.concatenate(
        [rng.uniform(-14, -11, 200), rng.uniform(-11, -7, 200), rng.normal(0, 1.5, 1600)]
    )
    y = np.concatenate(
        [rng.random(200) < 0.3, np.zeros(200, dtype=bool), rng.random(1600) < expit(z[400:])]
    ).astype(float)
    return expit(z), y


def test_spline_monotonicity_checked_over_the_whole_domain() -> None:
    """CAL-5: 0.3.x probed only linspace(0.002, 0.998) (logit ±6.2) while the
    knots reach logit -14; a dip down there passed as monotone."""
    s, y = _far_tail_inversion()
    with pytest.warns(UserWarning, match="not monotone"):
        cal = SplineCalibrator().fit(s, y)
    z = np.linspace(-14.0, -6.0, 4001)
    assert np.any(np.diff(cal.predict_proba(expit(z))) < 0.0)  # the dip is real
    assert cal.is_monotone_ is False
    again = SplineCalibrator.from_json(cal.to_json())
    assert again.is_monotone_ is False


@pytest.mark.filterwarnings("ignore:SplineCalibrator. fitted curve is not monotone")
def test_spline_exact_monotone_check_matches_dense_logit_grid() -> None:
    cal = SplineCalibrator().fit(*_curved_sample(3000))
    z = np.linspace(-27.0, 27.0, 200001)
    p = cal.predict_proba(expit(z))
    assert cal.is_monotone_ == bool(np.all(np.diff(p) >= -1e-15))


@pytest.mark.filterwarnings("ignore:SplineCalibrator. fitted curve is not monotone")
def test_spline_reports_convergence_and_final_edof() -> None:
    """CAL-9/10: the shared safeguarded solver exposes converged_, and edof_
    is the smoother trace at the *final* coefficients."""
    from probcal.spline import _second_difference_penalty

    s, y = _identity_sample(2000)
    cal = SplineCalibrator().fit(s, y)
    assert cal.converged_ is True
    assert cal.to_dict()["fit_meta"]["converged_"] is True
    basis = np.asarray(
        __import__("probcal._math", fromlist=["x"]).natural_cubic_basis(logit(s), cal.knots_)
    )
    mu = expit(basis @ cal.theta_)
    bwb = (basis * (mu * (1 - mu))[:, None]).T @ basis
    hess = bwb + cal.lambda_ * _second_difference_penalty(basis.shape[1])
    np.testing.assert_allclose(cal.edof_, np.trace(np.linalg.solve(hess, bwb)), rtol=1e-12)


def test_spline_cv_validated_at_fit() -> None:
    """CAL-14: cv must be an integer >= 2 (cv=1 used to fit on empty folds)."""
    s, y = _identity_sample(300)
    for bad in (1, 0, 2.5, True):
        cal = SplineCalibrator(cv=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cv"):
            cal.fit(s, y)
    with pytest.raises(ValueError, match="lambdas"):
        SplineCalibrator(lambdas=[-1.0, 1.0]).fit(s, y)
    with pytest.raises(ValueError, match="n_knots"):
        SplineCalibrator(n_knots=0).fit(s, y)


@pytest.mark.filterwarnings("ignore:SplineCalibrator. fitted curve is not monotone")
def test_spline_state_uses_public_names() -> None:
    cal = SplineCalibrator().fit(*_identity_sample(800))
    state = cal.to_dict()["state"]
    assert "knots_" in state and "theta_" in state
    assert not any(k.startswith("_") for k in state)
