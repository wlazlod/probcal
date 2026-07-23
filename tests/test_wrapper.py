"""Tests for probcal.wrapper.CalibratedModel."""

import numpy as np
import pytest

from probcal._math import expit, irls_logistic
from probcal.parametric import PlattCalibrator
from probcal.wrapper import CalibratedModel

RNG = np.random.default_rng(107)


class TinyLogit:
    """Minimal hand-rolled logistic model (no sklearn): fit / predict_proba."""

    def __init__(self) -> None:
        self.beta_ = None
        self.n_fit_rows_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TinyLogit":
        Xd = np.column_stack([np.ones(len(X)), X])
        self.beta_ = irls_logistic(Xd, y).beta
        self.n_fit_rows_ = len(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = expit(np.column_stack([np.ones(len(X)), X]) @ self.beta_)
        return np.column_stack([1.0 - p, p])


class MarginOnly:
    """Model exposing only decision_function."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarginOnly":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] - 1.0


def _data(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.normal(size=(n, 2))
    p = expit(-1.5 + 1.2 * X[:, 0] - 0.6 * X[:, 1])
    y = (RNG.random(n) < p).astype(float)
    return X, y


def _trained_model() -> tuple[TinyLogit, np.ndarray, np.ndarray]:
    X_tr, y_tr = _data(3000)
    model = TinyLogit().fit(X_tr, y_tr)
    X_cal, y_cal = _data(2000)
    return model, X_cal, y_cal


def test_prefit_matches_manual_calibration() -> None:
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    manual = PlattCalibrator().fit(model.predict_proba(X_cal)[:, 1], y_cal)
    X_new, _ = _data(200)
    np.testing.assert_allclose(
        wrapped.predict_proba(X_new),
        manual.predict_proba(model.predict_proba(X_new)[:, 1]),
        atol=1e-12,
    )


def test_decision_function_duck_typing() -> None:
    model = MarginOnly()
    X_cal, y_cal = _data(1500)
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    p = wrapped.predict_proba(X_cal[:50])
    assert p.shape == (50,)
    assert np.all((p > 0) & (p < 1))


def test_cv_pooled_flow() -> None:
    X, y = _data(2500)
    wrapped = CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=5).fit(X, y)
    # Final model refit on all rows; single pooled calibrator.
    assert wrapped.model_.n_fit_rows_ == 2500
    assert wrapped.calibrator_.fitted_
    p = wrapped.predict_proba(X[:100])
    assert np.all((p > 0) & (p < 1))


def test_cv_ensemble_flow() -> None:
    X, y = _data(2500)
    wrapped = CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=4, ensemble=True).fit(
        X, y
    )
    assert len(wrapped.ensemble_) == 4
    # Prediction equals the mean over fold pipelines.
    X_new = X[:60]
    expected = np.mean(
        [cal.predict_proba(m.predict_proba(X_new)[:, 1]) for m, cal in wrapped.ensemble_],
        axis=0,
    )
    np.testing.assert_allclose(wrapped.predict_proba(X_new), expected, atol=1e-12)


def test_cv_reproducible() -> None:
    X, y = _data(1200)
    a = CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", random_state=3).fit(X, y)
    b = CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", random_state=3).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X[:50]), b.predict_proba(X[:50]))


def test_offset_to_target_mean() -> None:
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    target = 0.05
    result = wrapped.offset_to(target_mean=target)
    assert result is wrapped
    assert len(wrapped.offsets_) == 1
    np.testing.assert_allclose(wrapped.predict_proba(X_cal).mean(), target, atol=1e-9)


def test_offset_composes_in_interval_inverse_and_affine() -> None:
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    lo0, hi0 = wrapped.interval_inverse(0.02, 0.10, space="logit")
    a0, b0 = wrapped.affine_logit_coeffs_
    wrapped.offset_to(delta=0.4)
    lo1, hi1 = wrapped.interval_inverse(0.02, 0.10, space="logit")
    a1, b1 = wrapped.affine_logit_coeffs_
    np.testing.assert_allclose([lo1, hi1], [lo0 - 0.4 / a0, hi0 - 0.4 / a0], atol=1e-9)
    assert a1 == a0
    np.testing.assert_allclose(b1, b0 + 0.4, atol=1e-12)


def test_predict_proba_2d() -> None:
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    p2 = wrapped.predict_proba_2d(X_cal[:10])
    assert p2.shape == (10, 2)
    np.testing.assert_allclose(p2.sum(axis=1), 1.0)


def test_interpret_includes_offset_stage() -> None:
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    wrapped.offset_to(delta=-0.2)
    interp = wrapped.interpret()
    assert "delta" in interp.param_names
    assert "a" in interp.param_names


def test_invalid_flow_raises() -> None:
    with pytest.raises(ValueError, match="flow"):
        CalibratedModel(TinyLogit(), PlattCalibrator(), flow="nope").fit(*_data(100))


def test_export() -> None:
    import probcal

    assert "CalibratedModel" in probcal.__all__
