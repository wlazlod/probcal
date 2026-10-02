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


# ---------------------------------------------------------------- 0.4.0 fixes


class LookupModel:
    """String-feature model: X is a 1-D array of category labels."""

    def __init__(self) -> None:
        self.seen_types_: list[type] = []

    def fit(self, X, y):
        self.seen_types_.append(type(X))
        self.rates_ = {k: float(np.mean(y[np.asarray(X) == k])) for k in set(np.asarray(X))}
        return self

    def predict_proba(self, X):
        self.seen_types_.append(type(X))
        p = np.clip([self.rates_.get(x, 0.5) for x in np.asarray(X)], 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


class WeightedTinyLogit(TinyLogit):
    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = sample_weight
        return super().fit(X, y)


def _string_data(n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.choice(np.array(["low", "mid", "high"]), size=n)
    rate = {"low": 0.05, "mid": 0.2, "high": 0.5}
    y = (RNG.random(n) < np.array([rate[x] for x in X])).astype(float)
    return X, y


@pytest.mark.parametrize("bad", [1, 0, 2.5, True, "5"])
def test_cv_must_be_an_integer_at_least_two(bad) -> None:
    # OFF-6: cv=1 trained every fold model on an empty set.
    X, y = _data(300)
    with pytest.raises(ValueError, match="cv must be an integer >= 2"):
        CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=bad).fit(X, y)


@pytest.mark.parametrize("flow", ["prefit", "cv"])
def test_x_is_passed_to_the_model_untouched(flow) -> None:
    # OFF-7: X was forced to float64, breaking string features and DataFrames.
    X, y = _string_data()
    model = LookupModel().fit(X, y) if flow == "prefit" else LookupModel()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow=flow, cv=3).fit(X, y)
    p = wrapped.predict_proba(np.array(["low", "high"]))
    assert p[0] < p[1]


def test_dataframe_rows_are_subset_with_iloc() -> None:
    pd = pytest.importorskip("pandas")
    X, y = _data(600)
    df = pd.DataFrame(X, columns=["a", "b"], index=np.arange(600) * 7)

    class FrameLogit(TinyLogit):
        def fit(self, X, y):
            assert isinstance(X, pd.DataFrame)
            return super().fit(X.to_numpy(), y)

        def predict_proba(self, X):
            assert isinstance(X, pd.DataFrame)
            return super().predict_proba(X.to_numpy())

    wrapped = CalibratedModel(FrameLogit(), PlattCalibrator(), flow="cv", cv=3).fit(df, y)
    assert wrapped.predict_proba(df.iloc[:5]).shape == (5,)


def test_sample_weight_reaches_model_fit_in_cv_flow() -> None:
    # OFF-8: weights went to the calibrator only.
    X, y = _data(900)
    w = RNG.uniform(0.5, 2.0, 900)
    wrapped = CalibratedModel(WeightedTinyLogit(), PlattCalibrator(), flow="cv", cv=3).fit(
        X, y, sample_weight=w
    )
    np.testing.assert_array_equal(wrapped.model_.sample_weight_, w)
    unweighted = CalibratedModel(WeightedTinyLogit(), PlattCalibrator(), flow="cv", cv=3)
    assert unweighted.fit(X, y).model_.sample_weight_ is None


def test_model_without_sample_weight_warns_once() -> None:
    X, y = _data(900)
    w = RNG.uniform(0.5, 2.0, 900)
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=3).fit(X, y, sample_weight=w)
    msgs = [r for r in rec if "does not accept sample_weight" in str(r.message)]
    assert len(msgs) == 1


def test_is_monotone_before_fit_raises_runtime_error() -> None:
    # OFF-16: AttributeError on a missing ensemble_.
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = CalibratedModel(TinyLogit(), PlattCalibrator()).is_monotone_


def test_offset_to_positional_is_deprecated() -> None:
    # OFF-14: offset_to(target_mean, delta) vs LogitOffset(delta, target_mean).
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator()).fit(X_cal, y_cal)
    with pytest.warns(DeprecationWarning, match="removed in 0.5.0"):
        wrapped.offset_to(0.05)
    assert wrapped.offsets_[-1].target_mean == 0.05
    with pytest.warns(DeprecationWarning):
        wrapped.offset_to(None, 0.1)
    assert wrapped.offsets_[-1].delta == 0.1


def test_protocol_delegates_to_chain() -> None:
    # OFF-19: one implementation (Chain) behind the wrapper's inverse maps.
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator()).fit(X_cal, y_cal)
    wrapped.offset_to(delta=0.3)
    chain = wrapped.chain_
    assert wrapped.interval_inverse(0.02, 0.1, buffer_logit=0.1) == chain.interval_inverse(
        0.02, 0.1, buffer_logit=0.1
    )
    q = np.array([0.03, 0.2])
    np.testing.assert_array_equal(wrapped.point_inverse(q), chain.point_inverse(q))
    s = wrapped.point_inverse(q)
    np.testing.assert_allclose(chain.predict_proba(s), q, rtol=1e-10)
    assert wrapped.affine_logit_coeffs_ == chain.affine_logit_coeffs_
    assert wrapped.is_monotone_
    with pytest.raises(ValueError, match="space"):
        wrapped.interval_inverse(0.02, 0.1, space="raw")
    with pytest.raises(ValueError, match="buffer_logit"):
        wrapped.interval_inverse(0.02, 0.1, buffer_logit=-1.0)


def test_with_offset_appends_a_fitted_offset_to_a_copy() -> None:
    # OFF-25: a data-free way to compose an already-fitted offset.
    from probcal.offset import LogitOffset

    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator()).fit(X_cal, y_cal)
    off = LogitOffset(delta=0.4).fit(np.array([0.1, 0.2]))
    new = wrapped.with_offset(off)
    assert wrapped.offsets_ == [] and len(new.offsets_) == 1
    assert new.offsets_[0] is not off and new.offsets_[0].delta_ == 0.4
    np.testing.assert_allclose(
        new.predict_proba(X_cal[:20]), off.transform(wrapped.predict_proba(X_cal[:20]))
    )
    loaded = CalibratedModel.from_json(wrapped.to_json(), model=model)
    assert len(loaded.with_offset(off).offsets_) == 1  # no calibration scores needed
    with pytest.raises(TypeError):
        wrapped.with_offset(0.4)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="not fitted"):
        wrapped.with_offset(LogitOffset(delta=0.1))


def test_x_and_y_length_mismatch_raises() -> None:
    model, X_cal, y_cal = _trained_model()
    with pytest.raises(ValueError, match="rows"):
        CalibratedModel(model, PlattCalibrator()).fit(X_cal[:10], y_cal)
