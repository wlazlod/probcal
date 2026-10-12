"""Tests for probcal.wrapper.CalibratedModel."""

import json
import warnings

import numpy as np
import pytest

from probcal._math import expit, irls_logistic
from probcal.offset import LogitOffset
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


class WeightedTinyLogit(TinyLogit):
    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = sample_weight
        return super().fit(X, y)


class KwargsTinyLogit(TinyLogit):
    """fit(**kwargs) must not be taken as accepting sample_weight."""

    def fit(self, X, y, **kwargs):
        if kwargs:
            raise TypeError(f"unexpected {sorted(kwargs)}")
        return super().fit(X, y)


class MarginOnly:
    """Model exposing only decision_function."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarginOnly":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] - 1.0


class LookupModel:
    """String-feature model: X is a 1-D array of category labels."""

    def fit(self, X, y):
        self.rates_ = {k: float(np.mean(y[np.asarray(X) == k])) for k in set(np.asarray(X))}
        return self

    def predict_proba(self, X):
        p = np.clip([self.rates_.get(x, 0.5) for x in np.asarray(X)], 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


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


def _string_data(n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.choice(np.array(["low", "mid", "high"]), size=n)
    rate = {"low": 0.05, "mid": 0.2, "high": 0.5}
    y = (RNG.random(n) < np.array([rate[x] for x in X])).astype(float)
    return X, y


# ------------------------------------------------------------------ flows


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
    X_cal, y_cal = _data(1500)
    wrapped = CalibratedModel(MarginOnly(), PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    p = wrapped.predict_proba(X_cal[:50])
    assert p.shape == (50,)
    assert np.all((p > 0) & (p < 1))


def test_cv_pooled_flow_refits_the_model_on_all_rows() -> None:
    X, y = _data(2500)
    wrapped = CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=5).fit(X, y)
    assert wrapped.model_.n_fit_rows_ == 2500
    assert wrapped.calibrator_.fitted_


def test_cv_ensemble_flow_averages_fold_pipelines() -> None:
    X, y = _data(2500)
    wrapped = CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=4, ensemble=True).fit(
        X, y
    )
    assert len(wrapped.ensemble_) == 4
    X_new = X[:60]
    expected = np.mean(
        [cal.predict_proba(m.predict_proba(X_new)[:, 1]) for m, cal in wrapped.ensemble_],
        axis=0,
    )
    np.testing.assert_allclose(wrapped.predict_proba(X_new), expected, atol=1e-12)


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


def test_prefit_model_must_be_trained_on_zero_one_labels() -> None:
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    X, y = _data(600)
    for labels in [(0, 1), (0.0, 1.0), (False, True)]:
        lr = sklearn_lm.LogisticRegression().fit(X, np.where(y == 1, labels[1], labels[0]))
        CalibratedModel(lr, PlattCalibrator()).fit(X, y)
    lr = sklearn_lm.LogisticRegression().fit(X, np.where(y == 1, "bad", "good"))
    with pytest.raises(ValueError, match=r"trained on classes \['bad', 'good'\]"):
        CalibratedModel(lr, PlattCalibrator()).fit(X, y)


# ------------------------------------------------------------------ validation


def _few_events() -> tuple[np.ndarray, np.ndarray]:
    X = np.random.default_rng(1).normal(size=(200, 2))
    y = np.zeros(200)
    y[:3] = 1
    return X, y


@pytest.mark.parametrize(
    ("make", "exc", "match"),
    [
        (lambda: CalibratedModel(TinyLogit(), PlattCalibrator(), flow="nope"), ValueError, "flow"),
        # OFF-6: cv=1 trained every fold model on an empty set.
        *[
            (
                lambda bad=bad: CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=bad),
                ValueError,
                "cv must be an integer >= 2",
            )
            for bad in (1, 2.5, True, "5")
        ],
        (
            lambda: CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=5, ensemble=True),
            ValueError,
            "cv <= 3",
        ),
    ],
    ids=["flow", "cv=1", "cv=2.5", "cv=True", "cv='5'", "ensemble-few-events"],
)
def test_fit_validation(make, exc, match) -> None:
    with pytest.raises(exc, match=match):
        make().fit(*_few_events())


def test_misc_validation() -> None:
    model, X_cal, y_cal = _trained_model()
    with pytest.raises(ValueError, match="rows"):
        CalibratedModel(model, PlattCalibrator()).fit(X_cal[:10], y_cal)
    # OFF-16: AttributeError on a missing ensemble_.
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = CalibratedModel(TinyLogit(), PlattCalibrator()).is_monotone_
    # The pooled flow only needs both classes in every training fold.
    CalibratedModel(TinyLogit(), PlattCalibrator(), flow="cv", cv=5).fit(*_few_events())


# ------------------------------------------------------------------ sample_weight routing


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


@pytest.mark.parametrize("model_cls", [TinyLogit, KwargsTinyLogit])
def test_model_without_sample_weight_warns_once_and_fits_unweighted(model_cls) -> None:
    X, y = _data(900)
    w = RNG.uniform(0.5, 2.0, 900)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = CalibratedModel(model_cls(), PlattCalibrator(), flow="cv", cv=3).fit(
            X, y, sample_weight=w
        )
    msgs = [r for r in rec if "does not accept sample_weight" in str(r.message)]
    assert len(msgs) == 1
    np.testing.assert_allclose(m.model_.beta_, TinyLogit().fit(X, y).beta_)


# ------------------------------------------------------------------ offsets


def test_offset_to_target_mean_uses_calibration_weights() -> None:
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
    assert wrapped.offset_to(target_mean=0.05) is wrapped
    assert len(wrapped.offsets_) == 1
    np.testing.assert_allclose(wrapped.predict_proba(X_cal).mean(), 0.05, atol=1e-9)

    w = np.where(y_cal == 1, 5.0, 1.0)
    weighted = CalibratedModel(model, PlattCalibrator()).fit(X_cal, y_cal, sample_weight=w)
    weighted.offset_to(target_mean=0.2)
    assert np.average(weighted.predict_proba(X_cal), weights=w) == pytest.approx(0.2, abs=1e-9)


def test_offset_to_with_x_accepts_sample_weight() -> None:
    model, X_cal, y_cal = _trained_model()
    w = RNG.uniform(0.5, 3.0, len(y_cal))
    m = CalibratedModel(model, PlattCalibrator()).fit(X_cal, y_cal)
    m2 = CalibratedModel.from_json(m.to_json(), model=model)
    m2.offset_to(target_mean=0.3, X=X_cal, sample_weight=w)
    assert np.average(m2.predict_proba(X_cal), weights=w) == pytest.approx(0.3, abs=1e-9)
    with pytest.raises(ValueError, match="only accepted together with X"):
        m.offset_to(target_mean=0.3, sample_weight=w)


def test_offset_to_positional_is_deprecated() -> None:
    # OFF-14: offset_to(target_mean, delta) vs LogitOffset(delta, target_mean).
    model, X_cal, y_cal = _trained_model()
    wrapped = CalibratedModel(model, PlattCalibrator()).fit(X_cal, y_cal)
    with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
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


def test_reload_without_model_keeps_model_ref_and_fingerprint() -> None:
    model, X_cal, y_cal = _trained_model()
    m = CalibratedModel(model, PlattCalibrator(), model_id="pd-v1").fit(X_cal, y_cal)
    d1 = m.to_dict()
    m2 = CalibratedModel.from_dict(json.loads(json.dumps(d1)))
    assert m2.to_dict()["state"]["model_ref"] == d1["state"]["model_ref"]
    assert m2.fingerprint() == m.fingerprint()
