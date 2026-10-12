"""Regression tests for the 0.3.5 wrapper, selector, clone and attribution fixes."""

import json
import warnings

import numpy as np
import pytest

from probcal import CalibratedModel, CalibratorSelector, Chain, LogitOffset, SegmentedCalibrator
from probcal.attribution import adjust_attributions
from probcal.base import clone_unfitted
from probcal.isotonic import IsotonicCalibrator
from probcal.parametric import BetaCalibrator, PlattCalibrator

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from probcal.sklearn import CalibratedClassifier, SklearnCalibrator  # noqa: E402


def _data(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0)))).astype(int)
    return X, y


# ------------------------------------------------------------------ sample_weight routing


def test_pipeline_without_routing_warns_and_fits_unweighted() -> None:
    X, y = _data()
    w = np.random.default_rng(1).uniform(0.5, 2.0, len(y))
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    with pytest.warns(UserWarning, match="does not accept sample_weight"):
        m = CalibratedModel(pipe, PlattCalibrator(), flow="cv", cv=3).fit(X, y, sample_weight=w)
    ref = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    np.testing.assert_allclose(m.model_.predict_proba(X), ref.predict_proba(X))


def test_var_keyword_fit_is_not_taken_as_accepting_weights() -> None:
    class KwargsModel:
        def fit(self, X, y, **kwargs):
            if kwargs:
                raise TypeError(f"unexpected {sorted(kwargs)}")
            self.lr_ = LogisticRegression().fit(X, y)
            return self

        def predict_proba(self, X):
            return self.lr_.predict_proba(X)

        def get_params(self, deep: bool = True):
            return {}

    X, y = _data()
    w = np.ones(len(y))
    with pytest.warns(UserWarning, match="does not accept sample_weight"):
        CalibratedModel(KwargsModel(), PlattCalibrator(), flow="cv", cv=3).fit(
            X, y, sample_weight=w
        )


# ------------------------------------------------------------------ offset_to weights


def test_offset_to_without_x_uses_calibration_weights() -> None:
    X, y = _data()
    w = np.where(y == 1, 5.0, 1.0)
    lr = LogisticRegression().fit(X, y)
    m = CalibratedModel(lr, PlattCalibrator()).fit(X, y, sample_weight=w)
    m.offset_to(target_mean=0.2)
    assert np.average(m.predict_proba(X), weights=w) == pytest.approx(0.2, abs=1e-9)


def test_offset_to_unweighted_fit_unchanged() -> None:
    X, y = _data()
    lr = LogisticRegression().fit(X, y)
    m = CalibratedModel(lr, PlattCalibrator()).fit(X, y).offset_to(target_mean=0.2)
    assert m.predict_proba(X).mean() == pytest.approx(0.2, abs=1e-9)


def test_offset_to_with_x_accepts_sample_weight() -> None:
    X, y = _data()
    w = np.random.default_rng(3).uniform(0.5, 3.0, len(y))
    lr = LogisticRegression().fit(X, y)
    m = CalibratedModel(lr, PlattCalibrator()).fit(X, y)
    m2 = CalibratedModel.from_json(m.to_json(), model=lr)
    m2.offset_to(target_mean=0.3, X=X, sample_weight=w)
    assert np.average(m2.predict_proba(X), weights=w) == pytest.approx(0.3, abs=1e-9)
    with pytest.raises(ValueError, match="only accepted together with X"):
        m.offset_to(target_mean=0.3, sample_weight=w)


# ------------------------------------------------------------------ model_ref on reload


def test_reload_without_model_keeps_model_ref_and_fingerprint() -> None:
    X, y = _data(300)
    lr = LogisticRegression(C=0.5).fit(X, y)
    m = CalibratedModel(lr, PlattCalibrator(), model_id="pd-v1").fit(X, y)
    d1 = m.to_dict()
    m2 = CalibratedModel.from_dict(json.loads(json.dumps(d1)))
    assert m2.to_dict()["state"]["model_ref"] == d1["state"]["model_ref"]
    assert m2.fingerprint() == m.fingerprint()


# ------------------------------------------------------------------ prefit classes_


def test_prefit_string_label_model_raises() -> None:
    X, y = _data()
    lr = LogisticRegression().fit(X, np.where(y == 1, "bad", "good"))
    with pytest.raises(ValueError, match=r"trained on classes \['bad', 'good'\]"):
        CalibratedModel(lr, PlattCalibrator()).fit(X, y)


@pytest.mark.parametrize("labels", [(0, 1), (0.0, 1.0), (False, True)])
def test_prefit_zero_one_labels_accepted(labels: tuple) -> None:
    X, y = _data()
    lr = LogisticRegression().fit(X, np.where(y == 1, labels[1], labels[0]))
    m = CalibratedModel(lr, PlattCalibrator()).fit(X, y)
    assert m.predict_proba(X).shape == (len(y),)


# ------------------------------------------------------------------ cv vs class counts


def test_selector_with_fewer_events_than_folds_raises_clearly() -> None:
    rng = np.random.default_rng(1)
    y = np.zeros(200, int)
    y[:4] = 1
    s = rng.random(200)
    with pytest.raises(ValueError, match=r"cv=5 needs at least 5 .* 4 positives .* cv <= 4"):
        CalibratorSelector(cv=5).fit(s, y)
    CalibratorSelector(candidates={"platt": PlattCalibrator()}, cv=4).fit(s, y)


def test_ensemble_with_fewer_events_than_folds_raises_clearly() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200, int)
    y[:3] = 1
    with pytest.raises(ValueError, match="cv <= 3"):
        CalibratedModel(
            LogisticRegression(), PlattCalibrator(), flow="cv", cv=5, ensemble=True
        ).fit(X, y)
    # The pooled flow only needs both classes in every training fold.
    CalibratedModel(LogisticRegression(), PlattCalibrator(), flow="cv", cv=5).fit(X, y)


# ------------------------------------------------------------------ Chain cloning


def test_clone_unfitted_chain_is_deep_and_fresh() -> None:
    proto = Chain([PlattCalibrator(), LogitOffset(delta=0.3)])
    c = clone_unfitted(proto)
    assert isinstance(c, Chain)
    assert c.stages is not proto.stages
    assert all(a is not b for a, b in zip(c.stages, proto.stages, strict=True))
    assert c.stages[1].delta == 0.3
    assert not c.fitted_


def test_chain_as_cv_calibrator_selector_candidate_and_segmented_base() -> None:
    X, y = _data()
    proto = Chain([PlattCalibrator(), LogitOffset(delta=0.3)])
    m = CalibratedModel(LogisticRegression(), proto, flow="cv", cv=3, ensemble=True).fit(X, y)
    cals = [c for _, c in m.ensemble_]
    assert cals[0] is not cals[1] and cals[0].stages[0] is not cals[1].stages[0]
    assert not proto.stages[0].fitted_

    s = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    sel = CalibratorSelector(candidates={"chain": proto, "beta": BetaCalibrator()}, cv=3).fit(s, y)
    assert np.all(np.isfinite(sel.predict_proba(s)))
    seg = np.random.default_rng(0).choice(["a", "b"], len(y))
    sc = SegmentedCalibrator(base=proto).fit(s, y, segments=seg)
    assert np.all(np.isfinite(sc.predict_proba(s, segments=seg)))
    assert not proto.stages[0].fitted_


# ------------------------------------------------------------------ attribution unwrapping


def test_adjust_attributions_unwraps_sklearn_adapters() -> None:
    X, y = _data(400, seed=1)
    lr = LogisticRegression().fit(X, y)
    s = lr.predict_proba(X)[:, 1]
    phi = X * lr.coef_[0]
    base = np.full(len(y), lr.intercept_[0])

    iso = IsotonicCalibrator().fit(s, y)
    ref = adjust_attributions(phi, base, iso)
    skc = SklearnCalibrator(IsotonicCalibrator()).fit(s, y)
    got = adjust_attributions(phi, base, skc)
    np.testing.assert_allclose(got.target, ref.target)
    np.testing.assert_allclose(got.phi_adj, ref.phi_adj)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = CalibratedClassifier(lr, calibrator=PlattCalibrator(), cv="prefit").fit(X, y)
    r = adjust_attributions(phi, base, cc)
    assert r.method_used == "affine-exact"
    from probcal._math import logit

    np.testing.assert_allclose(r.target, logit(cc.predict_proba(X)[:, 1]), atol=1e-8)
