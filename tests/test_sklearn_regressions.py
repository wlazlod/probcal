"""Regression tests for the 0.3.4 sklearn-adapter fixes (MON-* review items)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    KFold,
    StratifiedKFold,
    cross_val_predict,
)

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal.sklearn import CalibratedClassifier, SklearnCalibrator, SklearnOffset  # noqa: E402


def _feature_data(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    z = 1.4 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2] - 2.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    return X, y


def _manual(X, y, splitter):
    oof = cross_val_predict(
        LogisticRegression(max_iter=1000), X, y, cv=splitter, method="predict_proba"
    )[:, 1]
    cal = BetaCalibrator().fit(oof, y.astype(float))
    final = LogisticRegression(max_iter=1000).fit(X, y)
    return cal.predict_proba(final.predict_proba(X)[:, 1])


# ---------------------------------------------------------------- MON-1: stratify / random_state


def test_stratify_false_uses_shuffled_kfold_with_random_state() -> None:
    # stratify=False used to hand cv=int to cross_val_predict, which
    # stratifies anyway (StratifiedKFold, unshuffled) and ignores random_state.
    X, y = _feature_data()
    clf = CalibratedClassifier(
        LogisticRegression(max_iter=1000), cv=4, stratify=False, random_state=3
    ).fit(X, y)
    expected = _manual(X, y, KFold(n_splits=4, shuffle=True, random_state=3))
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)


def test_stratify_false_honours_random_state() -> None:
    X, y = _feature_data()
    a = CalibratedClassifier(cv=4, stratify=False, random_state=1).fit(X, y)
    b = CalibratedClassifier(cv=4, stratify=False, random_state=2).fit(X, y)
    assert not np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_stratify_true_keeps_the_shuffled_stratified_folds() -> None:
    X, y = _feature_data()
    clf = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=4, random_state=7).fit(X, y)
    expected = _manual(X, y, StratifiedKFold(n_splits=4, shuffle=True, random_state=7))
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)


# ---------------------------------------------------------------- MON-3: splitter cv


def test_cv_accepts_a_splitter_and_an_iterable_of_splits() -> None:
    # cv=<splitter> used to raise TypeError from int(self.cv)
    X, y = _feature_data()
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    expected = _manual(X, y, splitter)
    clf = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=splitter).fit(X, y)
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)
    splits = list(splitter.split(X, y))
    clf2 = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=splits).fit(X, y)
    np.testing.assert_allclose(clf2.predict_proba(X)[:, 1], expected, atol=1e-12)


@pytest.mark.parametrize("cv", [1, 0, True])
def test_integer_cv_below_two_raises(cv) -> None:
    X, y = _feature_data(300)
    with pytest.raises(ValueError):
        CalibratedClassifier(cv=cv).fit(X, y)


# ---------------------------------------------------------------- MON-2: X passes through


def test_dataframe_with_string_column_reaches_a_column_transformer() -> None:
    # X used to be forced to float64 first, which a string column cannot survive
    pd = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    X_num, y = _feature_data(900)
    X = pd.DataFrame({"a": X_num[:, 0], "b": X_num[:, 1], "seg": np.where(y == 1, "x", "y")})
    X.loc[::3, "seg"] = "z"
    pipe = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(), ["seg"])], remainder="passthrough"),
        LogisticRegression(max_iter=1000),
    )
    clf = CalibratedClassifier(pipe, cv=3, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (900, 2)
    assert list(clf.feature_names_in_) == ["a", "b", "seg"]


def test_sparse_x_reaches_the_estimator() -> None:
    sp = pytest.importorskip("scipy.sparse")
    X, y = _feature_data(600)
    clf = CalibratedClassifier(cv=3, random_state=0).fit(sp.csr_matrix(X), y)
    dense = CalibratedClassifier(cv=3, random_state=0).fit(X, y)
    np.testing.assert_allclose(
        clf.predict_proba(sp.csr_matrix(X)), dense.predict_proba(X), atol=1e-8
    )


# ---------------------------------------------------------------- MON-4: negative weights


def _negative_weights(n):
    w = np.ones(n)
    w[3] = -1.0
    return w


def test_negative_sample_weight_raises_in_every_adapter() -> None:
    # negative weights used to be silently dropped with the zero weights
    d = make_pd_portfolio(n=400, random_state=1)
    w = _negative_weights(400)
    with pytest.raises(ValueError, match="non-negative"):
        SklearnCalibrator().fit(d.scores, d.y, sample_weight=w)
    with pytest.raises(ValueError, match="non-negative"):
        SklearnOffset(delta=0.1).fit(d.scores, sample_weight=w)
    X, y = _feature_data(400)
    with pytest.raises(ValueError, match="non-negative"):
        CalibratedClassifier(cv=3).fit(X, y, sample_weight=w)


def test_zero_weights_are_still_dropped() -> None:
    d = make_pd_portfolio(n=400, random_state=2)
    w = np.ones(400)
    w[:50] = 0.0
    a = SklearnCalibrator().fit(d.scores, d.y, sample_weight=w)
    b = SklearnCalibrator().fit(d.scores[50:], d.y[50:])
    np.testing.assert_array_equal(a.predict_proba(d.scores), b.predict_proba(d.scores))


# ---------------------------------------------------------------- MON-12: prefit classes


def test_prefit_estimator_with_other_classes_raises() -> None:
    X, y = _feature_data(600)
    model = LogisticRegression(max_iter=1000).fit(X, y)  # classes [0, 1]
    with pytest.raises(ValueError, match="classes"):
        CalibratedClassifier(model, cv="prefit").fit(X, np.where(y == 1, "bad", "good"))


def test_prefit_estimator_with_matching_string_classes_fits() -> None:
    X, y = _feature_data(600)
    labels = np.where(y == 1, "bad", "good")
    model = LogisticRegression(max_iter=1000).fit(X, labels)
    clf = CalibratedClassifier(model, cv="prefit").fit(X, labels)
    assert list(clf.classes_) == ["bad", "good"]


# ---------------------------------------------------------------- MON-20: shared surface


def test_sklearn_calibrator_exposes_the_classifier_protocol_surface() -> None:
    d = make_pd_portfolio(n=1500, random_state=4)
    est = SklearnCalibrator().fit(d.scores, d.y)
    clf_like = est.calibrator_
    assert est.is_monotone_ is clf_like.is_monotone_
    assert est.affine_logit_coeffs_ == clf_like.affine_logit_coeffs_
    assert est.interval_inverse(0.0, 0.05) == clf_like.interval_inverse(0.0, 0.05)
    np.testing.assert_array_equal(
        est.point_inverse(np.array([0.1])), clf_like.point_inverse(np.array([0.1]))
    )
    assert est.fingerprint() == clf_like.fingerprint()
    assert est.to_dict() == clf_like.to_dict()
    assert est.to_json() == clf_like.to_json()
    assert repr(est.interpret()) == repr(clf_like.interpret())


def test_protocol_surface_requires_a_fitted_adapter() -> None:
    from sklearn.exceptions import NotFittedError

    for est in (SklearnCalibrator(), CalibratedClassifier()):
        with pytest.raises(NotFittedError):
            est.fingerprint()
        with pytest.raises(NotFittedError):
            _ = est.is_monotone_
