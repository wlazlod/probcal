"""Tests for the probcal.sklearn adapters. Skipped without sklearn."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict  # noqa: E402

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal._math import expit, logit  # noqa: E402
from probcal.sklearn import CalibratedClassifier, SklearnCalibrator, SklearnOffset  # noqa: E402

_D = make_pd_portfolio(n=2000, random_state=11)
_Q = make_pd_portfolio(n=500, random_state=12)


def _feature_data(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    z = 1.4 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2] - 2.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    return X, y


def _manual_oof(X, y, splitter):
    """The documented protocol: OOF scores -> BetaCalibrator, final model refit on all data."""
    oof = cross_val_predict(
        LogisticRegression(max_iter=1000), X, y, cv=splitter, method="predict_proba"
    )[:, 1]
    cal = BetaCalibrator().fit(oof, y.astype(float))
    final = LogisticRegression(max_iter=1000).fit(X, y)
    return cal.predict_proba(final.predict_proba(X)[:, 1])


# ---------------------------------------------------------------- SklearnCalibrator


def test_sklearn_calibrator_matches_direct_beta_fit() -> None:
    est = SklearnCalibrator().fit(_D.scores.reshape(-1, 1), _D.y)
    direct = BetaCalibrator().fit(_D.scores, _D.y)
    proba = est.predict_proba(_Q.scores.reshape(-1, 1))
    assert proba.shape == (500, 2)
    np.testing.assert_array_equal(proba[:, 1], direct.predict_proba(_Q.scores))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(est.predict(_Q.scores.reshape(-1, 1)), (proba[:, 1] >= 0.5))
    col = est.transform(_Q.scores.reshape(-1, 1))
    assert col.shape == (500, 1)
    np.testing.assert_array_equal(col[:, 0], proba[:, 1])
    # flat X is the same single score column
    flat = SklearnCalibrator().fit(_D.scores, _D.y)
    np.testing.assert_array_equal(flat.predict_proba(_Q.scores), proba)


def test_two_column_matrix_equals_column_one() -> None:
    X1 = _D.scores.reshape(-1, 1)
    X2 = np.column_stack([1.0 - _D.scores, _D.scores])
    a = SklearnCalibrator().fit(X1, _D.y)
    b = SklearnCalibrator().fit(X2, _D.y)
    np.testing.assert_array_equal(
        a.predict_proba(_Q.scores.reshape(-1, 1)),
        b.predict_proba(np.column_stack([1.0 - _Q.scores, _Q.scores])),
    )


def test_orientation_warning_fires_exactly_once_on_a_reversed_matrix() -> None:
    import warnings

    Xr = np.column_stack([_D.scores, 1.0 - _D.scores])  # default column 1 is wrong
    with pytest.warns(UserWarning, match="positive_column=0") as rec:
        SklearnCalibrator().fit(Xr, _D.y)
    assert len(rec) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # never on a correct matrix
        SklearnCalibrator().fit(np.column_stack([1.0 - _D.scores, _D.scores]), _D.y)


def test_sklearn_calibrator_classes_and_string_labels() -> None:
    labels = np.where(_D.y == 1.0, "bad", "good")
    est = SklearnCalibrator().fit(_D.scores, labels)
    assert list(est.classes_) == ["bad", "good"]  # np.unique order
    assert set(est.predict(_Q.scores)) <= {"bad", "good"}


def test_sklearn_calibrator_logit_input_is_exact_expit() -> None:
    z = logit(_D.scores)
    est = SklearnCalibrator(input="logit").fit(z, _D.y)
    est_p = SklearnCalibrator().fit(expit(z), _D.y)
    zq = logit(_Q.scores)
    np.testing.assert_array_equal(est.predict_proba(zq)[:, 1], est_p.predict_proba(expit(zq))[:, 1])


def test_import_error_names_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard message must name the extra; simulate a missing sklearn.
    import importlib
    import sys

    import probcal.sklearn as mod

    with monkeypatch.context() as m:
        for name in [k for k in sys.modules if k == "sklearn" or k.startswith("sklearn.")]:
            m.delitem(sys.modules, name)
        m.setitem(sys.modules, "sklearn", None)  # forces ImportError on re-import
        with pytest.raises(ImportError, match=r"probcal\[sklearn\]"):
            importlib.reload(mod)
    importlib.reload(mod)  # restore for the rest of the session


# ---------------------------------------------------------------- CalibratedClassifier


def test_calibrated_classifier_matches_manual_oof_protocol() -> None:
    X, y = _feature_data(1500)
    clf = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=5, random_state=0).fit(X, y)
    expected = _manual_oof(X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=0))
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)
    assert list(clf.classes_) == [0, 1]


def test_stratify_false_uses_shuffled_kfold_with_random_state() -> None:
    # MON-1: stratify=False used to hand cv=int to cross_val_predict, which
    # stratifies anyway (StratifiedKFold, unshuffled) and ignores random_state.
    X, y = _feature_data()
    clf = CalibratedClassifier(
        LogisticRegression(max_iter=1000), cv=4, stratify=False, random_state=3
    ).fit(X, y)
    expected = _manual_oof(X, y, KFold(n_splits=4, shuffle=True, random_state=3))
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)


def test_cv_accepts_a_splitter_and_an_iterable_of_splits() -> None:
    # MON-3: cv=<splitter> used to raise TypeError from int(self.cv)
    X, y = _feature_data()
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    expected = _manual_oof(X, y, splitter)
    clf = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=splitter).fit(X, y)
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)
    splits = list(splitter.split(X, y))
    clf2 = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=splits).fit(X, y)
    np.testing.assert_allclose(clf2.predict_proba(X)[:, 1], expected, atol=1e-12)


def test_calibrated_classifier_prefit() -> None:
    X, y = _feature_data(1500)
    model = LogisticRegression(max_iter=1000).fit(X[:1000], y[:1000])
    clf = CalibratedClassifier(model, cv="prefit").fit(X[1000:], y[1000:])
    assert clf.estimator_ is model
    manual = BetaCalibrator().fit(model.predict_proba(X[1000:])[:, 1], y[1000:].astype(float))
    np.testing.assert_allclose(
        clf.predict_proba(X)[:, 1],
        manual.predict_proba(model.predict_proba(X)[:, 1]),
        atol=1e-12,
    )


def test_calibrated_classifier_decision_function_maps_through_expit() -> None:
    from sklearn.svm import LinearSVC

    X, y = _feature_data(800)
    svc = LinearSVC().fit(X, y)
    clf = CalibratedClassifier(svc, cv="prefit", method="decision_function").fit(X, y)
    manual = BetaCalibrator().fit(expit(svc.decision_function(X)), y.astype(float))
    np.testing.assert_allclose(
        clf.predict_proba(X)[:, 1],
        manual.predict_proba(expit(svc.decision_function(X))),
        atol=1e-12,
    )


def test_dataframe_with_string_column_reaches_a_column_transformer() -> None:
    # MON-2: X used to be forced to float64 first, which a string column cannot survive
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
    assert clf.predict_proba(X).shape == (900, 2)
    assert list(clf.feature_names_in_) == ["a", "b", "seg"]


def test_sparse_x_reaches_the_estimator() -> None:
    sp = pytest.importorskip("scipy.sparse")
    X, y = _feature_data(600)
    clf = CalibratedClassifier(cv=3, random_state=0).fit(sp.csr_matrix(X), y)
    dense = CalibratedClassifier(cv=3, random_state=0).fit(X, y)
    np.testing.assert_allclose(
        clf.predict_proba(sp.csr_matrix(X)), dense.predict_proba(X), atol=1e-8
    )


# ---------------------------------------------------------------- shared surface


@pytest.mark.parametrize("kind", ["calibrator", "classifier"])
def test_adapters_delegate_the_calibrator_protocol(kind: str) -> None:
    # MON-20: both adapters expose the probcal surface of calibrator_.
    if kind == "calibrator":
        est = SklearnCalibrator().fit(_D.scores, _D.y)
    else:
        est = CalibratedClassifier(random_state=1).fit(*_feature_data(1500))
    cal = est.calibrator_
    assert isinstance(cal, BetaCalibrator)
    assert est.is_monotone_ is cal.is_monotone_ is True
    assert est.affine_logit_coeffs_ is None  # abm beta is not affine
    lo, hi = est.interval_inverse(0.0, 0.02, space="logit", buffer_logit=0.1)
    assert (lo, hi) == cal.interval_inverse(0.0, 0.02, space="logit", buffer_logit=0.1)
    assert np.isneginf(lo) and np.isfinite(hi)
    s = est.point_inverse(np.array([0.3]))
    np.testing.assert_allclose(cal.predict_proba(s), [0.3], atol=1e-9)
    assert est.fingerprint() == cal.fingerprint()
    assert est.to_dict() == cal.to_dict()
    assert est.to_json() == cal.to_json()
    assert repr(est.interpret()) == repr(cal.interpret())


def test_protocol_surface_requires_a_fitted_adapter() -> None:
    for est in (SklearnCalibrator(), CalibratedClassifier()):
        with pytest.raises(NotFittedError):
            est.fingerprint()
        with pytest.raises(NotFittedError):
            _ = est.is_monotone_


# ---------------------------------------------------------------- validation


def _negative_weights(n):
    w = np.ones(n)
    w[3] = -1.0
    return w


_X, _Y = _feature_data(300)
_Y3 = _Y.copy()
_Y3[:20] = 2
_D3 = _D.y.copy()
_D3[:10] = 2.0


@pytest.mark.parametrize(
    ("fit", "match"),
    [
        (lambda: SklearnCalibrator().fit(np.column_stack([_D.scores] * 3), _D.y), "one score"),
        (
            lambda: SklearnCalibrator(input="logit").fit(
                np.column_stack([logit(_D.scores), -logit(_D.scores)]), _D.y
            ),
            "one score column",
        ),
        (lambda: SklearnCalibrator().fit(_D.scores, _D3), "binary"),
        (lambda: CalibratedClassifier().fit(_X, _Y3), "binary"),
        (lambda: CalibratedClassifier(cv=1).fit(_X, _Y), None),
        (lambda: CalibratedClassifier(cv=0).fit(_X, _Y), None),
        (lambda: CalibratedClassifier(cv=True).fit(_X, _Y), None),
        # MON-4: negative weights used to be silently dropped with the zero weights
        (
            lambda: SklearnCalibrator().fit(_D.scores, _D.y, sample_weight=_negative_weights(2000)),
            "non-negative",
        ),
        (
            lambda: SklearnOffset(delta=0.1).fit(_D.scores, sample_weight=_negative_weights(2000)),
            "non-negative",
        ),
        (
            lambda: CalibratedClassifier(cv=3).fit(_X, _Y, sample_weight=_negative_weights(300)),
            "non-negative",
        ),
        # MON-12: a prefit estimator must have been fitted on the same classes
        (
            lambda: CalibratedClassifier(
                LogisticRegression(max_iter=1000).fit(_X, _Y), cv="prefit"
            ).fit(_X, np.where(_Y == 1, "bad", "good")),
            "classes",
        ),
    ],
)
def test_adapter_validation(fit, match) -> None:
    with pytest.raises(ValueError, match=match):
        fit()


# ---------------------------------------------------------------- estimator checks


def _compliance_estimators():
    from probcal.sklearn import CalibratedClassifier, SklearnOffset

    # SklearnCalibrator runs the suite on the logit scale: sklearn's generic
    # checks feed unbounded Gaussian features, which are legal logits but not
    # legal probabilities (the domain restriction sklearn special-cases its
    # own IsotonicRegression for). The default stays input="probability".
    return [SklearnCalibrator(input="logit"), CalibratedClassifier(), SklearnOffset(delta=0.1)]


# Expected-failure tables live in probcal.sklearn._xfail (one source of
# truth for the >=1.6 expected_failed_checks mechanism and the <1.6
# _xfail_checks tag).
from probcal.sklearn._xfail import (  # noqa: E402
    CALIBRATOR_XFAIL_CHECKS,
    CLASSIFIER_XFAIL_CHECKS,
    OFFSET_XFAIL_CHECKS,
)


def _expected_failures(estimator) -> dict:
    from probcal.sklearn import SklearnOffset

    if isinstance(estimator, SklearnCalibrator):
        return dict(CALIBRATOR_XFAIL_CHECKS)
    if isinstance(estimator, SklearnOffset):
        return dict(OFFSET_XFAIL_CHECKS)
    return dict(CLASSIFIER_XFAIL_CHECKS)


from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

try:
    _checks_decorator = parametrize_with_checks(
        _compliance_estimators(), expected_failed_checks=_expected_failures
    )
except TypeError:  # pragma: no cover - sklearn < 1.6 reads _xfail_checks tags instead
    _checks_decorator = parametrize_with_checks(_compliance_estimators())

# parametrize_with_checks hands pytest a generator of (estimator, check) pairs,
# which pytest deprecates (PytestRemovedIn10Warning); materialise it as a list.
_checks_argnames, _checks_argvalues = _checks_decorator.args
_checks_decorator = pytest.mark.parametrize(
    _checks_argnames, list(_checks_argvalues), **_checks_decorator.kwargs
)


@_checks_decorator
def test_sklearn_estimator_checks(estimator, check) -> None:
    check(estimator)
