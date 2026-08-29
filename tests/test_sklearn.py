"""Tests for the probcal.sklearn adapter (spec W6). Skipped without sklearn."""

import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal.sklearn import SklearnCalibrator  # noqa: E402

_D = make_pd_portfolio(n=2000, random_state=11)
_Q = make_pd_portfolio(n=500, random_state=12)


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


def test_sklearn_calibrator_accepts_flat_and_column_x() -> None:
    a = SklearnCalibrator().fit(_D.scores, _D.y)
    b = SklearnCalibrator().fit(_D.scores.reshape(-1, 1), _D.y)
    np.testing.assert_array_equal(
        a.predict_proba(_Q.scores), b.predict_proba(_Q.scores.reshape(-1, 1))
    )


def test_sklearn_calibrator_rejects_multicolumn_x() -> None:
    X = np.column_stack([_D.scores, _D.scores])
    with pytest.raises(ValueError, match="score-level"):
        SklearnCalibrator().fit(X, _D.y)


def test_sklearn_calibrator_classes_and_string_labels() -> None:
    labels = np.where(_D.y == 1.0, "bad", "good")
    est = SklearnCalibrator().fit(_D.scores, labels)
    assert list(est.classes_) == ["bad", "good"]  # np.unique order
    pred = est.predict(_Q.scores)
    assert set(pred) <= {"bad", "good"}


def test_sklearn_calibrator_exposes_probcal_object() -> None:
    est = SklearnCalibrator().fit(_D.scores, _D.y)
    cal = est.calibrator_
    assert isinstance(cal, BetaCalibrator)
    assert "a =" in repr(cal.interpret())
    lo, hi = cal.interval_inverse(0.0, 0.02, space="logit")
    assert np.isneginf(lo) and np.isfinite(hi)
    assert cal.to_dict()["class"] == "BetaCalibrator"
    assert len(cal.fingerprint()) == 64


def test_sklearn_calibrator_nested_params_and_clone() -> None:
    est = SklearnCalibrator(BetaCalibrator())
    assert est.get_params(deep=True)["calibrator__variant"] == "abm"
    est.set_params(calibrator__variant="ab")
    assert est.calibrator.variant == "ab"
    est2 = clone(est).fit(_D.scores, _D.y)
    assert est2.calibrator_.variant == "ab"


def test_sklearn_calibrator_logit_input_is_exact_expit() -> None:
    from probcal._math import expit, logit

    z = logit(_D.scores)
    est = SklearnCalibrator(input="logit").fit(z, _D.y)
    est_p = SklearnCalibrator().fit(expit(z), _D.y)
    zq = logit(_Q.scores)
    np.testing.assert_array_equal(est.predict_proba(zq)[:, 1], est_p.predict_proba(expit(zq))[:, 1])


def test_sklearn_calibrator_multiclass_raises() -> None:
    y3 = _D.y.copy()
    y3[:10] = 2.0
    with pytest.raises(ValueError, match="binary"):
        SklearnCalibrator().fit(_D.scores, y3)


def test_sklearn_calibrator_pickles() -> None:
    est = SklearnCalibrator().fit(_D.scores, _D.y)
    est2 = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(est.predict_proba(_Q.scores), est2.predict_proba(_Q.scores))


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


def _feature_data(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    z = 1.4 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2] - 2.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    return X, y


def test_calibrated_classifier_matches_manual_oof_protocol() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data()
    clf = CalibratedClassifier(LogisticRegression(max_iter=1000), cv=5, random_state=0).fit(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(
        LogisticRegression(max_iter=1000), X, y, cv=skf, method="predict_proba"
    )[:, 1]
    manual = BetaCalibrator().fit(oof, y.astype(float))
    final = LogisticRegression(max_iter=1000).fit(X, y)
    expected = manual.predict_proba(final.predict_proba(X)[:, 1])
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expected, atol=1e-12)
    assert list(clf.classes_) == [0, 1]


def test_calibrated_classifier_prefit() -> None:
    from sklearn.linear_model import LogisticRegression

    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data()
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

    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data(800)
    svc = LinearSVC().fit(X, y)
    clf = CalibratedClassifier(svc, cv="prefit", method="decision_function").fit(X, y)
    from probcal._math import expit as _expit

    manual = BetaCalibrator().fit(_expit(svc.decision_function(X)), y.astype(float))
    np.testing.assert_allclose(
        clf.predict_proba(X)[:, 1],
        manual.predict_proba(_expit(svc.decision_function(X))),
        atol=1e-12,
    )


def test_calibrated_classifier_multiclass_raises() -> None:
    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data(300)
    y3 = y.copy()
    y3[:20] = 2
    with pytest.raises(ValueError, match="binary"):
        CalibratedClassifier().fit(X, y3)


def test_calibrated_classifier_exposes_calibrator_protocol() -> None:
    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data()
    clf = CalibratedClassifier(random_state=1).fit(X, y)
    assert clf.is_monotone_ is True
    lo, hi = clf.interval_inverse(0.0, 0.02, space="logit", buffer_logit=0.1)
    assert np.isneginf(lo) and np.isfinite(hi)
    s = clf.point_inverse(np.array([0.3]))
    np.testing.assert_allclose(clf.calibrator_.predict_proba(s), [0.3], atol=1e-9)
    assert clf.affine_logit_coeffs_ is None  # abm beta is not affine
    assert len(clf.fingerprint()) == 64
    assert clf.to_dict()["class"] == "BetaCalibrator"
    assert "a =" in repr(clf.interpret())


def test_calibrated_classifier_to_json_delegates_to_calibrator() -> None:
    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data()
    clf = CalibratedClassifier(random_state=1).fit(X, y)
    assert clf.to_json() == clf.calibrator_.to_json()
    assert clf.to_dict() == clf.calibrator_.to_dict()


def test_calibrated_classifier_gridsearch_over_calibrator_variant() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data(600)
    gs = GridSearchCV(
        CalibratedClassifier(LogisticRegression(max_iter=1000), calibrator=BetaCalibrator()),
        {"calibrator__variant": ["ab", "abm"]},
        cv=3,
        scoring="neg_log_loss",
    ).fit(X, y)
    assert gs.best_params_["calibrator__variant"] in ("ab", "abm")


def test_calibrated_classifier_clone_and_pickle() -> None:
    from probcal.sklearn import CalibratedClassifier

    X, y = _feature_data(600)
    clf = CalibratedClassifier(random_state=3).fit(X, y)
    clf2 = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(clf.predict_proba(X), clf2.predict_proba(X))
    clone(clf).fit(X, y)  # clone-safe


# ---------------------------------------------------------------- estimator checks


def _compliance_estimators():
    from probcal.sklearn import CalibratedClassifier

    # SklearnCalibrator runs the suite on the logit scale: sklearn's generic
    # checks feed unbounded Gaussian features, which are legal logits but not
    # legal probabilities (the domain restriction sklearn special-cases its
    # own IsotonicRegression for). The default stays input="probability".
    return [SklearnCalibrator(input="logit"), CalibratedClassifier()]


# Expected-failure tables live in probcal.sklearn._compat (one source of
# truth for the >=1.6 expected_failed_checks mechanism and the <1.6
# _xfail_checks tag).
from probcal.sklearn._compat import (  # noqa: E402
    CALIBRATOR_XFAIL_CHECKS,
    CLASSIFIER_XFAIL_CHECKS,
)


def _expected_failures(estimator) -> dict:
    if isinstance(estimator, SklearnCalibrator):
        return dict(CALIBRATOR_XFAIL_CHECKS)
    return dict(CLASSIFIER_XFAIL_CHECKS)


from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

try:
    _checks_decorator = parametrize_with_checks(
        _compliance_estimators(), expected_failed_checks=_expected_failures
    )
except TypeError:  # pragma: no cover - sklearn < 1.6 reads _xfail_checks tags instead
    _checks_decorator = parametrize_with_checks(_compliance_estimators())


@_checks_decorator
def test_sklearn_estimator_checks(estimator, check) -> None:
    check(estimator)
