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
