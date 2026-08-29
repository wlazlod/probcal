"""Runnable snippets embedded in ``docs/guide/sklearn.md`` (spec K5).

Every code block in the guide's three-tier story is included from this file
via ``pymdownx.snippets`` (``--8<-- "tests/test_sklearn_guide_snippets.py:NAME"``),
never pasted, so the docs cannot drift from what is actually tested. Each
test runs the same ``# --8<-- [start:NAME]`` / ``[end:NAME]`` block the docs
include, then adds assertions below the closing marker (not shown in the
docs) to pin the claim.

Requires sklearn >= 1.6: the bare-duck tier (``clone``, ``get_tags``,
``check_is_fitted``, no adapter import) and the ``FrozenEstimator`` prefit
recipes are 1.6+ protocols, and keeping one minimum version for the whole
module keeps every snippet in the guide runnable verbatim from a single
environment.
"""

import pytest

pytest.importorskip("sklearn", minversion="1.6")

from sklearn.datasets import make_classification  # noqa: E402


def test_bare_duck() -> None:
    # --8<-- [start:bare_duck]
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import log_loss
    from sklearn.model_selection import cross_val_score
    from sklearn.utils import get_tags
    from sklearn.utils.validation import check_is_fitted

    from probcal import BetaCalibrator  # no probcal.sklearn import

    rng = np.random.default_rng(0)
    s = rng.uniform(0.05, 0.95, 300)
    y = rng.binomial(1, s).astype(float)

    cal = BetaCalibrator()
    assert get_tags(cal).requires_fit  # true of every probcal calibrator
    cal.fit(s, y)
    check_is_fitted(cal)  # would raise NotFittedError had fit not run
    unfitted_copy = clone(cal)  # same params, fitted_ is False

    def neg_log_loss(est: BetaCalibrator, X: np.ndarray, y_true: np.ndarray) -> float:
        p = est.predict_proba(X)  # 1-D, not the (n, 2) adapter convention
        return -log_loss(y_true, p, labels=[0.0, 1.0])

    scores = cross_val_score(BetaCalibrator(), s.reshape(-1, 1), y, cv=3, scoring=neg_log_loss)
    # --8<-- [end:bare_duck]
    assert unfitted_copy.get_params() == cal.get_params()
    assert unfitted_copy.fitted_ is False
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_sklearn_calibrator_pipeline() -> None:
    # --8<-- [start:sklearn_calibrator_pipeline]
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from probcal import BetaCalibrator
    from probcal.sklearn import SklearnCalibrator

    # X's column 0 is the model's score; the rest are context columns the
    # calibrator does not use.
    rng = np.random.default_rng(1)
    s = rng.uniform(0.05, 0.95, 300)
    y = rng.binomial(1, s).astype(float)
    X = np.column_stack([s, rng.normal(size=300)])

    pipe = Pipeline(
        [
            ("select_score", FunctionTransformer(lambda a: a[:, [0]])),
            ("cal", SklearnCalibrator(BetaCalibrator())),
        ]
    ).fit(X, y)
    pipe.predict_proba(X)  # (n, 2)
    # --8<-- [end:sklearn_calibrator_pipeline]
    proba = pipe.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sklearn_calibrator_voting() -> None:
    # --8<-- [start:sklearn_calibrator_voting]
    import numpy as np
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from probcal.sklearn import SklearnCalibrator

    rng = np.random.default_rng(2)
    s = rng.uniform(0.05, 0.95, 300)
    y = rng.binomial(1, s).astype(float)
    X = np.column_stack([s, rng.normal(size=(300, 3))])

    score_branch = Pipeline(
        [
            ("select_score", FunctionTransformer(lambda a: a[:, [0]])),
            ("cal", SklearnCalibrator()),
        ]
    )
    vote = VotingClassifier(
        [("calibrated_score", score_branch), ("lr", LogisticRegression())],
        voting="soft",
    ).fit(X, y)
    vote.predict_proba(X)
    # --8<-- [end:sklearn_calibrator_voting]
    proba = vote.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.all(np.isfinite(proba))


def test_calibrated_classifier() -> None:
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.85, 0.15], random_state=3)
    X_train, y_train, X_new = X[:400], y[:400], X[400:]
    # --8<-- [start:calibrated_classifier]
    from sklearn.linear_model import LogisticRegression

    from probcal.sklearn import CalibratedClassifier

    clf = CalibratedClassifier(LogisticRegression(), cv=5, random_state=0).fit(X_train, y_train)
    clf.predict_proba(X_new)  # (n, 2)

    # probcal calibrator protocol, delegated to calibrator_:
    clf.interpret()
    clf.calibrator_.to_json()
    clf.interval_inverse(0.0, 0.02, space="logit")
    # --8<-- [end:calibrated_classifier]
    proba = clf.predict_proba(X_new)
    assert proba.shape == (len(X_new), 2)
    assert isinstance(clf.calibrator_.to_json(), str)


def test_prefit_calibrated_classifier_cv() -> None:
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.85, 0.15], random_state=4)
    X_train, y_train = X[:300], y[:300]
    X_calib, y_calib = X[300:500], y[300:500]
    X_new = X[500:]
    # --8<-- [start:prefit_calibrated_classifier_cv]
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(X_train, y_train)
    cccv = CalibratedClassifierCV(FrozenEstimator(model)).fit(X_calib, y_calib)
    cccv.predict_proba(X_new)
    # --8<-- [end:prefit_calibrated_classifier_cv]
    proba = cccv.predict_proba(X_new)
    assert proba.shape == (len(X_new), 2)


def test_prefit_frozen_sklearn_calibrator() -> None:
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.85, 0.15], random_state=5)
    X_train, y_train = X[:300], y[:300]
    X_calib, y_calib = X[300:500], y[300:500]
    X_new = X[500:]
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(X_train, y_train)
    # --8<-- [start:prefit_frozen_sklearn_calibrator]
    from sklearn.frozen import FrozenEstimator

    from probcal.sklearn import SklearnCalibrator

    frozen = FrozenEstimator(model)  # a fitted estimator that clone()/fit() cannot touch
    cal = SklearnCalibrator().fit(frozen.predict_proba(X_calib)[:, [1]], y_calib)
    # --8<-- [end:prefit_frozen_sklearn_calibrator]
    cal.predict_proba(frozen.predict_proba(X_new)[:, [1]])
    proba = cal.predict_proba(frozen.predict_proba(X_new)[:, [1]])
    assert proba.shape == (len(X_new), 2)
