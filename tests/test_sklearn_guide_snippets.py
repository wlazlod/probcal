"""Runnable snippets embedded in ``docs/guide/sklearn.md``.

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
    pipe.transform(X)  # (n, 1) calibrated column
    # --8<-- [end:sklearn_calibrator_pipeline]
    proba = pipe.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    column = pipe.transform(X)
    assert column.shape == (300, 1)
    np.testing.assert_allclose(column[:, 0], proba[:, 1])


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


def test_sklearn_calibrator_two_column() -> None:
    import numpy as np

    rng = np.random.default_rng(9)
    s = rng.uniform(0.05, 0.95, 300)
    y = rng.binomial(1, s).astype(float)
    proba_matrix = np.column_stack([1.0 - s, s])  # column 1 holds P(y=1)
    reversed_matrix = proba_matrix[:, ::-1]  # column 0 holds P(y=1) here
    # --8<-- [start:sklearn_calibrator_two_column]
    from probcal.sklearn import SklearnCalibrator

    cal = SklearnCalibrator().fit(proba_matrix, y)  # positive_column=1 (default)
    cal.predict_proba(proba_matrix)  # (n, 2) in, (n, 2) out

    # A reversed matrix needs positive_column=0 to match:
    cal_reversed = SklearnCalibrator(positive_column=0).fit(reversed_matrix, y)
    # --8<-- [end:sklearn_calibrator_two_column]
    proba = cal.predict_proba(proba_matrix)
    assert proba.shape == (300, 2)
    np.testing.assert_array_equal(proba, cal_reversed.predict_proba(reversed_matrix))
    # Omitting positive_column on a reversed matrix warns once, rather than
    # silently flipping the column it calibrates:
    with pytest.warns(UserWarning, match="positive_column=0"):
        SklearnCalibrator().fit(reversed_matrix, y)


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
    clf.to_json()
    clf.interval_inverse(0.0, 0.02, space="logit")
    # --8<-- [end:calibrated_classifier]
    proba = clf.predict_proba(X_new)
    assert proba.shape == (len(X_new), 2)
    assert isinstance(clf.to_json(), str)


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
    cal.predict_proba(frozen.predict_proba(X_new)[:, [1]])
    # --8<-- [end:prefit_frozen_sklearn_calibrator]
    proba = cal.predict_proba(frozen.predict_proba(X_new)[:, [1]])
    assert proba.shape == (len(X_new), 2)


def test_grid_search_calibrator_variant() -> None:
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.85, 0.15], random_state=6)
    # --8<-- [start:grid_search_calibrator]
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    from probcal import BetaCalibrator
    from probcal.sklearn import CalibratedClassifier

    gs = GridSearchCV(
        CalibratedClassifier(LogisticRegression(max_iter=200), calibrator=BetaCalibrator()),
        {"calibrator__variant": ["a", "ab", "abm"]},
        scoring="neg_log_loss",
        cv=5,
    ).fit(X, y)
    # --8<-- [end:grid_search_calibrator]
    assert gs.best_params_["calibrator__variant"] in ("a", "ab", "abm")


def test_sklearn_offset_chain() -> None:
    import numpy as np

    rng = np.random.default_rng(10)
    s = rng.uniform(0.05, 0.95, 400)
    y = rng.binomial(1, s).astype(float)
    # --8<-- [start:sklearn_offset_chain]
    from sklearn.pipeline import Pipeline

    from probcal import BetaCalibrator, Chain, LogitOffset
    from probcal.sklearn import SklearnCalibrator

    est = SklearnCalibrator(calibrator=Chain([BetaCalibrator(), LogitOffset(target_mean=0.2)]))
    pipe = Pipeline([("cal", est)]).fit(s.reshape(-1, 1), y)
    pipe.predict_proba(s.reshape(-1, 1))  # (n, 2), through calibrator then offset

    fitted_chain = pipe.named_steps["cal"].calibrator_
    fitted_chain.interpret()  # both stages, concatenated
    # --8<-- [end:sklearn_offset_chain]
    assert isinstance(fitted_chain, Chain) and fitted_chain.fitted_
    proba = pipe.predict_proba(s.reshape(-1, 1))
    assert proba.shape == (400, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert est.calibrator.fitted_ is False  # the prototype itself was not mutated


def test_sklearn_offset_replace_step() -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    s = rng.uniform(0.05, 0.95, 400)
    y = rng.binomial(1, s).astype(float)
    # --8<-- [start:sklearn_offset_replace_step]
    from sklearn.pipeline import Pipeline

    from probcal.sklearn import SklearnCalibrator, SklearnOffset

    pipe = Pipeline([("cal", SklearnCalibrator()), ("off", SklearnOffset(target_mean=0.2))]).fit(
        s.reshape(-1, 1), y
    )
    fp_before = pipe.named_steps["cal"].calibrator_.fingerprint()

    # Governance produced a new delta (e.g. from mon.apply_recommendation()).
    # Swap only the offset step and refit it alone, on the calibrator's output:
    new_delta = 0.05
    pipe.set_params(off=SklearnOffset(delta=new_delta))
    p_cal = pipe.named_steps["cal"].transform(s.reshape(-1, 1))
    pipe.named_steps["off"].fit(p_cal)
    # --8<-- [end:sklearn_offset_replace_step]
    assert pipe.named_steps["cal"].calibrator_.fingerprint() == fp_before
    assert pipe.named_steps["off"].offset_.delta_ == new_delta


def test_routing_pipeline_sample_weight() -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    scores = rng.uniform(0.05, 0.95, 300)
    y = rng.binomial(1, scores).astype(float)
    w = np.where(y == 1.0, 3.0, 1.0)
    # --8<-- [start:routing_pipeline_sample_weight]
    import sklearn
    from sklearn.pipeline import Pipeline

    from probcal.sklearn import SklearnCalibrator

    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("cal", SklearnCalibrator().set_fit_request(sample_weight=True))])
        pipe.fit(scores.reshape(-1, 1), y, sample_weight=w)
    # --8<-- [end:routing_pipeline_sample_weight]
    assert pipe.named_steps["cal"].calibrator_.fitted_ is True


def test_routing_grid_search_sample_weight() -> None:
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.85, 0.15], random_state=8)
    import numpy as np

    w = np.where(y == 1, 3.0, 1.0)
    # --8<-- [start:routing_grid_search_sample_weight]
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    from probcal import BetaCalibrator
    from probcal.sklearn import CalibratedClassifier

    with sklearn.config_context(enable_metadata_routing=True):
        clf = CalibratedClassifier(
            LogisticRegression().set_fit_request(sample_weight=True),
            calibrator=BetaCalibrator(),
            cv=3,
        )
        gs = GridSearchCV(
            clf.set_fit_request(sample_weight=True).set_score_request(sample_weight=True),
            {"calibrator__variant": ["ab", "abm"]},
            cv=3,
        ).fit(X, y, sample_weight=w)
    # --8<-- [end:routing_grid_search_sample_weight]
    assert gs.best_params_["calibrator__variant"] in ("ab", "abm")
