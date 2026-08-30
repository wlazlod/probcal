"""Sample-weight delivery through the adapter, both metadata-routing modes.

What these tests pin down, verified identically on scikit-learn 1.4.2, 1.6.1
and 1.9.0:

* With routing **enabled**, a ``Pipeline`` ending in ``SklearnCalibrator`` (with
  ``set_fit_request(sample_weight=True)``) delivers the weights into the probcal
  ``fit`` bit-for-bit.
* With routing **enabled**, ``CalibratedClassifier`` hands the weights to the
  base estimator's cross-validated fits when that estimator requests them, and
  raises sklearn's ``UnsetMetadataPassedError`` when the request is unset —
  sklearn's routing machinery has the final word, exactly as it does inside
  ``CalibratedClassifierCV``.
* With routing **disabled** (the default), ``cross_val_predict(params=...)``
  passes the weights straight to the base estimator's ``fit``; no request
  declaration is involved.
* A base estimator whose ``fit`` has no ``sample_weight`` parameter gets exactly
  one ``UserWarning`` and unweighted inner fits; the calibrator fit stays
  weighted. Before this fix, it raised ``TypeError`` instead.
"""

import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", minversion="1.4")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal.sklearn import CalibratedClassifier, SklearnCalibrator  # noqa: E402

_D = make_pd_portfolio(n=800, random_state=5)
_S = _D.scores
_X1 = _S.reshape(-1, 1)
_Y = _D.y
# Deliberately imbalanced: events weigh ~23x a non-event, so the weighted and
# unweighted fits cannot coincide.
_W = np.where(_Y == 1.0, 7.0, 0.3)


def _features(seed: int = 5) -> np.ndarray:
    """Two features whose logistic fit reproduces the portfolio's signal."""
    rng = np.random.default_rng(seed)
    z = np.log(_S / (1.0 - _S))
    return np.column_stack([z, rng.normal(size=_S.size)])


class _SpyLogisticRegression(LogisticRegression):
    """LogisticRegression that records whether each ``fit`` received weights."""

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001, ANN201 - sklearn signature
        _SpyLogisticRegression.seen.append(sample_weight is not None)
        return super().fit(X, y, sample_weight=sample_weight)


_SpyLogisticRegression.seen = []


# --------------------------------------------------------------- SklearnCalibrator


def test_pipeline_routes_sample_weight_into_the_probcal_fit() -> None:
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline(
            [
                ("passthrough", "passthrough"),
                ("cal", SklearnCalibrator().set_fit_request(sample_weight=True)),
            ]
        )
        pipe.fit(_X1, _Y, sample_weight=_W)
    routed = pipe[-1].calibrator_.predict_proba(_S)

    weighted = BetaCalibrator().fit(_S, _Y, sample_weight=_W).predict_proba(_S)
    unweighted = BetaCalibrator().fit(_S, _Y).predict_proba(_S)
    np.testing.assert_array_equal(routed, weighted)
    assert not np.allclose(routed, unweighted)


def test_pipeline_without_request_does_not_reach_the_calibrator() -> None:
    """Routing is opt-in: an undeclared request is an error, never a silent drop."""
    from sklearn.exceptions import UnsetMetadataPassedError

    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("passthrough", "passthrough"), ("cal", SklearnCalibrator())])
        with pytest.raises(UnsetMetadataPassedError):
            pipe.fit(_X1, _Y, sample_weight=_W)


def test_sklearn_calibrator_takes_weights_without_routing() -> None:
    est = SklearnCalibrator().fit(_X1, _Y, sample_weight=_W)
    np.testing.assert_array_equal(
        est.calibrator_.predict_proba(_S),
        BetaCalibrator().fit(_S, _Y, sample_weight=_W).predict_proba(_S),
    )


# ------------------------------------------------------------ CalibratedClassifier


@pytest.mark.parametrize("routing", [False, True])
def test_grid_search_delivers_weights_to_base_and_calibrator(routing: bool) -> None:
    X = _features()
    grid = {"calibrator__variant": ["ab", "abm"]}
    with sklearn.config_context(enable_metadata_routing=routing):
        # The spy sits inside the search, so the assertion below is about the
        # fits the search actually ran.
        spy = _SpyLogisticRegression(max_iter=200)
        if routing:
            spy = spy.set_fit_request(sample_weight=True)
        clf = CalibratedClassifier(spy, calibrator=BetaCalibrator(), cv=3, random_state=0)
        if routing:
            # The wrapper is a consumer too: GridSearchCV only forwards what is asked
            # for, and it asks about `score` as well as `fit`.
            clf = clf.set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
        unweighted = GridSearchCV(clf, grid, cv=3).fit(X, _Y)
        _SpyLogisticRegression.seen = []
        weighted = GridSearchCV(clf, grid, cv=3).fit(X, _Y, sample_weight=_W)
        seen = list(_SpyLogisticRegression.seen)

    # 2 grid points x 3 search folds x (3 calibration folds + refit), plus the
    # search's own refit of the best estimator on all the data.
    assert len(seen) == 2 * 3 * 4 + 4
    assert all(seen)
    # And the weights changed the calibrated output.
    assert not np.allclose(
        weighted.best_estimator_.calibrator_.predict_proba(_S),
        unweighted.best_estimator_.calibrator_.predict_proba(_S),
    )


def test_routing_enabled_requires_an_explicit_request_on_the_base() -> None:
    from sklearn.exceptions import UnsetMetadataPassedError

    X = _features()
    with sklearn.config_context(enable_metadata_routing=True):
        clf = CalibratedClassifier(LogisticRegression(max_iter=200), cv=3, random_state=0)
        with pytest.raises(UnsetMetadataPassedError):
            clf.fit(X, _Y, sample_weight=_W)


@pytest.mark.parametrize("routing", [False, True])
def test_base_without_sample_weight_warns_once_and_still_weights_the_calibrator(
    routing: bool,
) -> None:
    X = _features()
    with sklearn.config_context(enable_metadata_routing=routing):
        clf = CalibratedClassifier(KNeighborsClassifier(15), cv=3, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(X, _Y, sample_weight=_W)
        unweighted = CalibratedClassifier(KNeighborsClassifier(15), cv=3, random_state=0).fit(X, _Y)

    mine = [
        w for w in caught if issubclass(w.category, UserWarning) and "KNeighbors" in str(w.message)
    ]
    assert len(mine) == 1
    assert "sample_weight" in str(mine[0].message)
    assert not np.allclose(
        clf.calibrator_.predict_proba(_S), unweighted.calibrator_.predict_proba(_S)
    )


def test_pipeline_base_estimator_routes_weights_through_its_steps() -> None:
    """A router base takes weights through ``**params``, not a named argument."""
    X = _features()
    with sklearn.config_context(enable_metadata_routing=True):
        _SpyLogisticRegression.seen = []
        base = Pipeline(
            [("lr", _SpyLogisticRegression(max_iter=200).set_fit_request(sample_weight=True))]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CalibratedClassifier(base, cv=3, random_state=0).fit(X, _Y, sample_weight=_W)
        seen = list(_SpyLogisticRegression.seen)
    assert len(seen) == 4
    assert all(seen)
    assert not [w for w in caught if "does not accept sample_weight" in str(w.message)]
