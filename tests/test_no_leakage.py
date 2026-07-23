"""Structural no-leakage tests (spec §13): the cv flow never scores a training
fold with a model that saw it."""

import numpy as np

from probcal._math import expit
from probcal.parametric import PlattCalibrator
from probcal.wrapper import CalibratedModel

RNG = np.random.default_rng(109)


class SpyModel:
    """Records (fitted row ids, scored row ids) pairs into a class-level
    registry that survives cloning by deepcopy. Row identity travels in the
    last feature column (an id column the model otherwise ignores)."""

    calls: list[tuple[frozenset, frozenset]] = []

    def __init__(self) -> None:
        self.fit_ids: frozenset = frozenset()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpyModel":
        self.fit_ids = frozenset(X[:, -1].astype(int).tolist())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ids = frozenset(X[:, -1].astype(int).tolist())
        SpyModel.calls.append((self.fit_ids, ids))
        p = expit(X[:, 0])
        return np.column_stack([1.0 - p, p])


def _spy_data(n: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([RNG.normal(size=n), np.arange(n, dtype=float)])
    y = (RNG.random(n) < expit(X[:, 0])).astype(float)
    return X, y


def _oof_calls(n: int) -> list[tuple[frozenset, frozenset]]:
    return [
        (fit_ids, scored)
        for fit_ids, scored in SpyModel.calls
        if fit_ids and len(fit_ids) < n  # fold models, not the final full-data refit
    ]


def test_cv_flow_out_of_fold_scoring_is_disjoint() -> None:
    n = 800
    X, y = _spy_data(n)
    SpyModel.calls = []
    CalibratedModel(SpyModel(), PlattCalibrator(), flow="cv", cv=5).fit(X, y)
    oof = _oof_calls(n)
    assert oof, "expected out-of-fold scoring calls to have been recorded"
    for fit_ids, scored in oof:
        assert not (fit_ids & scored), "a fold model scored rows it was trained on"


def test_cv_folds_cover_everything_exactly_once() -> None:
    n = 600
    X, y = _spy_data(n)
    SpyModel.calls = []
    CalibratedModel(SpyModel(), PlattCalibrator(), flow="cv", cv=4).fit(X, y)
    scored_union: set[int] = set()
    scored_total = 0
    for _, scored in _oof_calls(n):
        scored_union |= scored
        scored_total += len(scored)
    assert scored_union == set(range(n))
    assert scored_total == n  # each row scored exactly once out-of-fold
