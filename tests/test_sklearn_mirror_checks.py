"""Mirror of sklearn's generic checks on valid score-level data (spec K4.2).

`sklearn.utils.estimator_checks` generates multi-column feature matrices of
arbitrary reals, which no score-level estimator can accept — those checks are
declared inapplicable in `probcal.sklearn._compat`, each with its own reason.
This module re-implements the ones that *do* have a score-level analogue, on
valid `(n,)` probability data, against the adapter (`SklearnCalibrator`) and
the bare core (`BetaCalibrator`): fit idempotence, no mutation of the passed
arrays, `__dict__` unchanged by the predict-side methods, round-trip
persistence, clone-then-fit equivalence, subset invariance, and integer
`sample_weight` equal to row duplication (the last one over every registered
calibrator class).

Generic checks with no score-level analogue, and why:

* multiclass (`check_classifiers_classes`, the multi-class arm of
  `check_classifiers_train`) — a calibration map is binary by construction.
* pairwise / kernel inputs — `X` is a score, never a Gram matrix.
* sparse containers — one dense score column has no sparse form.
* `dtype=object`, pandas and `_NotAnArray` containers
  (`check_dtype_object`, `check_sample_weights_pandas_series`,
  `check_classifier_data_not_an_array`) — container handling is sklearn's
  `validate_data`, exercised by the checks that do run.
* readonly-memmap and F-contiguous layouts (`check_readonly_memmap_input`,
  `check_f_contiguous_array_estimator`) — a single column has no layout
  variants; the property they protect (fit must not write into its input) is
  mirrored by `test_fit_does_not_mutate_inputs`.
* `n_features_in_` bookkeeping (`check_n_features_in*`) — the adapter's is
  always 1 and is pinned in `tests/test_sklearn.py`; the bare core has no
  such attribute by design.
* `check_fit2d_predict1d` and `check_fit1d` — both expect a raise on 1-D `X`,
  which is the core's primary input form (spec W6/K2).
* `check_supervised_y_2d` — expects a `DataConversionWarning` and a silent
  ravel of column-vector `y`; `_validation.validate_binary_y` stays strictly
  1-D and raises instead.
* `check_positive_only_tag_during_fit` — the score domain is `[0, 1]`,
  stricter than the tag's "non-negative", so the tag cannot express it.

The score-level contract itself (interpret, inverses, fingerprints,
serialization) is pinned by `tests/test_calibrator_protocol.py`.
"""

import copy
import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal._registry import SERIALIZABLE  # noqa: E402
from probcal.base import BaseCalibrator  # noqa: E402
from probcal.sklearn import SklearnCalibrator  # noqa: E402

_D = make_pd_portfolio(n=600, random_state=21)
_S, _Y = _D.scores, _D.y
_Q = make_pd_portfolio(n=200, random_state=22).scores
_W = np.random.default_rng(0).integers(1, 4, size=_S.size).astype(np.float64)

CALIBRATOR_CLASSES = sorted(
    (c for c in SERIALIZABLE.values() if isinstance(c, type) and issubclass(c, BaseCalibrator)),
    key=lambda c: c.__name__,
)

# Calibrators for which integer weighting is not exactly row duplication, with
# the measured max absolute deviation on this module's fixture (n=600, weights
# drawn from {1, 2, 3}) and the reason. Everything not listed here is asserted
# bit-for-bit equal.
_WEIGHT_DUPLICATION_TOLERANCE: dict[str, tuple[float, str]] = {
    "BBQCalibrator": (
        5e-2,
        "the equal-mass bin grid is built from the rows (n triples under "
        "duplication) and the Beta-Binomial marginal likelihood that averages "
        "the bin counts is not scale-free; measured 2.8e-2",
    ),
    "BetaCalibrator": (
        1e-12,
        "IRLS on the weighted and the duplicated design converges to the same "
        "solution up to floating point; measured 1.3e-14",
    ),
    "CalibratorSelector": (
        2e-2,
        "selection runs an inner CV whose fold assignment depends on n, so "
        "equality is not structural; on this fixture both paths select the "
        "same calibrator and its full-data fit agrees exactly (measured 0)",
    ),
    "CrossVennAbersCalibrator": (
        2e-2,
        "cross Venn-Abers averages over folds whose assignment depends on n; " "measured 8.8e-3",
    ),
    "HistogramBinningCalibrator": (
        1.5e-1,
        "equal-mass edges are quantiles of the rows, which duplication moves, "
        "and the Jeffreys (k + 1/2)/(n + 1) shrinkage is not scale-free in the "
        "weighted counts; measured 1.1e-1",
    ),
    "PlattCalibrator": (
        3e-2,
        "Platt's target smoothing (N+ + 1)/(N+ + 2) and 1/(N- + 2) uses the "
        "class totals, which duplication triples; measured 2.0e-2",
    ),
    "ScalingBinningCalibrator": (
        1.2e-1,
        "a Platt stage (target smoothing, see PlattCalibrator) followed by "
        "equal-mass bins on its output; measured 8.5e-2",
    ),
    "SegmentedCalibrator": (
        1e-12,
        "delegates to the pooled base calibrator; IRLS floating point only, " "measured 1.3e-14",
    ),
    "SplineCalibrator": (
        2.5e-1,
        "the knot count is ceil(n ** (1/3)) and the knots are quantiles of the "
        "rows, both of which duplication moves, and the penalty weight is "
        "chosen by an inner CV; measured 1.9e-1",
    ),
}


# ----------------------------------------------------------------- helpers


def _make(kind: str) -> object:
    return SklearnCalibrator() if kind == "adapter" else BetaCalibrator()


def _as_X(kind: str, s: np.ndarray) -> np.ndarray:
    """The adapter's `X` is a score column; the core takes the scores themselves."""
    return s.reshape(-1, 1) if kind == "adapter" else s


def _proba(kind: str, est: object, s: np.ndarray) -> np.ndarray:
    """Positive-class probabilities, whichever convention the object follows."""
    p = est.predict_proba(_as_X(kind, s))
    return p[:, 1] if kind == "adapter" else p


def _state_equal(before: dict, after: dict) -> bool:
    if set(before) != set(after):
        return False
    for key, old in before.items():
        new = after[key]
        if isinstance(old, np.ndarray):
            if not np.array_equal(old, new):
                return False
        elif isinstance(old, BaseCalibrator):
            if old.to_dict()["state"] != new.to_dict()["state"]:
                return False
        elif old != new:
            return False
    return True


# ------------------------------------------------- mirrored generic checks


@pytest.mark.parametrize("kind", ["adapter", "core"])
def test_fit_idempotent(kind: str) -> None:
    """check_fit_idempotent: a second fit on the same data changes nothing."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    first = _proba(kind, est, _Q)
    est.fit(_as_X(kind, _S), _Y)
    np.testing.assert_array_equal(_proba(kind, est, _Q), first)


@pytest.mark.parametrize("kind", ["adapter", "core"])
def test_fit_does_not_mutate_inputs(kind: str) -> None:
    """check_sample_weights_not_overwritten / readonly input: fit writes nothing back."""
    X, y, w = _as_X(kind, _S).copy(), _Y.copy(), _W.copy()
    X_ref, y_ref, w_ref = X.copy(), y.copy(), w.copy()
    est = _make(kind).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(X, X_ref)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(w, w_ref)

    Q = _as_X(kind, _Q).copy()
    Q_ref = Q.copy()
    est.predict_proba(Q)
    np.testing.assert_array_equal(Q, Q_ref)


@pytest.mark.parametrize("kind", ["adapter", "core"])
def test_dict_unchanged_by_predict_side_methods(kind: str) -> None:
    """check_dict_unchanged: predict/transform must not touch fitted state."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    before = copy.deepcopy(vars(est))
    for method in ("predict", "predict_proba", "transform"):
        fn = getattr(est, method, None)
        if fn is not None:
            fn(_as_X(kind, _Q))
    assert _state_equal(before, vars(est))


def test_pickle_round_trip_predicts_identically() -> None:
    """check_estimators_pickle, adapter side: pickle is sklearn's persistence contract."""
    est = SklearnCalibrator().fit(_S.reshape(-1, 1), _Y)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(
        restored.predict_proba(_Q.reshape(-1, 1)), est.predict_proba(_Q.reshape(-1, 1))
    )
    np.testing.assert_array_equal(restored.classes_, est.classes_)


@pytest.mark.parametrize("cls", CALIBRATOR_CLASSES, ids=lambda c: c.__name__)
def test_json_round_trip_predicts_identically(cls: type) -> None:
    """check_estimators_pickle, core side: the core's persistence contract is JSON."""
    est = cls().fit(_S, _Y)
    restored = BaseCalibrator.from_json(est.to_json())
    assert type(restored) is cls
    np.testing.assert_array_equal(restored.predict_proba(_Q), est.predict_proba(_Q))


@pytest.mark.parametrize("kind", ["adapter", "core"])
def test_clone_then_fit_equals_fresh_fit(kind: str) -> None:
    """check_estimators_overwrite_params: a clone is a fresh estimator, params intact."""
    prototype = _make(kind)
    fitted = clone(prototype).fit(_as_X(kind, _S), _Y)
    fresh = _make(kind).fit(_as_X(kind, _S), _Y)
    np.testing.assert_array_equal(_proba(kind, fitted, _Q), _proba(kind, fresh, _Q))
    assert prototype.get_params() == _make(kind).get_params()


@pytest.mark.parametrize("kind", ["adapter", "core"])
def test_methods_subset_invariance(kind: str) -> None:
    """check_methods_subset_invariance: predicting a subset equals subsetting."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    mask = np.zeros(_Q.size, dtype=bool)
    mask[::3] = True
    np.testing.assert_array_equal(_proba(kind, est, _Q[mask]), _proba(kind, est, _Q)[mask])


@pytest.mark.parametrize("kind", ["adapter", "core"])
def test_methods_sample_order_invariance(kind: str) -> None:
    """check_methods_sample_order_invariance: the map is applied row by row."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    idx = np.random.default_rng(4).permutation(_Q.size)
    np.testing.assert_array_equal(_proba(kind, est, _Q[idx]), _proba(kind, est, _Q)[idx])


# --------------------------------------------- sample_weight ≡ duplication


@pytest.mark.parametrize("cls", CALIBRATOR_CLASSES, ids=lambda c: c.__name__)
def test_integer_sample_weight_equals_row_duplication(cls: type) -> None:
    """check_sample_weight_equivalence_on_dense_data, on valid score data."""
    repeats = _W.astype(int)
    weighted = cls().fit(_S, _Y, sample_weight=_W).predict_proba(_Q)
    duplicated = cls().fit(np.repeat(_S, repeats), np.repeat(_Y, repeats)).predict_proba(_Q)

    atol, _reason = _WEIGHT_DUPLICATION_TOLERANCE.get(cls.__name__, (0.0, ""))
    if atol == 0.0:
        np.testing.assert_array_equal(weighted, duplicated)
    else:
        np.testing.assert_allclose(weighted, duplicated, atol=atol, rtol=0.0)


def test_adapter_integer_sample_weight_equals_row_duplication() -> None:
    """The adapter inherits the wrapped calibrator's weighting exactly."""
    repeats = _W.astype(int)
    weighted = SklearnCalibrator().fit(_S.reshape(-1, 1), _Y, sample_weight=_W)
    duplicated = SklearnCalibrator().fit(
        np.repeat(_S, repeats).reshape(-1, 1), np.repeat(_Y, repeats)
    )
    atol = _WEIGHT_DUPLICATION_TOLERANCE["BetaCalibrator"][0]
    np.testing.assert_allclose(
        weighted.predict_proba(_Q.reshape(-1, 1)),
        duplicated.predict_proba(_Q.reshape(-1, 1)),
        atol=atol,
        rtol=0.0,
    )


def test_adapter_zero_weight_equals_dropping_the_row() -> None:
    """sklearn's zero-weight semantics: weight 0 excludes the observation."""
    w = np.ones(_S.size)
    w[::5] = 0.0
    keep = w > 0.0
    zeroed = SklearnCalibrator().fit(_S.reshape(-1, 1), _Y, sample_weight=w)
    dropped = SklearnCalibrator().fit(_S[keep].reshape(-1, 1), _Y[keep])
    np.testing.assert_array_equal(
        zeroed.predict_proba(_Q.reshape(-1, 1)), dropped.predict_proba(_Q.reshape(-1, 1))
    )


def test_every_tolerance_entry_names_a_registered_calibrator() -> None:
    """The exception table cannot rot: each entry must name a live class."""
    names = {c.__name__ for c in CALIBRATOR_CLASSES}
    assert set(_WEIGHT_DUPLICATION_TOLERANCE) <= names
    assert all(reason for _atol, reason in _WEIGHT_DUPLICATION_TOLERANCE.values())
