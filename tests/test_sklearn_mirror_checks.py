"""Mirror of sklearn's generic checks on valid score-level data.

`sklearn.utils.estimator_checks` generates multi-column feature matrices of
arbitrary reals, which no score-level estimator can accept — those checks are
declared inapplicable in `probcal.sklearn._compat`, each with its own reason.
This module re-implements the ones that *do* have a score-level analogue, on
valid `(n,)` probability data, against the adapter (`SklearnCalibrator`) and
the bare core (`BetaCalibrator`). Most mirrored checks also run on two-column
`(n, 2)` probability-matrix input (`_as_X`'s `"adapter-2col"`/`"core-2col"`
kinds), since `validate_scores` accepts that shape core-wide. `_MIRRORED_BY`
maps every mirrored check to the test that stands in for it, `_NO_ANALOGUE`
names the rest, and `test_every_declared_check_is_mirrored_or_explained`
asserts the two together cover the declared tables exactly, so neither list
can drift.

The adapter's `n_features_in_` is 1 or 2, depending on whether `X` had one
score column or a two-column probability matrix at fit, and is enforced at
predict/transform time; that is mirrored by
`test_mirror_n_features_in_is_recorded_at_fit` and
`test_mirror_n_features_in_enforced_after_fitting` below.

Checks with no score-level analogue, and why:

* `check_classifiers_classes`, `check_classifiers_train` — their multi-class
  arms; a calibration map is binary by construction.
* `check_classifier_data_not_an_array`, `check_dtype_object`,
  `check_sample_weights_not_an_array`, `check_sample_weights_pandas_series`,
  `check_transformer_data_not_an_array` — container and dtype-object
  handling, which is sklearn's `validate_data`, exercised by the checks that
  do run; the core takes numpy arrays only.
* `check_readonly_memmap_input`, `check_f_contiguous_array_estimator` — a
  single column has no layout variants; the property they protect (fit must
  not write into its input) is mirrored by `test_fit_does_not_mutate_inputs`.
* `check_fit1d`, `check_fit2d_predict1d` — both expect a raise on 1-D `X`,
  which is the core's primary input form.
* `check_supervised_y_2d` — expects a `DataConversionWarning` and a silent
  ravel of column-vector `y`; `_validation.validate_binary_y` stays strictly
  1-D and raises instead.
* `check_positive_only_tag_during_fit` — the score domain is `[0, 1]`,
  stricter than the tag's "non-negative", so the tag cannot express it.

Sparse and pairwise inputs never reach these tables: one dense score column
has no sparse form, and `X` is a score, never a Gram matrix.

The tolerances in `_WEIGHT_DUPLICATION_TOLERANCE` are measured values plus
headroom, not error budgets: the large ones (`SplineCalibrator` 0.25,
`HistogramBinningCalibrator` 0.15) only catch gross breakage of an already
non-exact equivalence. The score-level contract itself (interpret, inverses,
fingerprints, serialization) is pinned by `tests/test_calibrator_protocol.py`.
"""

import copy
import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal._registry import SERIALIZABLE  # noqa: E402
from probcal.base import BaseCalibrator  # noqa: E402
from probcal.sklearn import SklearnCalibrator, SklearnOffset  # noqa: E402
from probcal.sklearn._compat import (  # noqa: E402
    CALIBRATOR_XFAIL_CHECKS,
    CLASSIFIER_XFAIL_CHECKS,
    OFFSET_XFAIL_CHECKS,
)

_D = make_pd_portfolio(n=600, random_state=21)
_S, _Y = _D.scores, _D.y
_Q = make_pd_portfolio(n=200, random_state=22).scores
_W = np.random.default_rng(0).integers(1, 4, size=_S.size).astype(np.float64)

CALIBRATOR_CLASSES = sorted(
    (c for c in SERIALIZABLE.values() if isinstance(c, type) and issubclass(c, BaseCalibrator)),
    key=lambda c: c.__name__,
)

# Declared-inapplicable check -> the test in this module that stands in for it.
_MIRRORED_BY: dict[str, str] = {
    "check_classifiers_one_label_sample_weights": (
        "test_one_class_after_zero_weight_trimming_raises"
    ),
    "check_dict_unchanged": "test_dict_unchanged_by_predict_side_methods",
    "check_dont_overwrite_parameters": "test_fit_does_not_overwrite_init_parameters",
    "check_estimators_dtypes": "test_float32_input_gives_float64_output",
    "check_estimators_fit_returns_self": "test_fit_returns_self",
    "check_estimators_nan_inf": "test_non_finite_scores_raise",
    "check_estimators_overwrite_params": "test_fit_does_not_overwrite_init_parameters",
    "check_estimators_pickle": "test_pickle_round_trip_predicts_identically",
    "check_fit_check_is_fitted": "test_check_is_fitted_before_and_after_fit",
    "check_fit_idempotent": "test_fit_idempotent",
    "check_fit_score_takes_y": "test_fit_takes_y_and_score_scores",
    "check_methods_sample_order_invariance": "test_methods_sample_order_invariance",
    "check_methods_subset_invariance": "test_methods_subset_invariance",
    "check_n_features_in": "test_mirror_n_features_in_is_recorded_at_fit",
    "check_n_features_in_after_fitting": "test_mirror_n_features_in_enforced_after_fitting",
    "check_pipeline_consistency": "test_pipeline_consistency",
    "check_sample_weight_equivalence_on_dense_data": (
        "test_integer_sample_weight_equals_row_duplication"
    ),
    "check_sample_weights_invariance": "test_integer_sample_weight_equals_row_duplication",
    "check_sample_weights_list": "test_sample_weight_as_list_matches_array",
    "check_sample_weights_not_overwritten": "test_fit_does_not_mutate_inputs",
    "check_sample_weights_shape": "test_wrong_shape_sample_weight_raises",
    "check_transformer_general": "test_fit_transform_equals_fit_then_transform",
    "check_transformer_preserve_dtypes": "test_fit_transform_equals_fit_then_transform",
    "check_fit2d_1sample": "test_offset_fits_a_single_sample",
    "check_fit2d_1feature": "test_offset_fits_a_single_feature_column",
}

# Declared-inapplicable checks with no score-level analogue at all; each is
# spelled out in this module's docstring with the reason.
_NO_ANALOGUE: frozenset[str] = frozenset(
    {
        "check_classifier_data_not_an_array",
        "check_classifiers_classes",
        "check_classifiers_train",
        "check_dtype_object",
        "check_f_contiguous_array_estimator",
        "check_fit1d",
        "check_fit2d_predict1d",
        "check_positive_only_tag_during_fit",
        "check_readonly_memmap_input",
        "check_sample_weights_not_an_array",
        "check_sample_weights_pandas_series",
        "check_supervised_y_2d",
        "check_transformer_data_not_an_array",
    }
)

# Calibrators for which integer weighting is not exactly row duplication, with
# the measured max absolute deviation on this module's fixture (n=600, weights
# drawn from {1, 2, 3}) and the reason. Everything not listed here is asserted
# bit-for-bit equal.
_WEIGHT_DUPLICATION_TOLERANCE: dict[str, tuple[float, str]] = {
    "BBQCalibrator": (
        5e-2,
        "the equal-mass bin grids are quantiles of the rows, which duplication "
        "moves; measured 2.8e-2",
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
        "the default equal-mass edges are unweighted np.quantile values, which "
        "duplication moves (strategy='width' is exact, with or without the "
        "Jeffreys shrinkage); measured 1.1e-1",
    ),
    "PlattCalibrator": (
        1e-12,
        "weighted and duplicated IRLS differ in floating point only; measured " "2.5e-16",
    ),
    "ScalingBinningCalibrator": (
        1.2e-1,
        "equal-mass bins on the Platt stage's output, whose edges duplication "
        "moves (the Platt stage itself agrees to 2.5e-16); measured 8.5e-2",
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

# The four input shapes exercised by the mirrored checks below: adapter/core,
# each with a single score column and with a two-column probability matrix.
_KINDS: tuple[str, ...] = ("adapter", "core", "adapter-2col", "core-2col")


def _make(kind: str) -> object:
    return SklearnCalibrator() if kind.startswith("adapter") else BetaCalibrator()


def _as_X(kind: str, s: np.ndarray) -> np.ndarray:
    """The adapter's `X` is a score column; the core takes the scores themselves.

    The `"-2col"` kinds wrap the score as a two-column probability-simplex
    matrix `[1 - s, s]` — `validate_scores` accepts that shape core-wide.
    """
    if kind.endswith("-2col"):
        return np.column_stack([1.0 - s, s])
    return s.reshape(-1, 1) if kind == "adapter" else s


def _proba(kind: str, est: object, s: np.ndarray) -> np.ndarray:
    """Positive-class probabilities, whichever convention the object follows."""
    p = est.predict_proba(_as_X(kind, s))
    return p[:, 1] if kind.startswith("adapter") else p


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


@pytest.mark.parametrize("kind", _KINDS)
def test_fit_idempotent(kind: str) -> None:
    """check_fit_idempotent: a second fit on the same data changes nothing."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    first = _proba(kind, est, _Q)
    est.fit(_as_X(kind, _S), _Y)
    np.testing.assert_array_equal(_proba(kind, est, _Q), first)


@pytest.mark.parametrize("kind", _KINDS)
def test_fit_returns_self(kind: str) -> None:
    """check_estimators_fit_returns_self: fit returns the same object."""
    est = _make(kind)
    assert est.fit(_as_X(kind, _S), _Y) is est


@pytest.mark.parametrize("kind", _KINDS)
def test_fit_does_not_overwrite_init_parameters(kind: str) -> None:
    """check_dont_overwrite_parameters / check_estimators_overwrite_params."""
    est = _make(kind)
    params_before = copy.deepcopy(est.get_params())
    public_before = {k for k in vars(est) if not k.startswith("_")}
    est.fit(_as_X(kind, _S), _Y, sample_weight=_W)
    assert est.get_params() == params_before
    added = {k for k in vars(est) if not k.startswith("_")} - public_before
    assert all(k.endswith("_") for k in added), added


@pytest.mark.parametrize("kind", _KINDS)
def test_check_is_fitted_before_and_after_fit(kind: str) -> None:
    """check_fit_check_is_fitted (the bare core's hook is also covered in test_sklearn_duck)."""
    est = _make(kind)
    with pytest.raises(NotFittedError):
        check_is_fitted(est)
    est.fit(_as_X(kind, _S), _Y)
    check_is_fitted(est)


@pytest.mark.parametrize("kind", _KINDS)
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


@pytest.mark.parametrize("kind", _KINDS)
def test_dict_unchanged_by_predict_side_methods(kind: str) -> None:
    """check_dict_unchanged: predict/transform must not touch fitted state."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    before = copy.deepcopy(vars(est))
    for method in ("predict", "predict_proba", "transform"):
        fn = getattr(est, method, None)
        if fn is not None:
            fn(_as_X(kind, _Q))
    assert _state_equal(before, vars(est))


@pytest.mark.parametrize("kind", _KINDS)
def test_non_finite_scores_raise(kind: str) -> None:
    """check_estimators_nan_inf: NaN/inf scores are refused in fit and in predict."""
    for bad_value in (np.nan, np.inf):
        bad = _S.copy()
        bad[0] = bad_value
        with pytest.raises(ValueError):
            _make(kind).fit(_as_X(kind, bad), _Y)
        est = _make(kind).fit(_as_X(kind, _S), _Y)
        with pytest.raises(ValueError):
            est.predict_proba(_as_X(kind, bad))


@pytest.mark.parametrize("kind", _KINDS)
def test_float32_input_gives_float64_output(kind: str) -> None:
    """check_estimators_dtypes: float32 scores are accepted; probabilities stay float64."""
    est = _make(kind).fit(_as_X(kind, _S.astype(np.float32)), _Y)
    out = est.predict_proba(_as_X(kind, _Q.astype(np.float32)))
    assert out.dtype == np.float64


@pytest.mark.parametrize("kind", _KINDS)
def test_sample_weight_as_list_matches_array(kind: str) -> None:
    """check_sample_weights_list: a Python list of weights is accepted."""
    as_list = _make(kind).fit(_as_X(kind, _S), _Y, sample_weight=list(_W))
    as_array = _make(kind).fit(_as_X(kind, _S), _Y, sample_weight=_W)
    np.testing.assert_array_equal(_proba(kind, as_list, _Q), _proba(kind, as_array, _Q))


@pytest.mark.parametrize("kind", _KINDS)
def test_wrong_shape_sample_weight_raises(kind: str) -> None:
    """check_sample_weights_shape: length and dimension mismatches are refused."""
    for bad in (np.ones(_S.size // 2), np.ones((_S.size, 2))):
        with pytest.raises(ValueError):
            _make(kind).fit(_as_X(kind, _S), _Y, sample_weight=bad)


@pytest.mark.parametrize("kind", _KINDS)
def test_fit_takes_y_and_score_scores(kind: str) -> None:
    """check_fit_score_takes_y: y is the second positional argument of fit."""
    est = _make(kind)
    est.fit(_as_X(kind, _S), _Y)  # positionally, as a pipeline would call it
    if kind.startswith("adapter"):  # .score() is a sklearn ClassifierMixin method
        assert 0.0 <= est.score(_as_X(kind, _S), _Y) <= 1.0


def test_mirror_n_features_in_is_recorded_at_fit() -> None:
    """check_n_features_in: n_features_in_ reflects the column count seen at fit."""
    est2 = SklearnCalibrator().fit(_as_X("adapter-2col", _S), _Y)
    assert est2.n_features_in_ == 2
    est1 = SklearnCalibrator().fit(_as_X("adapter", _S), _Y)
    assert est1.n_features_in_ == 1


def test_mirror_n_features_in_enforced_after_fitting() -> None:
    """check_n_features_in_after_fitting: predicting with a different width raises."""
    est = SklearnCalibrator().fit(_as_X("adapter-2col", _S), _Y)
    with pytest.raises(ValueError):
        est.predict_proba(_as_X("adapter", _S))


@pytest.mark.parametrize("kind", ["adapter", "adapter-2col"])
def test_pipeline_consistency(kind: str) -> None:
    """check_pipeline_consistency: wrapping the adapter in a Pipeline changes nothing."""
    X, Xq = _as_X(kind, _S), _as_X(kind, _Q)
    pipe = make_pipeline(SklearnCalibrator()).fit(X, _Y)
    est = SklearnCalibrator().fit(X, _Y)
    np.testing.assert_array_equal(pipe.predict_proba(Xq), est.predict_proba(Xq))
    assert pipe.score(X, _Y) == est.score(X, _Y)


@pytest.mark.parametrize("kind", ["adapter", "adapter-2col"])
def test_fit_transform_equals_fit_then_transform(kind: str) -> None:
    """check_transformer_general / check_transformer_preserve_dtypes."""
    X = _as_X(kind, _S)
    combined = SklearnCalibrator().fit_transform(X, _Y)
    stepwise = SklearnCalibrator().fit(X, _Y).transform(X)
    np.testing.assert_array_equal(combined, stepwise)
    assert combined.shape == (X.shape[0], 1)
    assert combined.dtype == np.float64
    Xf32 = X.astype(np.float32)
    assert SklearnCalibrator().fit_transform(Xf32, _Y).dtype == np.float64


@pytest.mark.parametrize("kind", ["adapter", "adapter-2col"])
def test_one_class_after_zero_weight_trimming_raises(kind: str) -> None:
    """check_classifiers_one_label_sample_weights: the refusal names the class problem."""
    w = np.ones(_S.size)
    w[_Y == 1.0] = 0.0
    with pytest.raises(ValueError, match="class"):
        SklearnCalibrator().fit(_as_X(kind, _S), _Y, sample_weight=w)


@pytest.mark.parametrize("kind", ["adapter", "adapter-2col"])
def test_pickle_round_trip_predicts_identically(kind: str) -> None:
    """check_estimators_pickle, adapter side: pickle is sklearn's persistence contract."""
    est = SklearnCalibrator().fit(_as_X(kind, _S), _Y)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(
        restored.predict_proba(_as_X(kind, _Q)), est.predict_proba(_as_X(kind, _Q))
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
    """A clone is a fresh estimator with the same parameters (sklearn's clone contract)."""
    prototype = _make(kind)
    fitted = clone(prototype).fit(_as_X(kind, _S), _Y)
    fresh = _make(kind).fit(_as_X(kind, _S), _Y)
    np.testing.assert_array_equal(_proba(kind, fitted, _Q), _proba(kind, fresh, _Q))
    assert prototype.get_params() == _make(kind).get_params()


@pytest.mark.parametrize("kind", _KINDS)
def test_methods_subset_invariance(kind: str) -> None:
    """check_methods_subset_invariance: predicting a subset equals subsetting."""
    est = _make(kind).fit(_as_X(kind, _S), _Y)
    mask = np.zeros(_Q.size, dtype=bool)
    mask[::3] = True
    np.testing.assert_array_equal(_proba(kind, est, _Q[mask]), _proba(kind, est, _Q)[mask])


@pytest.mark.parametrize("kind", _KINDS)
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


# --------------------------------------------------- SklearnOffset-only mirrors
# check_fit2d_1sample and check_fit2d_1feature are declared inapplicable only
# for SklearnOffset (see OFFSET_XFAIL_CHECKS), for two different reasons:
#
# * check_fit2d_1sample generates a single sample. The calibrator's own
#   y-driven binary-class check happens to raise a message matching this
#   check's expected pattern first (one sample means one class), so the
#   check passes there without exercising anything offset-specific.
#   SklearnOffset has no y validation, so nothing short-circuits — fitting a
#   single sample is expected to genuinely work, which is what is mirrored
#   below.
# * check_fit2d_1feature generates one column of out-of-range values. The
#   calibrator's logit mode accepts arbitrary reals and raises nothing at
#   all (the same mechanism already excluded for check_fit1d), so the check
#   passes there too, again without exercising anything offset-specific. A
#   single *valid* probability column is expected to genuinely work for the
#   offset, which is what is mirrored below — the out-of-range values in
#   sklearn's generated data are a coincidental wording mismatch, not the
#   behaviour under test.


def test_offset_fits_a_single_sample() -> None:
    """check_fit2d_1sample: fitting a single row works (delta mode needs no solving)."""
    from probcal._math import expit, logit

    est = SklearnOffset(delta=0.2).fit(np.array([[0.3]]))
    expected = expit(logit(np.array([0.3])) + 0.2)
    np.testing.assert_allclose(est.transform(np.array([[0.3]]))[:, 0], expected)


def test_offset_fits_a_single_feature_column() -> None:
    """check_fit2d_1feature: a single valid probability column fits and predicts fine."""
    est = SklearnOffset(delta=0.2).fit(_S.reshape(-1, 1))
    np.testing.assert_array_equal(
        est.predict_proba(_Q.reshape(-1, 1))[:, 1], est.offset_.transform(_Q)
    )


# ------------------------------------------------------------ table guards


def test_every_tolerance_entry_names_a_registered_calibrator() -> None:
    """The exception table cannot rot: each entry must name a live class."""
    names = {c.__name__ for c in CALIBRATOR_CLASSES}
    assert set(_WEIGHT_DUPLICATION_TOLERANCE) <= names
    assert all(reason for _atol, reason in _WEIGHT_DUPLICATION_TOLERANCE.values())


def test_every_declared_check_is_mirrored_or_explained() -> None:
    """Every declared-inapplicable check is either mirrored here or explained above."""
    declared = (
        set(CALIBRATOR_XFAIL_CHECKS) | set(CLASSIFIER_XFAIL_CHECKS) | set(OFFSET_XFAIL_CHECKS)
    )
    accounted = set(_MIRRORED_BY) | _NO_ANALOGUE
    assert declared == accounted, {
        "declared but unaccounted": sorted(declared - accounted),
        "accounted but not declared": sorted(accounted - declared),
    }
    assert not set(_MIRRORED_BY) & _NO_ANALOGUE

    module = globals()
    for check, test_name in _MIRRORED_BY.items():
        assert callable(module.get(test_name)), f"{check} -> missing test {test_name}"
    for check in _NO_ANALOGUE:
        assert check in (__doc__ or ""), f"{check} is not explained in the module docstring"
