"""Mirror of sklearn's generic checks on valid score-level data.

`sklearn.utils.estimator_checks` generates multi-column feature matrices of
arbitrary reals, which no score-level estimator can accept — those checks are
declared inapplicable in `probcal.sklearn._xfail`, each with its own reason.
This module re-implements the ones that *do* have a score-level analogue, on
valid `(n,)` probability data, against the adapter (`SklearnCalibrator`) and
the bare core (`BetaCalibrator`); `SklearnOffset` has its own mirrors for the
offset-only declarations (`test_offset_single_sample_and_single_column`).
Most mirrored checks also run on two-column `(n, 2)` probability-matrix input
(`_as_X`'s `"adapter-2col"`/`"core-2col"` kinds), since `validate_scores`
accepts that shape core-wide. `_MIRRORED_BY` maps every mirrored check to the
test that stands in for it, `_NO_ANALOGUE` names the rest, and
`test_every_declared_check_is_mirrored_or_explained` asserts the two together
cover the declared tables exactly, so neither list can drift.

The adapter's `n_features_in_` is 1 or 2, depending on whether `X` had one
score column or a two-column probability matrix at fit, and is enforced at
predict/transform time; that is mirrored by `test_mirror_n_features_in`.

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
  not write into its input) is mirrored by `test_fit_contract`.
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

from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from probcal import BetaCalibrator, make_pd_portfolio  # noqa: E402
from probcal._registry import SERIALIZABLE  # noqa: E402
from probcal.base import BaseCalibrator  # noqa: E402
from probcal.sklearn import SklearnCalibrator, SklearnOffset  # noqa: E402
from probcal.sklearn._xfail import (  # noqa: E402
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
    "check_classifiers_one_label_sample_weights": "test_adapter_contract",
    "check_dict_unchanged": "test_predict_contract",
    "check_dont_overwrite_parameters": "test_fit_contract",
    "check_estimators_dtypes": "test_predict_contract",
    "check_estimators_fit_returns_self": "test_fit_returns_self",
    "check_estimators_nan_inf": "test_predict_contract",
    "check_estimators_overwrite_params": "test_fit_contract",
    "check_estimators_pickle": "test_adapter_contract",
    "check_fit_check_is_fitted": "test_fit_contract",
    "check_fit_idempotent": "test_fit_contract",
    "check_fit_score_takes_y": "test_fit_contract",
    "check_methods_sample_order_invariance": "test_predict_contract",
    "check_methods_subset_invariance": "test_predict_contract",
    "check_n_features_in": "test_mirror_n_features_in",
    "check_n_features_in_after_fitting": "test_mirror_n_features_in",
    "check_pipeline_consistency": "test_adapter_contract",
    "check_sample_weight_equivalence_on_dense_data": (
        "test_integer_sample_weight_equals_row_duplication"
    ),
    "check_sample_weights_invariance": "test_integer_sample_weight_equals_row_duplication",
    "check_sample_weights_list": "test_sample_weight_contract",
    "check_sample_weights_not_overwritten": "test_fit_contract",
    "check_sample_weights_shape": "test_sample_weight_contract",
    "check_transformer_general": "test_adapter_contract",
    "check_transformer_preserve_dtypes": "test_adapter_contract",
    "check_fit2d_1sample": "test_offset_single_sample_and_single_column",
    "check_fit2d_1feature": "test_offset_single_sample_and_single_column",
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
        "cross Venn-Abers averages over folds whose assignment depends on n; measured 8.8e-3",
    ),
    "HistogramBinningCalibrator": (
        1.5e-1,
        "the default equal-mass edges are unweighted np.quantile values, which "
        "duplication moves (strategy='width' is exact, with or without the "
        "Jeffreys shrinkage); measured 1.1e-1",
    ),
    "PlattCalibrator": (
        1e-12,
        "weighted and duplicated IRLS differ in floating point only; measured 2.5e-16",
    ),
    "ScalingBinningCalibrator": (
        1.2e-1,
        "equal-mass bins on the Platt stage's output, whose edges duplication "
        "moves (the Platt stage itself agrees to 2.5e-16); measured 8.5e-2",
    ),
    "SegmentedCalibrator": (
        1e-12,
        "delegates to the pooled base calibrator; IRLS floating point only, measured 1.3e-14",
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
def test_fit_contract(kind: str) -> None:
    """check_estimators_fit_returns_self, check_fit_idempotent, check_fit_check_is_fitted,
    check_dont_overwrite_parameters / check_estimators_overwrite_params,
    check_sample_weights_not_overwritten (fit writes nothing back), check_fit_score_takes_y."""
    X, y, w = _as_X(kind, _S).copy(), _Y.copy(), _W.copy()
    X_ref, y_ref, w_ref = X.copy(), y.copy(), w.copy()
    est = _make(kind)
    params_before = copy.deepcopy(est.get_params())
    public_before = {k for k in vars(est) if not k.startswith("_")}
    with pytest.raises(NotFittedError):
        check_is_fitted(est)
    assert est.fit(X, y, sample_weight=w) is est
    check_is_fitted(est)
    assert est.get_params() == params_before
    added = {k for k in vars(est) if not k.startswith("_")} - public_before
    assert all(k.endswith("_") for k in added), added
    np.testing.assert_array_equal(X, X_ref)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(w, w_ref)

    first = _proba(kind, est, _Q)
    est.fit(X, y, sample_weight=w)  # y positionally, as a pipeline would call it
    np.testing.assert_array_equal(_proba(kind, est, _Q), first)
    if kind.startswith("adapter"):  # .score() is a sklearn ClassifierMixin method
        assert 0.0 <= est.score(X, y) <= 1.0


@pytest.mark.parametrize("kind", _KINDS)
def test_fit_returns_self(kind: str) -> None:
    """check_estimators_fit_returns_self (also asserted inside test_fit_contract)."""
    est = _make(kind)
    assert est.fit(_as_X(kind, _S), _Y) is est


@pytest.mark.parametrize("kind", _KINDS)
def test_predict_contract(kind: str) -> None:
    """check_dict_unchanged, check_estimators_dtypes, check_estimators_nan_inf,
    check_methods_subset_invariance, check_methods_sample_order_invariance."""
    est = _make(kind).fit(_as_X(kind, _S.astype(np.float32)), _Y)
    Q = _as_X(kind, _Q).copy()
    Q_ref = Q.copy()
    before = copy.deepcopy(vars(est))
    for method in ("predict", "predict_proba", "transform"):
        fn = getattr(est, method, None)
        if fn is not None:
            fn(Q)
    assert _state_equal(before, vars(est))
    np.testing.assert_array_equal(Q, Q_ref)

    assert est.predict_proba(_as_X(kind, _Q.astype(np.float32))).dtype == np.float64
    full = _proba(kind, est, _Q)
    mask = np.zeros(_Q.size, dtype=bool)
    mask[::3] = True
    np.testing.assert_array_equal(_proba(kind, est, _Q[mask]), full[mask])
    idx = np.random.default_rng(4).permutation(_Q.size)
    np.testing.assert_array_equal(_proba(kind, est, _Q[idx]), full[idx])

    for bad_value in (np.nan, np.inf):
        bad = _S.copy()
        bad[0] = bad_value
        with pytest.raises(ValueError):
            _make(kind).fit(_as_X(kind, bad), _Y)
        with pytest.raises(ValueError):
            est.predict_proba(_as_X(kind, bad))


@pytest.mark.parametrize("kind", _KINDS)
def test_sample_weight_contract(kind: str) -> None:
    """check_sample_weights_list, check_sample_weights_shape."""
    as_list = _make(kind).fit(_as_X(kind, _S), _Y, sample_weight=list(_W))
    as_array = _make(kind).fit(_as_X(kind, _S), _Y, sample_weight=_W)
    np.testing.assert_array_equal(_proba(kind, as_list, _Q), _proba(kind, as_array, _Q))
    for bad in (np.ones(_S.size // 2), np.ones((_S.size, 2))):
        with pytest.raises(ValueError):
            _make(kind).fit(_as_X(kind, _S), _Y, sample_weight=bad)


def test_mirror_n_features_in() -> None:
    """check_n_features_in / check_n_features_in_after_fitting: recorded at fit, enforced after."""
    assert SklearnCalibrator().fit(_as_X("adapter", _S), _Y).n_features_in_ == 1
    est = SklearnCalibrator().fit(_as_X("adapter-2col", _S), _Y)
    assert est.n_features_in_ == 2
    with pytest.raises(ValueError):
        est.predict_proba(_as_X("adapter", _S))


@pytest.mark.parametrize("kind", ["adapter", "adapter-2col"])
def test_adapter_contract(kind: str) -> None:
    """check_pipeline_consistency, check_transformer_general / _preserve_dtypes,
    check_estimators_pickle (adapter side; the core's contract is JSON),
    check_classifiers_one_label_sample_weights."""
    X, Xq = _as_X(kind, _S), _as_X(kind, _Q)
    est = SklearnCalibrator().fit(X, _Y)
    pipe = make_pipeline(SklearnCalibrator()).fit(X, _Y)
    np.testing.assert_array_equal(pipe.predict_proba(Xq), est.predict_proba(Xq))
    assert pipe.score(X, _Y) == est.score(X, _Y)

    combined = SklearnCalibrator().fit_transform(X, _Y)
    np.testing.assert_array_equal(combined, est.transform(X))
    assert combined.shape == (X.shape[0], 1)
    assert combined.dtype == np.float64
    assert SklearnCalibrator().fit_transform(X.astype(np.float32), _Y).dtype == np.float64

    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(restored.predict_proba(Xq), est.predict_proba(Xq))
    np.testing.assert_array_equal(restored.classes_, est.classes_)

    w = np.ones(_S.size)
    w[_Y == 1.0] = 0.0
    with pytest.raises(ValueError, match="class"):  # one class left after zero-weight trimming
        SklearnCalibrator().fit(X, _Y, sample_weight=w)


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


def test_offset_single_sample_and_single_column() -> None:
    """check_fit2d_1sample (delta mode needs no solving) and check_fit2d_1feature."""
    from probcal._math import expit, logit

    est = SklearnOffset(delta=0.2).fit(np.array([[0.3]]))
    expected = expit(logit(np.array([0.3])) + 0.2)
    np.testing.assert_allclose(est.transform(np.array([[0.3]]))[:, 0], expected)
    est = SklearnOffset(delta=0.2).fit(_S.reshape(-1, 1))
    np.testing.assert_array_equal(
        est.predict_proba(_Q.reshape(-1, 1))[:, 1], est.offset_.transform(_Q)
    )


# ------------------------------------------------------------ table guards


def test_every_declared_check_is_mirrored_or_explained() -> None:
    """Every declared-inapplicable check is mirrored here or explained above, and every
    weight-duplication tolerance names a live class with a reason."""
    names = {c.__name__ for c in CALIBRATOR_CLASSES}
    assert set(_WEIGHT_DUPLICATION_TOLERANCE) <= names
    assert all(reason for _atol, reason in _WEIGHT_DUPLICATION_TOLERANCE.values())
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
