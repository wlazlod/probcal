"""Version shims for the supported scikit-learn range (>= 1.4)."""

import numpy as np

try:  # sklearn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:  # pragma: no cover - sklearn 1.4/1.5

    def _validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        if isinstance(y, str) and y == "no_validation":
            return estimator._validate_data(X, **kwargs)
        return estimator._validate_data(X, y, **kwargs)


def validate_X_y(
    estimator, X, y, *, reset: bool, allow_1d: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """2-D float64 X plus 1-D y, with ``n_features_in_`` bookkeeping."""
    X_arr = X if hasattr(X, "toarray") else np.asarray(X)
    if allow_1d and getattr(X_arr, "ndim", 2) == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _validate_data(estimator, X_arr, y, reset=reset, dtype=np.float64)


def validate_X(estimator, X, *, reset: bool = False, allow_1d: bool = False) -> np.ndarray:
    """2-D float64 X for predict-side entry points."""
    X_arr = X if hasattr(X, "toarray") else np.asarray(X)
    if allow_1d and getattr(X_arr, "ndim", 2) == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _validate_data(estimator, X_arr, reset=reset, dtype=np.float64)


# Checks that cannot apply to these estimators, declared through sklearn's
# own mechanisms: `expected_failed_checks` (>= 1.6) reads this table from the
# test suite; `_more_tags()["_xfail_checks"]` (< 1.6) reads it from the
# estimators. One source of truth for both.
#
# Every entry states why the data the check generates cannot represent a
# score-level estimator — these are inapplicable checks, not known failures.
# The checks with a score-level analogue are re-implemented on valid `(n,)`
# probability data in `tests/test_sklearn_mirror_checks.py`.

# Cause 1: multi-column X whose values happen to lie in [0, 1]. The column
# count alone puts the check outside the score-level contract.
_MULTI_COLUMN_DATA: dict[str, str] = {
    "check_classifier_data_not_an_array": "2 columns of small-integer coordinates",
    "check_classifiers_one_label_sample_weights": "10 columns of U(0, 1) noise",
    "check_dtype_object": "10 columns of U(0, 1) noise as dtype=object",
    "check_estimators_nan_inf": "3 columns of U(0, 1) noise",
    "check_fit_score_takes_y": "3 columns of U(0, 1) noise",
    "check_sample_weight_equivalence_on_dense_data": "30 columns of U(0, 1) noise",
    "check_sample_weights_invariance": "30 columns of U(0, 1) noise",  # < 1.6 name
    "check_sample_weights_list": "3 columns of U(0, 1) noise",
    "check_sample_weights_not_an_array": "2 columns of small-integer coordinates",
    "check_sample_weights_not_overwritten": "2 columns of small-integer coordinates",
    "check_sample_weights_pandas_series": "2 columns of small-integer coordinates",
    "check_sample_weights_shape": "2 columns of small-integer coordinates",
    "check_supervised_y_2d": "3 columns of U(0, 1) noise",
}

# Cause 2: multi-column X of arbitrary reals. Neither the shape nor the value
# range is a score — even one column of it would leave the [0, 1] domain.
_ARBITRARY_REAL_DATA: dict[str, str] = {
    "check_classifiers_classes": "2 standardized blob columns",
    "check_classifiers_train": "2 standardized blob columns",
    "check_dict_unchanged": "3 columns of 3 * U(0, 1)",
    "check_dont_overwrite_parameters": "3 columns of 3 * U(0, 1)",
    "check_estimators_dtypes": "5 columns of 3 * U(0, 1)",
    "check_estimators_fit_returns_self": "2 blob columns",
    "check_estimators_overwrite_params": "2 blob columns",
    "check_estimators_pickle": "3 blob columns",
    "check_f_contiguous_array_estimator": "3 columns of 3 * U(0, 1)",
    "check_fit_check_is_fitted": "2 columns of N(100, 1)",
    "check_fit_idempotent": "2 columns of N(100, 1)",
    "check_fit2d_predict1d": "3 columns of 3 * U(0, 1)",
    "check_methods_sample_order_invariance": "3 columns of 3 * U(0, 1)",
    "check_methods_subset_invariance": "3 columns of 3 * U(0, 1)",
    "check_n_features_in": "2 columns of N(100, 1)",
    "check_n_features_in_after_fitting": "4 columns of N(0, 1)",
    "check_pipeline_consistency": "3 blob columns",
    "check_positive_only_tag_during_fit": "the 4 mean-centred iris columns",
    "check_readonly_memmap_input": "2 blob columns",
    "check_transformer_data_not_an_array": "3 standardized blob columns",
    "check_transformer_general": "3 standardized blob columns",
    "check_transformer_preserve_dtypes": "3 standardized blob columns",
}

CALIBRATOR_XFAIL_CHECKS: dict[str, str] = {
    name: (
        f"inapplicable: the check generates {data}, and a score-level estimator "
        "takes exactly one score column by contract (spec W6)"
    )
    for name, data in _MULTI_COLUMN_DATA.items()
}
CALIBRATOR_XFAIL_CHECKS.update(
    {
        name: (
            f"inapplicable: the check generates {data} — arbitrary reals outside "
            "[0, 1], where a score-level estimator takes one column of "
            "probabilities (spec W6)"
        )
        for name, data in _ARBITRARY_REAL_DATA.items()
    }
)
CALIBRATOR_XFAIL_CHECKS["check_fit1d"] = (
    "inapplicable: the check expects a raise on 1-D X, which a score-level "
    "estimator accepts by design (spec W6)"
)

# Cause 3: an internal CV split. Its fold assignment depends on n, so weighting
# a row cannot equal repeating it — sklearn's own CV wrappers share this.
_CV_WEIGHT_REASON = (
    "inapplicable: weighting cannot equal duplication through a CV split whose "
    "fold assignment depends on n (sklearn's own CV wrappers share this)"
)
CLASSIFIER_XFAIL_CHECKS: dict[str, str] = {
    "check_sample_weight_equivalence_on_dense_data": _CV_WEIGHT_REASON,
    "check_sample_weights_invariance": _CV_WEIGHT_REASON,  # the < 1.6 name
}
