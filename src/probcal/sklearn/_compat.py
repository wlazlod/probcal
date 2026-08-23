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
SCORE_LEVEL_REASON = (
    "score-level estimator takes exactly one column by contract (spec W6); "
    "this check fits on generic multi-feature data"
)
CALIBRATOR_XFAIL_CHECKS: dict[str, str] = dict.fromkeys(
    [
        "check_classifier_data_not_an_array",
        "check_classifiers_classes",
        "check_classifiers_one_label_sample_weights",
        "check_classifiers_train",
        "check_dict_unchanged",
        "check_dont_overwrite_parameters",
        "check_dtype_object",
        "check_estimators_dtypes",
        "check_estimators_fit_returns_self",
        "check_estimators_nan_inf",
        "check_estimators_overwrite_params",
        "check_estimators_pickle",
        "check_f_contiguous_array_estimator",
        "check_fit_check_is_fitted",
        "check_fit_idempotent",
        "check_fit_score_takes_y",
        "check_fit2d_predict1d",
        "check_methods_sample_order_invariance",
        "check_methods_subset_invariance",
        "check_n_features_in",
        "check_n_features_in_after_fitting",
        "check_pipeline_consistency",
        "check_positive_only_tag_during_fit",
        "check_readonly_memmap_input",
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weights_invariance",  # the < 1.6 name for the same check
        "check_sample_weights_list",
        "check_sample_weights_not_an_array",
        "check_sample_weights_not_overwritten",
        "check_sample_weights_pandas_series",
        "check_sample_weights_shape",
        "check_supervised_y_2d",
        "check_transformer_data_not_an_array",
        "check_transformer_general",
        "check_transformer_preserve_dtypes",
    ],
    SCORE_LEVEL_REASON,
)
CALIBRATOR_XFAIL_CHECKS["check_fit1d"] = (
    "accepts 1-D score arrays by design (spec W6); the check expects a raise"
)
_CV_WEIGHT_REASON = (
    "weighting cannot equal duplication through a CV split whose fold "
    "assignment depends on n (sklearn's own CV wrappers share this)"
)
CLASSIFIER_XFAIL_CHECKS: dict[str, str] = {
    "check_sample_weight_equivalence_on_dense_data": _CV_WEIGHT_REASON,
    "check_sample_weights_invariance": _CV_WEIGHT_REASON,  # the < 1.6 name
}
