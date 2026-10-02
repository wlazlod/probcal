"""sklearn estimator-check declarations: which generated checks are inapplicable.

Data only, imported lazily: by the estimators' ``_more_tags()`` (sklearn
< 1.6 reads ``_xfail_checks`` from there at check time) and by the test suite
(``expected_failed_checks`` on sklearn >= 1.6). Nothing on the fit/predict
path imports this module.
"""

# Every entry states why the data the check generates cannot represent a
# score-level estimator — these are inapplicable checks, not known failures.
# The checks with a score-level analogue are re-implemented on valid `(n,)`
# probability data in `tests/test_sklearn_mirror_checks.py`.

# Cause 1: multi-column X. The column count alone puts the check outside the
# score-level contract, whatever its values are.
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
        "takes exactly one score column by contract"
    )
    for name, data in _MULTI_COLUMN_DATA.items()
}
CALIBRATOR_XFAIL_CHECKS.update(
    {
        name: (
            f"inapplicable: the check generates {data} — arbitrary reals outside "
            "[0, 1], where a score-level estimator takes one column of "
            "probabilities"
        )
        for name, data in _ARBITRARY_REAL_DATA.items()
    }
)
CALIBRATOR_XFAIL_CHECKS["check_fit1d"] = (
    "inapplicable: the check expects a raise on 1-D X, which a score-level "
    "estimator accepts by design"
)

# Cause 3: an internal CV split. Its fold assignment depends on n, so weighting
# a row cannot equal repeating it -- sklearn's own CV wrappers share this. The
# >= 1.6 check (check_sample_weight_equivalence_on_dense_data) passes explicit,
# n-independent splits through ``cv`` and runs live; only the < 1.6 check,
# which keeps the integer cv, is declared.
_CV_WEIGHT_REASON = (
    "inapplicable: weighting cannot equal duplication through a CV split whose "
    "fold assignment depends on n (sklearn's own CV wrappers share this)"
)
CLASSIFIER_XFAIL_CHECKS: dict[str, str] = {
    "check_sample_weights_invariance": _CV_WEIGHT_REASON,  # the < 1.6 name
}

# SklearnOffset is a y-optional, non-classifier transformer: the classifier-only
# checks (multi-class arms, the one-label sample-weight check, the supervised
# 2-D y check, container/dtype handling gated behind ClassifierMixin) are never
# generated for it by sklearn's own estimator-check machinery, so they are
# dropped from the calibrator's union rather than declared here.
_OFFSET_CLASSIFIER_ONLY: frozenset[str] = frozenset(
    {
        "check_classifier_data_not_an_array",
        "check_classifiers_classes",
        "check_classifiers_one_label_sample_weights",
        "check_classifiers_train",
        "check_supervised_y_2d",
    }
)
# check_fit1d generates 1-D X of 3 * U(0, 1): most values exceed 1, so
# SklearnOffset's own [0, 1] domain check raises ValueError as the check
# expects — unlike the calibrator (whose logit mode accepts arbitrary reals
# and so does not raise there), this check genuinely passes for SklearnOffset
# and is dropped rather than declared.
_OFFSET_NOT_APPLICABLE: frozenset[str] = _OFFSET_CLASSIFIER_ONLY | {"check_fit1d"}
# The offset's reasons are its own, not the calibrator's: SklearnOffset does
# accept a two-column matrix, so the operative refusal for the generated data
# is the column count (three or more) or the probability-simplex rule, never
# a one-column contract.
OFFSET_XFAIL_CHECKS: dict[str, str] = {
    name: (
        f"inapplicable: the check generates {data}, and this estimator takes one "
        "probability column or a two-column probability matrix (rows summing "
        "to 1) — the generated data is neither"
    )
    for name, data in _MULTI_COLUMN_DATA.items()
    if name not in _OFFSET_NOT_APPLICABLE
}
OFFSET_XFAIL_CHECKS.update(
    {
        name: (
            f"inapplicable: the check generates {data} — arbitrary reals outside "
            "[0, 1], never a probability column or simplex matrix"
        )
        for name, data in _ARBITRARY_REAL_DATA.items()
        if name not in _OFFSET_NOT_APPLICABLE
    }
)
# check_fit2d_1sample: the calibrator's own y-driven binary-class check
# happens to raise a message matching this check's expected pattern first (a
# single sample yields a single class); SklearnOffset has no y validation at
# all, so this reaches its own column-count check directly, whose message
# wording does not match the check's expected patterns.
# check_fit2d_1feature: the calibrator's logit mode accepts arbitrary reals
# outside [0, 1] without raising anything (the same mechanism already
# excluded above for check_fit1d); SklearnOffset's own [0, 1] domain check
# does raise here, but again with wording that does not match the check's
# expected patterns.
OFFSET_XFAIL_CHECKS["check_fit2d_1sample"] = (
    "inapplicable: the check generates 10 columns of 3 * U(0, 1) noise, and "
    "this estimator takes one probability column or a two-column probability "
    "matrix by contract"
)
OFFSET_XFAIL_CHECKS["check_fit2d_1feature"] = (
    "inapplicable: the check generates 1 column of 3 * U(0, 1) — values "
    "outside [0, 1], where this estimator takes one column of probabilities"
)


XFAIL_CHECKS: dict[str, dict[str, str]] = {
    "SklearnCalibrator": CALIBRATOR_XFAIL_CHECKS,
    "CalibratedClassifier": CLASSIFIER_XFAIL_CHECKS,
    "SklearnOffset": OFFSET_XFAIL_CHECKS,
}
"""Per-estimator table, keyed by the public class name."""
