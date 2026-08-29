# scikit-learn adapter

`probcal.sklearn` (extra: `pip install "probcal[sklearn]"`, scikit-learn ≥ 1.4)
provides two estimators. Neither is imported by `import probcal` — the core
stays numpy-only.

## Ending a pipeline with a calibrated column

`SklearnCalibrator` is *score-level*: its `X` is the score itself, one column
(`(n,)` or `(n, 1)`; more columns raise). Use it wherever sklearn expects an
estimator, or let `transform` end a `Pipeline`:

```python
from probcal import BetaCalibrator
from probcal.sklearn import SklearnCalibrator

est = SklearnCalibrator(BetaCalibrator()).fit(scores, y)
est.predict_proba(scores_new)     # (n, 2)
est.transform(scores_new)         # (n, 1) calibrated column

est.calibrator_.interpret()       # the full probcal audit surface
est.calibrator_.interval_inverse(0.0, 0.02, space="logit")
est.calibrator_.to_json("calibrator.json")
```

Score columns on the margin scale are declared, not guessed:
`SklearnCalibrator(input="logit")` maps them through `expit` exactly first.

## Cross-validated calibration of a classifier

`CalibratedClassifier` is the drop-in for
`sklearn.calibration.CalibratedClassifierCV(ensemble=False)`:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from probcal.sklearn import CalibratedClassifier

clf = CalibratedClassifier(
    HistGradientBoostingClassifier(), cv=5, random_state=0
).fit(X_train, y_train)
clf.predict_proba(X_new)

# probcal calibrator protocol, delegated — hand clf straight to treecf:
clf.interval_inverse(0.0, 0.02, space="logit")
clf.fingerprint()
```

`cv="prefit"` scores the calibration set with the already-fitted estimator;
`method="decision_function"` maps margins through `expit` before calibration
(the calibrator absorbs the monotone distortion this introduces).

## Grid search over the calibration map

The probcal calibrator is a nested estimator, so its parameters are
grid-searchable:

```python
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(
    CalibratedClassifier(model, calibrator=BetaCalibrator()),
    {"calibrator__variant": ["a", "ab", "abm"]},
    scoring="neg_log_loss",
    cv=5,
).fit(X, y)
```

## Sample weights and metadata routing

Both estimators declare `sample_weight` in `fit`, so the weights always reach
the probcal calibrator. What depends on `enable_metadata_routing` is how they
reach the *wrapped* classifier.

**Routing off (sklearn's default).** Weights are handed down directly:
`CalibratedClassifier` forwards them to `cross_val_predict`'s fold fits and to
the full-data refit whenever the base estimator's `fit` takes a
`sample_weight` argument.

**Routing on.** Every consumer must ask for the metadata, ours included:

```python
import sklearn
from sklearn.pipeline import Pipeline
from probcal.sklearn import SklearnCalibrator

with sklearn.config_context(enable_metadata_routing=True):
    pipe = Pipeline(
        [("cal", SklearnCalibrator().set_fit_request(sample_weight=True))]
    )
    pipe.fit(scores.reshape(-1, 1), y, sample_weight=w)
```

Inside a search, the base estimator declares its own request and the wrapper
declares one per method the search calls:

```python
with sklearn.config_context(enable_metadata_routing=True):
    clf = CalibratedClassifier(
        LogisticRegression().set_fit_request(sample_weight=True), cv=3
    )
    GridSearchCV(
        clf.set_fit_request(sample_weight=True).set_score_request(sample_weight=True),
        {"calibrator__variant": ["ab", "abm"]},
        cv=3,
    ).fit(X, y, sample_weight=w)
```

An undeclared request raises sklearn's `UnsetMetadataPassedError`; the adapter
does not soften that into a silent drop.

**Base estimators that cannot take weights.** If the base estimator's `fit` has
no `sample_weight` parameter (`KNeighborsClassifier`, for instance),
`CalibratedClassifier.fit` warns once (`UserWarning`, naming the class), runs
the fold fits and the refit unweighted, and still fits the calibrator with the
weights — the calibration map is weighted, the scores it calibrates are not.
Under routing, a router base (a `Pipeline`, say) counts as weight-capable: it
takes weights through `**params`, and sklearn's routing has the final word.

This behaviour is pinned by `tests/test_sklearn_routing.py`, verified on
scikit-learn 1.4.2, 1.6.1 and 1.9.0.

## Against `CalibratedClassifierCV`

| | `CalibratedClassifierCV(ensemble=False)` | `probcal.sklearn.CalibratedClassifier` |
|---|---|---|
| OOF protocol, one map, full refit | yes | yes (verified equivalent) |
| Calibration methods | sigmoid, isotonic | all 12 probcal calibrators + `CalibratorSelector` |
| Parameter interpretation | — | `interpret()` on the fitted map |
| Metric CIs | — | `probcal.metrics.evaluate` bootstrap |
| Exact inverse maps (policy → raw threshold) | — | `interval_inverse` / `point_inverse`, delegated |
| Serialization | pickle | versioned JSON + `fingerprint()` (and pickle) |
| Multiclass | yes | binary only, by design |

## Estimator-check compliance

Both estimators run `sklearn.utils.estimator_checks.parametrize_with_checks`
in CI on the pinned minimum (1.4) and the latest release.
`CalibratedClassifier` passes the full corpus except the sample-weight ≡
duplication equivalence, which cannot hold through a CV split whose fold
assignment depends on n (sklearn's own CV wrappers share this; declared via
`expected_failed_checks`). `SklearnCalibrator`'s one-column contract is
inapplicable to the generic multi-feature checks — those are declared through
sklearn's `expected_failed_checks`, each entry naming the data that check
generates and the part of the score-level contract it violates, the same
domain-restriction sklearn special-cases its own `IsotonicRegression` for; the
remaining convention checks run live. The inapplicable checks that do have a
score-level analogue are re-implemented on valid probability data in
`tests/test_sklearn_mirror_checks.py` — see *API stability* for the full
account.
