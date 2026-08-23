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
inapplicable to the generic multi-feature checks — those are declared
expected failures with the reason stated, the same domain-restriction sklearn
special-cases its own `IsotonicRegression` for; the remaining convention
checks run live.
