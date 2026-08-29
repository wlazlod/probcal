# scikit-learn adapter

probcal meets sklearn at three depths, from lightest to heaviest:

1. **Bare duck.** On sklearn >= 1.6, a plain probcal calibrator satisfies
   sklearn's estimator protocol directly — no import from `probcal.sklearn`
   at all.
2. **`SklearnCalibrator`.** You need the `(n, 2)`/`classes_` probability-matrix
   convention: pipelines, `VotingClassifier`, stacking.
3. **`CalibratedClassifier`.** You want the whole out-of-fold calibration
   protocol as one auditable object.

`probcal.sklearn` (extra: `pip install "probcal[sklearn]"`, scikit-learn >= 1.4)
holds tiers 2 and 3. Neither is imported by `import probcal` — the core
stays numpy-only regardless of which tier you use.

**Design position.** The core is duck-typed to sklearn's *semantic* contract
by design: `fit`/`predict_proba`, `get_params`/`set_params`,
`__sklearn_is_fitted__`, `__sklearn_tags__`. That is genuinely enough for
`clone`, `check_is_fitted`, `get_tags`, and CV loops with a custom scorer —
contexts that only ever call methods, never inspect array shape. What duck
typing cannot do is dissolve sklearn's *shape* conventions: a classifier's
`predict_proba` returns `(n, 2)` and carries `classes_`, and a probcal
calibrator's `predict_proba` returns `(n,)` because there is only one class's
probability to report. `SklearnCalibrator` and `CalibratedClassifier` exist
to translate that convention where something in the ecosystem — `Pipeline`,
`VotingClassifier`, `GridSearchCV` scoring on `"neg_log_loss"` — actually
requires it. Bare duckness targets **sklearn >= 1.6** (the versions that
define the `__sklearn_is_fitted__`/`__sklearn_tags__` protocol); on older
sklearn, use the adapter.

## Tier 1 — bare duck (sklearn >= 1.6)

No `probcal.sklearn` import. A bare calibrator clones, reports its tags,
passes `check_is_fitted`, and scores under `cross_val_score` with a scorer
that calls its native 1-D `predict_proba`:

```python
--8<-- "tests/test_sklearn_guide_snippets.py:bare_duck"
```

This is genuinely useful whenever the surrounding code only calls methods
and never asks for a `(n, 2)` matrix or `classes_` — a custom CV loop, a
hyperparameter search with a hand-written scorer, anything that treats the
calibrator as "an object with `fit`/`predict_proba`" rather than
"a classifier".

## Tier 2 — `SklearnCalibrator`: ending a pipeline with a calibrated column

`SklearnCalibrator` is *score-level*: its `X` is the score itself, one column
(`(n,)` or `(n, 1)`; more columns raise). Use it wherever sklearn expects an
estimator, or let `transform` end a `Pipeline`. A `FunctionTransformer` ahead
of it selects the score column out of a wider feature table:

```python
--8<-- "tests/test_sklearn_guide_snippets.py:sklearn_calibrator_pipeline"
```

`est.calibrator_` is the full probcal audit surface — `interpret()`,
`interval_inverse`, `to_json()` — one attribute away from the sklearn object.

Score columns on the margin scale are declared, not guessed:
`SklearnCalibrator(input="logit")` maps them through `expit` exactly first.

Because it is an ordinary sklearn classifier over its one score column, it
composes into ensembles that mix estimator types over the same `X`, each
branch selecting what it needs — here a `VotingClassifier` pairing the
calibrated score against a feature-based classifier:

```python
--8<-- "tests/test_sklearn_guide_snippets.py:sklearn_calibrator_voting"
```

## Tier 3 — `CalibratedClassifier`: cross-validated calibration as one object

`CalibratedClassifier` is the drop-in for
`sklearn.calibration.CalibratedClassifierCV(ensemble=False)`: out-of-fold
scores via `cross_val_predict`, one probcal calibrator fitted on the pooled
OOF scores, and the estimator refit on all data. What it adds is the whole
probcal calibrator protocol on the result — `interpret()`, `to_json()`,
`interval_inverse` — instead of an opaque fitted map:

```python
--8<-- "tests/test_sklearn_guide_snippets.py:calibrated_classifier"
```

`cv="prefit"` scores the calibration set with the already-fitted estimator;
`method="decision_function"` maps margins through `expit` before calibration
(the calibrator absorbs the monotone distortion this introduces). The
probcal calibrator protocol is delegated straight through — hand a fitted
`clf` to treecf's `Target.calibrated` the same way you would a bare
calibrator.

### Grid search over the calibration map

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

## The prefit recipe: calibrating an already-fitted model

Sometimes the model is already trained and you only have a held-out
calibration set. On sklearn >= 1.6, `sklearn.frozen.FrozenEstimator` wraps a
fitted estimator so `clone`-based tools (`CalibratedClassifierCV`, a CV
search) leave it untouched instead of refitting it:

```python
--8<-- "tests/test_sklearn_guide_snippets.py:prefit_calibrated_classifier_cv"
```

That is sklearn's own meta-estimator route, and it is a fine default. The
primitives-first alternative composes `FrozenEstimator` with the bare
`SklearnCalibrator` — the same idea with no meta-estimator in between, in
four lines:

```python
--8<-- "tests/test_sklearn_guide_snippets.py:prefit_frozen_sklearn_calibrator"
```

Both recipes are alternatives to `CalibratedClassifier(cv="prefit")`, not
replacements for it: some teams prefer composing sklearn primitives to
adopting a probcal meta-estimator, and both routes above serve that
preference. Reach for probcal's own meta-estimator (`CalibratedClassifier`,
any `cv`, including `"prefit"`) when you want `interpret()`, JSON
serialization and a fingerprint, or the exact inverse maps
(`interval_inverse`/`point_inverse`) — a `FrozenEstimator` composition still
gets you the fitted `SklearnCalibrator.calibrator_`'s full audit surface via
that one attribute, so the difference is packaging (one call vs. three
lines), not capability.

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
| Prefit route | `CalibratedClassifierCV(FrozenEstimator(m))` | `cv="prefit"`, or `FrozenEstimator(m)` + bare `SklearnCalibrator` |

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
