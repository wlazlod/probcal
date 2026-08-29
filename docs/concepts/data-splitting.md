# Data splitting

Post-hoc calibration is a second round of estimation, and it obeys the same law as the first:
parameters fitted on a sample look better on that sample than anywhere else. Every design
decision in probcal's data flows follows from taking that law seriously three times over —
once for fitting the calibrator, once for evaluating it, and once for choosing among
calibrators.

## Why calibrating on training data fails

A trained model's scores on its own training set are not the scores it produces in
deployment. The residuals \( y_i - s_i \) on training data are systematically too small —
the model has partially memorized its sample — so a calibration map fitted to them learns the
wrong correction. The direction of the error is predictable: on training data an overfitted
model looks *better calibrated than it is*, so the fitted map under-corrects, and the
deployed system inherits the very overconfidence calibration was meant to remove. The effect
is largest exactly where flexible calibrators are attractive: an
[isotonic map](methods-nonparametric.md) fitted on training scores happily reproduces
memorization artifacts block by block. Zadrozny and Elkan (2002) already fit calibrators on
held-out data as a matter of course; probcal makes the discipline structural rather than
customary.

## The prefit flow

The canonical arrangement — and the recommended one for credit risk — is a three-way split:
train the model on one set, fit the calibrator on a second **calibration set** the model never
saw, and report metrics on a third test set that neither estimation stage touched. In probcal
this is `flow="prefit"`: the wrapped model is already trained elsewhere,
`CalibratedModel(model, calibrator, flow="prefit").fit(X_cal, y_cal)` scores the calibration
set and fits the map on those scores. The flow also degrades gracefully to score level — when
the "model" is a rating engine in a data warehouse and only its scores are exportable, the
core API (`calibrator.fit(s_cal, y_cal)`) is the same computation without the wrapper.

The cost of prefit is data: the calibration set must be carved out of what the model could
otherwise train on. Its virtue is auditability — one dataset, one fitted map, one line in the
model documentation — and in regulated settings that virtue usually dominates.

## The cross-validation flow

When data are too scarce to reserve a calibration set, `flow="cv"` recovers one
synthetically. The wrapper clones the untrained model (via `sklearn.base.clone` when sklearn
is installed, `copy.deepcopy` otherwise — a deliberate duck-typing choice), trains
it on \( K - 1 \) folds, scores the held-out fold, and rotates, so that every observation
receives a score from a model that did not train on it. Folds are stratified on the outcome
and seeded.

What happens next is a genuine fork, controlled by `ensemble`:

With **`ensemble=True`**, each fold's calibrator is kept, and prediction averages the \( K \)
calibrated outputs — the behavior of scikit-learn's `CalibratedClassifierCV`. The averaging
reduces variance but the deployed object is \( K \) models and \( K \) maps.

With **`ensemble=False`** — probcal's recommended default — the out-of-fold scores are pooled,
**one** calibrator is fitted on the pool, and the final model is refitted on all data. The
statistical trade is a slight pessimism (out-of-fold scores come from models trained on
\( (K-1)/K \) of the data, marginally weaker than the final refit), which for calibration
purposes is the safe direction. The operational gain is decisive in credit risk: a single
model, a single auditable mapping, a single row of parameters in `interpret()` — the deployed
system is documentable in a way an ensemble of \( K \) maps is not.

## How large must a calibration set be?

The honest unit of calibration sample size is the **event count**, not the row count, and a
serviceable rule of thumb allocates a handful of events per parameter to be estimated. A
temperature or offset (one parameter) becomes estimable with a dozen events; Platt (two) and
beta (three) want a few dozen for their standard errors to stop dominating the fit; isotonic
regression and its relatives, whose effective complexity is data-driven, only begin to beat
the parametric families when events number in the hundreds — each pooled block needs enough
events for its level to mean something. These are orders of magnitude, not thresholds, and
the [selector](auto-selection.md) run on a restricted candidate menu is the empirical check
that supersedes them.

The split proportion follows from the same arithmetic. Carving 20–30% of development data
into a calibration set is conventional, but the right question is what the carve does to
*both* stages: too small a calibration set starves the map; too large a carve starves the
model, which then produces worse scores for the map to repair. When the development sample
cannot fund both stages at once, that is precisely the regime `flow="cv"` exists for — it
spends the same rows on both purposes at the cost of the fold machinery. And when even pooled
out-of-fold data yields too few events for anything beyond a level correction, the
[offset chapter's](offset.md) one-parameter path is the disciplined retreat.

## Selection needs its own nesting

Choosing *which* calibrator to use is itself estimation, and it consumes data like everything
else. The trap is subtle enough to state plainly: fit five calibrators on the calibration set,
evaluate all five on the same calibration set, pick the best — and the winner is
systematically flattered, because the pick rewards whichever method overfitted this
particular sample most. The flexible methods win these rigged contests disproportionately,
which is exactly backwards.

The [selector](auto-selection.md) therefore runs an inner cross-validation *within* the
calibration data: each candidate is repeatedly fitted on inner-training folds and scored on
inner-validation folds, and only those out-of-fold scores enter the comparison. probcal makes
the discipline structural — `CalibratorSelector`'s scoring path never receives in-fold
predictions, so selection on fitting data is not a misuse the documentation warns against but
a state the code cannot reach. The winner is then refitted on the full calibration set, and
`test_no_leakage.py` asserts both properties mechanically.

## Small-sample guidance

Low-default portfolios compress all three tensions at once. Concrete guidance, matched to
what the [nonparametric chapter](methods-nonparametric.md) says about method complexity:

With a few hundred calibration observations and a percent-level event rate, the binding
constraint is the number of *events*, not observations — 500 points at 3% is fifteen
defaults, enough to estimate two or three parameters honestly. Parametric maps, coarse
equal-mass binning, and [CVAP](methods-distribution-free.md) (which recycles data across
folds by construction) are the defensible candidates; give the selector this restricted menu
rather than the full catalog, because every additional candidate spends selection power. Keep
the inner fold count modest — five stratified folds keep at least a couple of events per
validation fold at these rates — and read the selector's per-fold standard deviations as
seriously as its means: a method that wins on average but swings fold to fold is a worse
deployment risk than a stable runner-up. When even this is too tight, drop to the
one-parameter [offset](offset.md) anchored to a long-run central tendency in the manner of
Tasche (2013), and let the per-grade backtests of
[Metrics and tests](metrics.md) monitor what the data cannot yet estimate; Pluto and Tasche
(2005) develop the limiting case where defaults are too few for any curve fitting at all.

## In probcal

```python
from probcal import CalibratedModel, PlattCalibrator

# Prefit: the model is trained, a separate calibration set exists (the canon).
wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)

# CV: no calibration set to spare — pooled out-of-fold scores, one auditable map.
wrapped = CalibratedModel(model, PlattCalibrator(), flow="cv", cv=5).fit(X, y)

# CalibratedClassifierCV-style fold ensemble, if you prefer variance reduction
# over a single auditable mapping:
wrapped = CalibratedModel(model, PlattCalibrator(), flow="cv", ensemble=True).fit(X, y)

p = wrapped.predict_proba(X_new)
```

## References

- Pluto, K., Tasche, D. (2005). "Estimating Probabilities of Default for Low Default Portfolios." In *The Basel II Risk Parameters*, Springer.
- Tasche, D. (2013). "The art of probability-of-default curve calibration." *Journal of Credit Risk* 9(4).
- Zadrozny, B., Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." KDD, 694–699.
