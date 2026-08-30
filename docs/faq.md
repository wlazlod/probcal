# FAQ

## Which calibrator should I use?

Diagnose first, then read the catalog: [Choose a calibrator](guide/choosing.md) has the
table (monotonicity, which inverse exists, data appetite, use when / avoid when) and the
decision path, ending at `CalibratorSelector` when the answer is not obvious.

## My model outputs logits, not probabilities. What do I pass?

Convert once: `cal.fit(probcal.expit(z), y)`. All calibrators accept probabilities in
`(0, 1)` only — one code path, no per-class ambiguity. Logit-based calibrators recover
your margins exactly, so nothing is lost in the round trip.

## Why is there no pandas/sklearn dependency?

The runtime is numpy-only by design: auditable installs, no version-conflict surface, and
results as frozen dataclasses of arrays with `as_dict()` when you want a DataFrame
(`pd.DataFrame(result.as_dict())`).

That is not a compatibility gap. On scikit-learn >= 1.6 a bare probcal calibrator already
*is* a valid sklearn estimator — `fit`/`predict_proba`, `get_params`/`set_params`,
`__sklearn_is_fitted__`, `__sklearn_tags__` — so `clone`, `check_is_fitted`, `get_tags`
and CV loops with a custom scorer work with no adapter and no import from
`probcal.sklearn`. What duck typing cannot dissolve is sklearn's *shape* convention: a
classifier's `predict_proba` returns `(n, 2)` and carries `classes_`, while a calibrator
returns `(n,)`. `SklearnCalibrator` and `CalibratedClassifier` (extra:
`pip install "probcal[sklearn]"`) exist exactly for the places that require the matrix
convention — `Pipeline`, `VotingClassifier`, `GridSearchCV` scoring on `"neg_log_loss"`.
The three tiers, and which one your situation needs:
[scikit-learn adapter](guide/sklearn.md).

## How do I translate a calibrated cutoff back to a raw score?

`cal.interval_inverse(lo, hi, space="probability" | "logit", buffer_logit=0.0)` on any
monotone calibrator — with `calibrated_bands_to_raw` for a whole masterscale and
`point_inverse` for an exact single preimage. Worked examples, both spaces, the
`UnattainableTargetError` case, and the points-scale hand-off:
[Set cutoffs and invert maps](guide/cutoffs.md); the theory is
[Inverse maps](concepts/inverse-maps.md).

## How does this interoperate with a counterfactual engine (treecf)?

Calibration does not change counterfactual geometry — only the target interval. The
recipe, for a "PD ≤ 2% after calibration" target:

```python
# docs: no-run — cal/treecf stand in for a fitted calibrator and the treecf module
lo_z, hi_z = cal.interval_inverse(0.0, 0.02, space="logit")
target = treecf.Target.raw(range=(lo_z, hi_z))
```

One trap: after deploying calibration, `Target.probability(...)` becomes a silent bug —
it inverts the model's own sigmoid link, not the calibrator, and therefore targets the
*uncalibrated* probability. Use `Target.raw` with bounds from `interval_inverse`.
Pass `buffer_logit=m` to keep counterfactuals valid under future re-anchoring of
magnitude up to `m`.

## How does probcal compare to netcal?

The two packages overlap on method names (temperature, Platt/logistic, beta, histogram
binning, BBQ, ENIR) but target different settings.

**netcal** is built for deep-learning pipelines: it covers multi-class and object-detection
confidence calibration and regression-uncertainty calibration, and it runs on the
PyTorch stack. If your model is a neural network, your problem is multi-class, or you need
detection/regression calibration, netcal is the right tool — probcal deliberately does none
of those (binary only, by design).

**probcal** is built for binary probabilities feeding regulated or audited decisions —
credit-risk PD models being the archetype. What it adds that netcal does not aim at: a
numpy-only runtime (no torch/scipy in the import path — a small, auditable dependency
surface), logit-scale diagnostics readable on low-event-rate portfolios, `interpret()` on
every fitted map, the first-class auditable [offset](concepts/offset.md) with pre/post
guardrail reports, structurally leak-free
[automatic selection](concepts/auto-selection.md), Venn–Abers interval predictions,
per-grade binomial/Jeffreys backtests, calibrated→raw
[threshold translation](concepts/inverse-maps.md), and SHAP additivity repair.

Rule of thumb: neural networks, multi-class, detection, or regression → netcal. Binary
scores feeding cutoffs, pricing, capital, or reason codes — especially under validation or
supervisory review → probcal.

## Are there other packages called "probcal"?

Yes — two, neither affiliated with this project. The R package
[probcal](https://prdm0.github.io/probcal/) (P. R. Diniz Marinho) offers binary and
multiclass calibrators with SKCE-based inference; its binary catalog (Platt, temperature,
beta, isotonic, histogram) is a subset of the eleven methods here, and probcal (Python)
covers the SKCE too — see [Metrics and tests](concepts/metrics.md). The GitHub repository
[spencermyoung513/probcal](https://github.com/spencermyoung513/probcal) is an ECAI 2025
research codebase around the Conditional Congruence Error for neural *regression* fit, and
is not on PyPI. `pip install probcal` installs this package; when citing, "probcal
(Python)" avoids the ambiguity.

## Can I select a calibrator by ECE or Hosmer–Lemeshow?

No — the selector refuses both. They are binning-sensitive, biased, non-proper report
metrics; optimizing them invites the optimizer to exploit the estimator. Select on
log loss (default) or Brier; report the ECE family and ICI alongside. The full argument
is the table in [Metrics and tests](concepts/metrics.md).

## Do Venn–Abers intervals come with a guarantee?

Yes — for the *interval* from `predict_interval()`, under exchangeability. The scalar
from `predict_proba` is the log-loss-minimax merger `p1/(1-p0+p1)` and is not itself
covered by the theorem. See
[Distribution-free methods](concepts/methods-distribution-free.md).
