# treecf: counterfactuals on calibrated targets

[treecf](https://github.com/wlazlod/treecf) computes exact counterfactuals
on tree ensembles; its `Target.calibrated` accepts any object with the
probcal calibrator protocol (`is_monotone_`, `interval_inverse`) — every
probcal calibrator, `LogitOffset`, `Chain`, `CalibratedModel`,
`probcal.sklearn.CalibratedClassifier`, and
`CalibratedScorecard` conform (pinned by
`tests/test_calibrator_protocol.py`). treecf never imports probcal at
runtime; the coupling is one duck-typed protocol. Extra:
`pip install "probcal[treecf]"` (treecf ≥ 0.2.3); the joint smoke test
runs whenever both are installed.

## Case 1 — parametric: "below 2% calibrated PD"

```python
from treecf import Explainer, Target
from probcal import BetaCalibrator

cal = BetaCalibrator().fit(model_scores_cal, y_cal)     # held-out data
exp = Explainer(model=model, background=X_train_sample)

res = exp.explain(x, target=Target.calibrated(cal, op="<=", value=0.02),
                  seed=0, backend="exact")
cal.predict_proba(model.predict_proba(res.x_cf[None])[:, 1])  # <= 0.02, certified
```

treecf resolves the calibrated target once, through
`cal.interval_inverse(..., space="logit")` — bounds on the *raw margin* —
and optimizes there; the calibrated read-out is exact because the inverse
is.

## Case 2 — isotonic: a target inside a plateau

Step calibrators map whole raw regions to one level. Ask for "at most the
plateau level" and the counterfactual lands at the *block boundary* — the
cheapest raw score whose level qualifies — because probcal's generalized
inverse returns the largest raw score **inside** the preimage (one float
below the next block's edge), so treecf's closed-bound treatment can never
overshoot into the next block. The full contract, with a worked example:
the *Inverse maps* chapter, "The step-calibrator contract".

```python
iso = IsotonicCalibrator().fit(model_scores_cal, y_cal)
res = exp.explain(x, target=Target.calibrated(iso, op="<=", value=plateau_level),
                  seed=0, backend="exact")
# res lands on the boundary of the block whose level == plateau_level
```

## Case 3 — after a macro shift: `Chain`

After a `LogitOffset` re-anchoring, recourse must invert
`offset ∘ calibrator` — inverting the calibrator alone answers yesterday's
policy. `Chain` composes the stages exactly, and
`CalibratedModel.chain_` hands you the composed map without the model:

```python
from probcal import Chain, LogitOffset

off = LogitOffset(target_mean=0.031).fit(cal.predict_proba(model_scores_now))
chain = Chain([cal, off])                       # or wrapped.chain_
res = exp.explain(x, target=Target.calibrated(chain, op="<=", value=0.02),
                  seed=0, backend="exact")
```

## Grade recourse and `buffer_logit`

`Target.bands(bands, space="calibrated", calibrator=cal)` solves a whole
rating ladder in one call — one counterfactual (or certified infeasibility)
per grade band.

`buffer_logit` shrinks the calibrated interval in logit space *before*
inversion, so a future central-tendency update of at most that magnitude
cannot invalidate the counterfactual. The principled value is the
**monitor's offset confidence-sequence half-width**: with
`mon = CalibrationMonitor(...)`, take
`0.5 * (delta_ci[1] - delta_ci[0])` from the latest `MonitorStep` — a
re-offset anywhere inside the anytime-valid sequence then leaves buffered
recourse certificates intact.

## Pitfall: `Target.probability` is not the calibrated probability

`Target.probability` inverts the *model's* sigmoid — the raw margin's
probability, not the calibrator's output. A "2% PD" policy defined on
calibrated probabilities but requested through `Target.probability(0.02)`
silently targets the wrong quantity whenever the calibrator is not the
identity. Calibrated policies go through `Target.calibrated` (or
`Target.bands(space="calibrated")`), always.

## Cross-repo work list (treecf side, spec W11 T1–T5)

These items live in the treecf repository; they are tracked here per the
release spec until merged there:

- **T1 — provenance:** `Target.calibrated`/`Target.bands` read the optional
  duck-typed `fingerprint()` (available on every probcal object) and store
  `calibrator_fingerprint`, the resolved raw interval, and `buffer_logit`
  in the counterfactual/batch proof record.
- **T2 — calibrated read-out:** when the calibrator exposes
  `predict_proba`, attach `calibrated_probability` for the original and
  counterfactual points (presentational; optimization stays on the raw
  margin).
- **T3 — plateau-aware feasibility tests:** targets equal to isotonic
  plateau values (`op="<="`, `op=">="`, ranges touching a plateau) against
  brute force over leaf combinations; probcal's generalized-inverse bounds
  are attained (closed-safe) as of this release.
- **T4 — test matrix** with probcal as an optional test dependency:
  {Platt, Temperature, Beta `abm`, Isotonic, CenteredIsotonic,
  `LogitOffset`, `Chain(Beta + offset)`, `CalibratedModel`} ×
  {`op="<="`, `op=">="`, `range`, `bands`} × {`buffer_logit` 0, 0.2} on
  LightGBM/XGBoost/sklearn ensembles. **Boundary-routing defect, found by
  these smoke tests and fixed upstream in
  [treecf#21](https://github.com/wlazlod/treecf/pull/21):** sklearn
  `tree_` ensembles route `float32(x) <= float64(threshold)` while
  treecf's IR evaluated in float64, so a counterfactual coordinate placed
  exactly on a split threshold (the natural optimum of a smallest-change
  search) could flip through many trees — in the reproducing case the
  exact backend stamped `proof="optimal"` on an `x_cf` whose true
  `decision_function` margin was 3.09 raw units away. treecf#21
  re-expresses thresholds as the exact float64 boundary of the float32
  cast, making routing bit-exact for every input; fixed as of treecf
  0.2.3, which the `probcal[treecf]` extra pins. XGBoost also casts
  features to float32 natively — the analogous fix is a treecf
  follow-up.
- **T5 — docs:** link this guide from treecf's `concepts/calibration.md`,
  mirror the three focus cases, and verify (add a test) that target
  inversion is computed once per `Target` in batch mode.
