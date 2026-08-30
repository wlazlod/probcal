# Choose a calibrator

The catalog, as a table you can decide from. Every column that is a
property of the code (monotonicity, which inverse exists, what serializes)
is read off the fitted object and pinned by a test; the source of this page
carries the test name next to each row, so a column that stops being true
breaks a build rather than misleading a reader.

Fit cost per method is a separate axis and lives in one place: the
README's [Calibrators at a
glance](https://github.com/wlazlod/probcal#calibrators-at-a-glance)
table. Two entries there change decisions rather than schedules: ENIR is
quadratic in unique scores (its `fit` warns above ~50,000) and Venn–Abers
stores its calibration set, so its JSON grows with *n*.

## The catalog

| Class | Fitted parameters | Monotone? | Exact `point_inverse`? | `interval_inverse`? | Data appetite | Serialization | Use when / avoid when |
|---|---|---|---|---|---|---|---|
| `TemperatureCalibrator` | `T_` (one) | by construction (`T > 0` always) | yes: affine, `(1/T, 0)` | yes, closed form | ~a dozen events | JSON, golden-pinned | Confidence too soft or too sharp with the base rate already right / cannot move the level at all: `s = 0.5` is a fixed point |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[TemperatureCalibrator]; tests/test_inverse_maps.py::test_point_inverse_temperature_round_trip; tests/test_golden.py::test_golden_loads_and_reproduces[TemperatureCalibrator] -->
| `PlattCalibrator` | `a_`, `b_`, `converged_` | checked at fit (`is_monotone_ = a_ > 0`) | yes: affine, `(a_, b_)` | yes, closed form | a few dozen events | JSON, golden-pinned | Level and slope both off, and you want two auditable numbers / curvature that a straight logit line cannot follow |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[PlattCalibrator]; tests/test_inverse_maps.py::test_point_inverse_platt_round_trip; tests/test_inverse_maps.py::test_point_inverse_non_monotone_platt_raises; tests/test_golden.py::test_golden_loads_and_reproduces[PlattCalibrator] -->
| `BetaCalibrator` (`"abm"`) | `a_`, `b_`, `c_`, `constraint_active_` | by construction (`a_, b_ >= 0` enforced) | yes: its own exact construction (seed + certified Halley) | yes, bisection | a few dozen events | JSON, golden-pinned | Asymmetric tail distortion, the low-PD default / nothing to gain when `a_ ≈ b_`, where the tie-break drops you to Platt anyway |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[BetaCalibrator]; tests/test_inverse_maps.py::test_beta_point_inverse_abm_fit_round_trip; tests/test_golden.py::test_golden_loads_and_reproduces[BetaCalibrator] -->
| `BetaCalibrator(variant="ab"/"a")` | same attributes, tied exponents | by construction | yes: affine (`affine_logit_coeffs_` is not `None`) | yes | a few dozen events | JSON, golden-pinned | You want the beta family with fewer parameters to defend / the tail asymmetry is the whole problem |  <!-- pinned: tests/test_inverse_maps.py::test_beta_variant_ab_override_matches_base_affine_path; tests/test_inverse_maps.py::test_beta_variant_a_point_inverse_round_trip -->
| `IsotonicCalibrator` | `block_mean_`, `block_first_s_`, `block_last_s_`, `block_center_s_`, `n_blocks_` | by construction (PAVA) | no: `NotImplementedError` naming `interval_inverse` | yes, `searchsorted` on blocks; plateau semantics | hundreds of events | JSON, golden-pinned | Visible curvature and events to fund it / recourse downstream, where plateaus and moving block edges make counterfactuals fragile |  <!-- pinned: tests/test_calibrator_protocol.py::test_plateau_generalized_inverse_contract[isotonic]; tests/test_inverse_maps.py::test_isotonic_block_edge_semantics; tests/test_golden.py::test_golden_loads_and_reproduces[IsotonicCalibrator] -->
| `CenteredIsotonicCalibrator` | as isotonic (interpolates between block centers) | by construction | no | yes: a *point* preimage, not a plateau | hundreds of events | JSON, golden-pinned | Isotonic's fit without its step function, the nonparametric choice when recourse is in scope / very few blocks, where interpolation barely differs |  <!-- pinned: tests/test_calibrator_protocol.py::test_centered_isotonic_is_pointwise_between_plateaus; tests/test_golden.py::test_golden_loads_and_reproduces[CenteredIsotonicCalibrator] -->
| `HistogramBinningCalibrator` | `edges_`, `bin_rate_`, `bin_weight_`, `is_monotone_` | checked at fit: bin rates need not increase, and often do not | no | only when the fit came out monotone; otherwise `NotImplementedError` | hundreds of events, *per bin* | JSON, golden-pinned | Grade-shaped output that mirrors a masterscale / thin bins: a non-monotone fit also loses every inverse |  <!-- pinned: tests/test_calibrator_protocol.py::test_plateau_generalized_inverse_contract[histogram]; tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[HistogramBinningCalibrator]; tests/test_golden.py::test_golden_loads_and_reproduces[HistogramBinningCalibrator] -->
| `ScalingBinningCalibrator` | `platt_` (nested), `edges_`, `bin_value_` | declared (class default; the bin values inherit the Platt stage's order and are not re-checked) | no | yes: pulls the bin edge back through the Platt stage | between Platt's and binning's: a Platt fit plus a bin count | JSON, golden-pinned (nested Platt envelope) | You want binned output whose calibration error is measurable at `O(1/ε² + B)` samples / you need a continuous map |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[ScalingBinningCalibrator]; tests/test_golden.py::test_golden_loads_and_reproduces[ScalingBinningCalibrator] -->
| `BBQCalibrator` | `bins_grid_`, `weights_`, `is_monotone_` | checked at fit on a probe grid | no | only when the fit came out monotone | hundreds of events; strong on small samples (see the benchmarks) | JSON, golden-pinned | Small portfolio where a single binning is a coin flip / thresholding or recourse, where a non-monotone average refuses to invert |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[BBQCalibrator]; tests/test_golden.py::test_golden_loads_and_reproduces[BBQCalibrator] -->
| `ENIRCalibrator` | `path_lambdas_`, `path_solutions_`, `kept_breakpoints_`, `weights_`, `dropped_weight_` | never: `is_monotone_ = False` at class level | no | no: `NotImplementedError`, the preimage may be a union of intervals | hundreds of events; quadratic in unique scores | JSON, golden-pinned | Pure predictive accuracy with no downstream inversion / anything on this page's cutoff and recourse paths |  <!-- pinned: tests/test_inverse_maps.py::test_enir_not_implemented; tests/test_golden.py::test_golden_loads_and_reproduces[ENIRCalibrator] -->
| `SplineCalibrator` | `n_knots_`, `lambda_`, `edof_`, spline coefficients, `is_monotone_` | checked at fit on a probe grid (warns when it fails) | no | yes when monotone, by bisection | hundreds of events (a penalized basis plus CV over `lambda`) | JSON, golden-pinned | Smooth curvature you want to show a validator as a curve, not a staircase / small samples, where the penalty is chosen on very little data |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[SplineCalibrator]; tests/test_inverse_maps.py::test_point_inverse_non_affine_monotone_raises; tests/test_golden.py::test_golden_loads_and_reproduces[SplineCalibrator] -->
| `VennAbersCalibrator` (IVAP) | the sorted calibration set plus both sweeps `F0_`, `F1_` | by construction | no | yes, by bisection | hundreds of events | JSON, golden-pinned, but O(n): the calibration set *is* the fitted map | You need a distribution-free interval, not just a number / file size and load time matter |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[VennAbersCalibrator]; tests/test_inverse_maps.py::test_vennabers_bisection_inverse; tests/test_golden.py::test_golden_loads_and_reproduces[VennAbersCalibrator] -->
| `CrossVennAbersCalibrator` (CVAP) | `_ivaps`, one fitted IVAP per fold | by construction | no | yes, inherited bisection | hundreds of events, spent across folds instead of carved out | JSON, golden-pinned; O(n) for the same reason | The calibration sample is too small to carve and you still want the VA guarantee / the fold ensemble is harder to describe in a model document |  <!-- pinned: tests/test_vennabers.py::test_cvap_reproducible_and_monotone; tests/test_golden.py::test_golden_loads_and_reproduces[CrossVennAbersCalibrator] -->
| `SegmentedCalibrator` | `base_` (nested), `segments_`, `delta_hat_`, `se_`, `delta_tilde_`, `shrink_`, `tau2_` | inherited from `base_` | yes whenever `base_` has one (default base: beta), taking `segment=` | yes, taking `segment=`, inverting through `base_` plus that segment's shrunk offset | the base's appetite, plus ~a dozen events per segment | JSON, golden-pinned (nested base envelope) | Segments that genuinely differ in level and a shrinkage story you can defend / segments that differ in *slope*, which is a re-fit, not an offset |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[SegmentedCalibrator]; tests/test_golden.py::test_golden_loads_and_reproduces[SegmentedCalibrator] -->
| `CalibratorSelector` | `best_name_`, `best_calibrator_` (nested), `report_`, `is_monotone_` | inherited from the winner | no: the selector exposes no affine coefficients even when the winner is affine; reach for `sel.best_calibrator_.point_inverse` | yes, by bisection through the winner | enough for an inner stratified K-fold (default `cv=5`) | JSON, golden-pinned (winner envelope plus the ranked report) | You want the choice made on out-of-fold evidence and written down / you already know the answer and the extra machinery only adds a layer to explain |  <!-- pinned: tests/test_calibrator_protocol.py::test_interval_inverse_keyword_protocol[CalibratorSelector]; tests/test_golden.py::test_golden_loads_and_reproduces[CalibratorSelector] -->

Three readings of the **monotone** column, and the difference matters:
*by construction* means the fit cannot produce a decreasing map;
*checked at fit* means `is_monotone_` is derived from this fit's
parameters, so the same class can invert on one portfolio and refuse on
the next; *declared* means the class default stands and nothing re-derives
it. Only `is_monotone_ = True` buys you `interval_inverse`; everything
else raises `NotImplementedError`, which is the whole reason the column is
in the table. `LogitOffset` is not a calibrator and so is not a row here,
but it carries the same protocol (closed-form both inverses) and composes
onto any of the above through `Chain`, which is not a row either, having
no fit of its own to choose: it inherits every column from the stages you
put in it.

Event counts are the honest unit: "hundreds of events" means events, not
rows, and 500 observations at a 3% base rate is fifteen. The reasoning is
in [Data splitting](../concepts/data-splitting.md#how-large-must-a-calibration-set-be);
head-to-head evidence at event rates from 1.5% to 30% is in
[Benchmarks](../benchmarks/comparison.md).

## Reading the columns off your own fit

The table is a summary of what the objects report. On your data, ask them:

```python
# s_cal, y_cal: held-out calibration scores and outcomes
from probcal import BetaCalibrator, HistogramBinningCalibrator

for cal in (BetaCalibrator().fit(s_cal, y_cal),
            HistogramBinningCalibrator(n_bins=10).fit(s_cal, y_cal)):
    print(type(cal).__name__,
          "monotone:", cal.is_monotone_,
          "affine:", cal.affine_logit_coeffs_ is not None,
          "rank:", cal.complexity_rank)
```

A binning fit that prints `monotone: False` has just told you it cannot
serve a cutoff or a counterfactual on this portfolio, before anyone
builds a policy on it. That check belongs in the same script as the fit.

## The decision path

**Start from the diagnosis, not the catalog.** A reliability curve on the
logit scale plus the guardrails (slope, intercept, Spiegelhalter) says
which failure you have; [Why calibration](../concepts/why-calibration.md)
and [Metrics and tests](../concepts/metrics.md) cover the reading.

- **Pure level error, and few events.** One parameter is all the data can
  fund: [`LogitOffset`](../concepts/offset.md), anchored to a long-run
  central tendency. This is the disciplined retreat, not a failure.
- **Slope error.** The parametric family: temperature if the base rate is
  already right, Platt when the level moved too.
- **Curvature, with events in the hundreds.** Isotonic, CIR, or the
  spline. Prefer CIR or the spline where a cutoff or recourse consumes the
  output: step functions turn a decision boundary into a plateau edge that
  moves at every refit (see [Set cutoffs](cutoffs.md)).
- **You need an interval, not a point.** Venn–Abers: `predict_interval()`
  carries a distribution-free guarantee under exchangeability; CVAP when
  the sample is too small to carve a calibration split.
- **The output feeds a masterscale or grade table.** Histogram binning or
  scaling-binning produce grade-shaped output natively, but check
  `is_monotone_` before promising anyone a grade-to-score translation.
- **Unsure, the usual case.** Hand the decision to
  [`CalibratorSelector`](../concepts/auto-selection.md), which scores every
  candidate on out-of-fold data only and writes down the contest. On a
  small portfolio give it a restricted menu: every extra candidate spends
  selection power.

```python
from probcal import (
    BetaCalibrator,
    CalibratorSelector,
    PlattCalibrator,
    TemperatureCalibrator,
)

sel = CalibratorSelector(
    candidates={"temperature": TemperatureCalibrator(),
                "platt": PlattCalibrator(),
                "beta": BetaCalibrator()},
    scoring="log_loss",
    cv=5,
).fit(s_cal, y_cal)

print(sel.best_name_)
print(sel.report_)          # ranked table, fold sd, guardrail flags
print(sel.interpret())      # the winner's parameters, in words
```

Read the runner-up's margin in units of the printed fold standard
deviations: inside one, the contest was a tie and parsimony already
adjudicated it through `complexity_rank`.

## What the choice commits you to

Picking a calibrator picks a downstream contract, which is why the inverse
columns sit in the same table as the fit quality:

- Cutoffs, masterscale bands, and counterfactual targets all run through
  the inverse: the how-to is [Set cutoffs and invert
  maps](cutoffs.md), the theory
  [Inverse maps](../concepts/inverse-maps.md).
- Whatever you fit, its parameters, provenance, and reproduction are
  evidence you can hand to a validator: see
  [Auditability](auditability.md) and
  [Build a validation report](report.md).
- After deployment the map is watched, not assumed:
  [Monitor and act](monitoring.md).
