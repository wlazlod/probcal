# API stability

**Status:** beta on PyPI. Until 1.0, breaking changes bump the minor
version, are listed in the changelog with the reasoning, and, where the
old behavior had legitimate uses, keep an explicit escape hatch (the
`bins=None`/`grid_size=None` pattern). Serialized artifacts have their own,
stronger promise: **every 0.x release reads schema 1**, enforced by
committed golden files in CI (see the *Serialization* chapter).

## Public surface

The public API is exactly the export lists below; anything prefixed with
`_` or not exported is internal and may change without notice.

- `probcal.__all__`: the calibrators (`PlattCalibrator`,
  `TemperatureCalibrator`, `BetaCalibrator`, `IsotonicCalibrator`,
  `CenteredIsotonicCalibrator`, `HistogramBinningCalibrator`,
  `ScalingBinningCalibrator`, `BBQCalibrator`, `ENIRCalibrator`,
  `VennAbersCalibrator`, `CrossVennAbersCalibrator`, `SplineCalibrator`,
  `SegmentedCalibrator`),
  `BaseCalibrator`, `UnattainableTargetError`, `CalibratorSelector`,
  `CalibratedModel`, `LogitOffset`, `OffsetEstimate`, `estimate_offset`,
  `offset_from_estimate`, `Chain`, the reliability curves and
  `calibration_belt`, threshold translation (`calibrated_interval_to_raw`,
  `calibrated_bands_to_raw`), attribution repair, `make_pd_portfolio`,
  `expit`/`logit`, and the `metrics`/`monitor` submodules.
- `probcal.metrics.__all__`: the 47-symbol metric catalog (proper
  scores, binned and binning-free calibration errors, per-grade backtests,
  the mixture-LR grade e-test, Pluto-Tasche most-prudent PDs, Jeffreys
  upper masterscale bands, the recalibration-regression framework, SKCE,
  `evaluate`).
- `probcal.monitor.__all__`: `CalibrationMonitor`, `MonitorStep`,
  `MonitorReport`, `AppliedAction`, `moc_offset`, `moc_offset_from_counts`.
- `probcal.sklearn` (extra `probcal[sklearn]`): `SklearnCalibrator`,
  `SklearnOffset`, `CalibratedClassifier`.
- `probcal.integrations.optbinning` (extra `probcal[optbinning]`):
  `calibrate_scorecard`, `CalibratedScorecard`.
- `probcal.plots` (extra `probcal[viz]`): the plotting catalog,
  including `plot_e_process`.
- `probcal.report` (extra `probcal[viz]`): `validation_report`. Not
  re-exported from top-level `probcal`, so `import probcal` and
  `import probcal.report` both stay matplotlib-free at import time.

## Added in 0.3.0

New public symbols relative to 0.2.0, kept here as one running list for
this release regardless of which chapter documents them (extend this list
rather than starting a new one for later 0.3.0 additions):

- `estimate_offset`, `offset_from_estimate`, `OffsetEstimate` (`probcal`):
  offset-only logistic MLE with a Fisher standard error; see *Offset*.
- `metrics.pluto_tasche`, `metrics.pluto_tasche_from_arrays`,
  `PlutoTascheResult`: one-period most-prudent PDs; see *Conservatism*.
- `metrics.jeffreys_upper_bands`: Jeffreys upper masterscale bands; see
  *Conservatism*.
- `monitor.moc_offset`, `monitor.moc_offset_from_counts`: margin-of-
  conservatism offsets; see *Conservatism* and *Monitoring*.
- `metrics.hl_e_test`, `HlEResult`: fixed-sample mixture-LR grade e-test
  (safe Hosmer–Lemeshow analogue); see *Monitoring*.
- `MonitorStep.grade_delta_ci`: per-grade time-uniform confidence sequences.
  `MonitorReport.onset_label` and the `CalibrationMonitor(recommendation_window=)`
  keyword-only constructor parameter (`"since_onset"` default, `"trailing"`
  escape hatch) give drift-onset localization and the window it feeds into
  `report()`'s trailing diagnostics; see *Monitoring*.
- `CalibrationMonitor.apply_recommendation`, `monitor.AppliedAction`: closes
  the report-to-action loop for `kind="re-offset"`; see *Monitoring*.
- `SegmentedCalibrator` (`probcal`): empirical-Bayes (DerSimonian-Laird)
  shrunken per-segment logit offsets on top of a shared base map; see
  *Segmented calibration*.
- `metrics.evaluate(by=)` and `GroupedMetricReport` (metric catalog 46 → 47
  symbols): per-group metric reports plus a pooled report; `plots.
  plot_reliability(by=)` adds the matching faceted reliability grid; see
  the new *Grouped evaluation* guide.
- `metrics.ecce(presorted=)`: keyword-only throughput switch (default
  `False`) declaring `p` already sorted ascending, so the internal sort is
  skipped; `evaluate`'s bootstrap sorts each replicate once and shares that
  order. Results are unchanged when the declaration holds; see *Metrics*'
  "Computational cost".
- `probcal.report.validation_report`: a self-contained HTML/markdown
  validation report (reliability, the metric report, the CORP
  decomposition, and, when given, the rating-grades, grouped-evaluation,
  monitoring, and calibrator-appendix sections); see the new *Validation
  report* guide.
- `corp_reliability` (`probcal`, `curves`), `CorpResult`: the CORP
  (consistent, optimally binned, reproducible) reliability diagram and its
  exact `score == mcb - dsc + unc` decomposition; `reliability_smooth`
  (`probcal`, `curves`), `KernelReliabilityCurve` is a fourth reliability
  construction sharing `smooth_ece`'s fixed-point bandwidth exactly; see
  the new *CORP and score decomposition* chapter.
- `plots.plot_corp`, `plots.plot_mcb_dsc`, `plots.plot_attributes`,
  `plots.plot_murphy`: four new plotting functions (CORP reliability, the
  MCB-DSC plane, the Hsu & Murphy attributes diagram, and Murphy diagrams);
  see *Visualization* and the new *CORP and score decomposition* chapter.
- `metrics.murphy_curve`, `MurphyCurve` (metric catalog 38 → 40 symbols):
  the Ehm, Gneiting, Jordan & Krüger (2016) elementary-score decomposition
  underlying `plots.plot_murphy`; see *Metrics and tests* and
  *Visualization*.
- `BaseCalibrator.__sklearn_is_fitted__` and
  `BaseCalibrator.__sklearn_tags__` (also `__sklearn_is_fitted__` on `Chain`,
  `LogitOffset` and `CalibratedModel`): sklearn's duck-typing hooks, so a
  bare calibrator works with `clone`, `get_tags`, `check_is_fitted` and CV
  loops on sklearn >= 1.6 without the adapter; sklearn is imported inside
  `__sklearn_tags__` only, so `import probcal` stays numpy-only. See the
  *sklearn* guide.

## Added in 0.3.1

New public symbols and behavior relative to 0.3.0, kept here as one running
list for this release regardless of which chapter documents them:

- `_validation.validate_scores` accepts a two-column probability matrix
  (`(n, 2)`, entries in `[0, 1]`, rows summing to 1 within `1e-6`), inherited
  everywhere scores enter the package — every calibrator, `LogitOffset`,
  `Chain`, the metric catalog, and `probcal.sklearn`'s probability mode; see
  the *sklearn* guide.
- `Chain(stages)` accepts unfitted stages, and `Chain.fit(s, y,
  sample_weight=None)` fits them sequentially (supersedes the 0.3.0 note that
  a chain composes only fitted stages); `get_params`/`set_params` gain
  `stages__i__param` nesting. See the *sklearn* guide.
- `SklearnOffset` (`probcal.sklearn`): a `TransformerMixin`/`BaseEstimator`
  wrapping `LogitOffset`; see the new "Stacking an offset" section of the
  *sklearn* guide.
- `positive_column` keyword-only parameter on `SklearnCalibrator` and
  `SklearnOffset`, plus the orientation `UserWarning` fired at fit; see the
  *sklearn* guide.
- `LogitOffset.fit(y=)` keyword-only parameter, accepted and ignored, giving
  `Chain.fit`'s sequential protocol one call shape across stages.

## Conventions that will not silently change

- New parameters are keyword-only; fitted attributes end in `_`.
- `import probcal` depends on numpy and the standard library only,
  enforced by a test rather than by review.
- No silent clamps or approximations: numeric shortcuts keep an exact
  escape hatch; refusals name the reason and the alternative
  (`UnattainableTargetError` doctrine).
- Serialization is JSON, never pickle.

## Deprecation policy

Pre-1.0: a deprecated symbol warns (`DeprecationWarning`) for at least one
minor release before removal, with the replacement named in the warning
and the changelog. Behavioral changes that alter numbers ship with a
detailed changelog entry and, where feasible, a parameter that
recovers the old values.

0.3.0 deprecated nothing: every new symbol above is additive, and no
existing symbol carries a `DeprecationWarning`; the policy stood unused
this release.

## scikit-learn estimator checks

`sklearn.utils.estimator_checks` generates multi-column feature matrices of
arbitrary reals. Its data model contains no score-level estimator, so most of
the corpus cannot be run against `SklearnCalibrator` at all, the same kind of
domain restriction imbalanced-learn declares for its resamplers. Those checks
are declared inapplicable through sklearn's own mechanism
(`expected_failed_checks` on 1.6+, `_more_tags()["_xfail_checks"]` below it),
each entry naming the data that check generates and which part of the
score-level contract it violates. They are inapplicable, not known failures;
the checks whose data the contract does admit run live and pass.

Every inapplicable check with a score-level analogue is re-implemented on
valid `(n,)` probability data in `tests/test_sklearn_mirror_checks.py`: fit
idempotence, no mutation of the passed arrays, `__dict__` unchanged by the
predict-side methods, pickle (adapter) and JSON (core) round trips,
clone-then-fit, subset and sample-order invariance, and integer
`sample_weight` equal to row duplication over every registered calibrator
class, with the exceptions (CV- and quantile-based internals) named in a
table together with their measured tolerance. Checks with no analogue at all
(multiclass, pairwise, sparse) are listed in that module's docstring with one
line each. The score-level contract itself is pinned by
`tests/test_calibrator_protocol.py`.

## Support matrix

| Dimension | Supported | Checked by |
|---|---|---|
| Python | 3.11, 3.12, 3.13 | CI matrix |
| numpy | ≥ 1.26, including 2.x | CI (lockfile tracks latest) |
| scikit-learn (adapter extra) | ≥ 1.4 | CI jobs at 1.4.2 and latest |
| scikit-learn (bare-core duck typing, no adapter) | ≥ 1.6 only | `tests/test_sklearn_duck.py` (`importorskip(minversion="1.6")`); runs in the main CI matrix (latest sklearn), skipped on the 1.4.2 `sklearn-min` job |
| optbinning (integration extra) | ≥ 0.21 | CI job at 0.21.0 |
| treecf (integration extra) | ≥ 0.2.4 | joint smoke test when installed |
