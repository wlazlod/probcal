# API stability

--8<-- "docs/_snippets/status.md"

## Two kinds of change

**API-breaking.** A public symbol is removed or renamed, a signature
changes incompatibly, or a documented behaviour changes meaning. Before
1.0 this bumps the minor version, is listed in the changelog with the
reasoning, and goes through the deprecation policy below.

**Numerically visible.** A default estimator, a numeric path, or a fitted
internal changes so that computed numbers move while every public call
still works. This may ship in a patch release. It always comes with a
changelog entry stating the size of the change and, where feasible, a
parameter that recovers the old values (the `bins=None`/`grid_size=None`
pattern).

What a patch release may therefore change: metric defaults and numeric
paths with an escape hatch; the shape of fitted internals that are not
part of the calibrator protocol (0.1.3 changed `ENIRCalibrator.path_solutions_`);
performance, documentation, and packaging. What it may not: public
signatures, exported names, the calibrator protocol, and the ability to
read every schema-1 artifact.

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
  `offset_from_estimate`, `Chain`, `Masterscale`, `GradeTable`, the
  reliability curves and `calibration_belt`, threshold translation
  (`calibrated_interval_to_raw`, `calibrated_bands_to_raw`,
  `build_masterscale`), attribution repair, `make_pd_portfolio`,
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

New public symbols per release are listed in the changelog under *Added*.

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

No symbol currently carries a `DeprecationWarning`; the changelog lists
any release in which one does.

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
