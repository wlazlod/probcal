# API stability

**Status:** beta on PyPI. Until 1.0, breaking changes bump the minor
version, are listed in the changelog with the reasoning, and — where the
old behavior had legitimate uses — keep an explicit escape hatch (the
`bins=None`/`grid_size=None` pattern). Serialized artifacts have their own,
stronger promise: **every 0.x release reads schema 1**, enforced by
committed golden files in CI (see the *Serialization* chapter).

## Public surface

The public API is exactly the export lists below; anything prefixed with
`_` or not exported is internal and may change without notice.

- **`probcal.__all__`** — the calibrators (`PlattCalibrator`,
  `TemperatureCalibrator`, `BetaCalibrator`, `IsotonicCalibrator`,
  `CenteredIsotonicCalibrator`, `HistogramBinningCalibrator`,
  `ScalingBinningCalibrator`, `BBQCalibrator`, `ENIRCalibrator`,
  `VennAbersCalibrator`, `CrossVennAbersCalibrator`, `SplineCalibrator`),
  `BaseCalibrator`, `UnattainableTargetError`, `CalibratorSelector`,
  `CalibratedModel`, `LogitOffset`, `Chain`, the reliability curves and
  `calibration_belt`, threshold translation (`calibrated_interval_to_raw`,
  `calibrated_bands_to_raw`), attribution repair, `make_pd_portfolio`,
  `expit`/`logit`, and the `metrics`/`monitor` submodules.
- **`probcal.metrics.__all__`** — the 44-symbol metric catalog (proper
  scores, binned and binning-free calibration errors, per-grade backtests,
  Pluto-Tasche most-prudent PDs, Jeffreys upper masterscale bands, the
  recalibration-regression framework, SKCE, `evaluate`).
- **`probcal.monitor.__all__`** — `CalibrationMonitor`, `MonitorStep`,
  `MonitorReport`, `moc_offset`, `moc_offset_from_counts`.
- **`probcal.sklearn`** (extra `probcal[sklearn]`) — `SklearnCalibrator`,
  `CalibratedClassifier`.
- **`probcal.integrations.optbinning`** (extra `probcal[optbinning]`) —
  `calibrate_scorecard`, `CalibratedScorecard`.
- **`probcal.plots`** (extra `probcal[viz]`) — the plotting catalog,
  including `plot_e_process`.

## Conventions that will not silently change

- New parameters are keyword-only; fitted attributes end in `_`.
- `import probcal` depends on numpy and the standard library only —
  enforced by a test, not by review.
- No silent clamps or approximations: numeric shortcuts keep an exact
  escape hatch; refusals name the reason and the alternative
  (`UnattainableTargetError` doctrine).
- Serialization is JSON, never pickle.

## Deprecation policy

Pre-1.0: a deprecated symbol warns (`DeprecationWarning`) for at least one
minor release before removal, with the replacement named in the warning
and the changelog. Behavioral changes that alter numbers ship with a
DECISIONS-referenced changelog entry and, where feasible, a parameter that
recovers the old values.

## Support matrix

| Dimension | Supported | Checked by |
|---|---|---|
| Python | 3.11, 3.12, 3.13 | CI matrix |
| numpy | ≥ 1.26, including 2.x | CI (lockfile tracks latest) |
| scikit-learn (adapter extra) | ≥ 1.4 | CI jobs at 1.4.2 and latest |
| optbinning (integration extra) | ≥ 0.21 | CI job at 0.21.0 |
| treecf (integration extra) | ≥ 0.2.1 | joint smoke test when installed |
