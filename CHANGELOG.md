# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).
Each release opens with a short summary; the itemized entries sit under *Details*.

## [Unreleased]

### Documentation

- Every page whose snippets draw on the held-out data vocabulary (`s_cal`, `y_cal`, ...) now opens with a collapsed box defining those names, each block names the ones it reads in its first comment, and a docs test enforces both. The monitoring guide states the assumptions behind the anytime-valid guarantee; the serialization chapter states what "bit-identical" covers and that metric estimates are not promised across releases. *API stability* states the two-tier change policy (API-breaking versus numerically visible) up front and drops the per-release symbol lists, which the changelog already carries; the status paragraph is one shared snippet. The calibrator catalog is split into a properties table and a guidance table. Every scikit-learn version floor links to the support matrix.

## [0.3.2] - 2026-09-07

Documentation only; no code changes. The README on PyPI now matches the docs site.

<details markdown="1">
<summary>Details</summary>

### Documentation

- README leads with what scikit-learn does not produce (logit-scale guardrails, the auditable offset stage, per-grade backtests) and carries a measured table against scikit-learn and netcal on the Taiwan credit-card default dataset; the calibrator table moved to `guide/choosing.md`.
- The comparison benchmark gains that dataset (scorecard base model with integer points, an 8-band masterscale), `PlattCalibrator` and `IsotonicCalibrator` rows, and per-method grade-size tables.
- Both quickstarts fit on one synthetic portfolio and report on a held-out draw; the previous versions scored the fitting rows. *Getting started* defines every name its blocks use.
- API reference adds `Chain`, `probcal.monitor`, `expit`/`logit`, and an *Integrations* page (`probcal.sklearn`, `probcal.integrations.optbinning`); a docs test fails if a public symbol is missing from the rendered reference.
- New *Glossary* page; one *Tutorials* section for both notebooks; plain phrasing on reader pages.
- This changelog opens each release with a summary and collapses the itemized entries.

</details>

## [0.3.1] - 2026-09-05

A `predict_proba` matrix now goes straight into any calibrator, metric, or offset; no column slice first. `Chain` can be fitted from unfitted stages, not only assembled from fitted ones, and works inside sklearn pipelines. `SklearnOffset` adds the offset as its own pipeline step with its own lifecycle. No breaking changes: every previously accepted input and object behaves identically.

<details markdown="1">
<summary>Details</summary>

### Added

- **Two-column probability-matrix input, core-wide.** Every place scores enter the package (calibrator `fit`/`predict_proba`/`point_inverse`, `LogitOffset`, `Chain`, the whole metric catalog, `probcal.sklearn`'s probability-input mode) now accepts an `(n, 2)` array whose rows sum to 1 within `1e-6` and uses column 1, the shape every `predict_proba` returns. One shared validator, one rejection message.
- **Fittable `Chain`.** `Chain(stages)` accepts unfitted stages and gains `Chain.fit(s, y, sample_weight=None)`, which fits the head calibrator and then each offset in turn on the running calibrated probabilities. Every stage is always refit; composing already-fitted objects is how a stage stays frozen. `stages` is stored verbatim, `calibrator_`/`offsets_` are read-only properties, and every reading method raises the standard not-fitted `RuntimeError` before a fit. `get_params`/`set_params` gain `stages__i__param` nesting (`LogitOffset` gained `get_params`/`set_params` to support it). The serialization payload is unchanged. Both sklearn adapters clone the calibrator prototype via `sklearn.base.clone`, so `SklearnCalibrator(calibrator=Chain([...]))` and `CalibratedClassifier(calibrator=Chain([...]))` work with an unfitted chain.
- **`SklearnOffset`.** `probcal.sklearn.SklearnOffset(delta=None, target_mean=None, *, positive_column=1)`, a `TransformerMixin`/`BaseEstimator`: `fit(X, y=None, sample_weight=None)` fits an inner `LogitOffset` as `offset_`; `transform` returns the shifted `(n, 1)` column, `predict_proba` the `(n, 2)` matrix, `to_dict` the fitted offset's payload. Runs the sklearn estimator checks with its own inapplicability table.
- **`positive_column`.** Keyword-only on `SklearnCalibrator` and `SklearnOffset`, naming which column of a two-column input is the positive class. At fit time a calibrator (which sees `y`) warns once if the selected column's event mean is below the non-event mean, naming the flip as the likely fix; probcal warns and never auto-flips. `input="logit"` stays single-column. `n_features_in_` (1 or 2) is recorded at fit and enforced at predict.
- **`LogitOffset.fit(y=)`.** `fit` accepts an optional keyword-only `y=None`, ignored in both modes, so `Chain.fit` has one call shape across calibrator and offset stages.

### Fixed

- A bare `(n, 2)` matrix from `model.predict_proba(X)` raised in every `fit`/`predict_proba` call, forcing a manual column slice before every calibration step. The two-column input rule removes that friction end to end, pinned against a real `LGBMClassifier` matrix flowing through `SklearnCalibrator` and `SklearnOffset` in one `Pipeline`.

### Documentation

- **Reader-oriented reorganization.** The nav now answers "who is arriving": *Learn* (install, how it works, the 20-minute walkthrough), *Guide* (task-ordered along calibrate → evaluate → invert/decide → monitor → act → report, with the integrations beside it), *Concepts* (regrouped theory). The front page states that lifecycle once, linking each stage to its guide page; no page changed URL. Three assembled guide pages are new: `guide/choosing.md` (the calibrator catalog, every factual column pinned by a test), `guide/cutoffs.md` (policy PD → raw cutoff → masterscale → recourse), `guide/auditability.md` (artifact → what it proves → how to verify it, plus a worked reviewer session). Every Python block in the docs is executed page by page in CI against a fixed data vocabulary (convention in `docs/README.md`). Every function in `probcal.plots` has a committed, alt-texted figure on a prose page. `guide/report.md` shows a full sample validation report.
- `guide/sklearn.md`: each tier states its input shape (`(n,)`, `(n, 1)`, or a `predict_proba` matrix `(n, 2)`), covers `positive_column` and the orientation warning, and gains "Stacking an offset": one `Chain` step, or a separate `SklearnOffset` step swapped in from `apply_recommendation`'s output without touching the calibrator. `guide/monitoring.md` gains "Re-offsetting inside a pipeline". `concepts/offset.md` describes `Chain.fit`'s sequential fitting: stages fit on the same calibration data, the offset anchoring the calibrator's in-sample output exactly as `CalibratedModel.offset_to` does; no cross-fitting, no automatic MLE offset.

</details>

## [0.3.0] - 2026-08-29

The governance release. Validation reports, grouped evaluation, segmented calibration with empirical-Bayes shrinkage, and the monitoring loop closed: `apply_recommendation`, a drift-onset estimate, per-grade confidence sequences, and margin-of-conservatism offsets. Pluto–Tasche most-prudent PDs and Jeffreys masterscale bands for low-default portfolios. CORP reliability diagrams, the MCB–DSC plane, attributes and Murphy diagrams. `evaluate` runs 3.5× faster with bit-identical point estimates. No breaking API changes; every 0.2.0 call and serialized file keeps working. One default changed: the monitor's post-alarm diagnostic window now starts at the estimated drift onset (`recommendation_window="trailing"` restores 0.2.0 behaviour). One numerical fix: a *weighted* `PlattCalibrator` fit differs from 0.2.0; every unweighted fit, and hence every default and golden file, is unchanged.

<details markdown="1">
<summary>Details</summary>

### Added

- **Grouped evaluation.** `metrics.evaluate(..., by=None)` and `GroupedMetricReport(pooled, groups, reports, counts)` (re-exported from `probcal.metrics`). `by=None` is the existing report, byte-identical to 0.2.0; given per-observation group labels, each sorted group gets its own report (seed `seed + 1000 * i`), identical to calling `evaluate()` by hand on that slice, plus a `pooled` report. A single-outcome-class group raises `ValueError` naming the group. `to_frame()` returns a list of `{group, metric, value, ci_low, ci_high}` dicts, or a pandas DataFrame when pandas is importable. Side-by-side reporting only: no cross-group test or multiplicity correction. `plots.plot_reliability(..., by=)` draws the matching faceted grid (pooled panel first) and returns the Figure. New chapter `guide/groups.md`.
- **Validation reports.** `probcal.report.validation_report(y, p, *, calibrator=None, monitor=None, grades=None, by=None, title=None, path=None, format="html", n_boot=200, seed=42) -> str`: a self-contained validation document assembled from the existing public API. HTML is a single file with figures embedded as data URIs and no scripts or external links; `format="markdown"` writes GFM tables plus PNG files to `<path stem>_figures/`. Sections are omitted when their input is absent: reliability (CORP and smooth) and the metric table always render; the rating-grades section (Jeffreys backtest, Pluto–Tasche, Jeffreys upper bands) with `grades`; grouped evaluation with `by`; monitoring with `monitor`; a collapsible calibrator appendix (`to_json()` plus `interpret()`) with `calibrator`. `n_boot`/`seed` drive every resampling site, so two builds from the same inputs are byte-identical apart from the timestamp line. Labels are HTML- or GFM-escaped. `import probcal.report` stays free of matplotlib until a figure renders; calling it needs `probcal[viz]`. Not re-exported from top-level `probcal`. New chapter `guide/report.md`.
- **Segmented calibration.** `probcal.SegmentedCalibrator(base=None, *, unseen="global")`: empirical-Bayes shrunken per-segment logit offsets on top of a shared base map (default `BetaCalibrator()`). Per segment, an offset-only logistic MLE against the base map's predictions; between-segment variance by the DerSimonian–Laird estimator; each segment's offset shrunk by `tau2/(tau2+se**2)`, so a noisy small segment shrinks toward the base map and `tau2=0` recovers complete pooling. `fit`/`predict_proba` take keyword-only `segments=`; an unseen label at predict time is `delta=0` (`unseen="global"`) or a `ValueError` (`unseen="raise"`). `interval_inverse`/`point_inverse` take `segment=`. `Chain` has no `segments=` slot, so `Chain([seg, ...])` always predicts through the global map (documented limitation). Fully serializable with its own golden file. On a 6-segment simulation with sizes 30 to 3,000, mean MSE 0.124 for empirical Bayes versus 0.235 (no pooling) and 0.168 (complete pooling). New chapter `concepts/segmented.md`.
- **Closing the monitoring loop.** `CalibrationMonitor.apply_recommendation(target=None) -> AppliedAction` (`monitor.AppliedAction`, re-exported from `probcal.monitor`): for `kind="re-offset"`, estimates the log-odds shift by maximum likelihood on the same window `report()`'s diagnostics use, composes the fitted `LogitOffset` onto `target` (a `Chain` or a `CalibratedModel`; `target` is never mutated), and returns a fresh `CalibrationMonitor` with the same constructor parameters, because the old e-process's guarantee no longer holds once the pipeline changes. `kind="re-fit"`/`"none"` return no offset; automatic re-fitting stays out of scope. `AppliedAction` carries `kind`, `offset`, `composed`, `monitor`, the batch `window`, and an `audit` dict of fingerprints plus `delta`/`se`, and is itself serializable.
- **Drift-onset localization.** Each `MonitorStep` records `log_e_increment`, the batch's additive log-LR contribution; `monitor._onset.estimate_onset(increments) -> int` is a backward-CUSUM argmax (an estimate, not a change-point test). `report()` runs it after an alarm and exposes `MonitorReport.onset_label: str | None`, with a matching sentence in `reasoning`. Steps loaded from a pre-0.3 payload carry `log_e_increment = None`; such a monitor reports `onset_label = None` and falls back to the trailing window. Median localization error 1 batch on a 24-batch simulation with drift at batch 12.
- **Per-grade confidence sequences.** Each grade in the `grade` array gets its own time-uniform confidence sequence for its own offset, surfaced as `MonitorStep.grade_delta_ci: dict[str, tuple[float, float] | None]` (empty without grades). `plots.plot_e_process(report, *, grades_panel=False, ax=None)`: `grades_panel=True` adds a second axes with every grade's band; the default renders pixel-identically to 0.2.0. Serialization is additive; a 0.2.0 monitor payload still loads and continues.
- **Per-grade e-test audit.** `metrics.hl_e_test(y, p, grades, *, mixture_grid=(0.1, 0.25, 0.5, 1.0), sample_weight=None) -> HlEResult`: a fixed-sample mixture-LR e-value per rating grade (the construction `CalibrationMonitor`'s offset e-process uses, without the predictable plug-in), multiplied across grades; `e_value`, `p_value = min(1, 1/e_value)`, per-grade `e_grade`, `construction="mixture-lr"`. Named a safe Hosmer–Lemeshow *analogue*: Henzi, Puke, Dimitriadis & Ziegel (2024) motivated it and is not the implemented construction. Simulated type-I `P(e >= 20) = 0.0025`. Re-exported from `probcal.metrics.grade`.
- **Audited offset MLE.** `probcal.offset.estimate_offset(y, p, *, sample_weight=None) -> OffsetEstimate` (`delta`, `se`, `n`, `events`, `weight_sum`) and `offset_from_estimate(est, p) -> LogitOffset`, both re-exported from `probcal`: the offset-only logistic MLE with a Fisher-information standard error, the companion to `LogitOffset`'s policy-driven `target_mean` mode. The MLE and `LogitOffset(target_mean=mean_w(y))` share one root-finder and agree bit-for-bit.
- **Margin-of-conservatism offsets.** `probcal.monitor.moc_offset(monitor_or_report, *, level=None) -> LogitOffset` reads the upper end of the monitor's confidence sequence for the current offset and returns a fitted `LogitOffset(delta=hi)` (a `level` needs the live monitor; a frozen `MonitorReport` raises `TypeError`). `moc_offset_from_counts(y, p, *, level=0.9, sample_weight=None)` re-anchors `p`'s mean at the one-sided Jeffreys posterior upper quantile of the observed event rate. Both re-exported from `probcal.monitor` and `probcal`.
- **Jeffreys masterscale bands.** `metrics.jeffreys_upper_bands(y, p, grades, *, level=0.9, order=None) -> dict[str, tuple[float, float]]`: the per-grade Jeffreys upper bound packaged as the contiguous `{grade: (lo, hi)}` table `thresholds.calibrated_bands_to_raw` consumes; a zero-default grade still gets a strictly positive `hi`. Bounds are monotonized by weighted isotonic regression in `order` (default: grades sorted by mean `p`, best first), with a `UserWarning` only when something changed.
- **Pluto–Tasche most-prudent PDs.** `metrics.pluto_tasche(grade_n, grade_d, *, confidence=0.9, grades=None) -> PlutoTascheResult` and `pluto_tasche_from_arrays(grades, y, *, order, confidence=0.9, sample_weight=None)`: the one-period most-prudent PD for ordered grades, pooling each grade with every worse grade and taking the one-sided Clopper–Pearson upper bound. Accepts an all-zero `y`. `pd_upper` is replaced by its cumulative maximum best to worst; `PlutoTascheResult.monotonized` reports whether that fired. Simulated minimum per-grade coverage 0.92 at confidence 0.9. New chapter `concepts/conservatism.md`.
- **CORP reliability diagrams.** `curves.corp_reliability(y, p, *, sample_weight=None, bands="consistency", level=0.9, n_resamples=200, random_state=42) -> CorpResult`: the PAV recalibration map of Dimitriadis, Gneiting & Jordan (2021), weighted throughout, plus the exact `score == mcb - dsc + unc` decomposition of both Brier score and log loss. `bands="consistency"` resamples under the calibration null, `"confidence"` bootstraps the data, `None` skips bands; both are pointwise, not simultaneous. `plots.plot_corp(result, *, scale="probability", show_decomposition=True, ax=None)`. New chapter `concepts/corp.md`.
- **MCB–DSC plane.** `plots.plot_mcb_dsc(candidates, *, score="brier", ax=None)`: each candidate at `(DSC, MCB)` with iso-score diagonals, from a `{name: (y, p)}` mapping or a fitted `CalibratorSelector`'s `report_`. `SelectionReport` gains `mcb`, `dsc`, `unc` columns (`None` on pre-0.3 reports).
- **Kernel reliability curves.** `curves.reliability_smooth` (`KernelReliabilityCurve`): a reliability construction whose bandwidth is `metrics.smooth_ece`'s fixed-point `sigma_star`, so `curve.smooth_ece` reproduces the metric bit-for-bit; Nadaraya–Watson event rate and density at that bandwidth, with a seeded bootstrap ribbon (`n_boot`, default 100; `0` disables).
- **Reliability plot upgrades.** `plots.plot_reliability` renders a `KernelReliabilityCurve` passed as `smooth` as a density-weighted variable-width curve with the miscalibration area shaded and an `smECE` readout; new keyword-only `stats` (`bool | MetricReport`) swaps the classic box for `n, events, intercept, slope, ICI, smECE, Brier`, or the report's values with CIs; new `risk_dist` (`"rug" | "split" | None`) selects the rug or a 30-bin spike histogram. The pre-existing default call renders pixel-identically to 0.2.0.
- **Attributes diagram.** `plots.plot_attributes(y, p, *, method="binned", n_bins=10, scale="probability", sample_weight=None, ax=None)`: identity, climatology lines, the no-skill line, and the shaded positive-Brier-skill region; `method="corp"` overlays the CORP PAV fit.
- **Murphy diagrams.** `metrics.murphy_curve(y, p, *, thresholds=513, sample_weight=None) -> MurphyCurve`: the Ehm, Gneiting, Jordan & Krüger (2016) elementary-score curve; `2 * integral` equals the Brier score. `plots.plot_murphy(curves, *, diff=False, n_boot=200, random_state=42, ax=None)`; `diff=True` on exactly two `(y, p)` pairs draws the pointwise difference with a paired-bootstrap band.
- **sklearn duck typing on the bare core.** `BaseCalibrator` gains `__sklearn_is_fitted__` and `__sklearn_tags__`; `Chain`, `LogitOffset`, `CalibratedModel` gain `__sklearn_is_fitted__`. On sklearn ≥ 1.6 a bare probcal calibrator works in `clone`, `get_tags`, `check_is_fitted`, and `cross_val_score(BetaCalibrator(), s.reshape(-1, 1), y, ...)` with no adapter import; contexts needing the `(n, 2)`/`classes_` convention still go through `probcal.sklearn`. The sklearn import lives inside `__sklearn_tags__`, so `import probcal` stays numpy + stdlib.
- **Mirror conformance suite for the sklearn adapter.** Every generic estimator check declared inapplicable to a score-level estimator is re-implemented on valid probability data: fit idempotence, no input mutation, pickle and JSON round trips, clone-then-fit equality, subset and order invariance, and integer `sample_weight` equal to row duplication over all 14 `BaseCalibrator` subclasses, each at a measured tolerance (exact for centered isotonic, ENIR, isotonic, temperature, Venn–Abers; structural deviations documented for the binning, spline, BBQ, CVAP, and selector classes).

### Changed

- treecf guide: the cross-repo work list is documented as implemented in treecf 0.2.4; the `probcal[treecf]`/`all` floor rises to `treecf>=0.2.4`.
- `CalibrationMonitor.report()` computes its post-alarm trailing-window diagnostics on batches from the estimated drift onset onward instead of the full history (`recommendation_window="since_onset"`, the new default); with `plug_in_window` also set, the window starts at the later of the two. `recommendation_window="trailing"` restores 0.2.0 behaviour exactly for those diagnostic inputs; `onset_label` is populated under both.
- **Single-column score input.** A `(n, 1)` column vector is accepted everywhere scores enter the package and ravelled first; results are bit-identical to passing the 1-D array. Every other 2-D shape still raises ("expected 1-D scores (or a single column); got shape ..."); `validate_binary_y` stays 1-D only.
- **Per-check applicability reasons for the sklearn estimator checks.** Each of the 36 declared checks carries its own reason naming the data it generates and which part of the score-level contract it violates; the declared set is unchanged. `docs/api-stability.md` gains a "scikit-learn estimator checks" section stating the position.
- **`guide/sklearn.md` rewritten around three tiers**: bare duck typing (sklearn ≥ 1.6), `SklearnCalibrator` (pipelines, `VotingClassifier`, stacking), `CalibratedClassifier` (out-of-fold orchestration), plus the prefit recipe via `FrozenEstimator`. Every code block is included from an executed test file, so the guide cannot drift. `CalibratedClassifier.to_json(path=None, *, indent=2)` now delegates to `calibrator_.to_json(...)`.

### Fixed

- **`PlattCalibrator`'s smoothing prior now reads weighted class mass.** The Lin–Lin–Weng targets used raw row counts while the IRLS was weighted, so integer weights did not equal row duplication (2.0e-2 apart). They now use weighted class totals: every unweighted fit, default, and golden file is bit-identical, and weighting a row matches repeating it to 2.5e-16. `ScalingBinningCalibrator` keeps its 8.5e-2 deviation, now entirely from the equal-mass bins on the Platt output.
- **Sample-weight delivery into `CalibratedClassifier`'s inner fits.** `fit(X, y, sample_weight=w)` raised `TypeError` when the base estimator's `fit` had no `sample_weight` parameter and recovered only through a silent unweighted refit. It now checks first (sklearn's own `has_fit_parameter` precedent, widened for metadata routing), warns once naming the class when the base cannot take weights, runs the fold fits and the refit unweighted, and fits the calibrator with the weights. With routing on, an undeclared request raises sklearn's `UnsetMetadataPassedError` instead of dropping the weights. Verified on scikit-learn 1.4.2, 1.6.1, and 1.9.0.

### Performance

- `metrics.evaluate`'s bootstrap loop sorts each replicate by prediction once and shares that order: `_math.loess` and `metrics.ecce` gain a keyword-only `presorted` flag (default `False`, so direct calls are unchanged); `ece`, `ece_debiased`, and `mce` share one binning pass; `ece_sweep`'s scan reads per-bin sums off prefix sums; and the LOESS anchor evaluation is vectorized in blocks. **Point estimates are unchanged bit-for-bit.** Bootstrap percentile CI bounds may move in their last bits (at most 3.9e-11 relative on a `n=10,000`/`n_boot=1000` run). The one intended numeric difference (tricube weights cubed by multiplication rather than `** 3`) is at most 2.3e-16 relative on a well-conditioned window; the rank-deficient corner is discussed in `concepts/metrics.md` under "Computational cost". Measured: `evaluate(n=10,000, n_boot=1000)` over the full catalog fell from **304.2s to 86.9s** (3.5×); a full-catalog replicate costs 0.089s, of which the ICI family is 58%, so `metrics=` excluding `ici`/`e50`/`e90`/`emax` remains the largest lever. `docs/scripts/benchmarks.py` gains per-metric rows.

### Documentation

- `concepts/conservatism.md` consolidated into one chapter for all four conservatism mechanisms, with a "When each mechanism applies" table, a "Margin of conservatism" section with a runnable `Chain([calibrator, moc_offset(mon)])` example, and an explicit warning that every bound on the page is a one-period estimator that understates uncertainty under cross-obligor correlation. `concepts/offset.md` gains "Estimating delta with a standard error". The end-to-end notebook gains a "Conservative margins" section. `docs/api-stability.md` gains an "Added in 0.3.0" running list.
- `concepts/monitoring.md` reorganized in build order: e-process and confidence-sequence theory, per-grade CS, the drift-onset estimate and its since-onset window, `apply_recommendation`, margin-of-conservatism offsets, the fixed-sample `hl_e_test` audit, and the simulation tables (per-grade CS coverage 1.000 for both drifted and stable grades; onset localization median error 1 batch). The end-to-end notebook gains a "Monitoring closed loop" section.

</details>

## [0.2.0] - 2026-08-23

Serialization, monitoring, and the sklearn adapter. Every fitted object round-trips through versioned JSON with fingerprints, never pickle, under a golden-file promise that every 0.x release reads schema 1. `probcal.monitor` watches deployed calibration with an e-process whose alarm keeps its type-I guarantee at every look. `probcal.sklearn` wraps calibrators as estimators and offers a `CalibratedClassifierCV` drop-in. `Chain` composes a calibrator with offsets for recourse engines; `probcal.integrations.optbinning` calibrates scorecards with exact point cut-offs. Three inverse-map fixes at the probability boundary. The measured comparison page and the end-to-end notebook arrive with this release. `smooth_ece` values for `n <= 8192` may differ from 0.1.3 at the ~1e-4 level (`bins=None` recovers them).

<details markdown="1">
<summary>Details</summary>

### Fixed

- Step calibrators' right generalized inverse (`IsotonicCalibrator`, `HistogramBinningCalibrator`) returned the next block's left edge, a supremum not in the preimage, so a consumer treating bounds as closed landed one block too high; it now returns the largest float below the boundary, which is attained.
- `point_inverse` at the probability boundary: `p ∈ {0, 1}` was clipped to `[1e-12, 1 − 1e-12]` before inversion, and a certified root with `|z| > logit(1 − 1e-12) ≈ 27.63` was returned as `σ(z)`, which rounds to 0.0/1.0 and cannot round-trip. Both now raise `UnattainableTargetError`: boundary targets with the offending values named; over-range roots only for `space="probability"`, pointing at `space="logit"`, where the answer is exact.
- `LogitOffset.interval_inverse` returned raw bounds below the `1e-12` clip that `transform` cannot round-trip; bounds beyond the representable output range now collapse to the full-range sentinels (0/1, ±inf) as in `BaseCalibrator.interval_inverse`, and an interval entirely outside that range raises `UnattainableTargetError`.

### Changed

- `smooth_ece`'s lattice path engages for every call with a non-degenerate logit range instead of only `n > bins`, removing the size cliff at typical calibration-set sizes (0.1.3: ~1s per call at n=4000). Path selection: bin at `bins` (default 8192), accept when `sigma* >= 8·width`, refine adaptively once, fall back to exact only when refinement is infeasible. Values for `n <= bins` may differ from the 0.1.2/0.1.3 exact grid at the ~1e-4 level on typical portfolios (measured ≤ 2.4e-4), larger on wide clipped-logit-range data where the lattice value is the better one; `bins=None` recovers old values bit-for-bit.

### Added

- Versioned JSON serialization on every fitted object: `to_dict`/`from_dict`/`to_json`/`from_json` with a schema-checked envelope (`probcal_schema` 1, `probcal_version`, class name, params, state, fit metadata) and a class registry for dispatch; `fingerprint()` (SHA-256, version- and timestamp-blind) plus `fit_meta["data_fingerprint"]` (permutation-invariant hash of the training triple). Covers all calibrators, `LogitOffset`, `CalibratorSelector`, and `CalibratedModel` (which stores a model *reference*, never the model object; reattach via `from_dict(d, model=...)`; the ensemble flow refuses serialization). Golden files verified in CI. New chapter `concepts/serialization.md`.
- `CalibratedModel(model_id=...)` names the wrapped model in the serialized reference.
- `probcal.monitor` (numpy + stdlib): `CalibrationMonitor`, anytime-valid calibration monitoring by e-processes (alarm at `E >= 1/alpha` has type-I error `<= alpha` at every stopping time). Components: offset plug-in with a `±{0.1, 0.25, 0.5, 1.0}` mixture, Cox shape plug-in, per-grade offset processes; a time-uniform confidence sequence for the current offset; sticky alarm, anytime p-value, `MonitorStep`/`MonitorReport`, a diagnostic re-offset/re-fit recommendation, JSON serialization with bit-exact resume, and `plots.plot_e_process`. Simulated at 2000 runs × 24 batches × n=2000: worst type-I 0.0275 at alpha 0.05, median detection delay 2 batches for `delta=0.4`, CS coverage 0.992+. New chapter `concepts/monitoring.md`.
- `probcal.sklearn` (extra `probcal[sklearn]`, scikit-learn ≥ 1.4): `SklearnCalibrator`, a probcal calibrator as an sklearn estimator over a single score column (`transform` ends a `Pipeline`; `input="logit"` accepts raw margins; the fitted object is `calibrator_`), and `CalibratedClassifier`, the `CalibratedClassifierCV(ensemble=False)` drop-in on probcal calibrators (OOF via `cross_val_predict` or `cv="prefit"`, one map, full refit; `method="decision_function"` maps through `expit`; binary only), which also exposes the probcal calibrator protocol by delegation. Estimator checks run in CI on sklearn 1.4 and latest with inapplicable checks declared. New chapter `guide/sklearn.md`.
- `probcal.Chain`: model-free composition of a fitted calibrator with `LogitOffset` stages, exposing the full calibrator protocol (exact inverses through every stage, composed affine coefficients, concatenated `interpret()`, JSON, fingerprints); `CalibratedModel.chain_` returns the equivalent chain.
- `probcal.integrations.optbinning` (extra `probcal[optbinning]`, ≥ 0.21): `calibrate_scorecard` → `CalibratedScorecard`, calibrated PD over an untouched scorecard with `points_affine_coeffs_` recovered to machine precision (`rounding=True` warns and falls back to `interval_inverse`), an exact `masterscale` from PD bands to point cut-offs, and provenance JSON binding the calibration layer to the scorecard by fingerprint. New chapters `guide/optbinning.md` and `guide/treecf.md`; `probcal[treecf]` extra.
- Protocol conformance: every inverse-capable object accepts `interval_inverse(lo, hi, space="logit", buffer_logit=...)` by keyword with `lo=0`/`hi=1` → ∓inf and buffered-empty refusal; the step-calibrator generalized-inverse contract is spelled out in the inverse-maps chapter.
- `CalibratorSelector` is now a `BaseCalibrator` subclass: same surface plus inherited `get_params`/`set_params`, serialization, and inverse maps delegating to the refitted winner; `is_monotone_` mirrors the winner.
- `ENIRCalibrator.fit` emits a single `UserWarning` above 50,000 unique scores stating the expected fit minutes.
- README "Calibrators at a glance" table: one-line scaling note per method.

### Documentation

- End-to-end notebook on a rare-event portfolio (`docs/notebooks/pd_end_to_end.ipynb`): GBM baseline → logit-scale reliability → selection with CIs → Jeffreys backtests → auditable offset → policy-to-raw translation and a treecf counterfactual → monitoring with an injected drift → JSON round-trips. Home Credit via `kagglehub` with a documented synthetic fallback that executes in CI.
- Comparison benchmark (`docs/benchmarks/comparison.md`, `docs/scripts/comparison.py` under `probcal[bench]`): probcal vs sklearn sigmoid/isotonic, netcal BBQ/ENIR/beta, and betacal on five OpenML datasets at 1.5–30% event rates, with bootstrap CIs, Jeffreys grade pass rates, and fit times, including where probcal loses.
- Docs restructured into Guide, Concepts, Tutorials, Benchmarks, and API reference, plus `docs/api-stability.md` (public surface, deprecation policy, support matrix). The internal decision log is not published.
- README front page rewritten around the wedge with the end-to-end notebook above the fold.

### Performance

- `smooth_ece` measures 1.0–3.6ms for n ∈ [64, 8192], 3–6ms at n=10⁴–10⁵, and ~43ms at n=10⁶. `evaluate(n=6000, n_boot=1000)` measures 198.9s (0.1.3: ≈10min).

</details>

## [0.1.3] - 2026-08-16

The performance release. The ICI family fits its LOESS smoother at 512 anchors instead of every observation (`ici` at n=50,000: 192s → 1.2s), `smooth_ece` smooths a pre-binned measure (seconds → milliseconds), and ENIR's memory is bounded at any size. `grid_size=None` and `bins=None` recover the exact pre-0.1.3 values and cost. `point_inverse` gives an exact single-point preimage for affine and beta maps. `evaluate`'s bootstrap now stratifies by class (`stratify=False` restores i.i.d.), and four silent defects in weighting, bootstrap degeneracy, and selector parsimony are fixed.

<details markdown="1">
<summary>Details</summary>

### Fixed

- `ENIRCalibrator` could exhaust memory well before m=50,000 because `path_solutions_` retained every breakpoint's full-length solution; the path solver now retains only the lowest-BIC `max_solutions` (default 256) breakpoints. A pruning bound that crashed on sub-unit total `sample_weight` is now disabled in that case instead of raising.
- `sample_weight` was accepted but silently had no effect on `e50`/`e90` and on `reliability_summary`'s internal `e90`; both now honor weights via a Hazen-position `weighted_quantile`.
- `evaluate`'s bootstrap silently substituted the point estimate for degenerate single-class resamples, narrowing the reported CI; a degenerate draw is now redrawn (up to 100 times, then `RuntimeError`).
- `CalibratorSelector`'s parsimony tie-break ranked every user-supplied candidate last regardless of complexity; it now reads `complexity_rank` off the candidate instance.
- `smooth_ece`'s binned path evaluated the smoothed measure on the exact path's 257-point grid, which aliased against the bin lattice and tripped the small-bandwidth guard on every call, so every reported value was correct but cost ~5.7–6.1s regardless of `n`. The binned path now evaluates natively on its own lattice with one adaptively sized refinement; measured n=10⁴: ~5.7s → ~3ms, n=10⁵: 19.7s → ~10ms, same fixed point.

### Changed

- `ici`/`e50`/`e90`/`emax` and `reliability_summary` fit their LOESS smoother at 512 equal-mass anchors by default (`loess`'s `grid_size`, interpolated between anchors, the device of R's `stats::lowess` `delta`); measured `|Δici| ≈ 1.3e-6` on `make_pd_portfolio(n=5000)`. The window search differs from the old core only at exact distance ties.
- `smooth_ece` smooths a pre-binned (default `bins=8192` equal-width logit bins) residual measure, with a small-bandwidth guard that refines once and then falls back to exact.
- `ENIRCalibrator.path_solutions_` is now shaped `(K, m)` over `kept_breakpoints_`; `path_lambdas_` still records every breakpoint.
- `evaluate`'s bootstrap resamples negative and positive classes separately by default (`stratify=True`); `stratify=False` restores i.i.d. resampling.
- `VennAbersCalibrator` predicts from precomputed `F0_`/`F1_` sweeps fit once instead of refitting two PAVA passes per query.

### Added

- `grid_size=` on `loess`/`ici`/`e50`/`e90`/`emax`/`reliability_summary` and `bins=` on `smooth_ece`; `None` recovers the exact pre-0.1.3 values and cost bit-for-bit.
- `evaluate(metrics=)`: bootstrap only the requested subset of the catalog; unknown names raise `ValueError`.
- `docs/scripts/benchmarks.py`: deterministic wall-time benchmarks for `ici`, `smooth_ece`, `evaluate`, ENIR, IVAP, and the selector, with matching `slow`-marked regression ceilings.
- `ENIRCalibrator(max_solutions=)` (default 256), plus `dropped_weight_` and `kept_breakpoints_`.
- `VennAbersCalibrator` fitted attributes `F0_`/`F1_`; `evaluate(stratify=)`; `probcal._math.weighted_quantile`; `BaseCalibrator.complexity_rank` (default 100.0), overridden by every built-in.
- `point_inverse(p, *, space)`: an exact single-point preimage alongside `interval_inverse`. On `BaseCalibrator`, any affine-logit map (`PlattCalibrator`, `TemperatureCalibrator`, `BetaCalibrator`'s `"a"`/`"ab"` variants) in closed form; `BetaCalibrator` overrides it for all three variants including the non-affine `"abm"` map, via a seed refined by up to 4 certified Halley steps; `LogitOffset` in closed form. Non-affine monotone and step calibrators raise `NotImplementedError` naming `interval_inverse`.

### Performance

- `ici` at n=50,000: 192.2s (0.1.2) → 1.2s. `loess(grid_size=512)` fits n=1,000,000 points in under 30s.
- ENIR fit at m=10,000: 68.6s / 783MB → 1.6s / 40.8MB peak. IVAP fit at n=100,000: 0.91s; `predict_interval` at m=100,000: 0.016s. `CalibratorSelector()`'s default menu: 34.5s at n=4,000 (0.1.2) → 7.9s at n=100,000.
- Corrected attribution: `evaluate`'s pre-fix cost was the `smooth_ece` aliasing defect above, not `ece_sweep`. Post-fix, `ici` is the largest single per-call contributor (0.24s at n=10⁴); `evaluate(n_boot=100)` at n=10⁴ measures ~44s. `metrics=` subsetting is the lever.

</details>

## [0.1.2] - 2026-08-12

Platt and beta fits on wide-score data no longer hit a false "separation" abort, and both calibrators now surface their IRLS convergence status instead of swallowing it.

<details markdown="1">
<summary>Details</summary>

### Fixed

- `irls_logistic` no longer aborts at `max|eta| > 30`, a false separation that biased Platt/beta fits on wide-score data (z ~ N(0, 8) with true slope 1.5: 0.1.1 returned a ≈ 1.18 plus a spurious warning). Newton now step-halves on an overflow-safe softplus objective; separation is detected only for the quasi-separation signature; the ridge fallback converges. `calibration_belt` may now select higher polynomial degrees on wide-score data where the old cap silently truncated the search.
- `PlattCalibrator` and `BetaCalibrator` no longer swallow IRLS convergence status: an unconverged fit raises a distinct `UserWarning` and is recorded by `interpret()`; `calibration_belt` stops forward degree extension at a separated fit.

### Added

- `IrlsResult.nll`, the final penalized objective value.
- `converged_` on `PlattCalibrator`/`BetaCalibrator` and `separation_fallback_` on `BetaCalibrator`, with matching `interpret()` audit lines; docs section "Separation, steep maps, and convergence".

</details>

## [0.1.1] - 2026-08-08

The kernel calibration error and its test (SKCE, after Widmann et al. 2019) join the metric catalog as report-only quantities. Four plots arrive: the annotated reliability diagram with a stats box and rug, ECCE drift walks, the per-grade traffic-light chart, and the offset audit chart, all under a per-call house style that never touches global matplotlib settings.

<details markdown="1">
<summary>Details</summary>

### Added

- `probcal.metrics.kernel`: `skce` (estimators `uq`/`ul`/`biased`, Laplacian/Gaussian kernels, deterministic median-heuristic bandwidth, probability/logit kernel scale) and `skce_test` (Arcones–Giné centered bootstrap and the O(n) asymptotic-normal method, one-sided, with the distribution-free `p_value_bound`). Report-only: excluded from `evaluate()` and the selector.
- Documentation: SKCE section and selection-table row in the metrics chapter, the kernel API page, and the namesake-disambiguation FAQ entry (the unaffiliated R package `probcal` and the ECAI 2025 research codebase).
- Tooling: mypy targets Python 3.12 (numpy ≥ 2 stubs use PEP 695 syntax); runtime 3.11 support unchanged.
- Visualization: annotated `plot_reliability` (stats box via the new `metrics.reliability_summary`, deterministic rug thinned to ≤ 1000 marks per class), `plot_ecce` cumulative-drift walks with the pointwise ±2 SD envelope (on the new `curves.ecce_curve`), `plot_grade_backtest` traffic-light chart with 90% display intervals (new `ci_low`/`ci_high` fields on both grade results, powered by the numpy-only `_math.beta_ppf`), and `plot_offset_audit`. `docs/scripts/generate_figures.py` regenerates every documentation figure deterministically.

### Changed

- All plots style themselves via a per-call `rc_context` house style; global matplotlib `rcParams` are never touched.
- `plot_reliability`: the twin-axis count-bar margin is opt-in (`counts=False` default); the rug is the density view.
- All documentation figures regenerated under the house style.

</details>

## [0.1.0] - 2026-08-07

First public release on PyPI: thirteen binary calibrators from Platt to Venn–Abers and spline, the full metric catalog with bootstrap CIs and per-grade backtests, exact inverse maps from a policy PD back to a raw score, the auditable `LogitOffset` stage, SHAP attribution repair, `CalibratedModel` prefit and cross-validation flows, `CalibratorSelector` under nested validation, and a four-chunk theory guidebook.

<details markdown="1">
<summary>Details</summary>

### Fixed

- `reliability_binned`: Wilson interval bounds are forced to contain the point estimate (floating-point noise at zero-event bins could push `ci_low` above a 0.0 event rate).
- `plot_reliability(scale="logit")`: bins with an event rate of exactly 0 or 1 are omitted from the point layer instead of rendering at the clipping floor; they remain visible in the count margin.

### Added

- Packaging and CI: PEP 639 license metadata, Python 3.13 classifier, beta status, a tag-vs-version check in the publish workflow, and a CI matrix on Python 3.11/3.12/3.13.
- Documentation: generated figures in the visualization chapter, "In probcal" snippets in every concept chapter, a probcal-vs-netcal note in the FAQ, README quickstart with printed output and a feature matrix, docs cross-linking.
- Datasets and tutorial: `make_pd_portfolio` (beta-family generative miscalibration with exact event-rate anchoring), the executed `pd_calibration_walkthrough.ipynb`, `getting-started.md`, `how-it-works.md`, the three-page API reference, and the FAQ.
- Automatic selection: `CalibratorSelector`, default 8-candidate menu, inner stratified seeded K-fold with structurally out-of-fold scoring, criteria log_loss/brier/ici/smooth_ece/ece_sweep (plain ECE and Hosmer–Lemeshow refused), guardrail flags, one-standard-error parsimony tie-break, winner refit, ranked `SelectionReport`.
- Wrapper flows: `CalibratedModel`, prefit flow and cv flow with duck-typed cloning and stratified seeded folds, `ensemble=False` pooled default vs `ensemble=True` fold averaging, `offset_to` appending inspectable `LogitOffset` stages, composed `interval_inverse` and `affine_logit_coeffs_`, `predict_proba`/`predict_proba_2d`.
- Inverse maps and thresholds: `interval_inverse(lo, hi, *, space, buffer_logit)` across the catalog (closed forms for Platt, temperature, `LogitOffset`; block-structure search for isotonic, CIR, histogram, scaling-binning; monotone bisection for beta, spline, Venn–Abers); `UnattainableTargetError` instead of silent clamping; non-monotone calibrators refuse with `NotImplementedError`; `thresholds.calibrated_interval_to_raw` and the masterscale `calibrated_bands_to_raw`.
- Attribution adjustment: `adjust_attributions` → `AdjustedAttribution`, affine-exact mode (exact composed Shapley values via `affine_logit_coeffs_`) and Aumann–Shapley mode (exact additivity for any calibrator), logit and probability scales, shap.Explanation duck-typing without a shap import.
- Logit offset: `LogitOffset` with explicit-delta and target-mean modes, audit trail (`delta_`, `pre_mean_`, `post_mean_`, timestamp), `audit_report()` with pre/post guardrails, `interpret()` with odds-factor and central-tendency readings.
- Curves, belt, and plots: `reliability_binned` (Wilson CIs, both scales), `reliability_loess`, `reliability_spline`, `calibration_belt` (GiViTI-style, forward degree selection ≤ 4, pointwise bands, identity-test p-value); `plots.py` (`[viz]`-guarded): `plot_reliability` with logit-scale probability-labeled ticks, `plot_belt`, `plot_comparison`, `plot_interval`, `plot_selection`.
- Metrics catalog: proper scores (`log_loss`, `brier_score`, `brier_skill_score`, Murphy decomposition, log-loss calibration/refinement split), binned estimators (`ece` l1/l2/max, `ece_debiased`, `ece_sweep`, `adaptive_ece`, `hosmer_lemeshow`), binning-free estimators (`smooth_ece`, `ecce`, `ici`/`e50`/`e90`/`emax`, `spiegelhalter_z`), the recalibration-regression framework (`calibration_intercept`/`_slope`/`_test`, `calibration_guardrails`), per-grade backtests (`binomial_grade_test`, `jeffreys_grade_test` with traffic lights), and `evaluate()` with seeded bootstrap percentile CIs.
- Calibrators: `PlattCalibrator` (Lin–Lin–Weng smoothing), `TemperatureCalibrator`, `BetaCalibrator` (variants abm/ab/a, `constraint_active_`), `IsotonicCalibrator` (PAVA, tie pooling, block-structure attributes), `CenteredIsotonicCalibrator`, `HistogramBinningCalibrator` (equal-mass/equal-width, Jeffreys shrinkage), `ScalingBinningCalibrator`, `BBQCalibrator`, `ENIRCalibrator` (`is_monotone_ = False`), `VennAbersCalibrator` (IVAP with `predict_interval()`), `CrossVennAbersCalibrator`, `SplineCalibrator` (natural cubic basis on the logit scale, penalized IRLS, λ by CV, monotonicity check); `BaseCalibrator` with fit/predict_proba/interpret, `get_params`/`set_params`, `is_monotone_`, `affine_logit_coeffs_`; public `logit`/`expit`.
- Numerical core: overflow-safe logit/expit, weighted PAVA, IRLS logistic regression with ridge stabilization and separation detection, safeguarded 1-D Newton and bisection, regularized incomplete beta and gamma, chi-square and normal quantiles, tricube LOESS, natural cubic spline basis; validation with `1e-12` clipping; frozen result dataclasses with `as_dict()`. Reference tests against scipy/scikit-learn/statsmodels.
- Repository scaffold: src layout, uv/ruff/black/mypy/pytest, CI workflows (docs deploy, tag-gated PyPI publish), mkdocs-material + MathJax, MIT license, citation metadata; all 12 flagged references verified against primary sources.
- Theory guidebook (~16,400 words): `concepts/why-calibration.md`, `methods-parametric.md`, `methods-nonparametric.md`, `methods-distribution-free.md`, `metrics.md`, `data-splitting.md`, `offset.md`, `shap-calibration.md`, `inverse-maps.md`, `auto-selection.md`, `visualization.md`.

</details>

[Unreleased]: https://github.com/wlazlod/probcal/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/wlazlod/probcal/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/wlazlod/probcal/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/wlazlod/probcal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/wlazlod/probcal/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/wlazlod/probcal/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/wlazlod/probcal/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/wlazlod/probcal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wlazlod/probcal/releases/tag/v0.1.0
