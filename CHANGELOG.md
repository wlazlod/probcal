# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `point_inverse` honoured neither end of the no-silent-clamp doctrine at the probability boundary: `p ∈ {0, 1}` was clipped to `[1e-12, 1 − 1e-12]` before inversion (a finite "inverse" for an unattainable target), and a certified root with `|z| > logit(1 − 1e-12) ≈ 27.63` was returned as `σ(z)`, which rounds to 0.0/1.0 and cannot round-trip through `predict_proba` (target 0.998 → raw 1.0 → forward 0.17, no warning). Both now raise `UnattainableTargetError`: boundary targets all-or-nothing with the offending values named; over-range roots only for `space="probability"`, with the error pointing at `space="logit"`, where the answer is exact (DECISIONS 69)
- `LogitOffset.interval_inverse` returned raw bounds below the `1e-12` clip (e.g. `raw_lo = 4.5e-14` for target `1e-9` at `delta = 10`) that `transform` cannot round-trip; bounds beyond the representable output range `[σ(δ − logit(1−1e-12)), σ(δ + logit(1−1e-12))]` now collapse to the full-range sentinels (0/1, ±inf) exactly as in `BaseCalibrator.interval_inverse`, and an interval entirely outside that range raises `UnattainableTargetError` (DECISIONS 72)

### Changed

- `smooth_ece`'s lattice path now engages for every call with a non-degenerate logit range instead of only `n > bins`, removing the size cliff at typical calibration-set sizes (0.1.3: exact path at n=4000 cost ~1s/call on the benchmark host and made `evaluate(n=6000)` bootstrap runs smECE-bound; n=8193 took ~2ms). Path selection is decoupled from n: bin at `bins` (default 8192), accept when the fixed point satisfies `sigma* >= 8·width`, refine adaptively once, fall back to exact only when refinement is infeasible or still under-resolved. Values for `n <= bins` may differ from the 0.1.2/0.1.3 exact grid at the ~1e-4 level on typical portfolios (measured ≤ 2.4e-4 on `make_pd_portfolio`; larger on wide clipped-logit-range data, where the exact path's fixed 257-point grid under-resolves small-sigma kernels and the lattice value — ≥ 8 samples per sigma — is the better one); `bins=None` recovers old values bit-for-bit (DECISIONS 68)

### Added

- `ENIRCalibrator.fit` emits a single `UserWarning` above 50,000 unique scores stating the expected fit minutes — the path solver is quadratic in unique scores (DECISIONS 70)
- README "Calibrators at a glance" table: one-line scaling note per method (DECISIONS 70)

### Documentation

- Removed conversation-context phrasing from the `_beta_point_inverse_z` docstring; internal decision log entry 67 rephrased to match

### Performance

- `smooth_ece` measures 1.0–3.6ms for n ∈ [64, 8192] on `make_pd_portfolio` (0.1.3: exact path, ~1s at n=4000 on the benchmark host), 3–6ms at n=10⁴–10⁵ and ~43ms at n=10⁶ (the O(n) pre-binning, unchanged from 0.1.3); observed lattice-vs-exact deviation ≤ 6.1e-5 across the swept sizes. `evaluate(n=6000, n_boot=1000)` measures 198.9s on the benchmark host (0.1.3: ≈10min, extrapolated from the 60s `n_boot=100` measurement)

## [0.1.3] - 2026-08-16

### Fixed

- ENIR (`ENIRCalibrator`) could exhaust memory well before m=50,000 because `path_solutions_` retained every breakpoint's full-length solution (`O(m^2)` memory); the path solver now retains only the lowest-BIC `max_solutions` (default 256) breakpoints, bounding memory at any m (DECISIONS 61); a pruning bound that skips provably negligible breakpoints previously crashed on sub-unit total `sample_weight` (`log(total weight) <= 0`) — the bound is now disabled in that case instead of raising, regression-tested against a sum-to-1.0 and a sum-below-1.0 case
- `sample_weight` was accepted but silently had no effect on `e50`/`e90` and on `reliability_summary`'s internal `e90` call — both now honor weights via the new Hazen-position `weighted_quantile` (DECISIONS 64)
- `evaluate`'s bootstrap silently substituted the point estimate for degenerate single-class i.i.d. resamples, artificially narrowing the reported CI; a degenerate draw is now redrawn (up to 100 times, then `RuntimeError`) instead of faked (DECISIONS 63)
- `CalibratorSelector`'s name-keyed parsimony table ranked every user-supplied candidate last on a tie regardless of actual complexity; parsimony now reads `complexity_rank` off the candidate instance, so a user override can win a tie against a built-in method (DECISIONS 65)
- `smooth_ece`'s binned path (default `bins=8192`) evaluated the smoothed measure on the exact path's 257-point grid, which aliases against the bin lattice: at small bandwidths the probe landed almost entirely between grid nodes, reporting ~1.7e-7 mass against a true total variation of 0.0749 (`n=10⁴`) — a spurious near-zero reading that always tripped the small-bandwidth guard and fell back to the exact O(n)-per-step computation, so **every reported value stayed correct**, but at a cost of ~5.7-6.1s/call regardless of `n`, roughly 3-6x slower than `bins=None` for `8192 < n ≲ 6×10⁴`. The binned path now evaluates the measure natively on its own lattice by direct Gaussian convolution (closed-form total variation when the kernel is isolated, which also makes the early-exit test non-spurious by construction) with one adaptively sized refinement instead of a blind 8x retry; measured `n=10⁴`: ~5.7s → ~3ms; `n=10⁵`: 19.7s (exact) → ~10ms (default), same fixed point as exact (DECISIONS 66)

### Changed

- `ici`/`e50`/`e90`/`emax` and `reliability_summary` fit their LOESS smoother at 512 equal-mass anchors by default instead of at every observation (`loess`'s new `grid_size` argument, interpolated between anchors, precedent R `stats::lowess`'s `delta`); measured `|Δici| ≈ 1.3e-6` on `make_pd_portfolio(n=5000)`, far below bootstrap CI width; the underlying `_loess_fit_sorted` core was rewritten from an `argpartition` r-nearest-neighbor search to a sorted two-pointer window walk, differing from the old core only at exact distance ties (leftmost minimal-width window) (DECISIONS 58)
- `smooth_ece` smooths a pre-binned (default `bins=8192` equal-width logit bins) residual measure instead of the raw per-observation one, cutting each bisection step from 257 x n to a lattice-native evaluation independent of n, with a small-bandwidth guard (`sigma* < 8 * bin_width` triggers one adaptively sized refinement, `bins <- ceil(range / (sigma*/8))`, then a silent exact fallback if the guard still trips) (DECISIONS 59, refined by 66)
- `ENIRCalibrator.path_solutions_` is now shaped `(K, m)` over `kept_breakpoints_` rather than every breakpoint; `path_lambdas_` still records every breakpoint (pre-1.0 attribute-shape change; DECISIONS 61)
- `evaluate`'s bootstrap resamples negative and positive classes separately by default (`stratify=True`), conditioning the CI on the observed class balance instead of also capturing base-rate sampling variability; `stratify=False` restores i.i.d. resampling (DECISIONS 63)
- `VennAbersCalibrator` predicts from precomputed `F0_`/`F1_` cumulative-sum-diagram sweeps fit once in `fit()` instead of refitting two PAVA passes per query (DECISIONS 62, amending entry 30)

### Added

- `grid_size=` on `loess`/`ici`/`e50`/`e90`/`emax`/`reliability_summary` and `bins=` on `smooth_ece`, both defaulting to the new fast paths; `grid_size=None`/`bins=None` recover the exact pre-0.1.3 values and cost bit-for-bit
- `evaluate` keyword-only `metrics=` subset of the catalog — computes and bootstraps only the requested metrics instead of the full catalog, with unknown names raising `ValueError` (DECISIONS 60)
- `docs/scripts/benchmarks.py`: deterministic wall-time benchmarks for `ici`, `smooth_ece`, and `evaluate` at several portfolio sizes; `tests/test_perf_smoke.py`: `slow`-marked regression ceilings for the grid-anchored LOESS and binned smECE fast paths
- `ENIRCalibrator(max_solutions=)` (default 256) bounding retained path solutions, plus `dropped_weight_` (retention loss, pruned breakpoints excluded) and `kept_breakpoints_`
- `VennAbersCalibrator` fitted attributes `F0_`/`F1_`
- `evaluate(stratify=)` keyword (default `True`)
- `probcal._math.weighted_quantile` (Hazen positions)
- `BaseCalibrator.complexity_rank` property (default 100.0), overridden by every built-in calibrator
- ENIR/IVAP/selector rows and matching `slow`-marked ceilings added to the same benchmark and perf-smoke suites
- `point_inverse(p, *, space)`: an exact single-point preimage (calibrated probability -> raw score/logit) alongside `interval_inverse`'s generalized-inverse interval. On `BaseCalibrator`, covers any affine-logit map (`PlattCalibrator`, `TemperatureCalibrator`, `BetaCalibrator`'s tied `"a"`/`"ab"` variants) in closed form; `BetaCalibrator` overrides it for all three variants, including the non-affine `"abm"` map, via a minimax-hyperbola seed refined by up to 4 certified Halley steps (DECISIONS 67); `LogitOffset` gets the same closed form as its `interval_inverse`. Non-affine monotone and step calibrators still raise `NotImplementedError` naming `interval_inverse`

### Performance

- `ici` at n=50,000: 192.2s (v0.1.2) to 1.2s, single-core, measured on the benchmark host
- `loess(grid_size=512)` fits n=1,000,000 points in under 30s
- ENIR fit at m=10,000: 68.6s / 783MB (v0.1.2) to 1.6s / 40.8MB tracemalloc peak, memory now bounded at any m; IVAP fit at n=100,000: 0.91s (v0.1.2 needed 188s for a fit of only 2,000 plus a predict of 10,000), `predict_interval` at m=100,000: 0.016s; `CalibratorSelector()`'s default menu: 34.5s at n=4,000 (v0.1.2) to 7.9s at n=100,000
- `evaluate`'s full-catalog cost at n=10⁴ was previously misattributed here to `ece_sweep` (~0.15s/call); the actual dominant cost pre-fix was the `smooth_ece` binned-path aliasing defect above (~5.7-6.1s/call, guard-triggered on every call once `n > 8192`). Post-fix, `smooth_ece`'s per-call cost is negligible (a few ms up to n=10⁵); measured single-call costs at n=10⁴ / n=5×10⁴: `ici` 0.24s / 1.09s, `ece_sweep` 0.12s / 0.49s — `ici`, not `ece_sweep`, is now the largest single per-call contributor to the full-catalog cost, though neither dominates outright (together with the rest of the catalog they roughly account for the measured totals). Corrected wall times: `evaluate(n_boot=100)` at n=10⁴ measures ~44s, `evaluate(n_boot=50)` at n=5×10⁴ measures ~88s, both post-fix, driven by the catalog's ordinary O(n)-ish metrics, not by this defect; `metrics=` subsetting (e.g. `metrics=("ici", "log_loss")`) is the lever for either (DECISIONS 66)

## [0.1.2] - 2026-08-12

### Fixed

- `irls_logistic` no longer aborts at `max|eta| > 30` — a false "separation" that biased Platt/beta fits on wide-score data (z ~ N(0, 8) with true slope 1.5: v0.1.1 returned a ≈ 1.18 plus a spurious separation warning); Newton now step-halves on an overflow-safe softplus objective, separation is detected only for effectively binary targets (all-correct-by-10-log-odds by the design's own contribution `eta - offset` with non-vanishing gradient, singular Hessian, or unconverged divergence — the quasi-separation signature), and the ridge-1e-6 fallback converges instead of hitting the same abort (DECISIONS 57); note `calibration_belt` may now select higher polynomial degrees on wide-score data where the old cap silently truncated the forward search
- `PlattCalibrator` and `BetaCalibrator` no longer swallow IRLS convergence status: an unconverged fit raises a distinct `UserWarning` (never the separation message) and is recorded by `interpret()`; `calibration_belt` stops forward degree extension at a separated fit instead of consuming its ridge-fallback coefficients

### Added

- `IrlsResult.nll` — final penalized objective value (backward-compatible field append)
- `converged_` on `PlattCalibrator`/`BetaCalibrator` and `separation_fallback_` on `BetaCalibrator`, with matching `interpret()` audit lines; docs section "Separation, steep maps, and convergence" in the parametric-methods chapter

## [0.1.1] - 2026-08-08

### Added

- `probcal.metrics.kernel`: squared kernel calibration error `skce` (estimators `uq`/`ul`/`biased`, Laplacian/Gaussian kernels, deterministic median-heuristic bandwidth with strided-subsample and tie fallbacks, probability/logit kernel scale, seeded `ul` pairing) and `skce_test` (Arcones–Giné centered bootstrap per the paper's Appendix G and the O(n) asymptotic-normal linear method, one-sided, with the distribution-free `p_value_bound`), after Widmann et al. (2019); report-only — deliberately excluded from `evaluate()` and the selector (DECISIONS 53); 21 tests incl. hand-computed anchors, brute-force references, bootstrap-identity check, and level/power studies
- Documentation: SKCE section and selection-table row in the metrics chapter, `probcal.metrics.kernel` API page, namesake-disambiguation FAQ entry (the unaffiliated R package `probcal` and the ECAI 2025 research codebase), and a README feature-matrix column for the R package
- Tooling: mypy targets Python 3.12 in config and CI (numpy ≥ 2 stubs use PEP 695 `type` statements that mypy only parses for target ≥ 3.12; DECISIONS 54); runtime 3.11 support unchanged
- Visualization: annotated `plot_reliability` (stats box via the new `probcal.metrics.reliability_summary` aggregate, deterministic event/non-event rug thinned to ≤ 1000 marks per class), `plot_ecce` cumulative-drift walks with the pointwise ±2 SD envelope (on the new `probcal.curves.ecce_curve`, whose `stat_max` agrees exactly with `metrics.ecce`), `plot_grade_backtest` traffic-light chart with 90% display intervals (new `ci_low`/`ci_high` fields on both grade results, Jeffreys central / Clopper–Pearson, powered by the new numpy-only `probcal._math.beta_ppf`), and `plot_offset_audit` for fitted `LogitOffset` stages; seeded `docs/scripts/generate_figures.py` regenerates every documentation figure deterministically; DECISIONS 55; 16 tests incl. hand anchors, a scipy `beta.ppf` reference, and rug/annotation determinism checks

### Changed

- All plots now style themselves via a per-call `rc_context` house style (muted palette, no top/right spines, light grid) — global matplotlib `rcParams` are never touched, verified by test
- `plot_reliability`: the twin-axis count-bar margin is now opt-in (`counts=False` default) — the rug replaces it as the density view
- All documentation figures regenerated under the house style by the new figure script (the logit-scale reliability figure is now the annotated variant)

## [0.1.0] - 2026-08-07

First public release on PyPI.

### Fixed

- `reliability_binned`: Wilson interval bounds are now forced to contain the point estimate (floating-point noise at zero-event bins could push `ci_low` above a 0.0 event rate, breaking error-bar rendering)
- `plot_reliability(scale="logit")`: bins with an event rate of exactly 0 or 1 (no finite logit) are omitted from the point layer instead of rendering at the clipping floor and crushing the axis; they remain visible in the count margin

### Added

- Packaging and CI: PEP 639 license metadata (SPDX `license = "MIT"` plus `license-files`, replacing the deprecated table form), Python 3.13 classifier, `Development Status :: 4 - Beta`, a tag-vs-version consistency check in the publish workflow, and a `ci.yml` matrix running lint, type-check, and the test suite on Python 3.11/3.12/3.13
- Documentation: generated example figures in the visualization chapter, "In probcal" code snippets in every concept chapter, and a probcal-vs-netcal guidance note in the FAQ

- Polish: README quickstart with real printed output, feature matrix vs scikit-learn/netcal/single-method packages, performance note on the deliberate absence of Rust acceleration, docs cross-linking pass
- Datasets, tutorial, and user docs: `make_pd_portfolio` (beta-family generative miscalibration with exact event-rate anchoring), executed tutorial notebook `pd_calibration_walkthrough.ipynb` (diagnose → select → fit → re-anchor → backtest → threshold translation, incl. the counterfactual-engine interop recipe), `getting-started.md` with a runnable quickstart and its printed output, `how-it-works.md` pipeline walkthrough with schema, three-page mkdocstrings API reference, and the FAQ (inverse-map protocol, `Target.probability` trap, selection rules)
- Automatic selection: `CalibratorSelector` — default 8-candidate menu per spec, inner stratified seeded K-fold with structurally out-of-fold-only scoring, criteria log_loss/brier/ici/smooth_ece/ece_sweep (plain ECE and Hosmer–Lemeshow refused), guardrail flags on pooled out-of-fold predictions, one-standard-error parsimony tie-break, winner refit on the full set, ranked `SelectionReport`; 9 tests incl. a spy-calibrator structural no-leakage check and the parsimony-on-calibrated-data behavior
- Wrapper flows: `CalibratedModel` — prefit flow (score + calibrate, the credit-risk canon), cv flow with duck-typed cloning (sklearn `clone` if installed, else deepcopy) and stratified seeded folds, `ensemble=False` pooled default (one calibrator on out-of-fold scores, final model refit) vs `ensemble=True` fold averaging; `offset_to` appending inspectable `LogitOffset` stages; composed `interval_inverse` and `affine_logit_coeffs_`; `predict_proba`/`predict_proba_2d`; structural no-leakage tests with a clone-surviving spy model (out-of-fold scoring disjoint from training rows, each row scored exactly once)
- Inverse maps and thresholds: `interval_inverse(lo, hi, *, space, buffer_logit)` implemented across the catalog — closed forms (Platt, temperature, `LogitOffset`), block-structure searchsorted (isotonic, CIR, histogram, scaling-binning), monotone bisection (beta, spline, Venn–Abers); `UnattainableTargetError` instead of silent clamping; non-monotone calibrators refuse with an explanatory `NotImplementedError`; `thresholds.py` with `calibrated_interval_to_raw` and the masterscale `calibrated_bands_to_raw`; 16 tests covering the full spec §10 list (round-trips, block-edge semantics, offset −δ shift, buffer monotonicity, space consistency)
- Attribution adjustment: `adjust_attributions` → `AdjustedAttribution` — affine-exact mode (exact composed Shapley values for Platt/temperature/offset via `affine_logit_coeffs_`) and Aumann–Shapley mode (exact additivity for any calibrator, central-difference fallback on degenerate rows); logit and probability scales; shap.Explanation duck-typing without a shap import; 10 tests incl. affine/AS 1e-12 equivalence and sign/rank invariance
- Logit offset: `LogitOffset` — explicit-delta and target-mean modes (unique bisection root, unit-tested against a brute-force grid), audit trail (`delta_`, `pre_mean_`, `post_mean_`, timestamp), `audit_report()` with pre/post guardrails, `interpret()` with odds-factor and central-tendency readings, affine logit coefficients for attribution composition
- Curves, belt, and plots: `curves.py` — `reliability_binned` (Wilson CIs, both scales), `reliability_loess`, `reliability_spline`, `calibration_belt` (GiViTI-style: forward LR degree selection ≤4, information-matrix pointwise bands, identity-test p-value; reimplemented from the Nattino papers); `plots.py` ([viz]-guarded) — `plot_reliability` with logit-scale probability-labeled ticks and count margin, `plot_belt`, `plot_comparison`, `plot_interval`, `plot_selection`; `SmoothReliabilityCurve` result dataclass; 11 tests incl. a hand-checked Wilson interval and belt null/alternative behavior
- Metrics catalog: `probcal.metrics` complete — proper scores (`log_loss`, `brier_score`, `brier_skill_score`, Murphy decomposition with optional bias correction, log-loss calibration/refinement split), binned estimators (`ece` l1/l2/max, `ece_debiased`, `ece_sweep`, `adaptive_ece`, `hosmer_lemeshow`), binning-free estimators (`smooth_ece` with self-consistent bandwidth, `ecce`, `ici`/`e50`/`e90`/`emax`, `spiegelhalter_z`), the recalibration-regression framework (`calibration_intercept`/`_slope`/`_test`, `calibration_guardrails`), per-grade backtests (`binomial_grade_test`, `jeffreys_grade_test` with traffic lights), and `evaluate()` with seeded bootstrap percentile CIs; 38 tests incl. hand-computed cases and sklearn/scipy references
- Spline calibrator: `SplineCalibrator` — natural cubic basis on the logit scale, penalized IRLS with second-difference roughness penalty, λ by stratified K-fold CV on log loss, effective degrees of freedom reported, post-fit monotonicity check with warning; completes the spec §6 calibrator catalog (11 of 11)
- Binning + Bayesian calibrators: `HistogramBinningCalibrator` (equal-mass/equal-width, Jeffreys shrinkage, empty-bin fallback, post-fit monotonicity flag), `ScalingBinningCalibrator` (Platt stage + equal-mass binning of fitted values), `BBQCalibrator` (Beta–Binomial marginal likelihood over a B-grid, Jeffreys prior, top-3 reporting), `ENIRCalibrator` (nearly-isotonic mPAVA path from raw data to the isotonic fit, BIC-weighted ensemble, `is_monotone_ = False`); 20 tests incl. hand-computed bin rates and the path-endpoint-equals-PAVA anchor
- Isotonic family + Venn–Abers: `IsotonicCalibrator` (PAVA step map, tie pooling, clamping, optional linear interpolation, block-structure attributes), `CenteredIsotonicCalibrator` (CIR through weight-centered block points), `VennAbersCalibrator` (IVAP with `predict_interval()`, log-loss-minimax scalarization, width reporting in `interpret()`), `CrossVennAbersCalibrator` (stratified seeded folds, geometric-mean merge, conservative envelope interval); 19 tests incl. direct-refit agreement and width-shrinkage checks
- Base API + parametric calibrators: `BaseCalibrator` (fit/predict_proba/predict_proba_2d/interpret, manual get_params/set_params, `is_monotone_`, `affine_logit_coeffs_`, `interval_inverse` contract stub), `PlattCalibrator` (Lin–Lin–Weng target smoothing), `TemperatureCalibrator` (safeguarded 1-D NLL solve), `BetaCalibrator` (variants abm/ab/a, betacal negative-coefficient refit, `constraint_active_`); public exports incl. `logit`/`expit`; 30 tests (identity/distortion recovery, monotonicity, small-sample stability, interpret contract)
- Numerical core: `_math.py` (overflow-safe logit/expit, weighted PAVA with block structure, IRLS logistic regression with ridge stabilization and separation detection, safeguarded 1-D Newton and bisection, vectorized lgamma/erf, regularized incomplete beta and lower incomplete gamma, chi-square and normal quantiles, tricube LOESS, natural cubic spline basis), `_validation.py` (score/target/weight validation with 1e-12 clipping), `_results.py` (frozen result dataclasses with `as_dict()` and aligned-table reprs); 49 unit tests plus 9 reference tests against scipy/scikit-learn/statsmodels (betainc and gammainc within 1e-12, IRLS within rtol 1e-8)
- Repository scaffold: src layout with docstring-only modules, tooling (uv, ruff, black, mypy, pytest), CI workflows (docs deploy, PyPI publish — tag-gated), documentation skeleton (mkdocs-material + MathJax), MIT license, citation metadata
- Reference verification: all 12 ⚠ references of the spec verified against primary sources; completed bibliographic records logged in `docs/DECISIONS.md` entries 10–21
- Theory guidebook, chunk 1: `concepts/why-calibration.md` (definitions, sources of miscalibration, decisioning and regulatory consequences, proper-scoring-rule lens) and `concepts/methods-parametric.md` (Platt, temperature, beta: derivations, parameter interpretation, worked reading)
- Theory guidebook, chunk 4 (completes the guidebook — ~16,400 words total): `concepts/shap-calibration.md` (identifiability obstacle, affine-exact class, Aumann–Shapley mode, invariance properties), `concepts/inverse-maps.md` (preimage identity, generalized inverses, attainability, buffer_logit drift robustness, masterscale workflow), `concepts/auto-selection.md` (selector protocol, structural no-leakage, report reading), `concepts/visualization.md` (reliability constructions, logit-scale rationale, calibration belt)
- Theory guidebook, chunk 3: `concepts/metrics.md` (full metric catalog with formulas and pathologies, bootstrap-CI protocol, report-reading order, selection-suitability table), `concepts/data-splitting.md` (prefit vs cv flows, ensemble vs pooled, calibration-set sizing, nested selection), `concepts/offset.md` (uniqueness of the bisection root, King–Zeng/Elkan/Tasche equivalences, worked re-anchoring, audit practice)
- Theory guidebook, chunk 2: `concepts/methods-nonparametric.md` (PAVA with worked micro-example, CIR, histogram binning, scaling-binning sample-complexity argument, BBQ, ENIR, spline calibration, properties table) and `concepts/methods-distribution-free.md` (IVAP construction, validity guarantee scope, scalarization caveat, exchangeability limits, CVAP geometric-mean merge)

[Unreleased]: https://github.com/wlazlod/probcal/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/wlazlod/probcal/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/wlazlod/probcal/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/wlazlod/probcal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wlazlod/probcal/releases/tag/v0.1.0
