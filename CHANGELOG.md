# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Base API + parametric calibrators (Task 3): `BaseCalibrator` (fit/predict_proba/predict_proba_2d/interpret, manual get_params/set_params, `is_monotone_`, `affine_logit_coeffs_`, `interval_inverse` contract stub), `PlattCalibrator` (Lin–Lin–Weng target smoothing), `TemperatureCalibrator` (safeguarded 1-D NLL solve), `BetaCalibrator` (variants abm/ab/a, betacal negative-coefficient refit, `constraint_active_`); public exports incl. `logit`/`expit`; 30 tests (identity/distortion recovery, monotonicity, small-sample stability, interpret contract)
- Numerical core (Task 2): `_math.py` (overflow-safe logit/expit, weighted PAVA with block structure, IRLS logistic regression with ridge stabilization and separation detection, safeguarded 1-D Newton and bisection, vectorized lgamma/erf, regularized incomplete beta and lower incomplete gamma, chi-square and normal quantiles, tricube LOESS, natural cubic spline basis), `_validation.py` (score/target/weight validation with 1e-12 clipping), `_results.py` (frozen result dataclasses with `as_dict()` and aligned-table reprs); 49 unit tests plus 9 reference tests against scipy/scikit-learn/statsmodels (betainc and gammainc within 1e-12, IRLS within rtol 1e-8)
- Repository scaffold: src layout with docstring-only modules, tooling (uv, ruff, black, mypy, pytest), CI workflows (docs deploy, PyPI publish — tag-gated), documentation skeleton (mkdocs-material + MathJax), MIT license, citation metadata
- Reference verification (Task 1a): all 12 ⚠ references of the spec verified against primary sources; completed bibliographic records logged in `docs/DECISIONS.md` entries 10–21
- Theory guidebook, chunk 1 (Task 1b): `concepts/why-calibration.md` (definitions, sources of miscalibration, decisioning and regulatory consequences, proper-scoring-rule lens) and `concepts/methods-parametric.md` (Platt, temperature, beta: derivations, parameter interpretation, worked reading)
- Theory guidebook, chunk 4 (Task 1e, completes Task 1 — ~16,400 words total): `concepts/shap-calibration.md` (identifiability obstacle, affine-exact class, Aumann–Shapley mode, invariance properties), `concepts/inverse-maps.md` (preimage identity, generalized inverses, attainability, buffer_logit drift robustness, masterscale workflow), `concepts/auto-selection.md` (selector protocol, structural no-leakage, report reading), `concepts/visualization.md` (reliability constructions, logit-scale rationale, calibration belt)
- Theory guidebook, chunk 3 (Task 1d): `concepts/metrics.md` (full metric catalog with formulas and pathologies, bootstrap-CI protocol, report-reading order, selection-suitability table), `concepts/data-splitting.md` (prefit vs cv flows, ensemble vs pooled, calibration-set sizing, nested selection), `concepts/offset.md` (uniqueness of the bisection root, King–Zeng/Elkan/Tasche equivalences, worked re-anchoring, audit practice)
- Theory guidebook, chunk 2 (Task 1c): `concepts/methods-nonparametric.md` (PAVA with worked micro-example, CIR, histogram binning, scaling-binning sample-complexity argument, BBQ, ENIR, spline calibration, properties table) and `concepts/methods-distribution-free.md` (IVAP construction, validity guarantee scope, scalarization caveat, exchangeability limits, CVAP geometric-mean merge)
