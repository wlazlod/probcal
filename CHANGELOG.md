# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository scaffold: src layout with docstring-only modules, tooling (uv, ruff, black, mypy, pytest), CI workflows (docs deploy, PyPI publish — tag-gated), documentation skeleton (mkdocs-material + MathJax), MIT license, citation metadata
- Reference verification (Task 1a): all 12 ⚠ references of the spec verified against primary sources; completed bibliographic records logged in `docs/DECISIONS.md` entries 10–21
- Theory guidebook, chunk 1 (Task 1b): `concepts/why-calibration.md` (definitions, sources of miscalibration, decisioning and regulatory consequences, proper-scoring-rule lens) and `concepts/methods-parametric.md` (Platt, temperature, beta: derivations, parameter interpretation, worked reading)
- Theory guidebook, chunk 2 (Task 1c): `concepts/methods-nonparametric.md` (PAVA with worked micro-example, CIR, histogram binning, scaling-binning sample-complexity argument, BBQ, ENIR, spline calibration, properties table) and `concepts/methods-distribution-free.md` (IVAP construction, validity guarantee scope, scalarization caveat, exchangeability limits, CVAP geometric-mean merge)
