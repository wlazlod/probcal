# probcal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Universal post-hoc probability calibration for binary classifiers: methods, metrics,
diagnostics, and auditable offsetting — **numpy-only**.

`probcal` unifies the binary calibration literature (Platt, temperature, beta, isotonic,
centered isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers, spline
calibration), an extensive catalog of calibration evaluation metrics and statistical tests,
calibration visualization on both probability and logit scales, an auditable logit-offset
(central tendency) adjustment, automatic method selection under nested validation, and two
data flows (prefit and cross-validation). Primary application domain: credit-risk PD models;
the package is fully general.

**Status:** pre-release (`0.0.1`), under active development. API may change until the first
PyPI release.

## Installation

```bash
pip install -e .            # runtime: numpy only
uv sync --extra dev         # development (tests, lint, type-check)
```

## Quickstart

*Coming with the first feature release — see the task plan in `PROBCAL_SPEC.md` §14.*

## Documentation

Built with mkdocs-material; run locally with `uv run mkdocs serve`.

## License

MIT. See [LICENSE](LICENSE) and `docs/LICENSING.md` for the conceptual-reference policy on
GPL-licensed R packages.
