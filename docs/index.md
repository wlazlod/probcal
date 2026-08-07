# probcal

Universal post-hoc probability calibration for binary classifiers: methods, metrics,
diagnostics, and auditable offsetting — **numpy-only**.

The package unifies the binary calibration literature — Platt, temperature, beta, isotonic,
centered isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers, and spline
calibration — with an extensive catalog of evaluation metrics and statistical tests,
visualization on both probability and logit scales, an auditable logit-offset (central
tendency) adjustment, automatic method selection under nested validation, and prefit /
cross-validation data flows.

**Status:** released on PyPI, beta. Breaking changes bump the minor version until 1.0.

```bash
pip install probcal
```

Start with [Getting started](getting-started.md), read the
[Concepts](concepts/why-calibration.md) chapters — the theoretical foundation of the
package, written before the corresponding code — and walk the executed
[PD calibration tutorial](notebooks/pd_calibration_walkthrough.ipynb). The
[How it works](how-it-works.md) page maps the full pipeline in one view.
