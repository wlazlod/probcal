# probcal

Universal post-hoc probability calibration for binary classifiers: methods, metrics,
diagnostics, and auditable offsetting — **numpy-only**.

The package unifies the binary calibration literature — Platt, temperature, beta, isotonic,
centered isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers, and spline
calibration — with an extensive catalog of evaluation metrics and statistical tests,
visualization on both probability and logit scales, an auditable logit-offset (central
tendency) adjustment, automatic method selection under nested validation, and prefit /
cross-validation data flows.

**Status:** pre-release (`0.0.1`). The [Concepts](concepts/why-calibration.md) chapters are the
theoretical foundation of the package and are written before the corresponding code.
