# probcal

Post-hoc probability calibration **and calibration governance** for binary
classifiers — the evidence a regulated PD model needs alongside the map itself.
The core is numpy-only; everything else is an optional extra.

One deployed calibration runs a loop, and probcal covers all of it:
[calibrate](guide/choosing.md) →
[evaluate](concepts/metrics.md) →
[invert and decide](guide/cutoffs.md) →
[monitor](guide/monitoring.md) →
[act](guide/auditability.md) →
[report](guide/report.md), then back to re-anchoring. The
[How it works](how-it-works.md) page draws that loop once, in one view.

**Calibrate.** Thirteen calibrators — Platt, temperature, beta, isotonic, centered
isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers (IVAP and CVAP),
spline, and segmented — plus a rigid, auditable logit offset, automatic selection
under nested cross-validation, and prefit / cross-validation data flows.
[Choose a calibrator](guide/choosing.md) is the catalog.

**Evaluate.** Proper scores, binned and binning-free calibration errors, the CORP
decomposition, per-grade supervisory backtests, calibration belts, reliability curves
on both probability and logit scales, [grouped evaluation](guide/groups.md) by segment,
and **conservatism** tooling for low-default portfolios (Pluto–Tasche most-prudent PDs,
Jeffreys upper bands, margin-of-conservatism offsets).

**Decide.** Exact and generalized inverses turn a policy PD into a raw-score cutoff or
a whole masterscale, refusing unattainable targets instead of clamping.
[Set cutoffs and invert maps](guide/cutoffs.md).

**Monitor and act.** Anytime-valid **monitoring** — an e-process whose alarm keeps its
type-I guarantee at every look — with drift-onset localization and a recommendation
that can be applied as a new offset, on the record.
[Monitor and act](guide/monitoring.md).

**Prove it.** JSON **serialization** (never pickle) with fingerprints and a golden-file
compatibility promise, self-contained HTML **reports**, and one page tying the artifacts
to what each of them actually proves: [Auditability](guide/auditability.md).

**Integrate.** **scikit-learn** (the bare core is a duck on ≥ 1.6; an adapter covers the
probability-matrix world), **optbinning** scorecards (calibrated PDs carried back to the
points scale), and **treecf** counterfactuals bound to a named calibrator.

```bash
pip install probcal
```

Three doors, by who is arriving:

- **New to calibration** → [Install and quickstart](getting-started.md), then the
  20-minute [PD calibration walkthrough](notebooks/pd_calibration_walkthrough.ipynb).
- **Validating someone's model** → [Auditability](guide/auditability.md) and
  [Build a validation report](guide/report.md).
- **Coming from scikit-learn** → the three-tier [sklearn guide](guide/sklearn.md).

**Status:** released on PyPI, beta. Until 1.0, breaking changes bump the minor version,
are listed in the changelog with the reasoning, and keep an explicit escape hatch where
the old behavior had legitimate uses; a deprecated symbol warns with a
`DeprecationWarning` for at least one minor release before removal, naming its
replacement. Serialized artifacts carry a stronger promise: every 0.x release reads
schema 1, pinned by golden files in CI. The full surface and its conventions:
[API stability](api-stability.md).
