# probcal

Post-hoc probability calibration **and calibration governance** for binary
classifiers: the evidence a regulated PD model needs alongside the map itself.
If you have binary scores that feed a threshold, a price, or a report, this
applies to you; PD models are the archetype because they are the most audited,
and the [glossary](glossary.md) maps their vocabulary to the generic one.
The core is numpy-only; everything else is an optional extra.

One deployed calibration runs a loop, and probcal covers all of it:
[calibrate](guide/choosing.md) →
[evaluate](concepts/metrics.md) →
[invert and decide](guide/cutoffs.md) →
[monitor](guide/monitoring.md) →
[act](guide/auditability.md) →
[report](guide/report.md), then back to re-anchoring. The
[How it works](how-it-works.md) page draws that loop once, in one view.

**Calibrate.** Thirteen calibrators: Platt, temperature, beta, isotonic, centered
isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers (IVAP and CVAP),
spline, and segmented. With them come a rigid, auditable logit offset, automatic
selection under nested cross-validation, and prefit / cross-validation data flows.
[Choose a calibrator](guide/choosing.md) is the catalog.

**Evaluate.** Proper scores, binned and binning-free calibration errors, the CORP
decomposition, per-grade supervisory backtests, calibration belts, reliability curves
on both probability and logit scales, [grouped evaluation](guide/groups.md) by segment,
and **conservatism** tooling for low-default portfolios (Pluto–Tasche most-prudent PDs,
Jeffreys upper bands, margin-of-conservatism offsets).

**Decide.** Exact and generalized inverses turn a policy PD into a raw-score cutoff or
a whole masterscale, refusing unattainable targets instead of clamping.
[Set cutoffs and invert maps](guide/cutoffs.md).

**Monitor and act.** Anytime-valid **monitoring** runs on an e-process whose alarm keeps
its type-I guarantee at every look, under
[stated assumptions](guide/monitoring.md#assumptions-and-scope). It localizes the onset
of drift and issues a
recommendation that can be applied as a new offset, on the record.
[Monitor and act](guide/monitoring.md).

**Prove it.** JSON **serialization** (never pickle) with fingerprints and a golden-file
compatibility promise, self-contained HTML **reports**, and one page tying the artifacts
to what each of them actually proves: [Auditability](guide/auditability.md).

**Integrate.** **scikit-learn** (on ≥ 1.6 a bare calibrator is accepted as an estimator
without any adapter; the adapter covers the probability-matrix API from 1.4, see the
[support matrix](api-stability.md#support-matrix)), **optbinning**
scorecards (calibrated PDs carried back to the points scale), and **treecf**
counterfactuals bound to a named calibrator.

```bash
pip install probcal
```

Where to start, by what you are doing:

- **New to calibration** → [Install and quickstart](getting-started.md), then the
  20-minute [PD calibration walkthrough](notebooks/pd_calibration_walkthrough.ipynb)
  (diagnose, select, fit, backtest). The longer
  [full lifecycle notebook](notebooks/pd_end_to_end.ipynb) adds offsetting, treecf
  recourse, monitoring, and JSON round-trips on a rare-event portfolio.
- **Validating someone's model** → [Auditability](guide/auditability.md) and
  [Build a validation report](guide/report.md).
- **Coming from scikit-learn** → the three-tier [sklearn guide](guide/sklearn.md).
- **Unsure what a word means** → the [glossary](glossary.md).

--8<-- "docs/_snippets/status.md"

The full policy, the public surface, and the support matrix:
[API stability](api-stability.md).
