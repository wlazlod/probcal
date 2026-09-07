# probcal

[![PyPI](https://img.shields.io/pypi/v/probcal.svg)](https://pypi.org/project/probcal/)
[![Downloads](https://img.shields.io/pypi/dm/probcal.svg)](https://pypistats.org/packages/probcal)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22069679.svg)](https://doi.org/10.5281/zenodo.22069679)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Probability calibration for binary classifiers, built for regulated PD models.
The runtime is numpy only. The calibration maps are the standard ones; what
probcal adds is the evidence a validator asks for next, which scikit-learn does
not produce:

- **Logit-scale diagnostics that keep 1% readable.** `calibration_guardrails`
  reports slope, intercept, and Spiegelhalter's test in log-odds, and every
  reliability curve draws on the logit axis, so a 3% portfolio is not crushed
  into the bottom-left corner.
- **An auditable offset stage.** A central-tendency adjustment is a
  `LogitOffset` object, not a refit: `audit_report()` records the mean before
  and after, the guardrails before and after, a timestamp, and a fingerprint.
- **Per-grade regulatory backtests.** `binomial_grade_test` and
  `jeffreys_grade_test` give the traffic-light table per rating grade, and
  `jeffreys_upper_bands` turns the same posteriors into a masterscale.

## Installation

```bash
pip install probcal            # runtime: numpy only
pip install "probcal[viz]"     # + matplotlib for probcal.plots
pip install "probcal[sklearn]" # + scikit-learn for probcal.sklearn adapters
```

Beta on PyPI. Breaking changes bump the minor version until 1.0 (see *API
stability* in the docs); serialized artifacts have a stronger promise, every
0.x release reads schema 1. Development setup:

```bash
git clone https://github.com/wlazlod/probcal && cd probcal
uv sync --extra dev
```

## Quickstart

The three things above, in one run. The calibrator is fitted on one synthetic
3% portfolio and every number below is measured on a second, held-out draw;
fitting and scoring the same rows would make the "after" line an identity
(see *Data splitting* in the docs):

```python
import numpy as np
from probcal import BetaCalibrator, LogitOffset, make_pd_portfolio
from probcal.metrics import calibration_guardrails, jeffreys_grade_test

cal_set = make_pd_portfolio(n=8000, random_state=42)   # synthetic 3% PD portfolio
test = make_pd_portfolio(n=8000, random_state=1)       # held-out draw, same distortion

# 1. Logit-scale diagnostics: slope, intercept, Spiegelhalter, before and after
g0 = calibration_guardrails(test.y, test.scores)
print(f"before: slope={g0.slope:.3f}  intercept={g0.intercept:+.3f}  ok={g0.all_ok}")
cal = BetaCalibrator().fit(cal_set.scores, cal_set.y)
p = cal.predict_proba(test.scores)                    # evaluated on the held-out set
g1 = calibration_guardrails(test.y, p)
print(f"after:  slope={g1.slope:.3f}  intercept={g1.intercept:+.3f}  ok={g1.all_ok}")

# 2. An auditable offset: re-anchor to a 3.5% policy PD and record what it cost
off = LogitOffset(target_mean=0.035).fit(p)
print(off.audit_report(test.y, p))
p_final = off.transform(p)

# 3. Per-grade regulatory backtest on a fixed PD masterscale
edges, labels = np.array([0, 0.01, 0.02, 0.05, 0.10, 1.0]), np.array(list("ABCDE"))
grades = labels[np.searchsorted(edges, p_final, side="right") - 1]
res = jeffreys_grade_test(test.y, p_final, grades)
for g, n, k, pd_, light in zip(res.grades, res.n, res.k, res.pd, res.light):
    print(f"grade {g}: n={n:5d}  defaults={k:3d}  PD={pd_:.4f}  {light}")
```

Output:

```text
before: slope=0.901  intercept=-0.743  ok=False
after:  slope=0.922  intercept=+0.021  ok=True
AuditReport(delta=+0.1238, odds factor 1.1317, fitted <timestamp>)
  portfolio mean: 0.03126 -> 0.03500
  slope:          +0.922 -> +0.922
  intercept:      +0.021 -> -0.103
  spiegelhalter p 0.460 -> 0.271
  guardrails ok:  True -> False
grade A: n= 1696  defaults=  7  PD=0.0063  green
grade B: n= 2043  defaults= 22  PD=0.0146  green
grade C: n= 2684  defaults= 97  PD=0.0319  green
grade D: n= 1114  defaults= 70  PD=0.0681  green
grade E: n=  463  defaults= 59  PD=0.1685  green
```

The audit report says what the policy offset cost: the intercept guardrail
now fails by exactly the applied shift, on the record. The executed
[end-to-end notebook](https://wlazlod.github.io/probcal/notebooks/pd_end_to_end/)
takes a rare-event portfolio through selection, backtests, offsetting,
threshold translation, and monitoring.

## Three calibrators, and the rest

`PlattCalibrator`, `BetaCalibrator`, and `IsotonicCalibrator` are the
production path: the quickstart and the benchmark use them, each has exact
inverse maps, `interpret()`, and JSON serialization pinned by golden files.
Ten more (temperature, centered isotonic, histogram binning, scaling-binning,
BBQ, ENIR, Venn–Abers IVAP and CVAP, spline, segmented) and the nested-CV
`CalibratorSelector` are in the
[calibrator catalog](https://wlazlod.github.io/probcal/guide/choosing/),
with monotonicity, inverses, data appetite, and fit cost per row. The
47-symbol metric catalog, with which metrics are safe to select on, is in
[Metrics and tests](https://wlazlod.github.io/probcal/concepts/metrics/).

## Serialization

Every fitted object round-trips through versioned, human-readable JSON, never
pickle (auditable; loading executes no code):

```python
cal.to_json("beta.json")
loaded = BetaCalibrator.from_json("beta.json")   # bit-identical predictions
cal.fingerprint()                                # sha-256 provenance id
```

Compatibility promise: every 0.x release reads schema 1, enforced by committed golden
files in CI; schema bumps ship only with a converter. Details: the *Serialization*
concepts chapter.

## Documentation

Built with mkdocs-material; run locally with `uv run mkdocs serve`. Start with
*Getting started*, then the *Concepts* chapters — the package's theoretical foundation.
The *Visualization* chapter is a
gallery of every plot, regenerated deterministically by `docs/scripts/generate_figures.py`;
the CORP reliability diagram, MCB-DSC plane, and score decomposition have their own
*CORP and score decomposition* chapter.

## License

MIT. See [LICENSE](LICENSE). GPL-licensed R packages are used as conceptual references
only; no GPL code is included.
