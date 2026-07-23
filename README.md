# probcal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Universal post-hoc probability calibration for binary classifiers: methods, metrics,
diagnostics, and auditable offsetting — **numpy-only**.

`probcal` unifies the binary calibration literature (Platt, temperature, beta, isotonic,
centered isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers, spline
calibration), an extensive catalog of calibration evaluation metrics and statistical tests,
calibration visualization on both probability and logit scales, an auditable logit-offset
(central tendency) adjustment, automatic method selection under nested validation, and two
data flows (prefit and cross-validation). Primary application domain: credit-risk PD models;
the package is fully general.

**Status:** pre-release, under active development. API may change until the first
PyPI release.

## Installation

```bash
pip install -e .            # runtime: numpy only
pip install -e ".[viz]"     # + matplotlib for probcal.plots
uv sync --extra dev         # development (tests, lint, type-check)
```

## Quickstart

```python
from probcal import BetaCalibrator, make_pd_portfolio
from probcal.metrics import calibration_guardrails

port = make_pd_portfolio(n=8000, random_state=42)   # synthetic 3% PD portfolio

g_before = calibration_guardrails(port.y, port.scores)
print(f"before: slope={g_before.slope:.3f}  intercept={g_before.intercept:+.3f}  ok={g_before.all_ok}")

cal = BetaCalibrator().fit(port.scores, port.y)
p = cal.predict_proba(port.scores)

g_after = calibration_guardrails(port.y, p)
print(f"after:  slope={g_after.slope:.3f}  intercept={g_after.intercept:+.3f}  ok={g_after.all_ok}")
print()
print(cal.interpret())
```

Output:

```text
before: slope=0.968  intercept=-0.765  ok=False
after:  slope=1.000  intercept=+0.000  ok=True

Interpretation[BetaCalibrator]
parameter  value
---------  --------
a          0.875054
b          1.58922
c          -1.15227
- a = 0.875: sensitivity near s -> 0; a < 1 raises the smallest probabilities (model was overconfident in the low tail), a > 1 deepens them
- b = 1.589: sensitivity near s -> 1; the mirrored reading for the high tail
- c = -1.152: base-rate shift of -1.152 log-odds, odds factor 0.316
- identity map corresponds to (a, b, c) = (1, 1, 0)
- a != b (gap -0.714): asymmetric tail distortion that no symmetric (Platt/temperature) map could express
```

Automatic selection, model wrapping, offsetting, and threshold translation:

```python
from probcal import CalibratedModel, CalibratorSelector, PlattCalibrator

sel = CalibratorSelector().fit(s_cal, y_cal)             # nested CV, log-loss criterion
wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, y_cal)
wrapped.offset_to(target_mean=0.031)                     # auditable central-tendency stage
lo_z, hi_z = wrapped.interval_inverse(0.0, 0.02, space="logit")   # "PD <= 2%" in raw margins
```

## Why probcal

| Capability | probcal | scikit-learn | netcal | single-method packages¹ |
|---|---|---|---|---|
| Calibration methods | 11 | 2 | many | 1 each |
| Runtime dependencies | numpy | scipy stack | torch stack | varies |
| Logit-scale diagnostics (low-PD readable) | yes | — | — | — |
| First-class auditable offset (central tendency) | yes | — | — | — |
| Automatic selection under nested validation | yes | — | — | — |
| Venn–Abers intervals | yes | — | — | venn-abers |
| Metric catalog with selection-suitability guidance | yes | partial | partial | — |
| Per-grade regulatory backtests (binomial, Jeffreys) | yes | — | — | — |
| Calibrated→raw threshold translation (`interval_inverse`) | yes | — | — | — |
| SHAP additivity repair on the calibrated scale | yes | — | — | — |
| Parameter interpretation (`interpret()`) on every method | yes | — | — | — |

¹ betacal, venn-abers, ml-insights.

Performance note: the PAVA family and special functions are hand-rolled numpy/stdlib;
profiling on real workloads has not shown them to be a bottleneck. Rust acceleration is
deliberately out of scope unless benchmarks on production-sized portfolios say otherwise.

## Documentation

Built with mkdocs-material; run locally with `uv run mkdocs serve`. Start with
*Getting started*, then the *Concepts* chapters — the package's theoretical foundation —
and the executed *PD calibration walkthrough* notebook.

## License

MIT. See [LICENSE](LICENSE) and `docs/LICENSING.md` for the conceptual-reference policy on
GPL-licensed R packages.
