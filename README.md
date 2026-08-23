# probcal

[![PyPI](https://img.shields.io/pypi/v/probcal.svg)](https://pypi.org/project/probcal/)
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

**Status:** released on PyPI, beta. The API is stable enough to build on; breaking changes
bump the minor version until 1.0.

## Installation

```bash
pip install probcal            # runtime: numpy only
pip install "probcal[viz]"     # + matplotlib for probcal.plots
```

Development setup (tests, lint, type-check):

```bash
git clone https://github.com/wlazlod/probcal && cd probcal
uv sync --extra dev
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

| Capability | probcal | scikit-learn | netcal | probcal (R)² | single-method packages¹ |
|---|---|---|---|---|---|
| Calibration methods | 11 | 2 | many | 5 binary³ | 1 each |
| Runtime dependencies | numpy | scipy stack | torch stack | native R | varies |
| Logit-scale diagnostics (low-PD readable) | yes | — | — | — | — |
| First-class auditable offset (central tendency) | yes | — | — | — | — |
| Automatic selection under nested validation | yes | — | — | — | — |
| Venn–Abers intervals | yes | — | — | — | venn-abers |
| Metric catalog with selection-suitability guidance | yes | partial | partial | partial | — |
| Per-grade regulatory backtests (binomial, Jeffreys) | yes | — | — | — | — |
| Kernel calibration error and test (SKCE, Widmann et al.) | yes | — | — | **yes** | — |
| Calibrated→raw threshold translation (`interval_inverse`) | yes | — | — | — | — |
| SHAP additivity repair on the calibrated scale | yes | — | — | — | — |
| Parameter interpretation (`interpret()`) on every method | yes | — | — | partial | — |

¹ betacal, venn-abers, ml-insights.
² prdm0/probcal (P. R. Diniz Marinho), unaffiliated — see the FAQ. Verified against v0.2.0, 2026-08-08.
³ Platt, temperature, beta, isotonic, histogram binning; its multiclass methods (Dirichlet, vector scaling, one-vs-rest) are out of probcal's binary scope.

### Calibrators at a glance

| Method | Class | Scaling |
|---|---|---|
| Platt scaling | `PlattCalibrator` | O(n) per IRLS iteration |
| Temperature scaling | `TemperatureCalibrator` | O(n) per IRLS iteration |
| Beta calibration | `BetaCalibrator` | O(n) per IRLS iteration |
| Isotonic regression | `IsotonicCalibrator` | O(n log n) fit (sort + PAVA) |
| Centered isotonic (CIR) | `CenteredIsotonicCalibrator` | O(n log n) fit |
| Histogram binning | `HistogramBinningCalibrator` | O(n log n) fit |
| Scaling-binning | `ScalingBinningCalibrator` | O(n log n) fit |
| BBQ | `BBQCalibrator` | O(n log n) fit per candidate binning |
| ENIR | `ENIRCalibrator` | quadratic in unique scores; intended for m ≲ 50,000 (`fit` warns above) |
| Venn–Abers (IVAP) | `VennAbersCalibrator` | O(n log n) fit, O(log n) per prediction |
| Spline calibration | `SplineCalibrator` | O(n · k) per IRLS iteration (k knots) |

Performance note: the ICI family (`ici`/`e50`/`e90`/`emax`) shares one LOESS fit anchored
to `grid_size=512` quantile points instead of refitting at every observation — the same
device R's `stats::lowess` uses via its `delta` parameter (fit at spaced points, interpolate
the rest) — and `smooth_ece` pre-aggregates its residual measure onto `bins=8192` cells
before the bandwidth bisection — for every n ≥ 64 as of this release (0.1.3 ran the exact
path for n ≤ 8192: ~1s at n=4000 on this host; now 1–4ms across n ∈ [64, 10⁵] and ~43ms at
n=10⁶, where the O(n) pre-binning dominates). Measured on this host: `ici` at n=50,000 dropped from 192.2s
(v0.1.2) to 1.2s, and `loess(grid_size=512)` now fits n=1,000,000 points in under 30s.
`grid_size=None` and `bins=None` recover the exact pre-0.1.3 values and cost, so nothing is
lost for portfolios small enough to afford it. Still numpy-only; Rust acceleration remains
out of scope unless a future workload demands it.

## Documentation

Built with mkdocs-material; run locally with `uv run mkdocs serve`. Start with
*Getting started*, then the *Concepts* chapters — the package's theoretical foundation —
and the executed *PD calibration walkthrough* notebook. The *Visualization* chapter is a
gallery of every plot, regenerated deterministically by `docs/scripts/generate_figures.py`.

## License

MIT. See [LICENSE](LICENSE). GPL-licensed R packages are used as conceptual references
only; no GPL code is included.
