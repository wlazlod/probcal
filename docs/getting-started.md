# Getting started

## Installation

```bash
pip install probcal            # runtime dependency: numpy only
pip install "probcal[viz]"     # + matplotlib for probcal.plots
```

To work on the package itself:

```bash
git clone https://github.com/wlazlod/probcal && cd probcal
uv sync --extra dev            # tests, lint, type-check
```

probcal requires Python ≥ 3.11. The core import path touches nothing beyond numpy and the
standard library: scipy, scikit-learn, pandas, and matplotlib are never imported by
`probcal` itself (matplotlib only inside the optional `probcal.plots`).

## Score-level quickstart

Calibrators work directly on scores; no model object is required. The example uses the
built-in synthetic PD portfolio (3% event rate, asymmetric tail distortion), fitted on
one draw and measured on another: a calibrator scored on its own fitting rows reports
slope 1 and intercept 0 by construction, which is no evidence at all.

```python
from probcal import BetaCalibrator, make_pd_portfolio
from probcal.metrics import calibration_guardrails

cal_set = make_pd_portfolio(n=8000, random_state=42)   # calibration set
test = make_pd_portfolio(n=8000, random_state=1)       # held-out set, same distortion

g_before = calibration_guardrails(test.y, test.scores)
print(f"before: slope={g_before.slope:.3f}  intercept={g_before.intercept:+.3f}  ok={g_before.all_ok}")

cal = BetaCalibrator().fit(cal_set.scores, cal_set.y)
p = cal.predict_proba(test.scores)                    # never the rows it was fitted on

g_after = calibration_guardrails(test.y, p)
print(f"after:  slope={g_after.slope:.3f}  intercept={g_after.intercept:+.3f}  ok={g_after.all_ok}")
print()
print(cal.interpret())
```

Output:

```text
before: slope=0.901  intercept=-0.743  ok=False
after:  slope=0.922  intercept=+0.021  ok=True

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

Every calibrator follows the same contract: `fit(s, y, sample_weight=None)`,
`predict_proba(s)`, `interpret()`. Inputs are probabilities in `(0, 1)`; if your model
emits raw logits, convert first with the exported `probcal.expit`.

## Model-level wrapper

`CalibratedModel` wraps any object with `predict_proba(X)` or `decision_function(X)`:

```python
import numpy as np
from probcal import CalibratedModel, PlattCalibrator


class ScoreModel:
    """Stand-in for a trained classifier: reads the score off column 0."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {}


model = ScoreModel()
X_cal, X_new = cal_set.scores.reshape(-1, 1), test.scores.reshape(-1, 1)  # from above
X_train, y_train = X_cal, cal_set.y

# Prefit flow: the model is already trained, a separate calibration set exists.
wrapped = CalibratedModel(model, PlattCalibrator(), flow="prefit").fit(X_cal, cal_set.y)
p = wrapped.predict_proba(X_new)

# CV flow: no calibration set to spare — clone/retrain per fold, pool out-of-fold scores.
wrapped = CalibratedModel(model, PlattCalibrator(), flow="cv", cv=5).fit(X_train, y_train)

# Central-tendency re-anchoring, kept as a separate auditable stage:
wrapped.offset_to(target_mean=0.031)
print(wrapped.offsets_[0].interpret())
```

## Automatic selection

```python
from probcal import CalibratorSelector

s_cal, y_cal, s_new = cal_set.scores, cal_set.y, test.scores   # from the quickstart
sel = CalibratorSelector().fit(s_cal, y_cal)   # nested CV, log-loss criterion
print(sel.report_)                             # ranked table with guardrail flags
p = sel.predict_proba(s_new)                   # the refitted winner
```

The selector never scores a candidate on the data it was fitted on. See
[Automatic selection](concepts/auto-selection.md) for the protocol and
[Data splitting](concepts/data-splitting.md) for why the nesting is mandatory.

## Where to go next

- The [Concepts](concepts/why-calibration.md) chapters are the package's theoretical
  foundation: method derivations, metric pathologies, and the selection rules.
- The [tutorial notebook](notebooks/pd_calibration_walkthrough.ipynb) walks a full PD
  calibration cycle: diagnose, select, fit, re-anchor, backtest, and translate cutoffs
  back to raw scores.
- The [FAQ](faq.md) covers the inverse-map protocol and interop with counterfactual
  engines.
