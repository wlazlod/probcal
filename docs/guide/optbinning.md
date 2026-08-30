# optbinning scorecards

<!-- docs: requires optbinning -->

`probcal.integrations.optbinning` (extra: `pip install "probcal[optbinning]"`,
optbinning ≥ 0.21) calibrates a fitted `Scorecard` without touching the
deployed points, and translates calibrated PD policies to the points scale
*exactly*. `Scorecard.score` is affine in the logistic regression's
log-odds unless `rounding=True`, and the integration recovers and verifies
that affine map to machine precision.

Scorecard → calibration → masterscale → monitoring, end to end:

```python
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from probcal._math import logit
from probcal.integrations.optbinning import calibrate_scorecard
from probcal.monitor import CalibrationMonitor

# X_train, y_train: training split; X_cal, y_cal: held-out calibration
# split; X_new: new applicants — built here from s_cal/s_new (one numeric
# feature, the model's own logit) since the walkthrough carries no raw
# tabular features.
X_train = pd.DataFrame({"z": logit(s_cal)})
y_train = y_cal.astype(int)
X_cal = X_train
X_new = pd.DataFrame({"z": logit(s_new)})

# 1. A fitted scorecard (train split).
sc = Scorecard(
    binning_process=BinningProcess(variable_names=list(X_train.columns)),
    estimator=LogisticRegression(),
    scaling_method="pdo_odds",
    scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
).fit(X_train, y_train)

# 2. Calibrate on held-out data. Points are untouched; predict_proba is
#    the calibrated PD; the fitted probcal object is one attribute away.
cs = calibrate_scorecard(sc, X_cal, y_cal)
print(cs.interpret())
pd_cal = cs.predict_proba(X_new)          # calibrated PD
points = cs.score(X_new)                  # unchanged deployed points

# 3. Masterscale: calibrated PD bands -> exact point cut-offs.
bands = {"A": (0.0, 0.005), "B": (0.005, 0.02), "C": (0.02, 0.08), "D": (0.08, 1.0)}
print(cs.masterscale(bands))
# {'A': (662.3, inf), 'B': (614.9, 662.3), ...}  — cut-offs on the points scale

# 4. Provenance: the JSON names both layers.
cs.to_json("scorecard-calibration.json")   # calibrator envelope +
cs.fingerprint()                           # scorecard-table fingerprint

# 5. Monitoring: matured cohorts through the anytime-valid monitor.
cohorts = {"m0": (X_cal.iloc[:1500], y_cal[:1500]), "m1": (X_cal.iloc[1500:], y_cal[1500:])}
mon = CalibrationMonitor(alpha=0.05)
for label, (X_b, y_b) in cohorts.items():
    mon.update(y_b, cs.predict_proba(X_b), label=label)
print(mon.report().recommendation)
```

With `rounding=True` the points are no longer affine in log-odds:
`calibrate_scorecard` warns, `points_affine_coeffs_` is `None`, and
`masterscale` refuses. Use `interval_inverse` on the raw model probability
instead (the exact escape hatch; no silent approximation).

Reattaching after a reload: persist the scorecard with optbinning's own
`save`/`load` and the calibration layer as JSON, then
`CalibratedScorecard.from_dict(d, scorecard=sc)`. The stored
scorecard-table fingerprint is checked, so a calibration layer can never be
silently attached to a different scorecard.
