# Set cutoffs and invert maps

Policies are written on calibrated probabilities — "approve below 2% PD",
"grade B is 0.5% to 2%" — and every consumer downstream of the calibrator
lives on the raw score: the decision engine, the masterscale, the
scorecard's points, the counterfactual generator. This page is the
translation service. The theory behind it (generalized inverses, plateau
semantics, the preimage identity) is
[Inverse maps](../concepts/inverse-maps.md); here it is only calls.

One prerequisite from [Choose a calibrator](choosing.md): translation
needs `is_monotone_ = True`, and an exact `point_inverse` needs an affine
or beta map. A non-monotone fit refuses, loudly, rather than returning a
cutoff nobody can defend.

## A policy PD to a raw cutoff

```python
# s_cal, y_cal: held-out calibration scores and outcomes
import numpy as np
from probcal import BetaCalibrator

cal = BetaCalibrator().fit(s_cal, y_cal)

# "PD <= 2%", as a raw-score interval and as a raw-margin (logit) interval:
lo_s, hi_s = cal.interval_inverse(0.0, 0.02)
lo_z, hi_z = cal.interval_inverse(0.0, 0.02, space="logit")
print(f"approve while score <= {hi_s:.4f}  (raw margin <= {hi_z:.4f})")

# The same boundary as a single exact point, available on affine and beta maps:
cut = cal.point_inverse(np.array([0.02]))[0]
print(cut, cal.predict_proba(np.array([cut]))[0])   # round-trips to 0.02
```

`lo=0` and `hi=1` mean "the full raw range on that side" — `0.0`/`1.0` in
probability space, `∓inf` in logit space — which is why a one-sided policy
is expressed as an interval with an open end rather than as a special
case. `space="logit"` is what a consumer holding the model's raw margin
wants (a SIGMOID-link scorer, a counterfactual engine); `space="probability"`
is what a threshold on the model's own output wants. The two are the same
number under `expit`, so pick by consumer, not by preference.

### When the policy is unattainable

A cutoff that the calibrator's output range does not contain is a policy
error, and probcal refuses to hide it behind a clamped number:

```python
from probcal import IsotonicCalibrator, UnattainableTargetError

iso = IsotonicCalibrator().fit(s_cal, y_cal)
print(iso.block_mean_.min(), iso.block_mean_.max())   # the attainable range

try:
    iso.interval_inverse(0.95, 1.0)
except UnattainableTargetError as err:
    print(err)      # names both intervals — never a silent clamp
```

This is routine on low-default portfolios: an isotonic map's range is the
span of its block means, so nothing above the top block is reachable. The
same refusal covers `point_inverse` targets at exactly 0 or 1, and
probability-space results whose raw logit exceeds `logit(1 - 1e-12)` —
there the error names `space="logit"`, where the answer is exact.

## A whole masterscale in one call

Grade edges are policy artifacts that outlive model versions; the
translation is what gets re-derived at each recalibration.

```python
from probcal import calibrated_bands_to_raw

masterscale = {"A": (0.0, 0.005), "B": (0.005, 0.02),
               "C": (0.02, 0.08), "D": (0.08, 1.0)}

raw_bands = calibrated_bands_to_raw(cal, masterscale, space="logit")
for grade, (band_lo, band_hi) in raw_bands.items():
    print(f"{grade}: raw margin in [{band_lo:.4f}, {band_hi:.4f})")
```

Adjacent grades share their edge exactly (`A`'s upper bound *is* `B`'s
lower bound), so the translated ladder covers the raw line without gaps or
overlaps. Store the output next to the calibrator's fingerprint: policy
fixed, mapping versioned — that pair is what makes a grade assignment
reproducible months later ([Auditability](auditability.md)).

## `buffer_logit`: cutoffs that survive the next re-anchoring

Every cutoff above is exact for *today's* calibrator. `buffer_logit`
shrinks the calibrated interval by a margin in logit space before
inverting, buying a conservative raw interval instead:

```python
# mon: a CalibrationMonitor with three seeded batches already applied
step = mon.steps_[-1]
half_width = 0.5 * (step.delta_ci[1] - step.delta_ci[0])
print(step.delta_ci, half_width)

lo_zb, hi_zb = cal.interval_inverse(0.0, 0.02, space="logit",
                                    buffer_logit=half_width)
print(hi_z, "->", hi_zb)        # the buffered cutoff is stricter
```

What is exact here: because a [`LogitOffset`](../concepts/offset.md) is a
pure translation on the logit scale, a future central-tendency update of
magnitude at most `m` cannot move a decision built with
`buffer_logit = m` across the boundary. What is a heuristic: reading `m`
off the monitor's anytime-valid confidence sequence for the current offset
(`MonitorStep.delta_ci`, described in
[Monitor and act](monitoring.md)). The confidence sequence covers the
plausible *offset* at level `1 - alpha` simultaneously at every stopping
time, which makes its width a defensible order of magnitude for "how far
might the next re-anchoring move things" — but it is not a bound on the
size of the update you will actually apply, and it says nothing at all
about a slope drift or a full re-fit, which change the map's shape rather
than translating it. Use it as a sized default, record the number you
used, and re-derive cutoffs after any re-fit regardless.

`probcal.monitor.moc_offset(mon)` takes the *upper end* of the same
confidence sequence as a margin-of-conservatism offset; the two uses are
siblings, one conservative in the cutoff, one conservative in the forecast.

## Carrying a cutoff to the points scale

A scorecard's deployed artifact is points, not probabilities.
`CalibratedScorecard.masterscale` translates calibrated PD bands to points
cut-offs exactly, because `Scorecard.score` is affine in the logistic
regression's log-odds (unless `rounding=True`, where the integration
refuses rather than approximating). Full workflow:
[optbinning scorecards](optbinning.md).

```python
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from probcal._math import logit
from probcal.integrations.optbinning import calibrate_scorecard

# X_cal: the calibration split's features — built here from s_cal alone
# (one numeric feature, the model's own logit), since this page carries no
# raw tabular features.
X_cal = pd.DataFrame({"z": logit(s_cal)})

sc = Scorecard(
    binning_process=BinningProcess(variable_names=["z"]),
    estimator=LogisticRegression(),
    scaling_method="pdo_odds",
    scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
).fit(X_cal, y_cal.astype(int))

cs = calibrate_scorecard(sc, X_cal, y_cal)
print(cs.masterscale(masterscale))   # {'A': (638.7, inf), 'B': (598.9, 638.7), ...}
```

The points are untouched by calibration — only the mapping from points to
PD changed, which is exactly what a recalibration should mean for a
deployed scorecard.

## Handing the cutoff to a counterfactual engine

[treecf](treecf.md) consumes calibrated targets through the same protocol;
the target resolves once, through `interval_inverse(..., space="logit")`,
and the search then runs on the raw margin:

```python
from treecf import Target

target = Target.calibrated(cal, op="<=", value=0.02)   # or buffer_logit=half_width
print(target.space, target.lo, target.hi)
```

Do not reach for `Target.probability(0.02)` once calibration is deployed:
it inverts the *model's* sigmoid link, so it silently targets the
uncalibrated probability. The [treecf guide](treecf.md) covers plateaus,
grade ladders, and the certificate that names the calibrator it was built
against.

## Related

- [Inverse maps](../concepts/inverse-maps.md) — generalized inverses,
  plateau semantics, the beta point-inverse construction.
- [Choose a calibrator](choosing.md) — which map gives you which inverse.
- [Auditability](auditability.md) — re-deriving a cutoff from the archived
  calibrator, months later.
- [Monitor and act](monitoring.md) — where `delta_ci` comes from.
