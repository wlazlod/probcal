# Monitoring a deployed calibration

How-to; the statistics (and why fixed-sample tests are invalid under
optional stopping) live in the *Monitoring* concepts chapter.

```python
# s_cal, y_cal, grades: held-out calibration scores, outcomes, rating labels
from probcal import BetaCalibrator
from probcal.monitor import CalibrationMonitor

deployed = BetaCalibrator().fit(s_cal, y_cal)   # the map actually in production
mon = CalibrationMonitor(alpha=0.05)

# Each time a cohort's outcomes mature (arrival order — never reordered).
# The monitor watches the *calibrated* forecast, never the raw score:
y_batch, p_batch, grade_batch = y_cal[:200], deployed.predict_proba(s_cal[:200]), grades[:200]
step = mon.update(y_batch, p_batch, grade=grade_batch, label="2026Q3")
step.e_global      # the alarm statistic (alarm when it ever reaches 1/alpha)
step.delta_ci      # anytime-valid CI for the current offset: (-3.0, 0.55) here,
                   # i.e. one 200-row batch buys almost no precision; the
                   # sequence tightens as cohorts accumulate

# Persist between batches; resuming reproduces the trajectory bit-for-bit:
mon.to_json("monitor-state.json")
mon = CalibrationMonitor.from_json("monitor-state.json")

rep = mon.report()
rep.alarm_at        # first crossing label, or None
rep.recommendation  # "none" | "re-offset" | "re-fit" (diagnostic, not a test)
rep.reasoning       # the plain-language trail behind it

from probcal.plots import plot_e_process   # probcal[viz]
plot_e_process(rep)
```

`plot_e_process(rep, grades_panel=True)` adds the per-grade confidence
sequences below the wealth curves; the
[monitoring chapter](../concepts/monitoring.md#components) shows the
rendered figure on a twelve-cohort drift scenario and reads it line by
line.

Three operational rules. Persist the state instead of recomputing from raw
data; predictability is what makes the guarantee hold. A portfolio-wide
macro shock *should* trip the alarm: that is the monitor working, not a
false positive. After re-calibrating, start a **new** monitor on the new
forecasts. The `delta_ci` half-width is also the principled `buffer_logit`
for recourse certificates; see the treecf guide.

## Re-offsetting inside a pipeline

<!-- docs: requires sklearn -->

When the deployed calibrator lives inside an sklearn pipeline (see the
[scikit-learn guide](sklearn.md#stacking-an-offset)), acting on
[Closing the loop](../concepts/monitoring.md#closing-the-loop-apply_recommendation)'s
recommendation means swapping the offset step and refitting it alone,
leaving the calibrator untouched:

```python
# s_cal, y_cal: held-out calibration scores and outcomes
from sklearn.pipeline import Pipeline

from probcal.sklearn import SklearnCalibrator, SklearnOffset

pipe = Pipeline(
    [("cal", SklearnCalibrator()), ("off", SklearnOffset(target_mean=0.03))]
).fit(s_cal.reshape(-1, 1), y_cal)
fp_before = pipe.named_steps["cal"].calibrator_.fingerprint()

# apply_recommendation() returned a new delta; swap only the offset step:
new_delta = 0.05
pipe.set_params(off=SklearnOffset(delta=new_delta))
p_cal = pipe.named_steps["cal"].transform(s_cal.reshape(-1, 1))
pipe.named_steps["off"].fit(p_cal)
assert pipe.named_steps["cal"].calibrator_.fingerprint() == fp_before
```

The calibrator's fingerprint survives the swap because nothing there called
its `fit` again. A single-step `Chain([BetaCalibrator(), LogitOffset(...)])`
inside `SklearnCalibrator` is the other option when one auditable object,
rather than an independently swappable offset, is what governance wants.
