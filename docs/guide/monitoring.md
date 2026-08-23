# Monitoring a deployed calibration

How-to; the statistics (and why fixed-sample tests are invalid under
optional stopping) live in the *Monitoring* concepts chapter.

```python
from probcal.monitor import CalibrationMonitor

mon = CalibrationMonitor(alpha=0.05)

# Each time a cohort's outcomes mature (arrival order — never reordered):
step = mon.update(y_batch, p_batch, grade=grades, label="2026Q3")
step.e_global      # the alarm statistic (alarm when it ever reaches 1/alpha)
step.delta_ci      # anytime-valid CI for the current offset, e.g. (0.15, 0.40)

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

Three operational rules: persist the state instead of recomputing from raw
data (predictability is what makes the guarantee hold); a portfolio-wide
macro shock *should* trip the alarm — that is the monitor working, not a
false positive; after re-calibrating, start a **new** monitor on the new
forecasts. The `delta_ci` half-width is also the principled `buffer_logit`
for recourse certificates — see the treecf guide.
