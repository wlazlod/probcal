# Grouped evaluation

How-to; the bootstrap protocol itself is documented in the *Metrics and
tests* concepts chapter — this page covers only what `by=` adds on top of
it.

```python
from probcal.metrics import evaluate
from probcal.plots import plot_reliability   # probcal[viz]
from probcal.curves import reliability_binned

segment = np.where(scores < 0.01, "retail", "corporate")

report = evaluate(y, scores, n_boot=1000, seed=42, by=segment)
report.groups          # sorted labels, e.g. ("corporate", "retail")
report.pooled          # MetricReport on the full data (seed unchanged)
report.reports         # one MetricReport per group, aligned with .groups
report.counts          # observation count per group
report.to_frame()      # long-format rows: group, metric, value, ci_low, ci_high

fig = plot_reliability(reliability_binned(y, scores), y=y, p=scores, by=segment)
```

Each group's report is the same call you would make by hand on that
group's slice — `evaluate(y[mask], scores[mask], seed=42 + 1000 * i, ...)`,
where `i` is the group's position in sorted-label order — so results are
identical whether you group with `by=` or slice manually, and deterministic
regardless of how many groups exist or what the labels are. A group with
only one outcome class raises the same `ValueError` a direct call on that
slice would, naming the offending group. `plot_reliability(by=...)` draws
the same panels: a pooled panel plus one per group, laid out on shared axes
(`curve` itself is ignored in this mode — each panel rebuilds its own
binned curve from that group's data).

**What this is not.** `by=` reports side by side; it runs no test of
whether groups differ, and applies no multiple-comparison correction
across the group reports it returns. Formal group-conditional calibration
*testing* is future work, not implemented here — read `report.reports`
descriptively, the way you would read several `evaluate()` calls made by
hand.
