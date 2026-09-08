# Glossary

The words the documentation uses with a fixed meaning. Credit-risk terms come
first with their generic equivalent, because PD models are the archetype the
package is built around; the package applies to any binary score.

**PD (probability of default).** The calibrated probability that an obligor
defaults within the horizon. Generic equivalent: the positive-class
probability of any binary classifier.

**Obligor.** One borrower, one row. Generic equivalent: an observation.

**Event, event rate.** A default (`y = 1`) and the share of rows that are
events. Sample size in this documentation is counted in events, not rows: 500
rows at a 3% event rate carry fifteen events.

**Score.** The model's output before calibration: a probability in `(0, 1)`,
or a raw logit converted with `probcal.expit`. Every calibrator maps scores to
calibrated probabilities.

**Calibration set, held-out set.** The rows a calibrator is fitted on, and
the rows every metric is reported on. They must be disjoint from each other
and from the model's training rows; see [Data
splitting](concepts/data-splitting.md).

**Grade.** A rating bucket: obligors whose calibrated PD falls in one band of
the masterscale. Per-grade backtests compare each grade's realized default
count with the PD assigned to it.

**Masterscale.** The fixed ladder of PD bands that defines the grades, shared
across models and reporting periods. `Masterscale` is the object: it assigns
grades (`lo <= p < hi`, top band closed), hands its bands to every translator,
and serializes with a fingerprint. `build_masterscale` designs one from data;
`jeffreys_upper_bands` builds a conservative band table from posteriors;
`calibrated_bands_to_raw` translates bands into raw-score cutoffs.

**Central tendency.** The long-run average default rate a portfolio's PDs are
required to aggregate to, set by policy rather than by the sample. Reaching it
is an offset, not a refit.

**Offset.** A rigid shift on the logit scale, `LogitOffset`, applied after the
calibration map and kept as its own auditable stage. `audit_report()` records
the mean, the guardrails, a timestamp, and a fingerprint before and after.

**Guardrails.** `calibration_guardrails`: the logit-scale calibration slope
(accepted in `[0.9, 1.1]`), the intercept (accepted within `±0.1`), and
Spiegelhalter's test (accepted at `p > 0.05`), reported together with an
overall `all_ok` flag.

**Traffic light.** The per-grade verdict of `binomial_grade_test` and
`jeffreys_grade_test`: green above `p = 0.05`, amber above `0.01`, red at or
below `0.01`. The tests are one-sided: a small value means the grade's PD is
likely understated.

**LDP (low-default portfolio).** A portfolio or grade with zero or very few
observed defaults, where the naive default rate is uninformative.

**Pluto–Tasche.** The most-prudent-estimate method for LDPs: an upper
confidence bound on each grade's PD that pools the grade with every worse
(higher-PD) grade, so that a zero-default grade still receives a positive PD.
`metrics.pluto_tasche` and `pluto_tasche_from_arrays`.

**MoC (margin of conservatism).** A deliberate upward shift of PDs to cover
estimation uncertainty. `monitor.moc_offset` derives one from a monitor's
confidence sequence for the current offset.

**Scorecard points.** The integer scale a deployed scorecard reports (for
instance PDO = 20, 600 points at 50:1 odds). The optbinning integration keeps
points unchanged and calibrates the PD behind them.

**Calibrator.** A fitted map from score to calibrated probability with the
contract `fit(s, y, sample_weight=None)`, `predict_proba(s)`, `interpret()`.
Thirteen are registered; the [catalog](guide/choosing.md) lists them.

**Reliability curve.** Observed event rate against predicted probability, drawn
on the probability or the logit axis. Binned, kernel, and CORP (isotonic)
constructions exist; see [Metrics and tests](concepts/metrics.md).

**Inverse map.** The raw score that produces a given calibrated probability.
`point_inverse` is the exact preimage, defined only for strictly monotone maps
with a closed form; `interval_inverse` is the generalized inverse of a
calibrated interval, `inf{s : g(s) >= lo}` to `sup{s : g(s) <= hi}`, defined
for every non-decreasing map, plateaus included. Unattainable targets raise
`UnattainableTargetError` instead of clamping.

**Cutoff.** A raw-score threshold obtained by inverting a policy PD, so the
decision rule can be expressed on the scale the deployed system uses.

**Segment.** A named subpopulation (a product line, a region) that receives its
own shrunken offset on top of a shared base map in `SegmentedCalibrator`, or
its own rows in grouped evaluation.

**E-process.** A non-negative process with initial value 1 whose expectation
under the null stays at most 1 at every look; crossing `1/alpha` is an alarm
that keeps its type-I error at `alpha` under optional stopping. The monitor's
null is that the deployed forecast is calibrated.

**Anytime-valid.** A guarantee that holds at every look at the data rather
than at one pre-planned sample size. Fixed-sample p-values lose this under
repeated checking; e-processes keep it.

**Onset.** The batch at which a monitored drift is estimated to have begun, as
opposed to the batch at which the alarm fired.

**Confidence sequence.** A sequence of intervals that all contain the true
value simultaneously with the stated probability, so it can be read after any
batch. The monitor keeps one for the current offset.

**Fingerprint.** SHA-256 of an object's canonical serialized form, blind to
library version and timestamp. Identical fits on identical data share it; it
is the provenance id monitors, registries, and recourse engines record.

**Golden file.** A committed JSON artifact in `tests/golden/` that a release
must still load and reproduce bit for bit. It is how "every 0.x release reads
schema 1" is enforced.

**Chain.** `probcal.Chain`: a calibrator followed by offsets, fitted and
inverted as one object, so a recourse engine sees a single map.
