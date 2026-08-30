# Conservatism: most-prudent PDs and margins of conservatism

The per-grade backtests in [Metrics and tests](metrics.md) — `binomial_grade_test`,
`jeffreys_grade_test` — ask whether a grade's *assigned* PD is consistent with its
realized defaults. On a low- or zero-default grade (routine for the best few grades of a
retail or sovereign scorecard) that question is close to unanswerable from the grade's own
data alone: with zero defaults, every exact tail probability or Jeffreys posterior reading
is uninformative about the true PD, and a naive point estimate of `0/n = 0` is not a usable
PD to assign going forward (regulators do not accept "the PD is exactly zero"). Pluto &
Tasche (2005) solve a different, decision-relevant problem instead: not "is the assigned PD
consistent with the data", but "what is the *most-prudent* (upper-bound) PD I can defend for
this grade, at a stated confidence, given everything I know about the whole rating scale".
`probcal.metrics.pluto_tasche` implements their one-period estimator; `jeffreys_upper_bands`
answers the related masterscale-wide question with a different tool; and `monitor.moc_offset`
/ `monitor.moc_offset_from_counts` carry the same most-prudent logic from a rating grade to
an already-calibrated portfolio, as an offset that stacks on top of calibration rather than
replacing it.

## When each mechanism applies

Four tools solve four distinct instances of "give me a defensible upper reading, not a point
estimate", and picking the right one depends on what is available and at what granularity the
number is needed:

| Mechanism | Applies when | Granularity | Needs |
|---|---|---|---|
| `metrics.pluto_tasche(_from_arrays)` | A rating scale has zero- or low-default grades and rating-order is trusted | Per grade | Ordered grades, own + pooled counts |
| `metrics.jeffreys_upper_bands` | Publishing a full, contiguous masterscale band table for `thresholds.calibrated_bands_to_raw` | Per grade (own counts only, no pooling) | Ordered grades, own counts |
| `monitor.moc_offset` | A `CalibrationMonitor` is already tracking the portfolio and its confidence sequence is live | Portfolio-wide | A live monitor, or its frozen report |
| `monitor.moc_offset_from_counts` | No monitor is running — only aggregate outcome counts are on hand | Portfolio-wide | Realized `(y, p)` |

Pluto-Tasche and the Jeffreys bands both answer "what PD may I *assign* to this grade" —
they are inputs to a masterscale, evaluated before or at the point of grading. The two
`monitor.moc_offset*` functions answer a different question, downstream of grading: "given
what I have observed about an *already-calibrated* portfolio's outcomes, by how much should
I conservatively shift it". They return a fitted `LogitOffset` rather than a per-grade
number, meant to be applied — see [Margin of conservatism](#margin-of-conservatism-composing-with-calibration-not-replacing-it)
below.

## The monotonicity assumption

A rating scale is presumed *ordered*: the true PD cannot decrease from a better grade to a
worse one. That assumption is doing the actual work. It licenses pooling a grade's own
(possibly zero-default) data with every worse grade's data before computing the bound — if
grade *i*'s true PD cannot exceed grade *i+1*'s, then observing few defaults across grades
*i* through *K* combined is still meaningful evidence that grade *i*'s own PD is low, even
when grade *i*'s own sample is too small or too clean to say anything by itself.

## The bound

For grades ordered best to worst (index 1..K, as given — the order is not inferred from the
data), pool grade *i* with every worse grade:

$$
n^*_i = \sum_{j=i}^{K} n_j, \qquad d^*_i = \sum_{j=i}^{K} d_j .
$$

The most-prudent PD for grade *i* is the one-sided Clopper-Pearson upper confidence bound of
the pooled default rate: the PD `p` at which observing at most `d*_i` defaults among `n*_i`
obligors has probability exactly `1 - confidence`,

$$
I_p(d^*_i + 1,\; n^*_i - d^*_i) = \text{confidence} \quad\Longrightarrow\quad
p = F^{-1}_{\mathrm{Beta}(d^*_i + 1,\; n^*_i - d^*_i)}(\text{confidence}),
$$

computed by `probcal._math.beta_ppf` (the same incomplete-beta quantile machinery
`binomial_grade_test`/`jeffreys_grade_test` use for their display intervals). Two edge cases
are handled directly rather than through the general solver: `d*_i == n*_i` (every pooled
obligor defaulted) gives `pd_upper = 1.0`, and a pooled set with `n*_i == 0` obligors raises
`ValueError` rather than silently returning a meaningless bound.

**Zero defaults, closed form.** When `d*_i = 0`, the incomplete-beta identity
`I_p(1, n) = 1 - (1 - p)^n` gives an exact closed form,

$$
p = 1 - (1 - \text{confidence})^{1/n^*_i},
$$

used in `tests/test_conservative.py` to pin `beta_ppf`'s bisection output to `1e-12`
independent of the general solver, and to reproduce the worked example from Pluto & Tasche
(2005) — grades A/B/C with `n = (100, 400, 300)`, all zero-default, at 90% confidence give
`pd_upper ≈ (0.29%, 0.33%, 0.76%)`.

**Weighted counts.** `pluto_tasche_from_arrays` accepts `sample_weight`; the resulting
per-grade counts (and their cumulative pools) are weighted sums, which can be non-integer.
These pass directly into the Beta shape parameters above — a documented convention, not an
approximation the caller needs to round away.

## Monotonicity of the output

Pooled sets are nested: grade *i*'s pooled set is grade *i*'s own data plus grade *i+1*'s
entire pooled set. For a portfolio whose observed per-grade default rates already respect
rating order, this makes `pd_upper` come out non-decreasing best to worst on its own. It is
not a guarantee for arbitrary input, though: a single noisy grade whose own rate happens to
exceed the worse-grade pool it joins can pull that grade's pooled rate — and so its bound —
above the next grade's. `pluto_tasche` corrects any such dip by taking the cumulative
maximum of `pd_upper` best to worst (a prudent hull: `hull[i] = max(pd_upper[0..i])`), so a
worse grade's bound is never below a better grade's. This is the only correction direction
consistent with a *most-prudent* estimator — it can only raise a grade's bound up to a
better grade's, never lower a grade's bound to match a worse one — and it never reduces any
individual grade's bound. `PlutoTascheResult.monotonized` reports whether the hull actually
changed anything. On realistic, already-monotone data it is a no-op (`monotonized = False`);
the coverage simulation below never triggers it either, since its true PD is deliberately
monotone with enough obligors per grade that sampling noise essentially cannot invert the
pooled-rate ordering.

## In probcal

```python
import numpy as np
from probcal.metrics import pluto_tasche, pluto_tasche_from_arrays

# From per-grade counts directly (Pluto & Tasche 2005 worked example).
res = pluto_tasche(
    np.array([100.0, 400.0, 300.0]),
    np.array([0.0, 0.0, 0.0]),
    confidence=0.9,
    grades=("A", "B", "C"),
)
print(res.pd_upper)          # [0.00287..., 0.00329..., 0.00758...]
print(res.interpret())       # one audit sentence per grade

# From observation-level grades and outcomes.
grades = np.array(["A"] * 100 + ["B"] * 400 + ["C"] * 300)
y = np.zeros(800)             # zero defaults observed
res2 = pluto_tasche_from_arrays(grades, y, order=("A", "B", "C"), confidence=0.9)
```

Both entry points return the same `PlutoTascheResult` and are re-exported from
`probcal.metrics.grade`, alongside the per-grade backtests they complement.

## Jeffreys upper bands: a masterscale band table

`jeffreys_upper_bands(y, p, grades, *, level=0.9, order=None)` packages the same per-grade
Jeffreys posterior upper bound `jeffreys_grade_test` already reports as its own-grade display
interval — `beta_ppf(level, k_i + 0.5, n_i - k_i + 0.5)` under a `Beta(k_i + 0.5, n_i - k_i +
0.5)` posterior on grade *i*'s own default rate, no cross-grade pooling — into the
`{grade: (lo, hi)}` masterscale table `thresholds.calibrated_bands_to_raw` consumes directly.
`lo_i` is the previous grade's `hi` (`0.0` for the best grade), so the bands are contiguous by
construction. Because each grade's bound uses only its own counts, a zero-default grade still
gets a strictly positive `hi` from the Jeffreys prior alone — unlike `pluto_tasche`, which
needs pooling with worse grades to say anything about a zero-default grade at all.

The own-grade `hi` sequence need not come out non-decreasing on its own (a noisy grade can post
a smaller posterior upper bound than a better grade), which would make adjacent bands overlap.
`jeffreys_upper_bands` monotonizes it with `_math.pava` (weighted isotonic regression, weight =
grade size) in the given `order` — the minimum-adjustment non-decreasing fit, not a running
maximum — and warns (`UserWarning`) only when that adjustment changed something.

```python
import numpy as np
from probcal import BetaCalibrator
from probcal.metrics import jeffreys_upper_bands
from probcal.thresholds import calibrated_bands_to_raw

grades = np.array(["A"] * 100 + ["B"] * 100)
y = np.array([0.0] * 100 + [1.0] * 5 + [0.0] * 95)
p = np.array([0.01] * 100 + [0.05] * 100)
bands = jeffreys_upper_bands(y, p, grades, level=0.9)
# {"A": (0.0, 0.0134...), "B": (0.0134..., 0.0846...)}

fitted_calibrator = BetaCalibrator().fit(s_cal, y_cal)  # s_cal, y_cal: held-out calibration scores and outcomes
raw_bands = calibrated_bands_to_raw(fitted_calibrator, bands)
```

## Margin of conservatism: composing with calibration, not replacing it

`monitor.moc_offset` and `monitor.moc_offset_from_counts` (theory: [Monitoring](monitoring.md#margin-of-conservatism-offsets))
turn monitoring or outcome evidence into a fitted `probcal.LogitOffset` that shifts an
*already-calibrated* portfolio further, conservatively. `moc_offset(mon)` reads the upper end
of a running `CalibrationMonitor`'s time-uniform confidence sequence for the current offset;
`moc_offset_from_counts(y, p)` re-anchors `p`'s mean at the one-sided Jeffreys posterior upper
quantile of the realized outcomes when no monitor is running — the same Jeffreys quantile
`jeffreys_upper_bands` uses, one level up, on the portfolio mean instead of a per-grade rate.

Both return a fitted `LogitOffset`, not a calibrator: they carry no shape correction, only a
uniform log-odds shift. **The margin of conservatism composes with calibration; it never
substitutes for it.** A calibrator still does the job of getting the reliability curve right
in the first place — MoC only adds a defensible margin on top, exactly as `probcal.Chain`
composes any calibrator with `LogitOffset` stages elsewhere in the package (see
[Offset](offset.md#audit-trail-and-composition)):

```python
import numpy as np

from probcal import BetaCalibrator, Chain, expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor, moc_offset

train = make_pd_portfolio(n=4000, random_state=0)
cal = BetaCalibrator().fit(train.scores, train.y)

# A monitor observes six batches drawn under an injected +0.3 log-odds drift.
mon = CalibrationMonitor(alpha=0.05)
rng = np.random.default_rng(1)
for seed in range(6):
    batch = make_pd_portfolio(n=500, random_state=100 + seed)
    p_batch = cal.predict_proba(batch.scores)
    y_batch = (rng.random(500) < expit(logit(p_batch) + 0.3)).astype(float)
    mon.update(y_batch, p_batch, label=f"m{seed}")

chain = Chain([cal, moc_offset(mon)])  # calibration first, MoC offset second

test = make_pd_portfolio(n=2000, random_state=999)
print(cal.predict_proba(test.scores).mean())    # ~0.0265 — calibrated, undercorrected
print(chain.predict_proba(test.scores).mean())  # ~0.0384 — conservatively re-anchored
```

`Chain` enforces the ordering in its constructor — the first stage must be a fitted
calibrator, every stage after it a fitted `LogitOffset` — so `Chain([moc_offset(mon), cal])`
is not expressible by accident; the calibration step always comes first, the conservatism
margin is layered on afterward, and both stages stay separately auditable
(`chain.calibrator_`, `chain.offsets_`).

## What the one-period bound does not cover

Every bound on this page — Pluto-Tasche's Clopper-Pearson upper bound, the Jeffreys upper
bands, and the confidence sequence `moc_offset` reads from — is a **one-period** estimator:
it treats each obligor's default as an independent Bernoulli draw within the observation
window. Real portfolios violate that assumption. Defaults share exposure to common risk
factors (the same macro cycle, sector, or geography), so the *effective* variance of the
portfolio default count is higher than the binomial variance these bounds are built on. Under
positive cross-obligor correlation, **the one-period most-prudent bound understates the true
uncertainty** — its nominal coverage overstates the actual coverage the confidence level would
suggest, because correlated defaults cluster together in the bad states that matter most for
tail risk exactly where independence assumes they scatter.

Extending these bounds to a multi-period setting with explicit default correlation (asset
correlation as in Pluto & Tasche's own follow-on work, or a systematic-factor model as in the
Basel IRB framework more broadly) is **deliberately out of scope for probcal 0.3**. The
one-period bound is a defensible, well-understood floor — not a correlation-robust one — and
should be documented as such wherever it informs a masterscale or a production offset.

## Simulation verification

Produced by `docs/scripts/conservative_sim.py`: a four-grade portfolio
(`n = (2000, 2000, 1000, 500)`, true PD `(0.5%, 1%, 3%, 8%)`, monotone by construction),
2000 seeded runs per confidence level. Coverage for grade *i* is the share of runs with
`pd_upper_i >= pd_true_i`; the gated quantity is the minimum over grades, since that is the
weakest guarantee the bound offers across the whole scale. A reduced-size version of the
same gate (300 runs, widened tolerance) runs in CI (`tests/test_conservative_sim.py`).

| confidence | runs | per-grade coverage (min over grades) | all-grades coverage | gate      |
|------------|------|---------------------------------------|----------------------|-----------|
| 0.9        | 2000 | 0.9225                                 | 0.9225               | >= 0.8866 |
| 0.95       | 2000 | 0.9610                                 | 0.9610               | >= 0.9403 |

Per-grade and all-grades coverage coincide in this run: with a monotone true PD and nested
pooled sets, the run that fails coverage for one grade tends to fail it for the binding
grade in every run, so the joint (all-grades) event and the minimum single-grade event
happen to line up here — not a general identity, just what this configuration produces.

## References

- Pluto, K., Tasche, D. (2005). "Estimating probabilities of default for low default
  portfolios." Deutsche Bundesbank Discussion Paper (also in *The Basel II Risk Parameters*,
  Springer, 2006).
