# Segmented calibration: shrunken per-segment offsets

A single calibration map is often right on average but wrong for a specific slice of the
portfolio (a product line, a vintage, a geography) because that slice's true residual
miscalibration differs from the pooled average the map was fit to correct. Fitting one offset
*per segment* from scratch overfits: a 20-obligor segment's offset MLE has a huge standard
error and mostly reflects sampling noise, not a real segment effect. Ignoring segments
entirely (complete pooling) throws away real heterogeneity when it exists. `SegmentedCalibrator`
sits between the two: one shared base map fit on all the data, plus a *shrunken* per-segment
logit offset that interpolates between "trust this segment's own data" and "trust the pooled
average", governed by how much genuine between-segment heterogeneity the data supports.

## Design rationale

Shrinking a per-segment *offset* toward the shared base map, rather than shrinking a full
per-segment refit toward a shared model, keeps four properties for free. The result stays
**monotone**: a level shift on the logit scale cannot un-sort scores that `base_` already
sorted. It stays **invertible**: inverting a scalar shift through `base_`'s own exact inverse
is closed-form for any `base_` (see *Protocol notes* below), whereas inverting a shrunk
nonlinear per-segment map has no general closed form. It stays **serializable**: the
per-segment state is six numbers (`delta_hat`, `se`, `delta_tilde`, `n`, `events`, `shrink`)
next to the one shared `base_`, not a second full calibrator's worth of parameters per
segment. And it stays **auditable per segment**: `interpret()` reads off, for any segment,
exactly how much of its own signal survived shrinkage versus how much fell back to the shared
map. A full per-segment refit (an independent `BetaCalibrator`, say, fit per segment, with
*its* parameters shrunk toward the pooled fit) does not shrink naturally to a well-defined
"no effect" point in parameter space, does not preserve monotonicity or exact invertibility
under shrinkage, and loses the one-number-per-segment audit trail; that design is out of
scope here.

## The model

`SegmentedCalibrator` fits a shared `base` calibrator (default `BetaCalibrator()`) on the full
dataset, exactly as if segments did not exist:

\[
p_0(s) = \text{base}(s).
\]

For each segment \( g \), it then fits the offset-only logistic MLE
(`probcal.offset.estimate_offset`) of the segment's residual log-odds shift against \( p_0 \):

\[
\hat\delta_g, \; \widehat{\mathrm{se}}_g = \texttt{estimate\_offset}\bigl(y_g,\, p_0(s_g)\bigr).
\]

\( \hat\delta_g \) is the natural, unbiased read of "how far off is `base` for segment \( g \)
specifically", but for a small segment its standard error is large, and using it directly
(no pooling) would put wide, noisy jumps into predictions for exactly the customers with the
least data behind them.

## Empirical-Bayes shrinkage (DerSimonian-Laird)

Treat the segment-level MLEs \( \hat\delta_g \) as noisy measurements of true, unknown
per-segment effects drawn from a common population with variance \( \tau^2 \): the classic
random-effects setup, here across segments instead of across independent studies
(DerSimonian & Laird, 1986). \( \tau^2 \) is estimated by their method-of-moments estimator,
restricted to the segments with a finite standard error (a single-class segment has no MLE,
see below, and contributes nothing to \( \tau^2 \)):

\[
w_g = \frac{1}{\widehat{\mathrm{se}}_g^{\,2}}, \qquad
\bar\delta_w = \frac{\sum_g w_g \hat\delta_g}{\sum_g w_g}, \qquad
Q = \sum_g w_g \bigl(\hat\delta_g - \bar\delta_w\bigr)^2,
\]

\[
\tau^2 = \max\!\left(0,\; \frac{Q - (G - 1)}{\sum_g w_g - \sum_g w_g^2 / \sum_g w_g}\right),
\]

with \( G \) the number of segments with a finite standard error; \( \tau^2 = 0 \) outright
when \( G < 2 \) (nothing to estimate heterogeneity from). Each segment's shrunk offset is
then the classic empirical-Bayes (precision-weighted) combination of its own estimate and the
population value 0 (the base map is already the pooled central estimate, so the population
mean of the *residual* offsets is 0 by construction):

\[
\text{shrink}_g = \frac{\tau^2}{\tau^2 + \widehat{\mathrm{se}}_g^{\,2}}, \qquad
\tilde\delta_g = \hat\delta_g \cdot \text{shrink}_g \in [0, 1).
\]

A small, noisy segment (large \( \widehat{\mathrm{se}}_g \)) has \( \text{shrink}_g \) near 0
and is pulled almost entirely back to the base map; a large, precise segment (small
\( \widehat{\mathrm{se}}_g \)) keeps most of its own estimate. When \( \tau^2 = 0 \) (no
detected heterogeneity beyond sampling noise), every segment shrinks fully to 0: complete
pooling, recovered exactly. Prediction applies the shrunk offset on the logit scale:

\[
p(s, g) = \sigma\bigl(\operatorname{logit}(p_0(s)) + \tilde\delta_g\bigr).
\]

A segment with only one outcome class has no offset MLE (`estimate_offset` raises
`ValueError`, since the score equation has no interior root); `SegmentedCalibrator` records
it as \( \hat\delta_g = 0 \), \( \widehat{\mathrm{se}}_g = \infty \), which shrinks fully
(\( \tau^2 / (\tau^2 + \infty) = 0 \)). That is the honest reading, since an
infinite-variance estimate carries zero weight in the pooling.

## Unseen segments and the `Chain` limitation

`fit` and `predict_proba` add a keyword-only `segments` argument on top of the base
calibrator signature: `segments=None` at fit time collapses to one segment `"__all__"`
(so the zero-argument protocol call `SegmentedCalibrator().fit(s, y)` still works), and
`segments=None` at predict time returns the plain base map (`delta=0`, no segment-specific
adjustment) rather than raising, since there is no segment information to look up with.
A label present in `segments` at predict time but never seen at fit time is handled by the
constructor's `unseen` policy: `"global"` (default) applies `delta=0`; `"raise"` raises
`ValueError`, for deployments where an unrecognized segment must not silently fall back.

Labels are compared as strings (`_coerce_segments` calls `.astype(str)`): fitting with
integer labels `0`, `1` stores them as `"0"`, `"1"`, but predicting with float labels `0.0`,
`1.0` looks up `"0.0"`, `"1.0"`. That is a silent mismatch, since every row then looks
"unseen" and, under the default `unseen="global"`, falls back to the base map without
raising. Pass `segments` with the *same* representation at fit and predict time (cast to
`str` yourself if the label type is not guaranteed to match). As a backstop, `predict_proba` and the
`segment=` inverse paths raise a `UserWarning` whenever *every* row of one call is unseen and
`unseen="global"`, the exact failure mode this int/float mismatch produces, and it names the
fitted `segments_` so the mismatch is easy to spot; a partial overlap (some rows match,
others are genuinely new segments) stays silent, since that is a legitimate use case.

`probcal.chain.Chain` has no `segments=` slot: every stage's `predict_proba` is called with
no extra arguments. `Chain([seg, ...])` therefore always predicts through `seg`'s global map
(`segments=None`, `delta=0`); the per-segment shift is never applied inside a `Chain`. Use
`SegmentedCalibrator` directly, passing `segments=`, whenever the per-segment offset must
apply; `Chain([seg, offset])` remains useful for composing `seg`'s *global* map with a
portfolio-wide offset (e.g. `monitor.moc_offset_from_counts`), same as any other calibrator.

## Protocol notes

`is_monotone_` is `base_.is_monotone_`, since segmentation adds a level shift per segment,
which does not change monotonicity in the raw score. `affine_logit_coeffs_` (the whole-calibrator
property external tooling, e.g. attribution repair, reads) is `(a, b + delta_tilde)` only when
exactly one segment was fitted and `base_` is itself affine on the logit scale; with more than
one segment there is no single affine map for the whole object (each segment has its own
intercept shift), so it is `None`. This does not affect `SegmentedCalibrator`'s own
`interval_inverse`/`point_inverse`, which always invert through `base_` directly (composed
with the requested segment's `delta_tilde` via `Chain([base_, LogitOffset(delta=delta_tilde_g)])`,
or `base_` alone when `segment=None` or the shrunk offset is exactly 0), and so work for any
number of segments as long as `base_` itself has an exact inverse.

## Example

Three segments of very different size and true miscalibration, `micro` (n=30), `mid`
(n=300) and `large` (n=3000), with `base_` fit on the pooled data:

```python
import numpy as np
from probcal import SegmentedCalibrator
from probcal._math import expit, logit

rng = np.random.default_rng(42)
sizes = {"micro": 30, "mid": 300, "large": 3000}
true_deltas = {"micro": -0.6, "mid": 0.12, "large": 0.5}

s_parts, y_parts, seg_parts = [], [], []
for name, n in sizes.items():
    s_g = expit(rng.normal(-1.0, 1.0, n))
    p_true = expit(logit(s_g) + true_deltas[name])
    y_g = (rng.random(n) < p_true).astype(float)
    s_parts.append(s_g)
    y_parts.append(y_g)
    seg_parts.append(np.full(n, name))

scores = np.concatenate(s_parts)
y = np.concatenate(y_parts)
segments = np.concatenate(seg_parts)

cal = SegmentedCalibrator().fit(scores, y, segments=segments)
print(cal.interpret())

p_global = cal.predict_proba(scores)                       # base map only (delta=0)
p_segmented = cal.predict_proba(scores, segments=segments)  # per-segment shrunk offset applied
```

```text
Interpretation[SegmentedCalibrator]
parameter    value
-----------  ---------
tau2         0.195006
delta.large  0.0598435
delta.micro  -0.51923
delta.mid    -0.484786
- tau2 = 0.1950: between-segment heterogeneity variance (DerSimonian-Laird method of moments on the per-segment offset MLEs); tau2 = 0 means complete pooling (every delta_tilde = 0)
- segment 'large': n=3000, events=1259.0, delta_hat=+0.0603, se=0.0402, delta_tilde=+0.0598, shrink=0.992
- segment 'micro': n=30, events=6.0, delta_hat=-1.1095, se=0.4708, delta_tilde=-0.5192, shrink=0.468
- segment 'mid': n=300, events=90.0, delta_hat=-0.5307, se=0.1359, delta_tilde=-0.4848, shrink=0.913
- unseen segments at predict/inverse time use delta=0 (unseen='global')
```

`large` is precise (`se=0.04`) and keeps 99.2% of its own offset MLE. `micro` is noisy
(`se=0.47`, only 30 observations) and is pulled halfway back toward the population: its raw
MLE overshoots the true `-0.6` at `-1.11`, but `delta_tilde=-0.52` is much closer.

## Recovery simulation

`docs/scripts/segmented_sim.py::recovery(runs, n_per_segment, true_deltas, seed)` draws six
segments with true residual offsets spread `-0.6 .. +0.6` and sizes `30 .. 3000`, fits a
`SegmentedCalibrator`, and compares the mean squared error (across segments and runs) of
three estimators of the true per-segment offset against no pooling (`delta_hat_`), complete
pooling (one offset MLE fit on the pooled data, ignoring segment identity), and the shipped
empirical-Bayes shrinkage (`delta_tilde_`); a second, homogeneous scenario (every segment's
true delta is 0, `n=3000`) checks that shrinkage degrades gracefully to complete pooling as
the true spread shrinks to 0. `tests/test_segmented_sim.py` (`pytest.mark.slow`) enforces the
same gates at a reduced run count in CI.

| scenario                                 | runs | MSE no-pooling | MSE complete-pooling | MSE / stat EB                  |
|------------------------------------------|------|----------------|-----------------------|---------------------------------|
| heterogeneous (spread -0.6..+0.6)        | 2000 | 0.2354         | 0.1680                | 0.1242                          |
| homogeneous (all true delta = 0, n=3000) | 2000 | -              | -                      | max\|mean delta_tilde\| = 0.0003 |

Empirical Bayes beats both no pooling and complete pooling on the heterogeneous scenario (it
never has to choose between "trust this tiny segment" and "ignore segments"; it blends the
two per segment, weighted by how much of a segment's disagreement with its peers survives
sampling noise), and collapses to complete pooling (shrunk offsets near 0) once there is no
real heterogeneity left to detect.
