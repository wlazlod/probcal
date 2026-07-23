# Distribution-free methods

Every method so far produces a point estimate of the calibrated probability and hopes, with
varying statistical justification, that it is close. Venn–Abers predictors make a different
kind of promise: a **validity guarantee that holds by construction**, under no assumption
beyond exchangeability of the data — no model of the score distribution, no smoothness, no
correctness of the underlying classifier. The price is honesty about form: the guaranteed
object is not a single probability but a pair of them, an interval that widens exactly where
the calibration data run thin. This chapter develops the construction of Vovk and Petej
(2014), its inductive and cross-validated variants as implemented in `probcal.vennabers`, and
the precise scope of the guarantee — including what is lost when the interval is collapsed to
a scalar.

## The inductive Venn–Abers predictor (IVAP)

Fix a calibration set \( (s_1, y_1), \ldots, (s_n, y_n) \) and a new score \( s \) whose label
is unknown. The IVAP asks two counterfactual questions. If the new observation's label were
\( 0 \), what would isotonic regression fitted on the augmented set
\( \{(s_i, y_i)\} \cup \{(s, 0)\} \) predict at \( s \)? Call it \( p_0 \). And if the label
were \( 1 \) — the same construction with \( (s, 1) \) appended — call the prediction
\( p_1 \). Each question is answered by a full PAVA fit on \( n + 1 \) points, and the pair
satisfies \( p_0 \le p_1 \) always.

The interval \( [p_0, p_1] \) is the Venn–Abers prediction. Its width is not noise — it is the
construction reporting how much influence a single label at \( s \) has over the isotonic fit
there. In dense, well-behaved regions of the score axis one appended point moves nothing and
the interval is tight; in sparse regions, or at scores near a block boundary, one label can
tip the pooling and the interval opens up. `VennAbersCalibrator.predict_interval()` returns
the pair, and `interpret()` summarizes the mean and maximum width over the scored set — a
direct, assumption-free reading of *where* the calibration is trustworthy.

**The guarantee.** Venn–Abers predictors are a special case of Venn predictors, and inherit
their defining property: under exchangeability, the multiprobability prediction is
**perfectly calibrated** in the precise sense that one of the two announced probabilities is
the output of a perfectly calibrated probability forecaster (Vovk and Petej, 2014). The
theorem is distribution-free — it requires only that the calibration observations and the test
observation are exchangeable — and it is a guarantee about the *pair*, which is what makes the
next section necessary.

## Scalarization, and what it costs

Most pipelines need one number. Vovk and Petej (2014) derive the merger that is minimax
optimal under log loss:

\[
p \;=\; \frac{p_1}{1 - p_0 + p_1}\,,
\]

which probcal uses as the scalar output of `predict_proba`. It lands inside \( [p_0, p_1] \),
behaves sensibly at the extremes, and in the regime \( p_0 \approx p_1 \) reduces to the
common value. But the validity theorem does **not** transfer to it: the guarantee attaches to
the interval object, and any scalarization — this one included — is a lossy summary whose
calibration is excellent in practice but no longer holds by construction. probcal's
documentation states this precisely wherever the scalar output appears, because the
distinction is exactly the kind that evaporates in second-hand summaries: *the package
guarantees the interval; it recommends the scalar.*

The scalarized map is monotone in the score. Both \( p_0(s) \) and \( p_1(s) \) are
non-decreasing — each is read off an isotonic fit — and the merger is non-decreasing in
\( p_1 \) and in \( p_0 \) (its partial derivatives are positive on the relevant domain), so
the composition is non-decreasing in \( s \). This is what allows `VennAbersCalibrator` to
participate in the [inverse-map machinery](inverse-maps.md) via its block structure, and the
claim is unit-tested rather than assumed.

## What exchangeability does and does not buy

The single assumption deserves scrutiny, because it is the entire foundation.
**Exchangeability** requires that the joint distribution of the calibration observations and
the test observation is invariant under permutation — informally, that the new case is "of the
same kind" as the calibration cases, with no distinguished ordering. It is weaker than
independence and identical distribution, and it makes no demand whatsoever on the shape of the
score distribution or the correctness of the model that produced the scores. That is the
strength: the guarantee survives arbitrary miscalibration of the underlying classifier.

What it does not survive is **distribution shift between calibration and deployment**. A PD
model calibrated on last year's originations and applied to this year's — after a change in
underwriting policy, marketing channel, or macroeconomic regime — is scoring observations that
are not exchangeable with the calibration set, and the Venn–Abers guarantee lapses along with
every other method's assumptions. The failure is not graceful degradation unique to
Venn–Abers; it is the shared failure mode of all post-hoc calibration, and the periodic
re-anchoring discussed in the [offset chapter](offset.md) exists precisely because deployment
populations drift. The honest statement: Venn–Abers removes every assumption *except* the one
that time inevitably breaks, and its intervals quantify sampling uncertainty, not drift.

A computational note completes the picture. Each test score costs two isotonic fits on
\( n + 1 \) points; probcal's implementation deduplicates query scores and pays exactly that
price per unique score, which is comfortably fast at calibration-set sizes this package
targets (hundreds to a few thousand points). Vovk and Petej (2014) show the augmented fits
can also be served from structures precomputed once from the calibration set — an
\( O((n+m)\log(n+m)) \) batch algorithm that probcal records as a planned optimization in
its DECISIONS log, to be adopted if profiling on real workloads ever makes the naive route
the bottleneck.

## The cross Venn–Abers predictor (CVAP)

The IVAP spends the calibration set once. When data are scarce — the standing condition of
this package — the **cross** variant stretches them: split the available data into \( K \)
folds, and for each fold \( k \) form an IVAP whose isotonic fits use the other \( K - 1 \)
folds, yielding pairs \( (p_0^{(k)}, p_1^{(k)}) \) for the test score. The \( K \) intervals
are then merged by the geometric-mean rule of Vovk and Petej (2014):

\[
p \;=\;
\frac{\mathrm{GM}\bigl(p_1^{(1)}, \ldots, p_1^{(K)}\bigr)}
     {\mathrm{GM}\bigl(1 - p_0^{(1)}, \ldots, 1 - p_0^{(K)}\bigr)
      + \mathrm{GM}\bigl(p_1^{(1)}, \ldots, p_1^{(K)}\bigr)}\,,
\]

the log-loss-motivated aggregation of the fold-wise multiprobabilities.
`CrossVennAbersCalibrator` implements this with stratified, seeded folds. The cross variant
trades a little of the inductive variant's conceptual purity — the folds are not independent,
and the merged scalar stands one further step from the raw guarantee — for markedly better
sample efficiency, and Vovk and Petej's empirical results favor it on small data. On a
few-hundred-point PD calibration set it is the variant to reach for.

## Where Venn–Abers sits in the toolbox

Three practical readings summarize the method's role in probcal.

As a **calibrator**, IVAP/CVAP is isotonic regression made cautious: it inherits the
shape-freedom of PAVA while the two-fit construction protects against the single most
damaging isotonic pathology, the infinitely confident 0 or 1 in the tail — appending the
counterfactual label pulls the extreme block off the boundary, so \( p_1 > 0 \) and
\( p_0 < 1 \) everywhere by construction.

As an **uncertainty instrument**, it is unique in the package: no other method reports a
per-score, assumption-free measure of its own reliability. A validator reviewing a low-default
portfolio learns more from "the Venn–Abers interval at the approval cutoff is
\( [0.014, 0.031] \)" than from any point estimate, and `plot_interval` (in the
[visualization](visualization.md) module) draws exactly this picture.

As a **selector candidate**, its scalarization competes on out-of-fold log loss like every
other method, with one caveat the [selector chapter](auto-selection.md) repeats: interval
width does not enter the scoring, so a Venn–Abers win says the scalar predicts well, not that
the intervals were needed. When the intervals are the point — regulatory conservatism,
low-default portfolios in the spirit of Pluto and Tasche (2005) — choose it directly rather
than through the selector.

## In probcal

```python
from probcal import CrossVennAbersCalibrator, VennAbersCalibrator

ivap = VennAbersCalibrator().fit(s_cal, y_cal)
intervals = ivap.predict_interval(s_new)      # (n, 2): the guaranteed object
p = ivap.predict_proba(s_new)                 # scalarized p1 / (1 - p0 + p1)
print(ivap.interpret())                       # mean/max interval width

cvap = CrossVennAbersCalibrator(cv=5, random_state=0).fit(s_cal, y_cal)
p = cvap.predict_proba(s_new)                 # geometric-mean merge across folds
```

## References

- Pluto, K., Tasche, D. (2005). "Estimating Probabilities of Default for Low Default Portfolios." In *The Basel II Risk Parameters*, Springer.
- Vovk, V., Petej, I. (2014). "Venn–Abers Predictors." UAI, 829–838.
