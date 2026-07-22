# Offset

Sometimes the entire calibration problem is one number. The model ranks well, its spread is
right, but the portfolio-level probability sits at the wrong height — the training sample was
rebalanced, the economy moved, or the long-run anchor changed. The repair is a uniform shift
in log-odds, and probcal promotes it to a first-class, separately auditable object:
`LogitOffset`. This chapter develops the theory of that shift, its equivalences in the
rare-events and cost-sensitive literatures, its credit-risk reading as central-tendency
re-anchoring, and the audit practice built around it.

## The transform

The offset applies

\[
p' = \sigma\bigl(\operatorname{logit}(p) + \delta\bigr),
\]

a rigid translation on the logit scale. Equivalently, in odds form:
\( \frac{p'}{1-p'} = e^{\delta}\, \frac{p}{1-p} \) — every observation's odds are multiplied
by the same factor \( e^{\delta} \). The map is strictly increasing, so ranking is untouched;
it has exactly one parameter, so nothing else can move; and it composes cleanly with any
calibrator, because adding \( \delta \) commutes with reading the map's output. Those three
properties are why the offset is kept *outside* the calibrators rather than folded into their
intercepts: a validator can see the recalibration and the re-anchoring as two separate,
separately justified steps.

`LogitOffset` operates in two modes. **Mode A** takes \( \delta \) explicitly — the case
where the shift is prescribed by policy or derived externally. **Mode B** takes a target mean
\( \pi^* \) and solves

\[
\frac{1}{n} \sum_{i=1}^{n} \sigma\bigl(\operatorname{logit}(p_i) + \delta\bigr) = \pi^*
\]

for \( \delta \). The left side is a strictly increasing, continuous function of \( \delta \)
(each summand is; a sum of strictly increasing functions is strictly increasing), running
from 0 to 1 as \( \delta \) spans the real line, so **the root exists and is unique** for any
\( \pi^* \in (0, 1) \). probcal solves it by bisection (`probcal._math.bisect`) — the
monotonicity that guarantees uniqueness also makes bisection unconditionally convergent — and
the uniqueness claim is unit-tested, not just asserted.

## Three derivations of the same number

The logit-additive correction appears independently in three literatures, and the
equivalence is worth spelling out because each gives a different justification for the same
arithmetic.

**Prior correction for choice-based sampling (King and Zeng, 2001).** Suppose a logistic
model is estimated on a sample where the event fraction is \( \bar{y} \) — perhaps events
were oversampled for estimation efficiency — while the population rate is \( \tau \). King
and Zeng show the sampling design biases only the intercept, and the correction subtracts
\( \ln\bigl[\bigl(\tfrac{1-\tau}{\tau}\bigr)\bigl(\tfrac{\bar{y}}{1-\bar{y}}\bigr)\bigr] \)
from it. In probcal's notation this is precisely an offset with

\[
\delta = -\ln\!\left[\frac{1-\tau}{\tau}\cdot\frac{\bar{y}}{1-\bar{y}}\right],
\]

applied uniformly to every prediction's logit. What the econometrician calls prior
correction, `LogitOffset` performs as a post-processing step — no refit required.

**Base-rate adjustment in cost-sensitive learning (Elkan, 2001).** Elkan's analysis of
learning under changed class priors yields an adjustment that converts probabilities
calibrated under one base rate into probabilities calibrated under another. On the odds
scale the conversion is exactly a uniform multiplication — the same object as the offset —
and probcal's documentation cites it descriptively as Elkan's base-rate adjustment. The
practical corollary matters for anyone who rebalanced training data: undersampling the
majority class *manufactures* a known miscalibration whose exact antidote is one offset.

**Central tendency in PD calibration (Tasche, 2013).** Credit-risk practice maintains a
**central tendency** — the long-run average default rate the portfolio's PDs should aggregate
to, set by through-the-cycle policy rather than by last year's realized rate. When the
portfolio mean PD drifts from the anchor, the re-anchoring adjustment shifts all PDs to match
it while preserving the rating order — mode B with \( \pi^* \) equal to the central tendency.
Tasche (2013) situates this operation within the broader family of PD-curve calibration
methods (quasi-moment matching among them) and supplies the framing probcal adopts: the
offset is the smallest, most transparent member of that family, appropriate exactly when the
diagnosis is a pure level error.

The three derivations answer different questions — *how was the sample drawn*, *what prior
will deployment face*, *what anchor does policy set* — and arrive at the same transform. The
offset chapter of a model's documentation should say which question motivated \( \delta \);
`interpret()` prompts for exactly that.

## A worked re-anchoring

A quarterly update makes the mechanics concrete. A portfolio's calibrated PDs average 4.2%,
while the policy central tendency stands at \( \pi^* = 3.1\% \). Mode B solves the
mean-matching equation by bisection. A first-order approximation locates the answer before
the solver runs: if all PDs sat exactly at the mean, the required shift would be the log-odds
difference

\[
\delta \;\approx\; \operatorname{logit}(0.031) - \operatorname{logit}(0.042)
\;=\; \ln\frac{0.031/0.969}{0.042/0.958} \;\approx\; -0.315 ,
\]

and because the portfolio's PDs are dispersed around their mean rather than concentrated at
it, the exact root lands near, but not exactly at, this value — the sigmoid's curvature makes
the mean of shifted probabilities differ slightly from the shifted mean. Bisection converges
to the exact \( \delta \) regardless; the approximation's role is the sanity check a reviewer
can do by hand.

The fitted object then reads: `delta_ ≈ −0.31`, odds factor \( e^{\delta} \approx 0.73 \) —
every obligor's odds cut by about 27% — `pre_mean_ = 0.042`, `post_mean_ = 0.031`, and
`audit_report()` shows the calibration intercept moving from about \( -0.31 \) (the
correction the data were asking for) to approximately zero while the slope and Spiegelhalter
columns stand still. That last line is the point of
the whole audit table: a level correction that also moved the slope would mean the diagnosis
was wrong and the defect was never a pure level shift. Note the asymmetric division of labor:
the same table printed after a *calibrator* refit answers "did the map repair the shape",
while printed after an offset it answers "did the shift do only what it claimed".

## What the offset cannot do

One parameter buys one repair. The offset moves calibration-in-the-large and nothing else: a
wrong calibration *slope* (spread), curvature, or tail asymmetry pass through untouched —
diagnosing those is the [regression framework's](metrics.md) job and repairing them is the
[calibrators'](methods-parametric.md). The converse discipline also holds: when the defect
*is* purely a level shift, fitting a full calibrator wastes events on parameters the data
does not need — see the [small-sample guidance](data-splitting.md). Temperature scaling is
the exact mirror image (spread without level), which is why the two are documented as
complements, never substitutes.

## Audit trail and composition

Because the offset is the adjustment most likely to be applied repeatedly in production —
quarterly central-tendency updates are common — `LogitOffset` keeps records. Fitting stores
`delta_`, the pre-adjustment mean `pre_mean_`, the post-adjustment mean `post_mean_`, and a
timestamp. `audit_report()` re-runs the [guardrails](metrics.md) before and after the shift,
so the reviewing validator sees in one table what moved, by how much, and what it did to
slope, intercept, and the Spiegelhalter statistic. `interpret()` renders \( \delta \) in all
three vocabularies: log-odds shift, multiplicative odds factor \( e^{\delta} \), and the
central-tendency re-anchoring statement with both means.

Composition is deliberately explicit. `CalibratedModel.offset_to(target_mean=...)` appends a
`LogitOffset` stage to the pipeline — calibrator first, offset second — and both stages
remain inspectable; the package never folds \( \delta \) into the calibrator's fitted
parameters. Beyond auditability, the separation has two technical dividends. The offset is
affine on the logit scale, so it composes into the exact
[attribution adjustment](shap-calibration.md) of any affine pipeline; and its inverse is a
subtraction, so [inverse maps](inverse-maps.md) pass through it exactly — a quarterly
\( \delta \) update of magnitude at most \( m \) moves every raw-score threshold by at most
\( m \) in logit units, which is the fact the `buffer_logit` robustness margin is built on.

## References

- Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." IJCAI, 973–978.
- King, G., Zeng, L. (2001). "Logistic Regression in Rare Events Data." *Political Analysis* 9(2), 137–163.
- Tasche, D. (2013). "The art of probability-of-default curve calibration." *Journal of Credit Risk* 9(4).
