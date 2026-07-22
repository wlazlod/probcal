# Why calibration

A binary classifier that outputs probabilities makes a quantitative promise: among all cases
scored \( p = 0.03 \), about 3% should turn out positive. Calibration is the discipline of
checking that promise and repairing it when it fails. This chapter fixes the definitions used
throughout probcal, explains where miscalibration comes from, and develops the decision-theoretic
and regulatory reasons why it matters — with credit-risk probability-of-default (PD) models as
the running example.

## Definitions

Let \( Y \in \{0, 1\} \) be the outcome and \( \hat{p} \in (0, 1) \) the predicted probability
of the event \( Y = 1 \). The model is **perfectly calibrated** when

\[
\Pr\bigl(Y = 1 \mid \hat{p} = p\bigr) = p \quad \text{for all } p \text{ in the support of } \hat{p}.
\]

Conditioning on the prediction is the essential point. Calibration is a property of the
*conditional* event rate given the score, not of the marginal event rate: a model can match the
portfolio-level default rate exactly while being badly miscalibrated region by region, and vice
versa. Two weaker notions recur in practice. **Calibration-in-the-large** requires only that the
mean prediction matches the mean outcome, \( \mathbb{E}[\hat{p}] = \Pr(Y = 1) \) — a single
scalar condition, necessary but far from sufficient. The recalibration-regression framework of
Cox (1958) interpolates between the two: fitting a logistic regression of \( Y \) on
\( \operatorname{logit}(\hat{p}) \) and asking whether the intercept is 0 and the slope is 1
tests calibration against the family of monotone logistic distortions, which captures the most
common failure modes without demanding the full conditional property. probcal implements this
family of diagnostics in `probcal.metrics.regression` and the corresponding tests are discussed
in [Metrics and tests](metrics.md).

Calibration says nothing about whether the model separates classes well. **Discrimination** —
the ability to rank positives above negatives — is a different axis entirely. The constant
predictor \( \hat{p} \equiv \bar{y} \) is perfectly calibrated and perfectly useless for
ranking; a distorted but strictly monotone transform of a strong score ranks flawlessly while
being arbitrarily miscalibrated. **Sharpness** names the third axis: how concentrated the
predictions are, i.e. how far the forecast distribution departs from the uninformative base
rate. A useful probabilistic model should be *as sharp as possible subject to calibration*. The
three properties are related but not exchangeable, and post-hoc calibration operates on
exactly one of them: every method in this package applies a (typically monotone) map
\( g : \hat{p} \mapsto g(\hat{p}) \) that changes calibration and sharpness while leaving
discrimination essentially untouched — a strictly monotone \( g \) preserves the ranking
exactly.

## Where miscalibration comes from

Miscalibration is the norm, not the exception, and its sources are mundane.

**Model class bias.** Naive Bayes pushes scores toward 0 and 1 because its independence
assumption double-counts correlated evidence; decision trees produce piecewise-constant scores
whose leaf frequencies are estimated on few observations. Both distortions were documented in
detail by Zadrozny and Elkan (2001), who introduced histogram binning and popularized isotonic
regression as remedies precisely because the distortions are not logistic in shape.

**Margin-based training.** Classifiers trained on hinge loss or similar margins — support
vector machines being the canonical case — do not produce probabilities at all; their outputs
are distances to a separating surface. Platt (1999) proposed mapping such outputs through a
fitted sigmoid, which is the origin of the whole post-hoc calibration family described in
[Parametric methods](methods-parametric.md).

**Regularization and overfitting.** Regularization shrinks fitted log-odds toward zero, which
makes predictions systematically underconfident; overfitting does the reverse. Boosted
ensembles are a well-known case of the latter pattern in the tails combined with
characteristic distortions induced by the loss: Zadrozny and Elkan (2002) treat boosted naive
Bayes explicitly. Modern deep networks miscalibrate for related reasons — Guo et al. (2017)
showed that depth, width, and weight decay all shift calibration even as accuracy improves,
and that the resulting distortion is often well repaired by a single temperature parameter.

**Class imbalance and sampling design.** When the minority class is rare, maximum-likelihood
logistic regression underestimates rare-event probabilities in small samples (King and Zeng,
2001), and any deliberate under- or over-sampling of the training data shifts the intercept of
the score distribution away from the population base rate. Elkan (2001) gives the standard
correction for a known shift in base rate. Both corrections are logit-additive, which is why
probcal exposes them through a first-class [offset](offset.md) rather than burying them inside
a calibrator.

**Distribution shift.** A model calibrated at development time drifts as the population
changes. In credit risk this is routine: the through-the-cycle average default rate moves with
the macroeconomy, so the *central tendency* of the portfolio must be re-anchored periodically
even when the ranking power of the score is stable. Tasche (2013) develops PD-curve calibration
under exactly this regime. The offset mechanism, again, is the auditable answer.

## Consequences in decisioning

If probabilities feed a decision rule, miscalibration is not a cosmetic defect — it changes the
decisions.

**Expected-loss pricing.** Risk-based pricing multiplies PD by exposure and loss-given-default
to obtain expected loss. A PD understated by a factor of two halves the risk premium: the
lender systematically underprices risky loans and overprices safe ones, and the resulting
adverse selection compounds the error, since mispriced risky applicants accept at higher rates.

**Cutoff policies.** A policy such as "approve when PD ≤ 2%" is stated on the calibrated
scale. Under miscalibration the effective cutoff sits somewhere else entirely, and the achieved
approval rate and bad rate both drift from their designed values. (The reverse translation —
carrying a calibrated-scale policy back to the raw score that a deployed model emits — is its
own problem, treated in [Inverse maps](inverse-maps.md).)

**Capital.** In the internal-ratings-based (IRB) approach, PD estimates enter the regulatory
risk-weight functions directly, so PD bias propagates into required capital. Supervisory
validation therefore treats calibration backtesting as a first-class exercise: the Basel
Committee's Working Paper No. 14 (BCBS, 2005) surveys the statistical machinery, and the
European Central Bank's reporting instructions (ECB, 2019) prescribe a concrete battery of
per-grade tests — notably the Jeffreys test that probcal implements in
`probcal.metrics.grade` — that banks must run against each rating grade of each IRB model.
An audit trail for every transformation applied to a PD is not optional in this setting, which
is why every probcal calibrator exposes `interpret()` and why the offset keeps its pre- and
post-adjustment state.

**Small samples and low event rates.** The same regulatory portfolios that demand calibrated
PDs make calibration statistically hard: a calibration set with a 3% event rate and a few
hundred observations contains a handful of defaults. Method choice becomes a bias–variance
question — a three-parameter parametric map may beat a nonparametric one simply because the
data cannot support more resolution. Pluto and Tasche (2005) treat the extreme case of
low-default portfolios. This tension motivates both the small-sample tests in probcal's suite
and the nested-validation [selector](auto-selection.md).

## Proper scoring rules as the organizing lens

A single principle organizes the zoo of calibration metrics. A **scoring rule**
\( S(\hat{p}, y) \) assigns a loss to the prediction–outcome pair; it is **proper** when the
expected loss \( \mathbb{E}_{Y \sim q}[S(p, Y)] \) is minimized at \( p = q \), and strictly
proper when the minimizer is unique. Under a strictly proper rule, honesty is optimal: no
systematic distortion of the true conditional probability can improve the expected score. The
two workhorses are the logarithmic loss and the Brier score (Brier, 1950),

\[
S_{\log} = -\bigl[y \log \hat{p} + (1 - y)\log(1 - \hat{p})\bigr],
\qquad
S_{\text{Brier}} = (\hat{p} - y)^2 .
\]

Propriety is what makes these scores safe *selection criteria*: a calibration map chosen to
minimize out-of-fold log loss cannot win by making predictions dishonest. Metrics that measure
calibration error directly — the ECE family, Hosmer–Lemeshow — are valuable *reports* but are
not proper and behave badly as objectives; the full argument, including estimator bias and
binning sensitivity, is developed in [Metrics and tests](metrics.md).

Proper scores also decompose. Murphy (1973) partitioned the expected Brier score into

\[
\underbrace{\text{reliability}}_{\text{calibration error}}
\;-\;
\underbrace{\text{resolution}}_{\text{useful variation}}
\;+\;
\underbrace{\text{uncertainty}}_{\text{outcome entropy}},
\]

where reliability measures the average squared gap between predicted probability and the
conditional event rate given the prediction, resolution measures how much the conditional
event rates vary across predictions, and uncertainty is a property of the outcomes alone. An
analogous calibration–refinement split applies to the log loss. The decomposition explains
precisely what post-hoc calibration can and cannot do: a monotone recalibration map drives the
reliability term toward zero and can only redistribute, never manufacture, resolution.
Bröcker (2009) put the decomposition of general proper scores on rigorous footing, and Ferro
and Fricker (2012) supplied the bias corrections needed when the terms are estimated from
finite samples by binning — both matter for the estimators implemented in
`probcal.metrics.scores`.

The practical reading of this chapter, and the stance taken throughout probcal: calibrate with
a map fitted on data the model has not seen, select the map by a strictly proper score
estimated out-of-fold, report the descriptive calibration metrics alongside, and keep every
adjustment inspectable. The remaining chapters fill in each of those steps.

## References

- Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review* 78(1), 1–3.
- Bröcker, J. (2009). "Reliability, sufficiency, and the decomposition of proper scores." *Quarterly Journal of the Royal Meteorological Society* 135(643), 1512–1519.
- BCBS (2005). *Studies on the Validation of Internal Rating Systems.* Working Paper No. 14, revised version, May 2005. Bank for International Settlements.
- Cox, D. R. (1958). "Two further applications of a model for binary regression." *Biometrika* 45, 562–565.
- ECB (2019). *Instructions for reporting the validation results of internal models — IRB Pillar I models for credit risk.* European Central Bank Banking Supervision, February 2019.
- Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." IJCAI, 973–978.
- Ferro, C. A. T., Fricker, T. E. (2012). "A bias-corrected decomposition of the Brier score." *Quarterly Journal of the Royal Meteorological Society* 138(668), 1954–1960.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." ICML, PMLR 70, 1321–1330.
- King, G., Zeng, L. (2001). "Logistic Regression in Rare Events Data." *Political Analysis* 9(2), 137–163.
- Murphy, A. H. (1973). "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology* 12(4), 595–600.
- Platt, J. C. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." In *Advances in Large Margin Classifiers*, MIT Press, 61–74.
- Pluto, K., Tasche, D. (2005). "Estimating Probabilities of Default for Low Default Portfolios." In *The Basel II Risk Parameters*, Springer.
- Tasche, D. (2013). "The art of probability-of-default curve calibration." *Journal of Credit Risk* 9(4).
- Zadrozny, B., Elkan, C. (2001). "Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers." ICML, 609–616.
- Zadrozny, B., Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." KDD, 694–699.
