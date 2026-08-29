# Parametric methods

Parametric calibrators repair miscalibration with a map chosen from a small, named family —
two or three interpretable parameters fitted by maximum likelihood. On the small calibration
sets typical of credit risk they are usually the right first choice: few parameters mean low
variance, the fitted values have direct diagnostic readings, and an auditor can reproduce the
map from the reported parameters alone. This chapter derives the three families implemented in
`probcal.parametric` — Platt scaling, temperature scaling, and beta calibration — and develops
the interpretation of every parameter.

Throughout, \( s \in (0, 1) \) is the raw score being calibrated, \( z =
\operatorname{logit}(s) = \ln\frac{s}{1-s} \) its logit, \( \sigma(t) = 1/(1 + e^{-t}) \) the
logistic function, and \( g(s) \) the calibrated probability. Following probcal's input
convention, calibrators accept probabilities and convert to logits internally; users holding
raw logits apply the exported `expit` first.

## Platt scaling

Platt (1999) faced scores that were not probabilities at all — support vector machine margins —
and proposed passing them through a fitted sigmoid. In probcal's formulation the feature is the
logit of the score, so the map is

\[
g(s) = \sigma\bigl(a z + b\bigr), \qquad z = \operatorname{logit}(s),
\]

with slope \( a \) and intercept \( b \) fitted by maximum likelihood — an ordinary logistic
regression of the outcome on a single covariate. `PlattCalibrator` fits it with the shared IRLS
core (`probcal._math.irls_logistic`), whose step-halved Newton iteration decreases the objective
at every step (see [Separation, steep maps, and convergence](#separation-steep-maps-and-convergence)).

**Fitting targets.** Platt's original recipe does not regress on the hard labels
\( y \in \{0, 1\} \). To keep the fitted sigmoid away from degenerate solutions when one class
is scarce, the positive and negative targets are smoothed to

\[
t_{+} = \frac{N_{+} + 1}{N_{+} + 2}, \qquad t_{-} = \frac{1}{N_{-} + 2},
\]

where \( N_{+} \) and \( N_{-} \) count positive and negative calibration observations. The
smoothing has a Bayesian reading — it is the posterior mean of a uniform-prior Bernoulli
probability within each class — and its practical value on small samples, together with a
numerically robust fitting procedure, is analyzed by Lin, Lin and Weng (2007), whose treatment
probcal follows. With a 3% event rate and 500 calibration points, \( N_{+} \approx 15 \), and
the difference between regressing on \( \{0,1\} \) and on the smoothed targets is material.

**Interpretation.** The slope corrects the spread of the score distribution: \( a < 1 \)
shrinks over-dispersed, overconfident scores toward the base rate, while \( a > 1 \) sharpens
underconfident ones. The intercept moves every prediction by \( b \) log-odds units — it is
calibration-in-the-large repair, the same role the standalone [offset](offset.md) plays. The
identity map corresponds exactly to \( (a, b) = (1, 0) \), so departures of the fitted pair
from \( (1, 0) \) *are* the diagnosis: probcal's `interpret()` states it in these terms, and
the calibration slope and intercept metrics of `probcal.metrics.regression` are the same two
quantities estimated as diagnostics rather than repairs.

**A caveat on families.** Fitted on logits, as here, the logistic family contains the identity
\( (a, b) = (1, 0) \): if the input is already calibrated, maximum likelihood can leave it
alone. Fitted on raw scores \( s \) directly — Platt's original setting, where \( g(s) =
\sigma(As + B) \) — the family contains *no* identity on \( (0,1) \), so the calibrator
distorts even perfectly calibrated inputs. This distinction, emphasized by Kull, Silva Filho
and Flach (2017), is the design reason probcal fits on the logit scale and one of the two
motivations for beta calibration below.

## Temperature scaling

Temperature scaling (Guo et al., 2017) is Platt scaling with the intercept removed and the
slope reparameterized:

\[
g(s) = \sigma\!\left(\frac{z}{T}\right), \qquad T > 0 .
\]

The single parameter \( T \) is fitted by minimizing the negative log-likelihood on the
calibration set. The objective is smooth and, on the reparameterization \( u = 1/T \), convex
— `TemperatureCalibrator` solves it with a guarded one-dimensional Newton iteration
(`probcal._math.newton_1d`) and falls back to bisection when the Newton step leaves the
bracket.

**Interpretation.** \( T > 1 \) divides every logit by more than one, pulling probabilities
toward \( 1/2 \): the model was overconfident and is being softened. \( T < 1 \) sharpens an
underconfident model. \( T = 1 \) is the identity. Because there is no intercept, temperature
scaling *cannot move the base rate*: the score \( s = 1/2 \) maps to \( 1/2 \) for every
\( T \), and more generally the map is symmetric about the logit origin. A model whose only
defect is a portfolio-level bias — the common credit-risk situation after a shift in central
tendency — is beyond temperature's reach; use Platt scaling or the [offset](offset.md) for
that. Guo et al. (2017) found this one-parameter family remarkably effective for modern neural
networks, whose miscalibration is often close to a pure confidence rescaling; on structured
tabular scores with asymmetric distortion, expect the richer families below to win.

**Degrees of freedom.** Platt has two parameters, temperature one. Under the nested-validation
[selector](auto-selection.md) ties on the scoring criterion break toward fewer parameters, so
temperature is preferred exactly when the data cannot distinguish the two — the parsimony
principle applied to calibration.

## Beta calibration

Beta calibration (Kull, Silva Filho and Flach, 2017) starts from a generative question: if the
score distributions within each class are Beta distributions — the natural family for
quantities living on \( (0,1) \) — what does the true calibration map look like? Let

\[
s \mid Y = 1 \sim \mathrm{Beta}(\alpha_1, \beta_1),
\qquad
s \mid Y = 0 \sim \mathrm{Beta}(\alpha_0, \beta_0),
\]

with class prior \( \pi = \Pr(Y = 1) \). Bayes' rule gives
\( \operatorname{logit} \Pr(Y = 1 \mid s) = \ln \frac{f_1(s)}{f_0(s)} +
\operatorname{logit}\pi \), and taking the log-ratio of the two Beta densities,

\[
\ln \frac{f_1(s)}{f_0(s)}
= (\alpha_1 - \alpha_0) \ln s \;-\; (\beta_0 - \beta_1) \ln(1 - s) \;+\; \text{const}.
\]

Writing \( a = \alpha_1 - \alpha_0 \), \( b = \beta_0 - \beta_1 \), and absorbing the constants
and the prior into \( c \), the calibration map is

\[
\operatorname{logit} g(s) = a \ln s \;-\; b \ln(1 - s) \;+\; c ,
\]

a logistic regression on the two features \( \ln s \) and \( -\ln(1-s) \). This is the "abm"
variant of `BetaCalibrator` — the full three-parameter family. The constrained variants tie or
drop parameters: `"ab"` imposes \( a = b \), which collapses the two features into
\( a \,\operatorname{logit}(s) + c \) — exactly Platt scaling on logits — and `"a"` retains a
single free exponent (the exact tying is fixed when the calibrator is
implemented). Temperature scaling is the further
special case \( a = b = 1/T \), \( c = 0 \). The parametric families of this chapter thus form
a nested hierarchy, which is what makes parsimony tie-breaking in the selector coherent.

**The identity property.** Setting \( (a, b, c) = (1, 1, 0) \) gives
\( \operatorname{logit} g(s) = \ln s - \ln(1-s) = \operatorname{logit}(s) \): the identity map
is an interior point of the beta family. Kull, Silva Filho and Flach (2017) stress the
consequence: beta calibration *cannot un-calibrate an already calibrated model* beyond
sampling noise, because maximum likelihood at the identity has no gradient pushing away from
it. The same authors' journal-length treatment (Kull et al., *Electronic Journal of
Statistics*, 2017) develops the full theory, and the multiclass generalization appears as
Dirichlet calibration (Kull et al., 2019) — out of probcal's binary scope but the natural
pointer for multiclass readers.

**Monotonicity constraint.** The map is non-decreasing in \( s \) iff \( a \ge 0 \) and
\( b \ge 0 \). Unconstrained maximum likelihood can violate this on noisy or tiny calibration
sets, producing a map that *reverses* ranking somewhere — unacceptable when downstream
decisions assume order preservation. probcal enforces the constraint by the refit strategy of
the reference betacal implementation: if the unconstrained fit yields \( a < 0 \), the
\( \ln s \) feature is dropped (fixing \( a = 0 \)) and the model refitted; symmetrically for
\( b < 0 \). The fitted object records which constraint was active, and `interpret()` reports it.

**Interpretation.** The exponents control tail sensitivity independently — the asymmetry that
Platt and temperature cannot express. Near \( s \to 0 \) the map behaves like
\( g(s) \propto s^{a} \), so \( a \) governs the low-probability tail: for a PD model,
\( a < 1 \) flattens and raises the smallest PDs (the model was too eager to assign near-zero
risk), \( a > 1 \) deepens them. Near \( s \to 1 \) the mirrored role belongs to \( b \). The
intercept \( c \) shifts all predictions in log-odds, as in Platt. A fit with
\( a \approx b \) says the distortion was symmetric and the extra parameter was not needed;
\( a \neq b \) quantifies exactly the kind of one-sided tail distortion that low-event-rate
portfolios exhibit.

## Parameter interpretation at a glance

| Calibrator | Parameters | Identity at | Reading of departures from identity |
|------------|-----------|-------------|--------------------------------------|
| `PlattCalibrator` | \( a, b \) | \( (1, 0) \) | \( a \lessgtr 1 \): over/underconfident spread; \( b \neq 0 \): base-rate shift in log-odds |
| `TemperatureCalibrator` | \( T \) | \( T = 1 \) | \( T > 1 \): overconfident (soften); \( T < 1 \): underconfident (sharpen); base rate untouchable |
| `BetaCalibrator` | \( a, b, c \) | \( (1, 1, 0) \) | \( a \): low-\(s\) tail sensitivity; \( b \): high-\(s\) tail sensitivity; \( c \): base-rate shift; \( a \neq b \): asymmetric distortion |

Every fitted calibrator returns this reading through `interpret()`, populated with the actual
fitted values — the table is the contract for what those interpretations must say.

## A worked reading

Numbers make the interpretation contract concrete. Suppose a PD model scores a calibration set
with a realized default rate of 3.1%, and the fitted calibrators come back as follows.

`BetaCalibrator(variant="abm")` reports \( (a, b, c) = (0.82,\; 1.31,\; -0.42) \). Reading each
parameter: \( a = 0.82 < 1 \) means that near the low-score tail the map behaves like
\( s^{0.82} \), which decays more slowly than \( s \) itself — the smallest raw PDs are pulled
*up*. The model was too confident about its safest accounts, a familiar pattern when the
development sample under-represents rare defaults among high-quality obligors. \( b = 1.31 > 1 \)
sharpens the approach to the high-risk end: raw scores near the top understated risk.
\( c = -0.42 \) multiplies every odds by \( e^{-0.42} \approx 0.66 \), correcting a
portfolio-level overestimation of about a third in odds terms. The asymmetry \( a \neq b \) is
the substantive finding — the distortion is different in the two tails, so no symmetric family
could have repaired it fully.

On the same data `PlattCalibrator` reports \( (a, b) = (1.05, -0.38) \). It captures nearly the
same base-rate correction (the intercept), sees an almost-correct spread on average
(\( a \approx 1 \)), and is structurally blind to the tail asymmetry: the single slope averages
the too-flat low tail against the too-steep high tail. If out-of-fold log loss barely
distinguishes the two fits, the asymmetry was not estimable from this sample and parsimony
favors Platt; if beta wins clearly, the tails carried real signal. This is precisely the
comparison the [selector](auto-selection.md) automates, and the reason its report shows the
per-candidate scores rather than a bare winner.

`TemperatureCalibrator` on the same data is the wrong tool: whatever \( T \) it fits, it cannot
express the \( -0.42 \) base-rate correction, and its fitted value will contort to compromise
between the spread and the shift. A fitted \( T \) far from 1 accompanied by a failing
calibration-in-the-large guardrail (see [Metrics and tests](metrics.md)) is the telltale of
this misuse.

## Choosing within the parametric family

The nesting temperature ⊂ Platt ⊂ beta orders the families by flexibility, and the choice is a
bias–variance decision made honestly by out-of-fold log loss in the
[selector](auto-selection.md). Two rules of thumb survive contact with practice. First, if the
diagnosed defect is purely a base-rate shift, none of these families is the parsimonious
answer — a one-parameter [logit offset](offset.md) fixes calibration-in-the-large without
touching spread, and it composes transparently with any calibrator. Second, when the reliability
diagram on the *logit scale* (see [Visualization](visualization.md)) shows curvature — not just
a wrong slope — the distortion is outside every family in this chapter, and the nonparametric
methods of the [next chapter](methods-nonparametric.md) apply.

## Separation, steep maps, and convergence

Logistic regression fitted by maximum likelihood has one genuine failure mode: **separation**.
When binary outcomes are perfectly split by the covariate, the likelihood keeps improving as the
slope grows without bound — no finite MLE exists. probcal's IRLS core detects this only where it
can actually occur: when the targets are effectively binary, it declares separation once every
observation sits on the correct side by more than 10 log-odds while the gradient shows the fit
still improving, when the Hessian degenerates outright, or when the iteration exhausts its
budget without converging (the divergence signature of quasi-separation, where tied boundary
points keep the margin small) — then warns and refits with a tiny ridge penalty
(\( 10^{-6} \)). The ridged objective always has a finite minimizer, so the fallback converges
by construction and its coefficients are finite.

**Platt cannot separate.** The Lin–Lin–Weng smoothed targets are strictly inside \( (0, 1) \),
which makes the cross-entropy objective coercive — a finite maximum-likelihood solution always
exists, however extreme the scores. A "separation" diagnostic for Platt would be a category
error, and probcal never raises one there. The same applies to any call with soft targets;
genuine separation is possible only for binary-target fits (the beta variants and the
calibration belt).

This matters because a legitimately *steep* calibration map is easy to mistake for separation.
Scores are clipped to \( [10^{-12}, 1 - 10^{-12}] \), so \( |\operatorname{logit}(s)| \le 27.6 \);
any true slope above about 1.1 pushes fitted log-odds past 30 at the extremes. Versions before
0.1.2 aborted the iteration at that point and returned a biased interior iterate: on wide-score
data with a true slope of 1.5, Platt reported \( a \approx 1.18 \) alongside a misleading
separation warning. The current core instead halves each Newton step until the objective
decreases, so steep maps are simply fitted.

What to check after fitting: `converged_` on `PlattCalibrator` and `BetaCalibrator` records
whether IRLS converged (an unconverged fit warns at fit time and is noted by `interpret()`), and
`separation_fallback_` on `BetaCalibrator` records that the ridge fallback produced the
coefficients — `interpret()` states both, so the audit trail is complete.

## In probcal

```python
from probcal import BetaCalibrator, PlattCalibrator, TemperatureCalibrator

platt = PlattCalibrator().fit(s_cal, y_cal)
print(platt.a_, platt.b_)                  # identity is (1, 0)
print(platt.converged_)                    # IRLS convergence status

temp = TemperatureCalibrator().fit(s_cal, y_cal)
print(temp.T_)                             # identity is 1; cannot move the base rate

beta = BetaCalibrator(variant="abm").fit(s_cal, y_cal)
print(beta.interpret())                    # a, b, c against the identity (1, 1, 0)

p = beta.predict_proba(s_new)
print(platt.affine_logit_coeffs_)          # (a, b) — the affine-exact attribution hook
```

## References

- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." ICML, PMLR 70, 1321–1330.
- Kull, M., Silva Filho, T., Flach, P. (2017). "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." AISTATS, PMLR 54, 623–631.
- Kull, M., Silva Filho, T., Flach, P. (2017). "Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration." *Electronic Journal of Statistics* 11(2), 5052–5080.
- Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., Flach, P. (2019). "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration." NeurIPS 32.
- Lin, H.-T., Lin, C.-J., Weng, R. C. (2007). "A Note on Platt's Probabilistic Outputs for Support Vector Machines." *Machine Learning* 68(3), 267–276.
- Platt, J. C. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." In *Advances in Large Margin Classifiers*, MIT Press, 61–74.
