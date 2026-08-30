# Nonparametric methods

The parametric families of the [previous chapter](methods-parametric.md) assume the distortion
has a known shape: a line in logit space, give or take a tail exponent. When the reliability
curve shows genuine curvature, the honest move is to stop assuming a shape and estimate the
map \( g \) directly, constrained only by weak structural requirements such as monotonicity.
That is the territory of this chapter: isotonic regression and its centered refinement,
histogram binning and its two-stage descendant, the two Bayesian ensembles BBQ and ENIR, and
penalized spline calibration. The price of flexibility is variance, and every method here is
best understood as a particular answer to the question *how much resolution can this
calibration set actually support?*

Notation continues from the previous chapter: \( s \) is the raw score, \( z =
\operatorname{logit}(s) \), \( g(s) \) the calibrated probability, and the calibration set is
\( (s_i, y_i)_{i=1}^{n} \) with optional weights \( w_i \).

## Isotonic regression and PAVA

Isotonic calibration makes exactly one assumption: the true calibration map is non-decreasing.
Sort the calibration pairs by score and solve

\[
\min_{m_1 \le m_2 \le \cdots \le m_n} \; \sum_{i=1}^{n} w_i\,(y_i - m_i)^2 ,
\]

the least-squares projection of the outcome sequence onto the monotone cone. The solution has
a closed combinatorial form computed by the **pool-adjacent-violators algorithm** (PAVA): scan
the sequence, and whenever a value exceeds its successor (a violation of monotonicity), merge
the two into a block carrying their weighted mean, then keep merging leftward while the new
block violates against its predecessor. The classical treatment is Barlow, Bartholomew,
Bremner and Brunk (1972); Zadrozny and Elkan (2002) brought the method into classifier
calibration.

**A worked micro-example.** Take five sorted scores with outcomes
\( y = (0,\, 1,\, 0,\, 0,\, 1) \) and unit weights. Initialize each \( m_i = y_i \). The pair
\( (m_2, m_3) = (1, 0) \) violates; pool observations 2–3 into a block with mean \( \tfrac12 \).
The block sequence is now \( 0,\ \tfrac12,\ 0,\ 1 \), and the new block still violates against
its right neighbor \( m_4 = 0 \), so pool again: observations 2–4 form one block with mean
\( (1 + 0 + 0)/3 = 1/3 \). The sequence \( 0,\ \tfrac13,\ 1 \) is monotone and the scan
finishes. The fitted values are

\[
\hat{m} = \Bigl(0,\; \tfrac13,\; \tfrac13,\; \tfrac13,\; 1\Bigr),
\]

three blocks in place of five free values. The pooling *is* the statistics: each block is a
region where the data cannot support a finer distinction, and the fitted level is the
empirical event rate of the pooled region. probcal's `pava` (in `probcal._math`) returns both
the expanded fitted values and this block structure, because the blocks are reused by CIR,
ENIR, and Venn–Abers.

**Using the fit.** `IsotonicCalibrator` predicts by locating a new score among the fitted
blocks: a step function, constant within each block. Scores outside the calibration range
clamp to the first or last block level. An optional `interpolation="linear"` mode joins block
midpoints to remove the discontinuities. `interpret()` reports the number of blocks (the
effective complexity actually estimated from the data), and flat steps translate into tied
predictions downstream.

**Failure modes to respect.** With \( n \) small, isotonic regression overfits at the
extremes: if the lowest-scored observation happens to be a non-event, the first block predicts
exactly 0, an infinitely confident statement no finite sample justifies. The output range is
limited to the span of block means, which matters for [inverse maps](inverse-maps.md): a
policy target below the lowest block level is simply unattainable. And the step shape means a
counterfactual explanation landing just past a block edge is fragile. None of this is fatal;
all of it argues for the guardrails and the range checks that probcal builds in.

## Centered isotonic regression

The step function's ties and flats are an artifact of the least-squares projection, not of the
data. **Centered isotonic regression** (Oron and Flournoy, 2017) post-processes the PAVA
solution: each block is collapsed to a single point at its weight-centered score coordinate
(the weighted mean of the scores in the block) with the block's fitted level as its value, and
the calibration map is the linear interpolation through these points. The result is strictly
increasing wherever the data permit, at negligible extra cost. `CenteredIsotonicCalibrator`
inherits everything from the isotonic fit, including its interpretation, and adds
strictness, which is the property to insist on when downstream consumers need distinct
predictions to remain distinct: risk-based pricing tiers, strict masterscale ordering, or the
round-trip inverses of [Inverse maps](inverse-maps.md).

## Histogram binning

Histogram binning (Zadrozny and Elkan, 2001) is the bluntest instrument in the package and
sometimes exactly the right one. Partition \( (0,1) \) into \( B \) bins, either **equal-width**
or **equal-mass** (quantile) in the calibration scores; within each bin, predict the empirical
event rate of the calibration observations that fell there. probcal optionally applies Jeffreys
shrinkage, replacing the raw rate \( k/n_b \) with \( (k + \tfrac12)/(n_b + 1) \), the
posterior mean under the Jeffreys Beta(½, ½) prior, which keeps small bins away from the
indefensible 0 and 1.

The single knob \( B \) is a transparent bias–variance dial: few bins, stable but coarse; many
bins, sharp but noisy. Equal-mass binning is the recommended default: it equalizes the
variance of the per-bin estimates and avoids empty bins where the score distribution is
sparse, which for low-PD portfolios is most of \( (0,1) \). Unlike isotonic regression,
binning does not even assume monotonicity, so a non-monotone fitted map is possible and is
worth reading as a diagnostic of noise rather than signal.

## Scaling-binning

Kumar, Liang and Ma (2019) observed that the two preceding ideas fix each other's weaknesses.
A parametric map is sample-efficient but its calibration error cannot be *measured* reliably
(its outputs take continuous values, so no bin ever accumulates repeats); histogram binning
has measurable error but needs many samples per bin. **Scaling-binning** runs both: first fit
a parametric map \( \hat{g} \) (Platt scaling in probcal's implementation), then form
equal-mass bins of the *fitted values* \( \hat{g}(s_i) \) and output the mean of \( \hat{g} \)
within each bin. Because the binning stage averages function values rather than raw outcomes,
the sample complexity to reach calibration error \( \varepsilon \) drops to
\( O(1/\varepsilon^2 + B) \) in place of histogram binning's \( O(B/\varepsilon^2) \); the
number of bins stops multiplying the sample requirement. The two-stage structure carries a
two-stage interpretation: read the parametric stage exactly as in the
[parametric chapter](methods-parametric.md), then read the binning stage as a discretization
that makes the residual calibration error estimable.

## Bayesian binning into quantiles (BBQ)

Choosing \( B \) by eye is uncomfortable. BBQ (Naeini, Cooper and Hauskrecht, 2015) removes
the choice by Bayesian model averaging: consider equal-mass binning models over a range of
\( B \), score each by its Bayesian marginal likelihood, and predict with the
posterior-weighted average of all models' outputs. Under a Beta prior per bin the
Beta–Binomial marginal has a closed form, computed with log-gamma functions
(`probcal._math.lgamma_vec`). The posterior weights are themselves diagnostic: concentrated
weight on one \( B \) says the data speak clearly about their own resolution; diffuse weight
says they do not, and
the averaging is doing real work. `BBQCalibrator.interpret()` reports the top three models by
weight. The averaged map is smoother than any single binning and typically monotone in
practice, though nothing enforces it.

## Ensemble of near-isotonic regressions (ENIR)

Isotonic regression enforces monotonicity as a hard wall. **Near-isotonic regression**
(Tibshirani, Hoefling and Tibshirani, 2011) softens the wall to a penalty:

\[
\min_{m} \; \tfrac12 \sum_i (y_i - m_i)^2 \; + \; \lambda \sum_i \bigl(m_i - m_{i+1}\bigr)_+ ,
\]

charging \( \lambda \) per unit of monotonicity violation. As \( \lambda \) grows from 0 to
the point where all violations vanish, the solutions trace a path (computable by a modified
PAVA that merges blocks at known breakpoints) that interpolates between the raw data and the
fully isotonic fit. ENIR (Naeini and Cooper, 2016) fits the whole path and combines the
solutions along it, weighted by BIC. The ensemble inherits flexibility from the low-\( \lambda \)
end and stability from the isotonic end, and the BIC weights again say where along that
spectrum the data place their trust.

The practical caveat: the combined map may be **non-monotone**. `ENIRCalibrator` sets
`is_monotone_ = False`, and consumers that require order preservation (counterfactual
targeting through [inverse maps](inverse-maps.md) above all) should prefer a monotone
calibrator. probcal raises rather than guesses when a non-monotone map is asked for a preimage.

**Retention bounds the ensemble's memory.** A calibration set with \( m \) distinct scores can
have up to \( m \) path breakpoints, and keeping every breakpoint's full-length solution around
for the BIC average costs \( O(m^2) \) memory; that is the failure mode `max_solutions` exists
to rule out. `ENIRCalibrator(max_solutions=256)` (the default) keeps only the 256 lowest-BIC
solutions as they are produced, so `path_solutions_` has shape `(K, m)` with \( K \le \)
`max_solutions` (fewer still when a breakpoint's BIC weight is provably negligible and is
pruned before ever being scored) rather than one row per breakpoint; `kept_breakpoints_`
indexes which breakpoints those rows came from, and `path_lambdas_` still records every
breakpoint regardless of retention. `dropped_weight_` reports the BIC weight lost to the cap
(evicted, already-scored solutions, not pruned ones), and `fit()` raises a `UserWarning` if
that loss exceeds 1e-6, the signal that `max_solutions` is cutting into the ensemble rather
than just bounding its memory. `max_solutions=None` recovers the unbounded,
one-row-per-breakpoint ensemble.

## Spline calibration

Between "a three-parameter formula" and "a step function per data block" sits a middle ground:
a smooth, flexible curve with a tunable budget of wiggliness. `SplineCalibrator` (after
Lucena, 2018) models the calibration map as a natural cubic spline in the logit of the score,

\[
\operatorname{logit} g(s) = \sum_{k} \theta_k \, N_k(z), \qquad z = \operatorname{logit}(s),
\]

where \( \{N_k\} \) is the natural cubic basis of Hastie, Tibshirani and Friedman (2009,
§5.2.1): cubic between knots, linear beyond the boundary knots, which is exactly the tail
behavior a calibration map should have where data run out. The coefficients are fitted by
penalized IRLS with a second-difference roughness penalty, and the penalty weight \( \lambda \)
is chosen by K-fold cross-validated log loss within the calibration set. The **effective
degrees of freedom**, the trace of the smoother matrix, is reported by `interpret()` and is
the honest complexity measure: a fitted spline using 2.3 effective degrees of freedom has
found nothing a parametric family could not, while 6 degrees of freedom says the curvature is
real. Regions where the fitted curve runs steeper than the identity are regions of local
underconfidence; shallower, local overconfidence. Smoothness makes the spline the most
pleasant map to invert and to explain, with one caveat: the penalty does not enforce
monotonicity, so probcal checks the fitted curve and flags the rare non-monotone outcome.

## Properties at a glance

The methods of this chapter differ along axes that matter operationally, not just
statistically. Continuity decides whether nearby scores can receive identical calibrated
values (ties feed through to pricing tiers and cutoff behavior). Strict monotonicity decides
whether the map can be inverted cleanly for [threshold translation](inverse-maps.md). The
output range decides whether extreme policy targets are attainable at all. And the complexity
knob names what must be justified to a validator.

| Method | Monotone | Continuous | Output range | Complexity knob |
|--------|----------|------------|--------------|-----------------|
| Isotonic (PAVA) | yes (weak) | no (steps) | span of block means | none (data-driven blocks) |
| Centered isotonic | yes (strict where data permit) | yes | span of block points | none |
| Histogram binning | not guaranteed | no (steps) | span of bin rates | \( B \), bin strategy |
| Scaling-binning | yes (inherits Platt) | no (steps) | span of binned \( \hat g \) values | \( B \) + parametric stage |
| BBQ | typical, not guaranteed | no (averaged steps) | span of averaged rates | prior over \( B \) |
| ENIR | **no** | no | data-driven | \( \lambda \) path, BIC weights |
| Spline | checked, not enforced | yes (smooth) | unbounded in logit | \( \lambda \) via CV (effective d.o.f.) |

Two rows deserve a second look. ENIR is the only method that *deliberately* admits
non-monotonicity, which is why `is_monotone_` exists on every calibrator rather than being
assumed. And the spline is the only nonparametric map whose logit is unbounded. Like the
parametric families, it can extrapolate beyond the observed event rates, which is either a
feature (sensible tail behavior) or a risk (unsupported extrapolation) depending on how far
out of range it is asked to predict.

## Choosing among them

A rough field guide, to be overridden by the [selector's](auto-selection.md) out-of-fold
evidence. Below a few hundred calibration points, parametric methods and coarse equal-mass
binning are the defensible options. Around a thousand, isotonic regression and CIR become
competitive, with CIR preferred whenever ties matter; scaling-binning is attractive when the
calibration error itself must be certified. BBQ and ENIR buy insensitivity to the resolution
choice at the cost of ensemble opacity; the spline buys smoothness at the cost of a
cross-validation loop. All of them, unlike the parametric families, can repair curvature,
and all of them make the [data-splitting discipline](data-splitting.md) more important, not
less, because flexible maps are precisely the ones that overfit a reused calibration set.

## In probcal

```python
from probcal import (
    BBQCalibrator,
    CenteredIsotonicCalibrator,
    ENIRCalibrator,
    HistogramBinningCalibrator,
    IsotonicCalibrator,
    ScalingBinningCalibrator,
    SplineCalibrator,
)

iso = IsotonicCalibrator().fit(s_cal, y_cal)
print(iso.n_blocks_)                          # effective complexity from the data

cir = CenteredIsotonicCalibrator().fit(s_cal, y_cal)   # strict where data permit
hist = HistogramBinningCalibrator(n_bins=10).fit(s_cal, y_cal)  # equal-mass, Jeffreys
sb = ScalingBinningCalibrator(n_bins=10).fit(s_cal, y_cal)      # Platt stage + binning

bbq = BBQCalibrator().fit(s_cal, y_cal)
print(bbq.interpret())                        # top-3 binnings by posterior weight

enir = ENIRCalibrator().fit(s_cal, y_cal)     # is_monotone_ = False, by design
spline = SplineCalibrator().fit(s_cal, y_cal)
print(spline.edof_, spline.lambda_)           # honest complexity + CV-chosen penalty
```

## References

- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., Brunk, H. D. (1972). *Statistical Inference under Order Restrictions.* Wiley.
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer.
- Kumar, A., Liang, P., Ma, T. (2019). "Verified Uncertainty Calibration." NeurIPS 32.
- Lucena, B. (2018). "Spline-Based Probability Calibration." arXiv:1809.07751.
- Naeini, M. P., Cooper, G. F. (2016). "Binary Classifier Calibration using an Ensemble of Near Isotonic Regression Models." IEEE ICDM, 360–369.
- Naeini, M. P., Cooper, G. F., Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 29, 2901–2907.
- Oron, A. P., Flournoy, N. (2017). "Centered Isotonic Regression: Point and Interval Estimation for Dose–Response Studies." *Statistics in Biopharmaceutical Research* 9(3), 258–267.
- Tibshirani, R. J., Hoefling, H., Tibshirani, R. (2011). "Nearly-Isotonic Regression." *Technometrics* 53(1), 54–61.
- Zadrozny, B., Elkan, C. (2001). "Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers." ICML, 609–616.
- Zadrozny, B., Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." KDD, 694–699.
