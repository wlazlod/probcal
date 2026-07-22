# Metrics and tests

Measuring calibration is harder than fixing it. The quantity of interest —
\( \Pr(Y = 1 \mid \hat{p}) \) — is a conditional expectation that no finite sample reveals
directly, so every metric estimates it through some smoothing device, and every smoothing
device imports bias, sensitivity, or both. This chapter walks the full catalog implemented in
`probcal.metrics`, states each estimator's formula and known pathologies, and ends with the
table that answers the operational question: *which of these may I select a calibrator on, and
which are for reporting only?* The short answer, argued in detail below: select on log loss
(default) or Brier; report the ECE family and ICI; never select on ECE or Hosmer–Lemeshow.

All metrics share the signature `metric(y, p, *, sample_weight=None, **kw)`, and
`evaluate(y, p)` assembles everything into a `MetricReport` with seeded bootstrap percentile
confidence intervals.

## Proper scoring rules

The **log loss** and **Brier score** (Brier, 1950) were defined in
[Why calibration](why-calibration.md):

\[
\mathrm{LL} = -\frac{1}{n}\sum_i \bigl[y_i \ln p_i + (1-y_i)\ln(1-p_i)\bigr],
\qquad
\mathrm{BS} = \frac{1}{n}\sum_i (p_i - y_i)^2 .
\]

Both are strictly proper: their expectation is uniquely minimized by the true conditional
probability, so no calibration map can improve them by lying. Their sample versions are
unbiased estimators of the expected loss, which is the property none of the direct calibration
metrics below share, and the reason they anchor selection. Log loss penalizes tail
overconfidence harshly (a confident wrong prediction costs unboundedly); the Brier score is
bounded and correspondingly gentler. For a portfolio-level orientation the **Brier skill
score**

\[
\mathrm{BSS} = 1 - \frac{\mathrm{BS}}{\mathrm{BS}_{\text{ref}}},
\qquad \mathrm{BS}_{\text{ref}} = \bar{y}(1 - \bar{y}),
\]

references the climatology forecast \( p \equiv \bar{y} \): positive values beat the base
rate, and on a 3% portfolio the reference is small, so seemingly modest Brier differences are
large skill differences.

**Murphy decomposition.** The binned estimator of Murphy's (1973) partition,

\[
\mathrm{BS} \approx
\underbrace{\frac{1}{n}\sum_b n_b\,(\bar{p}_b - \bar{y}_b)^2}_{\text{reliability}}
\;-\;
\underbrace{\frac{1}{n}\sum_b n_b\,(\bar{y}_b - \bar{y})^2}_{\text{resolution}}
\;+\;
\underbrace{\bar{y}(1-\bar{y})}_{\text{uncertainty}},
\]

with \( \bar p_b, \bar y_b \) the mean prediction and event rate in bin \( b \), makes the
calibration–sharpness trade visible in two numbers. It inherits every bias of its binning:
the plug-in reliability term is biased upward and resolution downward, exactly the effect
Bröcker (2009) formalized and Ferro and Fricker (2012) corrected. probcal implements the
corrected variant alongside the naive one and documents the binning dependence rather than
hiding it. The analogous **calibration–refinement split of the log loss** replaces squared
gaps with Kullback–Leibler terms: with \( c(p) = \Pr(Y=1 \mid \hat p = p) \) estimated by a
recalibration curve, calibration is the mean divergence between \( \mathrm{Bernoulli}(c(p)) \)
and \( \mathrm{Bernoulli}(p) \), refinement the mean entropy of \( \mathrm{Bernoulli}(c(p)) \).
The split is only as good as the plug-in estimate of \( c \); the estimator choice is recorded
as a DECISIONS entry when implemented.

## Binned estimators

The **expected calibration error** family discretizes the conditional expectation with
\( B \) bins:

\[
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}\,\bigl|\bar{p}_b - \bar{y}_b\bigr|,
\]

with variants: `strategy="mass"` (equal-count bins, the recommended default) or `"width"`;
`norm="l2"` squares the gaps; `norm="max"` takes the worst bin, which is the **maximum
calibration error** (MCE). Two pathologies are structural, not incidental. First, **binning
sensitivity**: ECE is a function of \( B \) and the bin edges, and rankings of models can flip
under a different, equally defensible binning. Second, **finite-sample bias**: within each bin
the absolute difference of two noisy means is biased upward — a *perfectly calibrated*
model has positive expected ECE, and the bias grows with \( B \) and shrinks portfolio-size
slowly. `ece_debiased` applies the bias correction in the spirit of Bröcker (2009) and Ferro
and Fricker (2012); `ece_sweep` implements the monotonic-sweep calibration error of Roelofs
et al. (2022), which chooses the largest equal-mass \( B \) whose bin means remain monotone —
a principled, data-driven resolution choice that markedly reduces bias. `adaptive_ece` is an
explicit alias for equal-mass ECE, provided because the literature uses the name; the
documentation states the equivalence.

The **Hosmer–Lemeshow test** (Hosmer and Lemeshow, 1980) groups observations into \( g \)
risk deciles and forms

\[
C = \sum_{b=1}^{g} \frac{(O_b - E_b)^2}{E_b\,\bigl(1 - E_b/n_b\bigr)} \;\sim\; \chi^2_{g-2},
\]

with \( O_b \) observed and \( E_b \) expected events per group. It is the historical
workhorse of clinical model validation and it carries the same two diseases in sharper form:
the statistic depends on an essentially arbitrary grouping (changing \( g \), or the tie
handling at decile boundaries, changes the p-value), and its power scales with \( n \) so that
on large portfolios it rejects calibration defects of no practical consequence, while on small
ones it detects almost nothing. probcal ships it because validators expect it, marks it
report-only, and never lets the [selector](auto-selection.md) see it.

## Binning-free estimators

Four estimators avoid the binning choice altogether.

**Smooth ECE** (Błasiok and Nakkiran, 2024) replaces hard bins with kernel smoothing: the
residuals \( y_i - p_i \) are smoothed with a reflected Gaussian kernel (probcal applies it on
the logit scale) and the calibration error is read from the smoothed curve, with the bandwidth
chosen by the paper's self-consistency principle — the reported error is the fixed point where
the measurement scale matches the error magnitude. The result is a continuous, reparametrization-
robust quantity with none of ECE's edge artifacts; any implementation simplification is
recorded as a DECISIONS entry.

**ECCE**, the empirical cumulative calibration error (Arrieta-Ibarra, Gujral, Tannen, Tygert
and Xu, 2022), sorts observations by \( p \) and tracks the cumulative deviation
\( C_k = \sum_{i \le k} (y_{(i)} - p_{(i)}) \). Under calibration this walk is a martingale
hovering near zero; systematic over- or under-prediction makes it drift. The Kolmogorov-style
maximum \( \max_k |C_k|/n \) and the mean absolute deviation summarize the drift, and the plot
of \( C_k \) against sorted \( p \) localizes *where* the miscalibration lives without any
smoothing parameter at all.

**ICI and its quantiles** (Austin and Steyerberg, 2019). Fit a LOESS smoother
\( \hat{c}(p) \) of outcome on prediction (Austin and Steyerberg, 2014, established the
graphical practice) and average the absolute distance to the diagonal:

\[
\mathrm{ICI} = \frac{1}{n} \sum_i \bigl|\hat{c}(p_i) - p_i\bigr| ,
\]

with E50, E90 and Emax the median, 90th percentile, and maximum of the same distances. The
family is smooth, interpretable in probability units, and inherits only the mild
LOESS-bandwidth dependence (`frac=0.75` by default, stated openly).

**Spiegelhalter's z** (Spiegelhalter, 1986) is the classical unbiasedness test built directly
on the Brier score. Its numerator \( \sum_i (y_i - p_i)(1 - 2p_i) \) has expectation zero
under calibration, and standardizing by its variance under the null,

\[
z = \frac{\sum_i (y_i - p_i)(1 - 2p_i)}
         {\sqrt{\sum_i (1 - 2p_i)^2\, p_i (1 - p_i)}} ,
\]

gives an asymptotically standard normal statistic with a two-sided p-value. No binning, no
smoothing; the trade is that it aggregates over the whole range and can miss compensating
regional errors.

## The recalibration-regression framework

The most decision-relevant diagnostics come from Cox's (1958) idea of regressing the outcome
on the prediction. Three quantities, all fitted by the shared IRLS core:

The **calibration intercept** fits \( \operatorname{logit} \Pr(Y=1) = \alpha +
\operatorname{logit}(p) \) with the slope fixed at 1 (an offset-term logistic regression):
\( \alpha \) is calibration-in-the-large in log-odds — on a PD portfolio,
\( \alpha = -0.3 \) says the model overestimates portfolio risk by a factor
\( e^{0.3} \approx 1.35 \) in odds.

The **calibration slope** fits \( \operatorname{logit} \Pr(Y=1) = \alpha + \beta\,
\operatorname{logit}(p) \) and reads \( \beta \): values below 1 mean predictions are too
spread out — the signature of overfitting — and values above 1 mean underfitting. These are
the same parameters a [Platt calibrator](methods-parametric.md) would *fit as repairs*, here
estimated as *diagnoses*.

The **calibration test** is the likelihood-ratio test of \( (\alpha, \beta) = (0, 1) \)
jointly, on 2 degrees of freedom — the Cox-framed "weak calibration" test, in the lineage
running through Miller, Hui and Tierney (1991). Its χ² p-value comes from
`probcal._math.gammainc_lower`, keeping the runtime numpy-only.

**Guardrails.** `calibration_guardrails(y, p)` condenses the framework into three flags used
across the package and printed in every selection report: slope within \( [0.9, 1.1] \),
intercept within \( \pm 0.1 \), Spiegelhalter p-value above 0.05. The thresholds are
conventions, not theorems; they encode "no deviation a validator would flag" and are
documented as such.

## Per-grade backtesting

Credit-risk validation operates on rating grades, not on continuous scores. Given grade
assignments and per-grade PDs, `binomial_grade_test` computes for each grade the exact
binomial tail probability of observing at least the realized number of defaults under the
grade's PD — the incomplete-beta representation via `probcal._math.betainc` keeps it exact at
any \( n \) — alongside the normal approximation, in the traffic-light style summary
supervisors expect (BCBS, 2005). `jeffreys_grade_test` implements the ECB's preferred
formulation (ECB, 2019): the posterior for the grade's true default rate under the Jeffreys
prior is \( \mathrm{Beta}(k + \tfrac12,\; n - k + \tfrac12) \), and the reported p-value is
the posterior probability that the true rate lies at or below the assigned PD. The reading is
one-sided and conservative by design — a small value flags a grade whose PD is likely
understated — and the documentation says so explicitly, because two-sided misreadings of the
Jeffreys test are a recurring validation error.

## Uncertainty: the bootstrap protocol

A metric without an uncertainty statement invites overreading, and calibration metrics on
percent-level event rates are noisy in ways intuition underestimates. `evaluate(y, p)`
therefore attaches confidence intervals to every scalar it reports, by the case-resampling
bootstrap: draw \( n \) observations with replacement from the evaluation pairs, recompute
the metric, repeat `n_boot=1000` times with a seeded generator, and report the 2.5th and
97.5th percentiles of the bootstrap distribution. The percentile method is chosen for
robustness of implementation and transparency — it makes no normality assumption, matters for
bounded and skewed statistics like ECE near zero, and is reproducible bit for bit given the
seed.

Two honest caveats accompany the protocol. First, on low-event-rate data the resamples vary
substantially in event count, and a resample can lose the tail entirely; the resulting
intervals are wide because the uncertainty is real, and narrowing them by stratifying the
bootstrap on the outcome is a deliberate deviation that would understate total sampling
variation (probcal does not do it by default). Second, bootstrap intervals for *biased*
estimators center on the biased value — a bootstrap CI around plain ECE quantifies its
variance, not its bias, so the interval can exclude zero for a perfectly calibrated model.
The report pairs ECE with its debiased variant precisely so this artifact is visible rather
than misread. Bootstrap-heavy computations carry the `slow` pytest marker and a fixed default
seed, per the package's reproducibility conventions.

## Reading a report

A `MetricReport` is designed to be read in a fixed order. Start with log loss and Brier
against their pre-calibration values — did the repair help at all, and is the improvement
larger than the bootstrap intervals overlap? Then the guardrail triplet — slope, intercept,
Spiegelhalter — which localizes any remaining defect to spread, level, or neither. Then the
descriptive family — debiased ECE, smooth ECE, ICI, ECCE — read as a cross-check: these
should broadly agree, and when they do not, the disagreement itself is diagnostic (a large
MCE with small ICI means one bad region, not global miscalibration; a large ECCE maximum with
small mean means a localized drift). Per-grade tests come last, because they answer a
different question — not "is the map good" but "which grades would a supervisor flag". The
[visualization chapter](visualization.md) pairs each layer of this reading with a plot.

## What to select on — the table

| Metric | Proper | Binning-sensitive | Finite-sample bias | Formal test | Selection use |
|--------|--------|-------------------|--------------------|-------------|---------------|
| Log loss | strictly | no | unbiased | no | **default criterion** |
| Brier score | strictly | no | unbiased | no | **alternative criterion** |
| Brier skill score | derived | no | mild (ratio) | no | report |
| Murphy / LL decompositions | — | yes | corrected variant available | no | report |
| ECE (mass/width, MCE) | no | **yes** | **upward** | no | **never** |
| Debiased ECE | no | yes | reduced | no | report |
| ECE sweep (Roelofs) | no | reduced | reduced | no | optional, with care |
| Smooth ECE | no | no | low | no | optional, with care |
| ECCE | no | no | low | max-statistic | report |
| ICI / E50 / E90 / Emax | no | no (LOESS frac) | low | no | optional |
| Spiegelhalter z | — | no | — | **yes** | never (it is a test) |
| Hosmer–Lemeshow | no | **yes** | — | **yes** | **never** |
| Calibration intercept/slope | — | no | — | yes (LR/Wald) | guardrails |
| Binomial / Jeffreys per grade | — | grades fixed | — | **yes** | never (backtest) |

The logic behind the verdicts compresses to one principle. Selection is optimization, and
optimizing a biased, binning-dependent, non-proper quantity invites the optimizer to exploit
the estimator rather than improve the calibration — a calibrator can win an ECE contest by
emitting values that straddle bin edges favorably, and win a Hosmer–Lemeshow contest by
blurring predictions until the test loses power. Strictly proper scores close that loophole by
construction. The [selector](auto-selection.md) therefore defaults to out-of-fold log loss,
accepts Brier, ICI, smooth ECE and ECE-sweep as deliberate alternatives, refuses plain ECE and
Hosmer–Lemeshow entirely, and prints the guardrail flags next to whatever criterion was used.

## References

- Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., Xu, C. (2022). "Metrics of Calibration for Probabilistic Predictions." *Journal of Machine Learning Research* 23(351), 1–54.
- Austin, P. C., Steyerberg, E. W. (2014). "Graphical assessment of internal and external calibration of logistic regression models by using loess smoothers." *Statistics in Medicine* 33(3), 517–535.
- Austin, P. C., Steyerberg, E. W. (2019). "The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models." *Statistics in Medicine* 38(21), 4051–4065.
- BCBS (2005). *Studies on the Validation of Internal Rating Systems.* Working Paper No. 14, revised version, May 2005. Bank for International Settlements.
- Błasiok, J., Nakkiran, P. (2024). "Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing." ICLR.
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review* 78(1), 1–3.
- Bröcker, J. (2009). "Reliability, sufficiency, and the decomposition of proper scores." *Quarterly Journal of the Royal Meteorological Society* 135(643), 1512–1519.
- Cox, D. R. (1958). "Two further applications of a model for binary regression." *Biometrika* 45, 562–565.
- ECB (2019). *Instructions for reporting the validation results of internal models — IRB Pillar I models for credit risk.* European Central Bank Banking Supervision, February 2019.
- Ferro, C. A. T., Fricker, T. E. (2012). "A bias-corrected decomposition of the Brier score." *Quarterly Journal of the Royal Meteorological Society* 138(668), 1954–1960.
- Hosmer, D. W., Lemeshow, S. (1980). "Goodness of fit tests for the multiple logistic regression model." *Communications in Statistics — Theory and Methods* 9(10), 1043–1069.
- Miller, M. E., Hui, S. L., Tierney, W. M. (1991). "Validation techniques for logistic regression models." *Statistics in Medicine* 10(8), 1213–1226.
- Murphy, A. H. (1973). "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology* 12(4), 595–600.
- Roelofs, R., Cain, N., Shlens, J., Mozer, M. C. (2022). "Mitigating Bias in Calibration Error Estimation." AISTATS, PMLR 151, 4036–4054.
- Spiegelhalter, D. J. (1986). "Probabilistic prediction in patient management and clinical trials." *Statistics in Medicine* 5(5), 421–433.
