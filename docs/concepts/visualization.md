# Visualization

Every calibration claim in this package has a picture, and the pictures are built in two
layers: `probcal.curves` computes plotting-ready dataclasses with numpy alone, and
`probcal.plots` renders them when matplotlib (the `[viz]` extra) is installed. The
separation means every curve is available to any backend — or to no backend, in a batch
report — and the rendering layer adds convention, not computation.

## Reliability constructions

The reliability diagram plots estimated event rate against predicted probability; a
calibrated model traces the diagonal. probcal builds it three ways, because the
construction *is* the estimator and inherits its trade-offs. `reliability_binned` groups
predictions (equal-mass by default), plotting each bin's mean prediction against its event
rate with a **Wilson confidence interval** — the binomial interval that behaves sensibly at
the small counts and extreme rates a PD portfolio produces — and the bin counts as a
histogram margin, so sparse regions announce themselves. `reliability_loess` and
`reliability_spline` draw the smooth versions in the tradition of Austin and Steyerberg
(2014), trading the binning artifacts for a bandwidth choice; plotted together with the
binned points they distinguish real curvature from bin noise. Every curve object carries
both probability-scale and logit-scale coordinate arrays, so the choice of scale is made at
plot time, not at computation time.

## Why the logit scale is the flagship

On a low-PD portfolio, the probability-scale reliability diagram is a picture of almost
nothing: the entire book lives between 0 and 0.1, the region a linear axis compresses into
its left margin. Plotting on the logit scale stretches exactly where the decisions are —
the difference between 0.5% and 1% PD is a factor of two in price and a full grade on a
masterscale, and it is invisible on \( [0,1] \) but a fixed distance in log-odds.
`plot_reliability(scale="logit")` is therefore the package's default recommendation for
credit work, with axis ticks *labeled in probabilities* at logit positions, so the reader
keeps probability intuition while the geometry keeps resolution. Parametric calibrators are
straight lines on this scale (slope and intercept readable by eye), the
[offset](offset.md) is a vertical translation, and tail miscalibration — the kind that
costs money at the approval cutoff — stops hiding in the corner.

## The calibration belt

A smoothed reliability curve without uncertainty invites overinterpretation. The **GiViTI
calibration belt** (Nattino, Finazzi and Bertolini, 2014) answers with a confidence region:
fit a polynomial logistic recalibration of the outcome on \( \operatorname{logit}(p) \),
selecting the polynomial degree by forward likelihood-ratio testing (capped at degree 4),
then invert the likelihood-ratio acceptance region pointwise into a band around the fitted
curve at the requested confidence levels (80% and 95% by default). Where the band excludes
the diagonal, the data affirmatively reject calibration in that region — a localized,
test-backed statement no eyeballed curve provides — and the associated p-value summarizes
the global test. The construction (Nattino et al., 2017, describe the practitioner-facing
version) is reimplemented in probcal from the papers, on the numpy-only χ² machinery of
`probcal._math`.

## The remaining plots

Three purpose-built views complete `probcal.plots`. `plot_comparison(before, after)` puts
pre- and post-calibration (or pre- and post-offset) reliability on one axis pair — the
picture a validation report leads with. `plot_interval` draws
[Venn–Abers interval widths](methods-distribution-free.md) against score, localizing where
calibration uncertainty concentrates. `plot_selection` renders the
[SelectionReport](auto-selection.md) as a ranked dot plot with fold-spread whiskers and
guardrail markers — the table, made presentable. All of them accept the dataclasses from
`probcal.curves`, and none of them is importable without the `[viz]` extra; the import
guard raises with the install instruction rather than a bare `ImportError`.

## In probcal

```python
from probcal import calibration_belt, reliability_binned, reliability_loess
from probcal.plots import plot_belt, plot_comparison, plot_reliability  # [viz] extra

curve = reliability_binned(y, p, n_bins=10)        # Wilson CIs, both scales
smooth = reliability_loess(y, p)
belt = calibration_belt(y, p)
print(belt.degree, belt.p_value)

ax = plot_reliability(curve, smooth=smooth, scale="logit")   # the flagship view
ax = plot_belt(belt)
fig = plot_comparison(reliability_binned(y, s_raw), reliability_binned(y, p))
```

## References

- Austin, P. C., Steyerberg, E. W. (2014). "Graphical assessment of internal and external calibration of logistic regression models by using loess smoothers." *Statistics in Medicine* 33(3), 517–535.
- Nattino, G., Finazzi, S., Bertolini, G. (2014). "A new calibration test and a reappraisal of the calibration belt for the assessment of prediction models based on dichotomous outcomes." *Statistics in Medicine* 33(14), 2390–2407.
- Nattino, G., Lemeshow, S., Phillips, G., Finazzi, S., Bertolini, G. (2017). "Assessing the Calibration of Dichotomous Outcome Models with the Calibration Belt." *Stata Journal* 17(4), 1003–1014.
