# SHAP and calibration

Additive attributions and post-hoc calibration are both standard equipment in deployed credit
models, and they break each other quietly. SHAP values (Lundberg and Lee, 2017) satisfy
**local accuracy**: the base value plus the per-feature attributions reconstructs the model's
output for the explained row. Calibrate that output afterwards, and the reconstruction now
lands on the *raw* score — a number the decision no longer uses. Reason codes derived from
those attributions explain the wrong quantity, which in adverse-action contexts is not a
cosmetic gap. `probcal.attribution` restores additivity on the calibrated scale, and this
chapter states exactly what can and cannot be restored.

## The identifiability obstacle

Write the model as \( f \) with attributions \( \phi_1, \ldots, \phi_d \) and base value
\( \phi_0 \), so that \( s = \phi_0 + \sum_i \phi_i \) for the explained row, and let \( g \)
be the fitted calibration map. One might hope to compute the Shapley values of the composed
model \( g \circ f \) from \( (\phi_0, \phi) \) alone. In general this is **impossible**. The
Shapley value of feature \( i \) under \( g \circ f \) averages differences
\( g(\mathbb{E}[f \mid S \cup \{i\}]) - g(\mathbb{E}[f \mid S]) \) over coalitions \( S \), and
those coalition expectations \( \mathbb{E}[f \mid S] \) — the entire intermediate structure
that TreeSHAP (Lundberg et al., 2020) traverses internally — are not recoverable from their
Shapley-weighted sums. Two models with identical \( (\phi_0, \phi) \) on a row can have
different coalition structure and therefore different exact attributions after composition
with a nonlinear \( g \). Any method operating on arrays of attributions must therefore
choose between exactness on a restricted class of calibrators and a principled approximation
on all of them. probcal offers both, explicitly labeled, and refuses to blur the line.

## Affine-exact mode

There is one clean escape. The Shapley value is a linear operator on games: if the composed
output is an *affine* function of the additive representation, the composed attributions are
the same affine rescaling. Suppose the calibrator is affine on the logit scale,
\( \operatorname{logit} g = a z + b \) where \( z \) is the score's logit, and attributions
live on that logit scale (the reason-code convention). Then

\[
\phi_i' = a\,\phi_i, \qquad \phi_0' = a\,\phi_0 + b ,
\]

and these are the **exact** Shapley values of \( g \circ f \) — no approximation, by
linearity. The affine class covers exactly: `TemperatureCalibrator`
(\( a = 1/T,\ b = 0 \)), `PlattCalibrator` fitted on logits (\( a, b \)), `LogitOffset`
(\( a = 1,\ b = \delta \)), and any composition of them — composing affine maps multiplies
the slopes and accumulates the intercepts, which is how `CalibratedModel` combines a
calibrator stage with an [offset stage](offset.md). Every calibrator exposes
`affine_logit_coeffs_`, returning \( (a, b) \) or `None`, and the wrapper composes the
coefficients across the pipeline when all stages are affine.

The boundary of the class must be stated as plainly as its interior. Beta calibration is
*not* affine in \( z \): substituting \( s = \sigma(z) \) into its map gives

\[
\operatorname{logit} g = -a \,\mathrm{softplus}(-z) \; + \; b \,\mathrm{softplus}(z) \; + \; c ,
\]

which is affine only in the degenerate case \( a = b \). Isotonic maps, binning, splines and
Venn–Abers are further still from affine. For all of these, the exact composed Shapley values
are unidentifiable from arrays, and the next mode is the honest tool.

## Aumann–Shapley mode

The general mode rescales each row's attributions by the *secant slope* of the calibrator
across that row's move from the base value:

\[
\phi_i' = \phi_i \cdot \frac{g(s) - g(s_0)}{s - s_0},
\qquad \phi_0' = g(s_0),
\]

where \( s_0 \) is the base value and \( s = s_0 + \sum_i \phi_i \) the explained output.
Additivity is restored *identically*: summing the adjusted attributions telescopes to
\( g(s) \) by construction, for any calibrator whatsoever — including piecewise-constant maps,
where the derivative is zero almost everywhere but the secant is not. The construction is not
ad hoc: applied to the univariate outer map \( g \) along the straight-line path from
\( s_0 \) to \( s \), it is exactly the Aumann–Shapley value (Aumann and Shapley, 1974), the
same object that integrated gradients (Sundararajan, Taly and Yan, 2017) computes for deep
networks. The honest caveat, stated wherever the mode is documented: it is *not* the Shapley
value of \( g \circ f \) in general — the calibrator's nonlinearity is distributed across
features **proportionally to** \( \phi_i \), which is a modeling choice, not a theorem.
Degenerate rows with \( s \approx s_0 \) replace the ill-conditioned secant with a central
difference estimate of \( g'(s_0) \); the switching threshold is a DECISIONS entry.

## Properties that survive adjustment

Three facts make the adjusted attributions safe for their primary consumer, reason-code
generation. First, on the affine class the two modes agree to numerical precision — the
equivalence is unit-tested at \( 10^{-12} \), so `method="auto"` (affine-exact when
available, Aumann–Shapley otherwise) never introduces a discontinuity of meaning. Second, for
*monotone* \( g \) the row multiplier is positive, so every attribution keeps its sign and
the within-row ranking of \( |\phi_i| \) is preserved — the ordered list of adverse-action
reasons is **invariant** under calibration adjustment, which is the property Regulation B
compliance workflows actually rely on. Third, reconstruction is verified, not assumed: the
returned `AdjustedAttribution` carries the maximum row-wise reconstruction error, tested
below \( 10^{-10} \).

On scale: `scale="logit"` is the default because reason codes conventionally rank log-odds
contributions, and it is the scale on which affine-exactness exists at all;
`scale="probability"` runs the same machinery on \( g \) expressed in probability units, with
the documentation noting that no calibrator is affine there. The module takes plain arrays,
duck-types `shap.Explanation` objects by reading `.values` and `.base_values`, and never
imports shap.

## The exact-but-heavy alternative, and related work

When exact composed attributions are genuinely required, the route is recomputation with
coalition access: explain the composed pipeline directly (for tree models, TreeSHAP with the
calibrated output as the model's output), paying the full explanation cost per calibrator
update. That path needs the model's internals and a SHAP implementation, so it sits outside a
numpy-only package; probcal documents it as the exact alternative rather than imitating it
approximately. Adjacent rather than overlapping: Calibrated Explanations (Löfström, Löfström,
Johansson and Sönströd, 2024) builds uncertainty-aware explanations on Venn–Abers foundations
— a different object (explanations with uncertainty semantics) than the array-level
additivity repair this module performs.

## In probcal

```python
from probcal import adjust_attributions

# phi: (n, d) SHAP values on the margin (logit) scale; base: scalar or (n,).
adj = adjust_attributions(phi, base_value, calibrator)   # method="auto"
print(adj.method_used)                  # "affine-exact" or "aumann-shapley"
print(adj.max_reconstruction_error)     # < 1e-10 by construction

# base + rows now reconstruct the calibrated log-odds:
recon = adj.base_adj + adj.phi_adj.sum(axis=1)   # equals adj.target

# shap.Explanation objects are duck-typed — no shap import needed:
adj = adjust_attributions(explanation, None, calibrator, scale="logit")
```

## References

- Aumann, R. J., Shapley, L. S. (1974). *Values of Non-Atomic Games.* Princeton University Press.
- Löfström, H., Löfström, T., Johansson, U., Sönströd, C. (2024). "Calibrated explanations: With uncertainty information and counterfactuals." *Expert Systems with Applications* 246, 123154.
- Lundberg, S. M., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 30.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence* 2(1), 56–67.
- Sundararajan, M., Taly, A., Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." ICML, PMLR 70.
