# Inverse maps

Once decisions are made on calibrated probabilities, every consumer that lives on the raw
score needs a translation service. A credit policy says "approve below 2% PD"; the deployed
model emits raw margins; the cutoff engine, the masterscale, and the counterfactual generator
all operate upstream of the calibrator. The capability to translate calibrated intervals back
to raw-score intervals belongs to the calibrator itself — only it knows its functional form,
its block structure, and its output range — so probcal builds it into the base contract:
`interval_inverse(lo, hi, *, space, buffer_logit)`, with `thresholds.py` providing thin
functional wrappers.

## The preimage identity

For a **monotone** calibration map \( g \), inverting a decision interval commutes with
composition:

\[
\{\,x : g(f(x)) \in [\mathrm{lo}, \mathrm{hi}]\,\}
\;=\;
\{\,x : f(x) \in [\,g^{-1}(\mathrm{lo}),\; g^{-1}(\mathrm{hi})\,]\,\}.
\]

The consequence reaches further than notation: calibration does not change counterfactual
*geometry*, only the target interval. A counterfactual engine searching for the smallest
feature change that lands \( f(x) \) in a raw interval works unchanged after calibration is
deployed — it just needs the translated interval. This is the designed hand-off to treecf:
probcal computes `lo_z, hi_z = cal.interval_inverse(0.0, 0.02, space="logit")` and the
engine consumes `Target.raw(range=(lo_z, hi_z))`. The division of responsibilities is
deliberate and recorded as a DECISIONS entry — probcal owns the capability and publishes the
duck-typed protocol (`interval_inverse` plus `is_monotone_`); target-construction ergonomics
belong to the consumer.

One trap deserves its single sentence here and a longer one in the FAQ: after calibration is
deployed, a consumer-side `Target.probability(...)` silently inverts the *model's own*
sigmoid link rather than the calibrator, and therefore targets the uncalibrated probability.

## Generalized inverses and plateau semantics

Step calibrators have no literal inverse, so the contract is defined by generalized
inverses: for non-decreasing \( g \),

\[
\mathrm{raw\_lo} = \inf\{\, s : g(s) \ge \mathrm{lo} \,\},
\qquad
\mathrm{raw\_hi} = \sup\{\, s : g(s) \le \mathrm{hi} \,\},
\]

with \( \mathrm{lo} = 0 \) and \( \mathrm{hi} = 1 \) mapping to \( -\infty \) and
\( +\infty \). For an [isotonic map](methods-nonparametric.md) this means the preimage of a
target begins exactly at the left edge of the first qualifying block — a semantics the tests
pin down explicitly. Implementations follow the map's structure: closed form for Platt,
temperature and the offset; monotone bisection for beta and monotone splines;
`searchsorted` on the block structure for isotonic, CIR, binning, scaling-binning and the
scalarized Venn–Abers (whose monotonicity the
[distribution-free chapter](methods-distribution-free.md) establishes). The wrapper composes
right-to-left through the pipeline — the offset inverts first, subtracting \( \delta \) on
the logit, then the calibrator's inverse applies.

Two refusals are part of the contract. If \( [\mathrm{lo}, \mathrm{hi}] \) does not
intersect the calibrator's output range — routine for isotonic maps on low-PD data, whose
range is the span of block means — the method raises `UnattainableTargetError` naming both
intervals; silent clamping would convert a policy error into a wrong cutoff. And a
non-monotone calibrator (`is_monotone_ = False`, ENIR being the resident example) has
preimages that may be unions of intervals; the method raises `NotImplementedError` with an
explanation, and the documented recommendation is simply to use a monotone calibrator when
recourse or thresholding is downstream.

## Robustness: plateaus and drift

A counterfactual that lands just past a block edge of a step calibrator is fragile twice
over: the calibrated value jumps discretely at the edge, and any refit moves the edge. The
first fragility argues for continuous calibrators — beta or [CIR](methods-nonparametric.md)
— wherever recourse is in scope. The second is addressed by `buffer_logit`: the calibrated
interval is shrunk by a margin in logit space *before* inversion, so the produced raw
interval is conservative by that margin. The guarantee it buys is concrete because the
[offset](offset.md) is a pure logit translation: a central-tendency update of magnitude at
most \( m \) cannot invalidate a counterfactual built with `buffer_logit = m`. Larger buffers
give tighter raw intervals — a monotonicity the tests check — and the trade is explicit:
robustness against recalibration drift, paid in recourse difficulty. The concern is the one
the algorithmic-recourse literature formalizes: recourse recommendations invalidated by
model updates and distribution shift (Rawal, Kamar and Lakkaraju, 2020), and the case for
building recourse robust to such shifts (Upadhyay, Joshi and Lakkaraju, 2021).

## The masterscale workflow

Rating systems define grades on calibrated PD — a masterscale
\( \{\text{grade } j : [\mathrm{lo}_j, \mathrm{hi}_j)\} \) — while the model emits raw
margins, with calibration sitting between. `calibrated_bands_to_raw(calibrator, bands, ...)`
maps the entire masterscale to raw intervals in one call, which is the canonical workflow:
grade edges translated once per recalibration, consumed by cutoff engines and counterfactual
targeting alike (the output plugs directly into band-style targets on the raw scale). Since
grade boundaries are policy artifacts that outlive model versions, the translation — not the
masterscale — is what changes when the calibrator is refitted, and the audit story stays
clean: policy fixed, mapping versioned.

## In probcal

```python
from probcal import UnattainableTargetError, calibrated_bands_to_raw

lo_s, hi_s = cal.interval_inverse(0.0, 0.02)               # "PD <= 2%" in score space
lo_z, hi_z = cal.interval_inverse(0.0, 0.02, space="logit")  # ... in raw margins

# Robust to the next re-anchoring of magnitude <= 0.1 log-odds:
lo_z, hi_z = cal.interval_inverse(0.0, 0.02, space="logit", buffer_logit=0.1)

masterscale = {"A": (0.0, 0.01), "B": (0.01, 0.03), "C": (0.03, 1.0)}
raw_bands = calibrated_bands_to_raw(cal, masterscale, space="logit")

try:
    cal.interval_inverse(0.95, 1.0)     # beyond an isotonic map's output range
except UnattainableTargetError as err:
    print(err)                           # named intervals — never a silent clamp
```

## Point inverses: an exact preimage for the parametric family

`interval_inverse` answers "what raw range maps into this calibrated interval"; a second
question is narrower and, for the affine and beta families, answerable exactly: "what raw
score maps to *this one* calibrated probability". `point_inverse(p, *, space)` is that exact
preimage — a single closed-form (or fixed, certified-precision) computation, not bisection.

It is defined on:

- `BaseCalibrator` (inherited by every calibrator): the affine-logit path. Any calibrator
  whose `affine_logit_coeffs_` is not `None` — `PlattCalibrator`, `TemperatureCalibrator`,
  and `BetaCalibrator`'s tied `"a"`/`"ab"` variants — inverts
  `logit g(s) = a * logit(s) + b` in one line: `z = (logit(p) - b) / a`.
- `BetaCalibrator` (all three variants, overriding the base method): the full `"abm"` map is
  not affine on the logit scale, so it gets its own layered exact construction — see below.
- `LogitOffset`: `z = logit(p) - delta`, the same closed form as its `interval_inverse`.

A non-monotone calibrator, or a monotone one with no affine-logit or beta closed form
(isotonic/CIR/binning/scaling-binning, splines, Venn–Abers, ENIR), raises
`NotImplementedError` naming `interval_inverse` — the right tool wherever the map is a step
function (plateaus have no single preimage) or an otherwise non-affine monotone curve
(only a generalized inverse is well-defined).

## The beta inverse: a layered exact construction

`BetaCalibrator.point_inverse` solves, with `z = logit(s)` and `K = logit(p) - c`,

\[
h(z) = a z + (b - a)\,\mathrm{softplus}(z) = K, \qquad \mathrm{softplus}(z) = \ln(1 + e^{z}),
\]

which has no elementary closed form when `a != b` (it is transcendental — see the Lambert-W
connection below). The construction layers three ideas, each exact in a different regime:

**Layer 1 — minimax hyperbola seed.** Replacing `softplus(z)` by the minimax hyperbola
`(z + sqrt(z^2 + kappa)) / 2` with `kappa = 1.524` (max pointwise deviation 0.076) turns the
equation quadratic, with admissible root

\[
z_0 = \frac{(a+b) K - (b-a)\sqrt{K^2 + \kappa a b}}{2 a b},
\]

exact at `a = b` (collapses to `K / a`) and in both tails, with error bounded by
`0.076 * |b - a| / min(a, b)` elsewhere.

**Layer 2 — certified Halley correction.** Up to 4 fixed Newton–Halley steps refine `z_0` to
machine precision, exiting as soon as the residual certificate `|h(z) - K| <= 1e-13 *
max(1, |K|)` is met — a bounded, finite expression, not open-ended iteration. The residual is
itself a certificate: `|z - z*| <= |h(z) - K| / min(a, b)`.

**Layer 3 — the Lambert-W tail form (theory, not shipped code).** As `|z| -> infinity` the
equation admits a closed form in the Lambert-W function. For the left tail (`z -> -infinity`,
`softplus(z) -> 0`):

\[
z = \frac{K}{a} - W_0\!\left(\frac{b-a}{a}\, e^{K/a}\right),
\]

with the symmetric right-tail form (`z -> +infinity`) obtained by the `a <-> b`, `z -> -z`,
`K -> -K` swap. Both are exact only in the limit; away from the tails the correct branch and
argument regime are case-dependent, so probcal ships the seed-plus-Halley construction above
rather than a Lambert-W evaluator — recorded here as the closed form the numerics are
approximating, not as an implementation.

Degenerate exponents fall back to their own closed forms: `a == b` uses the affine formula
directly; `a == 0` (`h` ranges over `(0, infinity)`, so `p` above `sigma(c)` only) and
`b == 0` (range `(-infinity, 0)`, `p` below `sigma(c)` only) invert via `expm1`/`log`, and
raise `UnattainableTargetError` outside the attainable range; `a == b == 0` (a constant map)
raises `NotImplementedError` — no point has a well-defined preimage.

## References

- Rawal, K., Kamar, E., Lakkaraju, H. (2020). "Algorithmic Recourse in the Wild: Understanding the Impact of Data and Model Shifts." arXiv:2012.11788.
- Upadhyay, S., Joshi, S., Lakkaraju, H. (2021). "Towards Robust and Reliable Algorithmic Recourse." *Advances in Neural Information Processing Systems* 34 (NeurIPS 2021).
