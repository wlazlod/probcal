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

## References

- Rawal, K., Kamar, E., Lakkaraju, H. (2020). "Algorithmic Recourse in the Wild: Understanding the Impact of Data and Model Shifts." arXiv:2012.11788.
- Upadhyay, S., Joshi, S., Lakkaraju, H. (2021). "Towards Robust and Reliable Algorithmic Recourse." *Advances in Neural Information Processing Systems* 34 (NeurIPS 2021).
