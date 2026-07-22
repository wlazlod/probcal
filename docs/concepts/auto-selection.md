# Automatic selection

Eleven calibrators is a catalog, not a recommendation. `CalibratorSelector` turns the catalog
into a defensible choice by running the comparison the way the
[data-splitting chapter](data-splitting.md) demands: every candidate scored only on data it
did not fit, the criterion a strictly proper score by default, and the whole contest
documented in a report a validator can re-derive.

## The protocol

The selector receives the calibration data and a candidate list — by default Platt,
temperature, beta ("abm"), isotonic, centered isotonic, equal-mass histogram binning,
scaling-binning, and the inductive Venn–Abers; the heavier ensemble and spline methods join
by explicit opt-in. It runs an inner stratified K-fold entirely *within* the calibration
data: each candidate is fitted on the inner-training folds and produces predictions on the
held-out folds, and only those out-of-fold predictions are scored. The default criterion is
log loss, with Brier, ICI, smooth ECE, and ECE-sweep as deliberate alternatives — the
[metrics chapter's table](metrics.md) explains why plain ECE and Hosmer–Lemeshow are not on
that list. Alongside the criterion, each candidate's out-of-fold predictions pass through the
guardrails (slope, intercept, Spiegelhalter), which do not affect the ranking but are flagged
in the report — a candidate can win the score and still arrive with a warning attached.

Ties, within the resolution the fold spread supports, break toward **fewer parameters**: if
beta cannot beat Platt by more than noise, the extra tail parameter was not estimable and
parsimony takes it away; the same logic runs down the nested family to temperature and, at
the limit, to the [offset](offset.md). The winner is then refitted on the full calibration
set — the inner folds existed to rank, not to produce the deployed map — and returned
alongside a `SelectionReport`: the ranked table of candidates with mean and standard
deviation of the criterion across folds, guardrail flags, and the chosen-flag column.

## Why the nesting is structural

The trap the selector exists to prevent — scoring candidates on the data they were fitted on
— is described in the [data-splitting chapter](data-splitting.md); the design point here is
*how* it is prevented. The selector's scoring path receives out-of-fold predictions only;
there is no code path by which an in-fold prediction reaches the criterion, so the leakage is
not a documented misuse but an unrepresentable state, and `test_no_leakage.py` asserts it
mechanically. This is worth a sentence of justification because the failure it forecloses is
the quiet kind: selection bias does not crash, it just systematically crowns the most
flexible candidate, and the deployed map is a little worse forever.

## Reading a SelectionReport

The report is designed around three questions. *Who won, and by how much?* — read the mean
criterion against the runner-up's, in units of the fold standard deviations printed beside
them; a margin inside one standard deviation is a tie that parsimony already adjudicated.
*Is the winner healthy?* — the guardrail columns answer for the winner what they answer for
any calibrator, and a winning method with a failing intercept flag usually means the
candidate menu should have included the offset composition. *Was the contest fair to the
data?* — a report where every nonparametric method trails the parametric block is the
selector saying the sample could not support flexibility, which on a few hundred low-rate
observations (see the [sizing guidance](data-splitting.md)) is the expected verdict, not a
malfunction. The report prints; nothing needs a plotting backend; and the
[selection plot](visualization.md) exists for the deck where a table will not land.

## References

- Roelofs, R., Cain, N., Shlens, J., Mozer, M. C. (2022). "Mitigating Bias in Calibration Error Estimation." AISTATS, PMLR 151, 4036–4054.
