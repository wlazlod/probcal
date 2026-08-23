# Anytime-valid calibration monitoring

## The question

A calibrated forecast is deployed. Matured outcome batches arrive
periodically — monthly cohorts of `(p_i, y_i)` pairs. Is the forecast *still*
calibrated — and if not, is a level shift enough (re-offset with
`LogitOffset`) or has the shape changed (re-fit the calibrator)?

The obvious procedure — run `binomial_grade_test` or Hosmer–Lemeshow every
month and act on the first rejection — is invalid. Fixed-sample tests
control type-I error only for a pre-specified evaluation window; testing
*every* window and stopping at the first rejection inflates the error
without bound (with enough looks, a true null is rejected almost surely).
Monitoring is optional stopping by construction, so it needs a guarantee
that survives optional stopping.

## E-processes and Ville's inequality

An **e-process** is a non-negative process `E_k` with `E_0 = 1` that is a
supermartingale under the null. Ville's inequality gives, for any `alpha`:

$$
P_{H_0}\bigl(\exists k:\; E_k \ge 1/\alpha\bigr) \le \alpha .
$$

The rule "alarm the first time `E_k >= 1/alpha`" therefore has type-I error
at most `alpha` *however long monitoring runs and whenever it stops* — the
property fixed-sample p-values cannot provide. `p_anytime = min(1, 1/max_k
E_k)` is a p-value valid at every stopping time.

## The null

Observations arrive in a fixed order (arrival order within a batch is
arbitrary but frozen). The monitored hypothesis is **conditional
calibration** of the deployed forecast (Henzi & Ziegel 2022; Arnold, Henzi &
Ziegel 2023):

$$
H_0:\quad \mathbb{E}\left[y_i \mid \mathcal{F}_{i-1},\, p_i\right] = p_i .
$$

Every component below multiplies Bernoulli likelihood-ratio factors

$$
\mathrm{LR}_i(q_i) \;=\; \frac{q_i^{\,y_i}(1-q_i)^{1-y_i}}{p_i^{\,y_i}(1-p_i)^{1-y_i}},
$$

whose conditional expectation under `H_0` is exactly 1 for **any
predictable** alternative `q_i` (computed from strictly earlier data). The
running product is then a test martingale, and — the second tool — the
*average* of e-values is an e-value (Vovk & Wang 2021), which is how
components and variants combine below.

## Components

**Offset (level) process** `E_off`. Alternative: the true PD is
`sigma(logit(p_i) + delta)`. Two variants, averaged:

- the **predictable plug-in**: before batch `k`, `delta_hat_k` is the
  `LogitOffset` mode-B solution on batches `1..k-1` (the shift matching the
  past outcome rate) — it adapts to sustained drift;
- the **mixture** over a fixed grid `delta in {+-0.1, +-0.25, +-0.5,
  +-1.0}` with uniform prior — it protects the first batches, when the
  plug-in is still noise.

**Shape process** `E_shape`. Alternative: Cox recalibration
`logit(true) = c + a * logit(p_i)` with the predictable plug-in
`(c_hat, a_hat)` fitted by IRLS on past batches (the same engine as
`probcal.metrics.regression`). This detects slope drift that no offset can
repair.

**Per-grade processes** `E_grade[g]` (when a `grade` array is passed): the
offset process restricted to grade `g`, combined by averaging into
`E_grades`. A single drifting grade can then trip the alarm even when the
portfolio-level mean is preserved.

**Global alarm** on `E = mean(E_off, E_shape[, E_grades])` at level
`alpha`; component e-values are always reported for diagnosis.

## A confidence sequence for the current offset

Inverting the offset process against a grid of shifted nulls `delta_0 in
[-3, 3]` — "the deployed forecast, moved by `delta_0`, is calibrated" —
yields a **time-uniform** `(1 - alpha)` confidence sequence for the current
calibration-in-the-large offset: with 95% *anytime* confidence, statements
like "the model is currently 0.15–0.40 log-odds too optimistic". Each null
keeps its own e-process (plug-in alternative; cost `O(grid × n)` per
batch); a `delta_0` once rejected stays rejected, so the sequence is a
running intersection. This half-width is also the natural value for a
recourse engine's `buffer_logit` (see the treecf guide): a re-offset within
the sequence cannot invalidate a buffered counterfactual.

## The recommendation rule — a diagnostic, not a test

After an alarm the monitor reports one of `re-offset` / `re-fit`, from two
trailing-window diagnostics: recommend **re-offset** when the Cox slope
bootstrap CI contains 1 *and* the Cox-vs-offset residual likelihood ratio
(does the 2-parameter correction explain the window materially better than
the offset-only correction?) stays within the chi-square(1) 5% bound;
otherwise **re-fit**. The shape *e-process* is always reported alongside
but is deliberately not the discriminator: its alternative family contains
the intercept, so it fires under pure level drift too and cannot separate
the two failure modes on its own. The rule is a *diagnostic summary with no
error guarantee* — every component process is reported so the reader can
disagree with it.

## Validity conditions

- **Predictability.** Parameters for batch `k` use only batches `< k`; the
  observation order within a batch is fixed and arbitrary; past batches are
  never re-run or reordered — persist the monitor state (`to_json`) between
  updates rather than recomputing from raw data.
- **Within-cohort dependence.** Macro shocks that hit a whole cohort are
  *not* covered by the martingale null: the guarantee is conditional
  calibration given the forecast, and a portfolio-wide shock will trip the
  alarm — which is the desired behavior, not a false positive to engineer
  away.
- **Delayed labels.** The process advances only when labels mature; batch
  labels are opaque strings, and batches whose outcomes arrive out of
  calendar order are simply processed in arrival order.
- **After re-calibration**, start a **new** monitor on the new forecasts —
  the old null no longer describes production.
- **Weights.** Sample weights enter as exponents on the Bernoulli factors
  for reporting parity with the rest of probcal, but non-integer weights
  break the exact martingale property — the monitor warns once when it
  sees them.

## Relation to existing tools

`optbinning.ScorecardMonitoring` provides PSI and fixed-sample
characteristic tests — *population stability*, complementary to sequential
calibration validity: PSI asks "has the input mix moved", this monitor asks
"are the probabilities still right, accounting for every look we have
taken". Fixed-sample e-value tests such as the safe Hosmer–Lemeshow test
(Henzi, Puke, Dimitriadis & Ziegel 2024) answer a third question — a
one-shot audit with e-value semantics — and are a natural later addition to
`probcal.metrics.grade`.

## Simulation verification

Produced by `docs/scripts/monitor_sim.py` (spec W9): 2000 seeded runs × 24
monthly batches of n=2000 at a 5% event rate; drift experiments inject the
shift at batch 12; reduced-size versions of the same gates run in CI
(`tests/test_monitor_sim.py`), which also cross-checks the vectorized
simulator against the shipped `CalibrationMonitor` class.

| experiment                               | result | gate |
|------------------------------------------|--------|------|
| type-I offset (alpha=0.05)               | 0.0135 | <= 0.0597 |
| type-I shape (alpha=0.05)                | 0.0195 | <= 0.0597 |
| type-I global (alpha=0.05)               | 0.0155 | <= 0.0597 |
| type-I offset (alpha=0.01)               | 0.0025 | <= 0.0144 |
| type-I shape (alpha=0.01)                | 0.0040 | <= 0.0144 |
| type-I global (alpha=0.01)               | 0.0040 | <= 0.0144 |
| type-I global, per-grade (alpha=0.05)    | 0.0155 | <= 0.0597 |
| type-I global, hetero sizes (alpha=0.05) | 0.0275 | <= 0.0597 |
| type-I global, per-grade (alpha=0.01)    | 0.0020 | <= 0.0144 |
| type-I global, hetero sizes (alpha=0.01) | 0.0050 | <= 0.0144 |
| power delta=0.2                          | detect 0.94, median delay 6.0 | reported |
| power delta=0.4                          | detect 0.99, median delay 2.0 | median delay <= 6 |
| power slope=0.8                          | detect 0.99, median delay 2.0 | median delay <= 12 |
| power slope=1.25                         | detect 0.99, median delay 3.0 | reported |
| CS time-uniform coverage (delta=0)       | 0.9920 | >= 0.95 |
| CS time-uniform coverage (delta=0.4)     | 1.0000 | >= 0.95 |

The recommendation gate (correct call in ≥ 90% of pure-offset and
pure-slope runs) is enforced through the real `CalibrationMonitor` in the
same CI suite.

## References

- Henzi, A., Ziegel, J. F. (2022). "Valid sequential inference on
  probability forecast performance." *Biometrika* 109(3), 647–663.
- Arnold, S., Henzi, A., Ziegel, J. F. (2023). "Sequentially valid tests
  for forecast calibration." *Annals of Applied Statistics* 17(3),
  1909–1935.
- Vovk, V., Wang, R. (2021). "E-values: Calibration, combination and
  applications." *Annals of Statistics* 49(3), 1736–1754.
- Ville, J. (1939). *Étude critique de la notion de collectif.*
  Gauthier-Villars. (Ville's inequality.)
- Henzi, A., Puke, M., Dimitriadis, T., Ziegel, J. (2024). "A safe
  Hosmer–Lemeshow test." *The New England Journal of Statistics in Data
  Science* 2(2), 175–189.
