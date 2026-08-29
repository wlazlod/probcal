# Conservative most-prudent PDs

The per-grade backtests in [Metrics and tests](metrics.md) — `binomial_grade_test`,
`jeffreys_grade_test` — ask whether a grade's *assigned* PD is consistent with its
realized defaults. On a low- or zero-default grade (routine for the best few grades of a
retail or sovereign scorecard) that question is close to unanswerable from the grade's own
data alone: with zero defaults, every exact tail probability or Jeffreys posterior reading
is uninformative about the true PD, and a naive point estimate of `0/n = 0` is not a usable
PD to assign going forward (regulators do not accept "the PD is exactly zero"). Pluto &
Tasche (2005) solve a different, decision-relevant problem instead: not "is the assigned PD
consistent with the data", but "what is the *most-prudent* (upper-bound) PD I can defend for
this grade, at a stated confidence, given everything I know about the whole rating scale".
`probcal.metrics.pluto_tasche` implements their one-period estimator.

## The monotonicity assumption

A rating scale is presumed *ordered*: the true PD cannot decrease from a better grade to a
worse one. That assumption is doing the actual work. It licenses pooling a grade's own
(possibly zero-default) data with every worse grade's data before computing the bound — if
grade *i*'s true PD cannot exceed grade *i+1*'s, then observing few defaults across grades
*i* through *K* combined is still meaningful evidence that grade *i*'s own PD is low, even
when grade *i*'s own sample is too small or too clean to say anything by itself.

## The bound

For grades ordered best to worst (index 1..K, as given — the order is not inferred from the
data), pool grade *i* with every worse grade:

$$
n^*_i = \sum_{j=i}^{K} n_j, \qquad d^*_i = \sum_{j=i}^{K} d_j .
$$

The most-prudent PD for grade *i* is the one-sided Clopper-Pearson upper confidence bound of
the pooled default rate: the PD `p` at which observing at most `d*_i` defaults among `n*_i`
obligors has probability exactly `1 - confidence`,

$$
I_p(d^*_i + 1,\; n^*_i - d^*_i) = \text{confidence} \quad\Longrightarrow\quad
p = F^{-1}_{\mathrm{Beta}(d^*_i + 1,\; n^*_i - d^*_i)}(\text{confidence}),
$$

computed by `probcal._math.beta_ppf` (the same incomplete-beta quantile machinery
`binomial_grade_test`/`jeffreys_grade_test` use for their display intervals). Two edge cases
are handled directly rather than through the general solver: `d*_i == n*_i` (every pooled
obligor defaulted) gives `pd_upper = 1.0`, and a pooled set with `n*_i == 0` obligors raises
`ValueError` rather than silently returning a meaningless bound.

**Zero defaults, closed form.** When `d*_i = 0`, the incomplete-beta identity
`I_p(1, n) = 1 - (1 - p)^n` gives an exact closed form,

$$
p = 1 - (1 - \text{confidence})^{1/n^*_i},
$$

used in `tests/test_conservative.py` to pin `beta_ppf`'s bisection output to `1e-12`
independent of the general solver, and to reproduce the worked example from Pluto & Tasche
(2005) — grades A/B/C with `n = (100, 400, 300)`, all zero-default, at 90% confidence give
`pd_upper ≈ (0.29%, 0.33%, 0.76%)`.

**Weighted counts.** `pluto_tasche_from_arrays` accepts `sample_weight`; the resulting
per-grade counts (and their cumulative pools) are weighted sums, which can be non-integer.
These pass directly into the Beta shape parameters above — a documented convention, not an
approximation the caller needs to round away.

## Monotonicity of the output

Pooled sets are nested: grade *i*'s pooled set is grade *i*'s own data plus grade *i+1*'s
entire pooled set. For a portfolio whose observed per-grade default rates already respect
rating order, this makes `pd_upper` come out non-decreasing best to worst on its own. It is
not a guarantee for arbitrary input, though: a single noisy grade whose own rate happens to
exceed the worse-grade pool it joins can pull that grade's pooled rate — and so its bound —
above the next grade's. `pluto_tasche` corrects any such dip by taking the cumulative
maximum of `pd_upper` best to worst (a prudent hull: `hull[i] = max(pd_upper[0..i])`), so a
worse grade's bound is never below a better grade's. This is the only correction direction
consistent with a *most-prudent* estimator — it can only raise a grade's bound up to a
better grade's, never lower a grade's bound to match a worse one — and it never reduces any
individual grade's bound. `PlutoTascheResult.monotonized` reports whether the hull actually
changed anything. On realistic, already-monotone data it is a no-op (`monotonized = False`);
the coverage simulation below never triggers it either, since its true PD is deliberately
monotone with enough obligors per grade that sampling noise essentially cannot invert the
pooled-rate ordering.

## In probcal

```python
import numpy as np
from probcal.metrics import pluto_tasche, pluto_tasche_from_arrays

# From per-grade counts directly (Pluto & Tasche 2005 worked example).
res = pluto_tasche(
    np.array([100.0, 400.0, 300.0]),
    np.array([0.0, 0.0, 0.0]),
    confidence=0.9,
    grades=("A", "B", "C"),
)
print(res.pd_upper)          # [0.00287..., 0.00329..., 0.00758...]
print(res.interpret())       # one audit sentence per grade

# From observation-level grades and outcomes.
grades = np.array(["A"] * 100 + ["B"] * 400 + ["C"] * 300)
y = np.zeros(800)             # zero defaults observed
res2 = pluto_tasche_from_arrays(grades, y, order=("A", "B", "C"), confidence=0.9)
```

Both entry points return the same `PlutoTascheResult` and are re-exported from
`probcal.metrics.grade`, alongside the per-grade backtests they complement.

## Simulation verification

Produced by `docs/scripts/conservative_sim.py` (spec C1): a four-grade portfolio
(`n = (2000, 2000, 1000, 500)`, true PD `(0.5%, 1%, 3%, 8%)`, monotone by construction),
2000 seeded runs per confidence level. Coverage for grade *i* is the share of runs with
`pd_upper_i >= pd_true_i`; the gated quantity is the minimum over grades, since that is the
weakest guarantee the bound offers across the whole scale. A reduced-size version of the
same gate (300 runs, widened tolerance) runs in CI (`tests/test_conservative_sim.py`).

| confidence | runs | per-grade coverage (min over grades) | all-grades coverage | gate      |
|------------|------|---------------------------------------|----------------------|-----------|
| 0.9        | 2000 | 0.9225                                 | 0.9225               | >= 0.8866 |
| 0.95       | 2000 | 0.9610                                 | 0.9610               | >= 0.9403 |

Per-grade and all-grades coverage coincide in this run: with a monotone true PD and nested
pooled sets, the run that fails coverage for one grade tends to fail it for the binding
grade in every run, so the joint (all-grades) event and the minimum single-grade event
happen to line up here — not a general identity, just what this configuration produces.

## References

- Pluto, K., Tasche, D. (2005). "Estimating probabilities of default for low default
  portfolios." Deutsche Bundesbank Discussion Paper (also in *The Basel II Risk Parameters*,
  Springer, 2006).
