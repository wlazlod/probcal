# PROBCAL_SPEC.md — Implementation Specification for `probcal`

**Version of this spec:** 1.0 (2026-07-22)
**Audience:** Claude Code, starting implementation from an empty repository.
**Language of all deliverables:** English (code, docs, docstrings, commit messages).

---

## 0. Project summary

`probcal` is a universal Python package for **post-hoc probability calibration of binary
classifiers**. It unifies the full literature of binary calibration methods (Platt, temperature,
beta, isotonic, centered isotonic, histogram binning, scaling-binning, BBQ, ENIR, Venn–Abers,
spline calibration), an extensive catalog of calibration **evaluation metrics and statistical
tests**, calibration **visualization** on both probability and logit scales, an auditable
**logit-offset (central tendency) adjustment**, **automatic method selection** under nested
validation, and two data flows (**prefit** on a separate calibration set, and **cross-validation**).

Primary application domain: credit risk PD models (small calibration sets, low event rates,
regulatory audit requirements), but the package is fully general.

Positioning: no existing package unifies all of the above (sklearn: 2 methods; netcal: torch
dependency; betacal/venn-abers/ml-insights: single-method). A SoftwareX/JOSS-style software
paper is planned. The differentiators to protect at all costs: **numpy-only runtime**, logit-scale
diagnostics, first-class offset, transparent auto-selection.

---

## 1. Hard constraints (non-negotiable)

1. **Version is `0.0.1` and stays `0.0.1` until the package is published on PyPI.** Do not bump
   the version for any change before publication. `CHANGELOG.md` accumulates everything under
   `[Unreleased]`. Publication (and the bump to `0.1.0`) is a decision reserved for the owner.
2. **Runtime dependency: `numpy` only.** Nothing else. No scipy, no pandas, no sklearn, no
   matplotlib in the core import path.
   - `matplotlib` lives in the `[viz]` extra; `probcal.plots` guards its import and raises a
     clear `ImportError` with install instructions if missing.
   - `scipy`, `scikit-learn`, `statsmodels` are **dev/test-only** dependencies used exclusively
     as numerical references in tests (see §13). They must never be imported by `src/probcal`.
   - Special functions come from the stdlib (`math.lgamma`, `math.erf`) vectorized via
     `np.frompyfunc`, or from hand-rolled implementations in `probcal._math` (see §5).
   - No pandas: all tabular results are returned as frozen dataclasses of numpy arrays with an
     `as_dict()` method and a readable `__repr__` (aligned table).
3. **License: MIT.** Add `LICENSE` (MIT, copyright Daniel Wlazło). All GPL-licensed R packages
   (givitiR, rms, CalibratR) may be used as *conceptual* references only — reimplement from the
   primary papers, never port code. Record this in `docs/LICENSING.md`.
4. **Python `>=3.11`.** Rationale: consistency with FlagGAM, `typing.Self` for fluent
   `fit() -> Self`, better tracebacks/perf. Record as DECISIONS entry #1.
5. **Citation integrity.** Only the references listed in §15 may be cited. References marked
   ⚠ must be double-checked against the primary source (web search) before appearing in docs;
   if a ⚠ reference cannot be verified, cite the method descriptively without the reference and
   log a DECISIONS entry. **Never fabricate or guess a citation.** This is an absolute rule.
6. **Theory before code.** Task 1 (§14) — the long-form theoretical documentation — must be
   completed before any calibrator is implemented. The theory text is the design contract.
7. **No data leakage in any code path.** Calibrators must never be fitted and evaluated on the
   same observations inside selection or reported metrics; `CalibratorSelector` must make this
   structurally impossible (see §11).

---

## 2. Repository conventions (mirror of FlagGAM)

Replicate the structure and tooling of `github.com/wlazlod/FlagGAM` exactly, adapted to MIT and
numpy-only. Target tree:

```
probcal/
├── .github/workflows/
│   ├── docs.yml            # mkdocs gh-deploy on push to main
│   └── publish.yml         # PyPI publish on release tag (inactive until 0.1.0)
├── .gitignore
├── CHANGELOG.md            # Keep a Changelog 1.1.0 + SemVer; everything under [Unreleased]
├── CITATION.cff
├── LICENSE                 # MIT
├── PROBCAL_SPEC.md         # this file
├── README.md
├── conftest.py             # empty (root marker), as in FlagGAM
├── mkdocs.yml
├── pyproject.toml
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── how-it-works.md     # narrative pipeline walkthrough with math
│   ├── DECISIONS.md        # numbered implementation decisions (FlagGAM format)
│   ├── LICENSING.md
│   ├── api.md
│   ├── faq.md
│   ├── changelog.md        # includes ../CHANGELOG.md via snippet
│   ├── javascripts/mathjax.js
│   ├── concepts/           # ← the long-form THEORY (Task 1, §14)
│   │   ├── why-calibration.md
│   │   ├── methods-parametric.md
│   │   ├── methods-nonparametric.md
│   │   ├── methods-distribution-free.md
│   │   ├── metrics.md
│   │   ├── data-splitting.md
│   │   ├── offset.md
│   │   ├── shap-calibration.md
│   │   ├── inverse-maps.md
│   │   ├── auto-selection.md
│   │   └── visualization.md
│   └── notebooks/
│       └── pd_calibration_walkthrough.ipynb
├── src/probcal/            # src layout, hatchling wheel target
│   └── ... (see §4)
└── tests/                  # one test file per source module + cross-cutting tests
```

**pyproject.toml** — copy FlagGAM's template with these deltas:

```toml
[project]
name = "probcal"
version = "0.0.1"
description = "Universal post-hoc probability calibration for binary classifiers: methods, metrics, diagnostics, and auditable offsetting — numpy-only."
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
viz = ["matplotlib>=3.8"]
dev = ["pytest>=8", "ruff>=0.4", "black>=24", "mypy>=1.10",
       "scipy>=1.11", "scikit-learn>=1.4", "statsmodels>=0.14"]  # references for tests ONLY
docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.24", "mkdocs-jupyter>=0.24"]
```

Keep from FlagGAM verbatim: hatchling build backend with `packages = ["src/probcal"]`; ruff
(`E,F,I,UP,B`, line-length 100); black line-length 100; mypy on `probcal`; pytest with
`testpaths = ["tests"]` and markers `slow` and `reference` (reference = tests importing
scipy/sklearn/statsmodels; run in dev CI, skipped if libs absent). Ship `py.typed`. Docstrings:
**numpy style** throughout (rendered by mkdocstrings). `mkdocs.yml`: copy FlagGAM's theme,
plugins, mathjax setup and strict mode; nav mirrors the tree above.

**DECISIONS.md protocol** (FlagGAM format): numbered bold-title entries; every resolved
ambiguity, every deviation from this spec, every ⚠-reference outcome gets an entry citing the
spec section that drove it.

**README.md**: badges (MIT), one-paragraph description, installation (`pip install -e .`,
`uv sync --extra dev`), quickstart with a **runnable example including printed output** (FlagGAM
convention), feature matrix vs sklearn/netcal, link to docs.

---

## 3. Public API overview

Two levels, both exported from `probcal.__init__`:

**Score-level (core).** Every calibrator maps raw scores/probabilities `s ∈ (0,1)` (or logits)
to calibrated probabilities:

```python
cal = BetaCalibrator(variant="abm")            # "abm" | "ab" | "a"
cal.fit(s_cal, y_cal, sample_weight=None)      # -> Self
p = cal.predict_proba(s_new)                   # -> np.ndarray shape (n,)
cal.interpret()                                # -> Interpretation dataclass (see §6)
```

**Model-level (wrapper).** Duck-typed wrapper around any object with `predict_proba(X)` or
`decision_function(X)`:

```python
wrapped = CalibratedModel(model, calibrator=BetaCalibrator(), flow="prefit")
wrapped.fit(X_cal, y_cal)                      # flow="prefit": model already trained
wrapped = CalibratedModel(model, calibrator=..., flow="cv", cv=5, ensemble=False)
wrapped.fit(X_train, y_train)                  # flow="cv": clones+retrains per fold
wrapped.predict_proba(X_new)
wrapped.offset_to(target_mean=0.031)           # §8; returns Self, logs delta
```

Manual `get_params`/`set_params` on `BaseCalibrator` (no sklearn import) so everything is
sklearn-compatible *if* sklearn happens to be installed. `predict_proba` returns 1-D array of
P(y=1); a `predict_proba_2d()` helper returns the sklearn-style (n,2) matrix.

---

## 4. Module layout

```
src/probcal/
├── __init__.py         # public exports + __version__ = "0.0.1"
├── _validation.py      # binary y checks, score clipping to [eps,1-eps], eps=1e-12, logit/expit
├── _math.py            # numerical core (§5)
├── _results.py         # frozen dataclasses: ReliabilityCurve, MetricReport, SelectionReport,
│                       #   Interpretation, BeltResult — arrays + as_dict() + aligned __repr__
├── base.py             # BaseCalibrator (fit/predict_proba/interpret, get_params/set_params)
├── parametric.py       # PlattCalibrator, TemperatureCalibrator, BetaCalibrator
├── isotonic.py         # IsotonicCalibrator (PAVA), CenteredIsotonicCalibrator (CIR)
├── binning.py          # HistogramBinningCalibrator, ScalingBinningCalibrator
├── bayesian.py         # BBQCalibrator, ENIRCalibrator
├── vennabers.py        # VennAbersCalibrator (inductive), CrossVennAbersCalibrator
├── spline.py           # SplineCalibrator (natural cubic basis + penalized IRLS, CV lambda)
├── offset.py           # LogitOffset transform + solve_offset(target_mean) via bisection
├── attribution.py      # SHAP/additive-attribution adjustment to calibrated outputs (§9)
├── thresholds.py       # calibrated→raw interval and masterscale-band mapping (cutoffs, CF targets) (§10)
├── wrapper.py          # CalibratedModel (flows prefit / cv, ensemble True/False)
├── selection.py        # CalibratorSelector (§11)
├── curves.py           # reliability-curve builders (binned / LOESS / spline), GiViTI-style belt
├── metrics/
│   ├── __init__.py     # flat re-exports
│   ├── scores.py       # log loss, Brier, Brier skill score, decompositions
│   ├── binned.py       # ECE family, MCE, Hosmer–Lemeshow
│   ├── smooth.py       # smoothECE, ECCE, ICI/E50/E90/Emax (LOESS), Spiegelhalter z
│   ├── regression.py   # calibration intercept & slope + tests
│   └── grade.py        # per-grade binomial and Jeffreys tests (credit-risk backtesting)
├── datasets.py         # make_pd_portfolio(): synthetic miscalibrated PD scores generator
└── plots.py            # matplotlib helpers ([viz] extra, import-guarded)
```

---

## 5. Numerical core (`_math.py`) — pure numpy + stdlib

All functions vectorized, documented, and covered by `reference`-marked tests against scipy
(§13). Contents:

1. `logit`, `expit` (clipped, overflow-safe).
2. `pava(y, w)` — pool-adjacent-violators, amortized O(n), preallocated arrays, no Python
   appends; returns block means expanded to observations + block structure (needed by CIR,
   Venn–Abers, ENIR).
3. `irls_logistic(X, y, w=None, ridge=0.0, offset=None, max_iter=100, tol=1e-10)` — Newton/IRLS
   logistic regression with ridge stabilization and separation detection (warn + return
   ridge-regularized fit). This single routine powers Platt, beta, spline, calibration
   slope/intercept, belt, and LOESS-free logistic recalibration.
4. `newton_1d` and `bisect` — for temperature fitting and offset solving.
5. `lgamma_vec`, `erf_vec` — `np.frompyfunc(math.lgamma/erf)` wrappers cast to float64.
6. `betainc(a, b, x)` — regularized incomplete beta via continued fraction (Lentz's algorithm);
   needed for exact binomial p-values and the Jeffreys test. Accuracy target: max abs error
   < 1e-12 vs `scipy.special.betainc` on a dense grid (reference test).
7. `gammainc_lower(s, x)` — regularized lower incomplete gamma (series + continued fraction);
   gives the χ² CDF for Hosmer–Lemeshow and the belt. Same accuracy protocol.
8. `chi2_ppf(q, df)` — bisection on `gammainc_lower`.
9. `norm_ppf(q)` — Acklam-style rational approximation refined by one Halley step on `erf_vec`;
   `norm_cdf` via `erf_vec`.
10. `loess(x, y, frac=0.75, degree=1)` — tricube-weighted local linear regression evaluated on a
    grid; used by ICI and smoothed reliability curves.
11. `natural_cubic_basis(x, knots)` — closed-form N_k basis (Hastie–Tibshirani–Friedman §5.2.1
    construction) for `SplineCalibrator` and smoothed curves.

---

## 6. Calibrator catalog

Common contract: `fit(s, y, sample_weight=None) -> Self`; `predict_proba(s) -> (n,)`; monotone
methods expose `is_monotone_ = True` and `interval_inverse(lo, hi, *, space, buffer_logit)` (§10); every calibrator implements `interpret()` returning an
`Interpretation` dataclass: fitted parameter values + a plain-language, domain-aware reading of
each (strings below are the required content, verbatim spirit not verbatim text). Docstrings
carry the same interpretations.

| # | Class | Algorithm & fitting | Parameters and REQUIRED interpretation |
|---|-------|--------------------|----------------------------------------|
| 1 | `PlattCalibrator` | Logistic regression of y on z=logit(s) via `irls_logistic`, Lin–Lin–Weng target smoothing (targets (N₊+1)/(N₊+2), 1/(N₋+2)) for stability. | slope `a`: spread correction — a<1 shrinks overconfident scores toward the base rate, a>1 sharpens underconfident ones; intercept `b`: base-rate (calibration-in-the-large) shift in log-odds; identity ⇔ (a,b)=(1,0). Note in docs: the logistic family fitted on raw SVM outputs (Platt 1999) does not contain identity; on logits it does. |
| 2 | `TemperatureCalibrator` | p=σ(z/T); T minimizes NLL via 1-D Newton with bisection fallback. | `T`: T>1 = model was overconfident (softening), T<1 = underconfident (sharpening); T cannot fix base-rate error — docs must state this and point to Platt/offset. |
| 3 | `BetaCalibrator` | Logistic regression on features [ln s, −ln(1−s)] (variant "abm": both + intercept; "ab": a=b; "a": single). Constraint a,b≥0 enforced by the betacal refit strategy: if a<0 drop ln s and refit, if b<0 drop −ln(1−s) and refit (DECISIONS entry). | `a`: sensitivity near s→0 — governs the low-PD tail (critical for credit risk); `b`: sensitivity near s→1; `c`: base-rate shift; (a,b,c)=(1,1,0) is the identity ⇒ beta cannot un-calibrate an already calibrated model (contrast with Platt); a≠b captures asymmetric distortion. Temperature = special case a=b, c=0. |
| 4 | `IsotonicCalibrator` | PAVA on (s,y); step function; out-of-range clamped; optional interpolation="linear" between block midpoints. | Step levels are empirical event rates of pooled score blocks; number of blocks = effective complexity (report it); flat steps ⇒ ties in predictions. |
| 5 | `CenteredIsotonicCalibrator` | CIR post-processing of PAVA: collapse each pooled block to its weight-centered point, linear interpolation ⇒ strictly increasing. | Same as isotonic + strictness removes tied predictions; preferred when downstream ranking must be strict. |
| 6 | `HistogramBinningCalibrator` | Equal-width or equal-mass bins; per-bin event rate with optional Jeffreys shrinkage ((k+0.5)/(n+1)). | Bin rates = local event frequencies; B controls bias-variance; equal-mass recommended default (lower estimator bias). |
| 7 | `ScalingBinningCalibrator` | Kumar–Liang–Ma: fit Platt first, then equal-mass binning of the fitted *function values*; outputs bin means of g(s). | Two-stage interpretation: parametric stage as Platt; binning stage yields measurable calibration error with O(1/ε²+B) samples vs O(B/ε²) for histogram. |
| 8 | `BBQCalibrator` | Bayesian averaging over equal-mass binning models with different B; Bayesian marginal-likelihood score via `lgamma_vec`. | Posterior weights over B = uncertainty about resolution; report top-3 weighted models. |
| 9 | `ENIRCalibrator` | Modified PAVA path of near-isotonic solutions (Tibshirani et al. nearly-isotonic fits), BIC-weighted ensemble. | Path parameter λ trades monotonicity strictness vs fit; BIC weights = model plausibility; output may be non-monotone — set `is_monotone_=False` and warn in docs. |
| 10 | `VennAbersCalibrator` / `CrossVennAbersCalibrator` | IVAP: two isotonic fits (label 0 and 1 appended) ⇒ interval [p0,p1]; scalarized as p = p1/(1−p0+p1). CVAP: fold-wise IVAPs merged (geometric-mean rule from Vovk–Petej). Expose `predict_interval()`. | Interval width = calibration uncertainty at that score (report mean/max width); validity guarantee holds for the interval object, not the scalarization — docs must state this precisely. |
| 11 | `SplineCalibrator` | Natural cubic basis on z=logit(s), penalized IRLS (ridge on second-difference penalty), λ by K-fold CV grid on log loss. | Effective d.o.f. = trace of smoother (report); fitted curve slope vs identity read as local over/underconfidence by score region. |

Every calibrator's docstring ends with a `References` section using §15 entries only.

---

## 7. Metric & test catalog (`probcal.metrics`)

All metrics: signature `metric(y, p, *, sample_weight=None, **kw)`, return float or a small
frozen dataclass; every one gets a docstring with formula (LaTeX), interpretation, known
pathologies, and reference. `evaluate(y, p)` returns a `MetricReport` with everything below +
bootstrap percentile CIs (`n_boot=1000`, seeded RNG).

**Proper scoring rules** (`scores.py`)
1. `log_loss` — primary selection criterion.
2. `brier_score`; `brier_skill_score` (reference = climatology p̄).
3. `murphy_decomposition(y, p, bins)` → reliability − resolution + uncertainty (document that
   the binned decomposition inherits binning bias).
4. `logloss_calibration_refinement(y, p)` — calibration/refinement split of log loss via the
   recalibration-curve plug-in (document estimator choice; DECISIONS entry).

**Binned estimators** (`binned.py`)
5. `ece(y, p, n_bins=15, strategy="mass"|"width", norm="l1"|"l2"|"max")` — `norm="max"` ⇒ MCE.
6. `ece_debiased` — Bröcker/Ferro–Fricker bias correction.
7. `ece_sweep` — Roelofs et al.: equal-mass bins, largest B preserving monotone bin means.
8. `adaptive_ece` — equal-mass alias with explicit name (document equivalence).
9. `hosmer_lemeshow(y, p, g=10)` — χ² via `gammainc_lower`; docs: arbitrary grouping, power
   pathologies at large n; report-only, never a selection criterion.

**Binning-free** (`smooth.py`)
10. `smooth_ece(y, p)` — Błasiok–Nakkiran kernel-smoothed ECE (reflected Gaussian kernel on
    logit scale, principled bandwidth per paper; document any simplification as DECISIONS).
11. `ecce(y, p)` — empirical cumulative calibration error (Kolmogorov-style max and mean of the
    cumulative deviation ∑(yᵢ−pᵢ) over sorted p); ⚠ reference — verify before citing.
12. `ici(y, p, frac=0.75)` + `e50`, `e90`, `emax` — LOESS-based Austin–Steyerberg family.
13. `spiegelhalter_z(y, p)` — z statistic + two-sided p-value via `norm_cdf`.

**Recalibration-regression framework** (`regression.py`)
14. `calibration_intercept(y, p)` — logistic fit of y on offset(logit p) (slope fixed at 1):
    calibration-in-the-large in log-odds; interpretation: portfolio-level over/underestimation.
15. `calibration_slope(y, p)` — Cox slope; <1 ⇒ overfitting/overconfidence, >1 ⇒ underfitting.
16. `calibration_test(y, p)` — likelihood-ratio test of (α,β)=(0,1) jointly (2 df), the
    Cox/Miller "weak calibration" test; χ² p-value via `gammainc_lower`.

**Per-grade backtesting (credit risk)** (`grade.py`)
17. `binomial_grade_test(y, p, grades)` — exact binomial p-value per rating grade via `betainc`,
    plus normal approximation; traffic-light style summary table.
18. `jeffreys_grade_test(y, p, grades)` — Jeffreys posterior Beta(k+½, n−k+½) tail probability
    per grade (ECB IRB backtesting practice); document one-sided conservative reading.

**Guardrail summary.** `calibration_guardrails(y, p)` → dataclass: slope∈[0.9,1.1]?,
|intercept|≤0.1?, Spiegelhalter p>0.05? — used by `CalibratorSelector` and printed in reports.

Docs (`concepts/metrics.md`) must include the comparison table: metric × {binning-sensitive?,
biased?, proper?, test?, selection-suitable?} and the explicit recommendation: **select on log
loss (default) or Brier; report ECE-family and ICI; never select on ECE or HL.**

---

## 8. Offset feature (`offset.py`) — first-class and auditable

`LogitOffset(delta=None, target_mean=None)`:
- mode A (explicit): p' = σ(logit(p) + δ).
- mode B (target): solve mean(σ(logit(pᵢ)+δ)) = π* for δ by `bisect`; the portfolio mean is
  strictly increasing in δ, so the root is unique — state and unit-test this.
- `interpret()`: δ in log-odds; exp(δ) = multiplicative odds factor applied uniformly; the
  credit-risk **central tendency** re-anchoring; equivalent to King–Zeng prior correction with
  δ = −ln[((1−τ)/τ)·(ȳ/(1−ȳ))] and to Elkan's base-rate adjustment (cite §15).
- Audit trail: fitting stores `delta_`, `pre_mean_`, `post_mean_`, timestamp; `audit_report()`
  returns pre/post guardrails (§7) so a validator sees exactly what the shift did.
- Composability: `CalibratedModel.offset_to(...)` appends a LogitOffset to the pipeline and
  keeps both stages inspectable — never fold δ into the calibrator's parameters.

---

## 9. Attribution adjustment (`attribution.py`) — SHAP that sums to the calibrated value

Post-hoc calibration breaks SHAP's local accuracy: base + Σφᵢ reconstructs the *raw* score, not
the calibrated PD. This module restores additivity on the calibrated scale — a headline feature
(reason codes must explain the number actually used in the decision).

**Theoretical framing (required in the theory chapter and docstrings).** From `(base_value, φ)`
alone, the exact Shapley values of the composition g∘f are **not identifiable** — they require
coalition expectations E[f|S], which TreeSHAP outputs no longer contain. The package therefore
offers two principled modes and refuses to pretend otherwise:

1. **Affine-exact mode.** For calibrators affine on the logit scale — `TemperatureCalibrator`
   (a=1/T, b=0), `PlattCalibrator` fitted on logits (a, b), `LogitOffset` (a=1, b=δ), and any
   composition thereof — logit(p') = a·z + b. By linearity of the Shapley value: φ'ᵢ = a·φᵢ and
   base' = a·base + b. These ARE the exact Shapley values of the composed model. Beta, isotonic,
   spline, and binning are **not** affine in z (for beta: logit(p') = −a·softplus(−z) +
   b·softplus(z) + c); the docs must state this explicitly.
2. **Aumann–Shapley mode (general).** φ'ᵢ = φᵢ · [g(s) − g(s₀)] / (s − s₀) with s₀ = base value,
   base' = g(s₀). Exact additivity by construction for ANY calibrator (the difference quotient
   also covers piecewise-constant g where g' = 0 a.e.). Grounding: this is exactly the
   Aumann–Shapley / integrated-gradients attribution of the univariate outer map applied to the
   additive representation s = s₀ + Σφᵢ (straight-line path) — cite Aumann & Shapley (1974) and
   Sundararajan et al. (2017). Honest caveat for docs: it is *not* the Shapley value of g∘f in
   general; the nonlinearity is distributed proportionally to φᵢ.

**Properties to state and unit-test.**
- For affine calibrators the two modes coincide to 1e-12 (equivalence test).
- For monotone g the row multiplier is positive ⇒ signs and within-row |φ| ranking preserved ⇒
  adverse-action reason-code ordering is invariant under adjustment (Regulation B relevance —
  state in docs).
- Reconstruction: max |base' + Σφ'ᵢ − target| < 1e-10 on every row (returned in the result).
- Degenerate rows s ≈ s₀: fall back to the local slope g'(s₀) estimated by central difference;
  DECISIONS entry for the epsilon.

**API.**
```python
adjust_attributions(
    phi,                    # (n, d) raw SHAP values on the model's score scale
    base_value,             # scalar or (n,)
    calibrator,             # fitted BaseCalibrator or CalibratedModel (incl. offset stage)
    scale="logit",          # "logit" (default; reason-code convention) | "probability"
    method="auto",          # "auto": affine-exact if available, else Aumann–Shapley
) -> AdjustedAttribution   # phi_adj, base_adj, target, method_used, max_reconstruction_error
```
- Calibrators gain a property `affine_logit_coeffs_` → `(a, b)` or `None`; `CalibratedModel`
  composes coefficients across pipeline stages (calibrator ∘ offset) when all stages are affine.
- `scale="probability"`: same machinery with g expressed on the probability scale; document
  that affine-exactness holds only on the logit scale.
- No `shap` dependency: accepts plain arrays; duck-types `shap.Explanation` (reads `.values`,
  `.base_values`) without importing shap. Never import shap in `src/probcal`.
- The recompute-on-composition alternative (e.g. TreeSHAP with `model_output="probability"`) is
  documented in the theory chapter as the exact-but-heavy route, out of scope for numpy-only.

## 10. Inverse maps and decision thresholds (calibrated → raw)

Once decisions are made on calibrated PD, every raw-score consumer needs the inverse map:
policy cutoffs (“approve below 2% PD”) translated to score cutoffs, masterscale grade edges
expressed in score space, and counterfactual targets for CF engines. The capability is intrinsic
to the calibrator — only it knows its functional form, block structure, and output range — so it
lives here; consumers stay decoupled. For CF engines the bridge is one identity: for **monotone**
g, `{x : g(f(x)) ∈ [lo, hi]} = {x : f(x) ∈ [g⁻¹(lo), g⁻¹(hi)]}` — calibration does not change
counterfactual geometry, only the target interval, so treecf works unchanged via `Target.raw`.

**Calibrator contract additions (base.py; implemented per calibrator as they are built).**
- `interval_inverse(lo, hi, *, space="probability", buffer_logit=0.0)` → `(raw_lo, raw_hi)`,
  the generalized-inverse preimage: `raw_lo = inf{s : g(s) ≥ lo}`, `raw_hi = sup{s : g(s) ≤ hi}`
  for non-decreasing g. `space="probability"` returns bounds on the model's probability output;
  `space="logit"` returns their logits (what treecf's SIGMOID-link `Target.raw` expects).
  `lo=0.0` / `hi=1.0` map to −inf / +inf.
- Implementations: closed-form for Platt/temperature/`LogitOffset`; monotone bisection for beta
  and monotone splines; `searchsorted` on the block structure for isotonic/CIR/binning/
  scaling-binning and scalarized Venn–Abers (p = p1/(1−p0+p1) is non-decreasing in s — both
  partials are positive and p0, p1 are non-decreasing; state and test this).
- `CalibratedModel.interval_inverse` composes through the pipeline right-to-left (offset first:
  subtract δ on the logit, then the calibrator's inverse).
- **Attainability check**: if `[lo, hi]` does not intersect the calibrator's output range
  (common for isotonic on low-PD data, where the range is `[min block, max block]`), raise
  `UnattainableTargetError` naming both intervals. Never silently clamp.
- **Non-monotone calibrators** (`is_monotone_ = False`, e.g. ENIR): the preimage may be a union
  of intervals; raise `NotImplementedError` with an explanatory message. Document that CF
  pipelines should use monotone calibrators.
- **Plateau caveat** (step calibrators): a counterfactual landing just past a block edge is
  fragile — the calibrated value jumps discretely and any refit moves the edge. Recommend
  `buffer_logit > 0` or a continuous calibrator (beta, CIR) for recourse use; state in docs.
- `buffer_logit`: shrinks the calibrated interval by a margin in logit space *before* inverting,
  producing counterfactuals robust to future recalibration drift — in particular to central
  tendency updates of `LogitOffset` (a quarterly δ update of magnitude ≤ m cannot invalidate a
  CF built with `buffer_logit = m`). Connect to the recourse-robustness literature (⚠ refs).

**Module `thresholds.py`** (numpy-only; arrays and floats only — no knowledge of any consumer):
- `calibrated_interval_to_raw(calibrator, lo, hi, *, space, buffer_logit)` — thin functional
  wrapper over the method above.
- `calibrated_bands_to_raw(calibrator, bands, *, space, buffer_logit)` — maps a masterscale
  `{grade: (lo, hi)}` defined on calibrated PD to raw intervals; output plugs directly into
  treecf `Target.bands(..., space="raw")`. This is the canonical rating-grade workflow: grades
  are defined on calibrated PD, the model emits raw margins, calibration sits between.

**Interop recipe (docs + tested example in the tutorial notebook, works with treecf as-is):**
```python
lo_z, hi_z = cal.interval_inverse(0.0, 0.02, space="logit")   # "PD ≤ 2%" after calibration
target = treecf.Target.raw(range=(lo_z, hi_z))
```
**Division of responsibilities (record as a DECISIONS entry).** probcal owns the capability and
publishes the duck-typed protocol — `interval_inverse(lo, hi, *, space)` plus `is_monotone_` —
in `docs/faq.md`. Ergonomics of target construction belong to treecf and are tracked in that
repository, not here: `Target.calibrated(..., calibrator=...)` duck-typing the protocol, and the
user-facing warning that after deploying calibration `Target.probability(...)` becomes a silent
bug (it inverts the model's own sigmoid link, not g, and thus targets the *uncalibrated*
probability). probcal's docs carry the `Target.raw` recipe above and a single sentence pointing
at the trap.

**Tests.** Round-trip `g(interval_inverse(τ)) ≈ τ` for strictly monotone calibrators; block-edge
semantics for isotonic (preimage starts exactly at the left edge of the first qualifying
block); empty-preimage raises; offset composition (δ shift moves raw bounds by exactly −δ in
logit space); probability/logit space consistency; buffer monotonicity (larger buffer ⇒ tighter
raw interval).

---

## 11. Flows and automatic selection

**Flow "prefit"** (separate calibration set): model is already trained elsewhere; `fit(X_cal,
y_cal)` scores `X_cal` with the model and fits the calibrator on (s, y). This is the canonical
credit-risk flow (train/calibration/test split). Also usable purely at score level via the core
API when the model is not a Python object at all (scores from a data warehouse).

**Flow "cv"**: `fit(X, y)` clones the model (duck-typed: `sklearn.base.clone` if available, else
`copy.deepcopy` — DECISIONS entry), trains on k−1 folds, scores the held-out fold; then either
`ensemble=True` — per-fold calibrators averaged at predict time (sklearn `CalibratedClassifierCV`
behavior), or `ensemble=False` — one calibrator on pooled out-of-fold scores, final model
refitted on all data (recommended default for credit risk; document why: single auditable
mapping). Stratified folds by default; seeded.

**`CalibratorSelector`**: candidates (default: Platt, temperature, beta-abm, isotonic, CIR,
histogram-mass, scaling-binning, IVAP; full list opt-in), inner K-fold on the calibration data
only; per-candidate out-of-fold `scoring` (default `"log_loss"`; options: `"brier"`, `"ici"`,
`"smooth_ece"`, `"ece_sweep"`) + guardrails; returns `SelectionReport` (ranked table: method,
mean±sd score, guardrail flags, chosen flag) and the winner refitted on the full calibration
set. **Refuses** to score on the fitting data (structural: scoring only ever sees out-of-fold
predictions). Ties broken by fewer parameters (parsimony). Docs must explain the
selection-on-fitting-data trap and why nesting is mandatory.

---

## 12. Curves, belt, and plots

`curves.py` (numpy-only, returns dataclasses — all plotting-backend-agnostic):
- `reliability_binned(y, p, n_bins, strategy)` — per-bin mean p, event rate, count, Wilson CI.
- `reliability_loess(y, p, frac)` and `reliability_spline(y, p)` — smoothed curves on a grid.
- `calibration_belt(y, p, confidence=(0.8,0.95))` — GiViTI-style: polynomial logistic fit on
  logit(p) with forward LR selection of degree (≤4), pointwise confidence band by inverting the
  LR region (χ² via `_math`), + associated test p-value. Reimplemented from Nattino et al.
  papers only (GPL code untouched).
- Every result carries both `p`-scale and `logit`-scale coordinate arrays.

`plots.py` ([viz]): `plot_reliability` (binned+smoothed overlay, score histogram margin,
`scale="probability"|"logit"`), `plot_belt`, `plot_comparison(before, after)` (pre/post
calibration and pre/post offset), `plot_interval(vennabers)` (interval widths vs score),
`plot_selection(report)`. Logit-scale plots are the flagship feature — the low-PD region must be
readable; axis ticks labeled in probabilities at logit positions.

---

## 13. Testing strategy

Mirror FlagGAM: one `tests/test_<module>.py` per source module + cross-cutting files.
- `test_package.py`: version == "0.0.1", `__all__` complete, py.typed present, **no forbidden
  imports** — walk `sys.modules` after `import probcal` and assert scipy/sklearn/pandas/
  matplotlib absent.
- `test_math_reference.py` (`@pytest.mark.reference`): betainc/gammainc/chi2_ppf/norm_ppf/
  lgamma/erf vs scipy on dense grids (tolerances stated per function); `pava` vs
  `sklearn.isotonic`; `irls_logistic` vs `statsmodels.GLM` coefficients (rtol 1e-8); `loess` vs
  statsmodels lowess (loose rtol, documented).
- Per-calibrator tests: identity recovery on perfectly calibrated synthetic data (beta ⇒
  (1,1,0), Platt ⇒ (1,0), T ⇒ 1, δ ⇒ 0); monotonicity property tests; known-distortion recovery
  (generate y ~ Bernoulli(σ(a·z+b)) and check parameter recovery); small-sample stability (no
  NaN/separation crashes at n=200, 20 events); Venn–Abers interval validity on simulation.
- `test_no_leakage.py`: selector never evaluates in-fold; wrapper cv flow never scores training
  folds with a model that saw them.
- Metrics: closed-form checks on hand-computable tiny cases + reference comparisons where
  available (sklearn log_loss/brier, statsmodels HL if available).
- Seeded RNG everywhere; `slow` marker for bootstrap/CV-heavy tests.

---

## 14. Task order (all within version 0.0.1)

Each task = one coherent PR-sized unit; finish with tests green, `ruff`+`black`+`mypy` clean,
CHANGELOG `[Unreleased]` updated, DECISIONS entries added.

- **Task 0 — Scaffold.** Full tree of §2, pyproject, tooling configs, CI workflows, empty
  modules with docstrings, LICENSE, CITATION.cff, README skeleton. DoD: `uv sync --extra dev`,
  `pytest` (collects trivially), `mkdocs build --strict` all pass.
- **Task 1 — Theory guidebook (BEFORE any algorithm code).** Write the long-form theoretical
  documentation in `docs/concepts/` per the outline in §14.1. DoD: all nine chapters complete,
  `mkdocs build --strict` clean, every citation from §15, every ⚠ resolved or descoped with a
  DECISIONS entry. Target total length: **12,000–18,000 words**, full mathematical notation.
- **Task 2 — Numerical core.** `_math.py`, `_validation.py`, `_results.py` + reference tests.
- **Task 3 — Base API + parametric calibrators** (Platt, temperature, beta) + `interpret()`.
- **Task 4 — Isotonic family + Venn–Abers** (PAVA, CIR, IVAP, CVAP).
- **Task 5 — Binning + Bayesian** (histogram, scaling-binning, BBQ, ENIR).
- **Task 6 — Spline calibrator.**
- **Task 7 — Metrics catalog** (§7 complete, incl. bootstrap CIs and `evaluate`).
- **Task 8 — Curves + belt + plots.**
- **Task 9 — Offset** (+ audit report).
- **Task 10 — Attribution adjustment** (`attribution.py`, §9) + affine-flag plumbing on calibrators.
- **Task 11 — Inverse maps & thresholds** (`interval_inverse` on all calibrators, `thresholds.py`, §10).
- **Task 12 — Wrapper flows** (prefit, cv/ensemble) + `offset_to`.
- **Task 13 — CalibratorSelector** + `SelectionReport`.
- **Task 14 — Datasets + tutorial notebook + remaining docs** (`make_pd_portfolio` with
  controllable miscalibration: slope/intercept distortion, asymmetric tail distortion, target
  event rate ~3%; executed notebook committed; getting-started, how-it-works, api.md, faq).
- **Task 15 — Polish.** README quickstart with printed outputs, feature-matrix table,
  docs cross-linking, final DECISIONS/LICENSING pass.

### 14.1 Theory guidebook — required chapter outline (Task 1)

1. `why-calibration.md` (~2000 w): definitions (perfect calibration, calibration vs
   discrimination/sharpness); why miscalibration arises (model class bias, regularization,
   class imbalance, distribution shift, boosting/overfitting); consequences in decisioning —
   expected-loss pricing, cutoff policies, credit-risk capital (PD feeds RWA), regulatory
   context (IRB backtesting, model validation); proper scoring rules as the organizing lens;
   the calibration–refinement decomposition.
2. `methods-parametric.md` (~2500 w): Platt (incl. Lin–Lin–Weng), temperature, beta (full
   derivation from Beta-distributed class-conditional scores; identity property; monotonicity
   constraint), parameter interpretation tables (§6 content, expanded).
3. `methods-nonparametric.md` (~2500 w): isotonic/PAVA (with the pooling intuition and a worked
   micro-example), CIR, histogram/quantile binning, scaling-binning (sample-complexity
   argument), BBQ, ENIR (near-isotonic path), spline calibration.
4. `methods-distribution-free.md` (~1500 w): Venn–Abers theory — validity under exchangeability,
   IVAP construction, CVAP merging, interval semantics and scalarization caveat.
5. `metrics.md` (~3000 w): full §7 catalog with formulas, pathologies (ECE bias and binning
   sensitivity, Roelofs findings, HL power issues), decompositions, per-grade backtesting
   practice, the selection-suitability table.
6. `data-splitting.md` (~1500 w): why calibrating on training data fails; prefit vs cv flows;
   ensemble vs pooled; nested validation for selection; small-sample guidance for credit risk.
7. `offset.md` (~1500 w): central tendency adjustment theory; King–Zeng and Elkan derivations;
   Tasche's PD-curve calibration framing (QMM, scaled PDs); uniqueness of the bisection root;
   audit practice.
8. `auto-selection.md` (~800 w): the selector protocol, criteria, guardrails, parsimony ties.
9. `visualization.md` (~700 w): reliability construction variants, why the logit scale matters
   for low-PD portfolios, the calibration belt idea.
10. `shap-calibration.md` (~1500 w): why calibration breaks SHAP local accuracy;
    non-identifiability of Shapley values of g∘f from (base, φ); the affine-exact class and the
    linearity argument; Aumann–Shapley grounding of proportional rescaling; sign/rank
    preservation and reason-code invariance; scale choice; the recompute-on-composition
    alternative; related work (Calibrated Explanations, ⚠).
11. `inverse-maps.md` (~1000 w): cutoff and masterscale translation to score space;
    the preimage identity for monotone g and why CF
    geometry is calibration-invariant; generalized inverses and plateau semantics;
    attainability; the Target.probability trap after calibration; buffer_logit and
    recourse robustness to recalibration/offset drift (⚠ refs); the masterscale
    band workflow.

Style: direct prose, numpy-style math via MathJax, no bullet-point padding, every method chapter
ends with a `References` block from §15.

---

## 15. Reference list (the ONLY permitted citations)

Verified (safe to cite as given):
- Platt, J. C. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." In *Advances in Large Margin Classifiers*, MIT Press, 61–74.
- Lin, H.-T., Lin, C.-J., Weng, R. C. (2007). "A Note on Platt's Probabilistic Outputs for Support Vector Machines." *Machine Learning* 68(3), 267–276.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." ICML, PMLR 70, 1321–1330.
- Zadrozny, B., Elkan, C. (2001). "Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers." ICML, 609–616.
- Zadrozny, B., Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." KDD, 694–699.
- Kull, M., Silva Filho, T., Flach, P. (2017). "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." AISTATS, PMLR 54, 623–631.
- Kull, M., Silva Filho, T., Flach, P. (2017). "Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration." *Electronic Journal of Statistics* 11(2), 5052–5080.
- Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., Flach, P. (2019). "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration." NeurIPS 32.
- Naeini, M. P., Cooper, G. F., Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 29, 2901–2907.
- Naeini, M. P., Cooper, G. F. (2016). "Binary Classifier Calibration using an Ensemble of Near Isotonic Regression Models." IEEE ICDM, 360–369.
- Vovk, V., Petej, I. (2014). "Venn–Abers Predictors." UAI, 829–838.
- Kumar, A., Liang, P., Ma, T. (2019). "Verified Uncertainty Calibration." NeurIPS 32.
- Lucena, B. (2018). "Spline-Based Probability Calibration." arXiv:1809.07751.
- Oron, A. P., Flournoy, N. (2017). "Centered Isotonic Regression: Point and Interval Estimation for Dose–Response Studies." *Statistics in Biopharmaceutical Research* 9(3), 258–267.
- Roelofs, R., Cain, N., Shlens, J., Mozer, M. C. (2022). "Mitigating Bias in Calibration Error Estimation." AISTATS, PMLR 151, 4036–4054.
- Błasiok, J., Nakkiran, P. (2024). "Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing." ICLR.
- Austin, P. C., Steyerberg, E. W. (2019). "The Integrated Calibration Index (ICI) and related metrics..." *Statistics in Medicine* 38(21), 4051–4065.
- Austin, P. C., Steyerberg, E. W. (2014). "Graphical assessment of internal and external calibration of logistic regression models by using loess smoothers." *Statistics in Medicine* 33(3), 517–535.
- Spiegelhalter, D. J. (1986). "Probabilistic prediction in patient management and clinical trials." *Statistics in Medicine* 5(5), 421–433.
- Cox, D. R. (1958). "Two further applications of a model for binary regression." *Biometrika* 45, 562–565.
- Hosmer, D. W., Lemeshow, S. (1980). "Goodness of fit tests for the multiple logistic regression model." *Communications in Statistics — Theory and Methods* 9(10), 1043–1069.
- King, G., Zeng, L. (2001). "Logistic Regression in Rare Events Data." *Political Analysis* 9(2), 137–163.
- Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." IJCAI, 973–978. (Cite the base-rate adjustment as "Elkan's base-rate adjustment"; do not quote a closed-form equation without re-verifying it.)
- Tasche, D. (2013). "The art of probability-of-default curve calibration." *Journal of Credit Risk* 9(4). (arXiv:1212.3716)
- Pluto, K., Tasche, D. (2005). "Estimating Probabilities of Default for Low Default Portfolios." (in *The Basel II Risk Parameters*, Springer; arXiv version exists)
- Nattino, G., Finazzi, S., Bertolini, G. (2014). "A new calibration test and a reappraisal of the calibration belt..." *Statistics in Medicine* 33(14), 2390–2407.
- Nattino, G., Lemeshow, S., Phillips, G., Finazzi, S., Bertolini, G. (2017). "Assessing the Calibration of Dichotomous Outcome Models with the Calibration Belt." *Stata Journal* 17(4), 1003–1014.
- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., Brunk, H. D. (1972). *Statistical Inference under Order Restrictions.* Wiley.
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review* 78(1), 1–3.
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer. (natural cubic spline basis, §5.2.1)
- Lundberg, S. M., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 30.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence* 2(1), 56–67.
- Sundararajan, M., Taly, A., Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." ICML, PMLR 70.
- Aumann, R. J., Shapley, L. S. (1974). *Values of Non-Atomic Games.* Princeton University Press.

⚠ Verify against the primary source before citing (web search during Task 1; log outcome in DECISIONS):
- ⚠ Bröcker, J. (2009). "Reliability, sufficiency, and the decomposition of proper scores." *QJRMS* 135 — confirm volume/pages.
- ⚠ Ferro, C. A. T., Fricker, T. E. (2012). "A bias-corrected decomposition of the Brier score." *QJRMS* — confirm details.
- ⚠ Murphy, A. H. (1973). "A new vector partition of the probability score." *Journal of Applied Meteorology* 12 — confirm pages.
- ⚠ Tibshirani, R. J., Hoefling, H., Tibshirani, R. (2011). "Nearly-Isotonic Regression." *Technometrics* 53(1) — needed for ENIR's mPAVA; confirm.
- ⚠ Arrieta-Ibarra et al. (2022). "Metrics of Calibration for Probabilistic Predictions." *JMLR* 23 — confirm author list before citing ECCE.
- ⚠ Miller, M. E., Hui, S. L., Tierney, W. M. (1991) — recalibration-test lineage for `calibration_test`; if unverifiable, attribute the 2-df LR test to Cox (1958) framing only.
- ⚠ ECB (2019). "Instructions for reporting the validation results of internal models — IRB Pillar I models" (Jeffreys test source) — confirm exact title/year.
- ⚠ BCBS (2005). Working Paper No. 14, "Studies on the Validation of Internal Rating Systems" — confirm.
- ⚠ van der Burgt, M. (2008). "Calibrating low-default portfolios, using the cumulative accuracy profile." *Journal of Risk Model Validation* — confirm volume/pages if cited.
- ⚠ Löfström, H., Löfström, T., Johansson, U., Sönströd, C. "Calibrated Explanations" (uncertainty-aware explanations via Venn–Abers; believed published in *Expert Systems with Applications*, ~2024; arXiv:2305.02305) — verify venue/year before citing as related work.
- ⚠ Upadhyay, S., Joshi, S., Lakkaraju, H. (2021). "Towards Robust and Reliable Algorithmic Recourse." NeurIPS 34 — verify before citing (recourse robustness, §10).
- ⚠ Rawal, K., Kamar, E., Lakkaraju, H. (2020). "Algorithmic Recourse in the Wild: Understanding the Impact of Data and Model Shifts." arXiv:2012.11788 — verify before citing (§10).

---

## 16. Explicit non-goals (v0.0.1)

Multiclass calibration; GP calibration (heavy); torch/GPU paths; Rust acceleration (revisit only
if profiling on real workloads shows PAVA-family pain — see benchmark note in README); pandas
interop beyond `as_dict()`; conformal prediction beyond Venn–Abers; exact Shapley values of the composed model via coalition access or shap-library integration (array-level adjustment only, §9); modifying treecf itself (the interop is one-directional: probcal exposes the protocol, treecf may adopt it later, §10).
