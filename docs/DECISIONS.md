# Implementation Decisions

Dated log of ambiguities resolved and design choices made during implementation.

---

1. **Python floor is `>=3.11`; development runs on 3.12.** Rationale: consistency with
   FlagGAM, `typing.Self` for fluent `fit() -> Self`, better tracebacks and performance.
   `mypy` runs with `python_version = "3.11"` to guard against 3.12-only constructs.
   *(grooming 2026-07-22)*

2. **Input convention: probabilities only.** All calibrators accept scores
   `s ∈ (0,1)`; methods defined on logits (temperature, Platt-on-logits) convert internally
   via `z = logit(s)`. Users holding raw logits convert explicitly with the exported
   `expit`. No `input=` switch on calibrators. *(grooming 2026-07-22)*

3. **The theory guidebook executed as: reference verification first, then four writing
   chunks.** All flagged (⚠) references are web-verified in one pass before any chapter is written;
   outcomes are logged here. Chunks: (1) why-calibration + methods-parametric,
   (2) methods-nonparametric + methods-distribution-free, (3) metrics + data-splitting +
   offset, (4) shap-calibration + inverse-maps + auto-selection + visualization.
   *(grooming 2026-07-22)*

4. **GitHub repository `wlazlod/probcal` created private; public flip is the owner's
   decision.** While private, `docs.yml` (gh-pages deploy) is dormant; `publish.yml` is
   tag-gated and inactive until `0.1.0`. *(grooming 2026-07-22)*
   — Flipped public on the owner's instruction; GitHub Pages enabled from the `gh-pages`
   branch and the docs deploy activated. *(2026-07-23)*

5. **Repository is bound to the Obsidian project knowledge base at scaffold time.**
   Daily notes and reference-verification evidence live in the vault; canonical engineering
   decisions live here. *(grooming 2026-07-22)*

6. **Planning artifacts live in `.plan/` (git-ignored), one dated file per task unit.**
   Mirrors FlagGAM's untracked `plan/` convention, renamed per owner preference.
   *(grooming 2026-07-22)*

7. **Hyperparameters left open by the original design** (spline knot count/placement and λ grid,
   histogram default bin count, LOESS evaluation grid, bootstrap internals beyond
   n_boot=1000/percentile/seeded) **are decided inside their implementing task**, each with
   its own entry here. *(grooming 2026-07-22)*

8. **Tree deltas vs the FlagGAM template:** no `benchmarks/`, no `scripts/`, no `NOTICE`;
   the planned repository tree is authoritative. *(grooming 2026-07-22)*

9. **Commit conventions:** Conventional Commits; no AI-attribution trailers or taglines in
   any artifact. *(owner's global convention; grooming 2026-07-22)*

10. **⚠ Bröcker (2009) — verified, record completed.** Bröcker, J. (2009). "Reliability,
    sufficiency, and the decomposition of proper scores." *Quarterly Journal of the Royal
    Meteorological Society* 135(643), 1512–1519. DOI 10.1002/qj.456. Confirmed via Crossref
    and the Wiley landing page. *(2026-07-22)*

11. **⚠ Ferro & Fricker (2012) — verified, record completed.** Ferro, C. A. T., Fricker,
    T. E. (2012). "A bias-corrected decomposition of the Brier score." *Quarterly Journal of
    the Royal Meteorological Society* 138(668), 1954–1960. DOI 10.1002/qj.1924. Confirmed via
    Crossref, Wiley, and the author's manuscript copy. *(2026-07-22)*

12. **⚠ Murphy (1973) — verified, record completed.** Murphy, A. H. (1973). "A New Vector
    Partition of the Probability Score." *Journal of Applied Meteorology* 12(4), 595–600.
    DOI 10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2. Cite the journal under its 1973 name
    (now *Journal of Applied Meteorology and Climatology*). Confirmed via Crossref and the AMS
    journal archive. *(2026-07-22)*

13. **⚠ Tibshirani, Hoefling & Tibshirani (2011) — verified, record completed.**
    Tibshirani, R. J., Hoefling, H., Tibshirani, R. (2011). "Nearly-Isotonic Regression."
    *Technometrics* 53(1), 54–61. DOI 10.1198/TECH.2010.10111. Author order as given is
    correct; the published spelling is "Hoefling" (not "Höfling"), per the paper byline,
    Crossref, and Taylor & Francis. Safe to cite for ENIR's modified PAVA. *(2026-07-22)*

14. **⚠ Arrieta-Ibarra et al. (2022) — verified as given; full author list confirmed.**
    Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., Xu, C. (2022). "Metrics of
    Calibration for Probabilistic Predictions." *Journal of Machine Learning Research* 23(351),
    1–54. arXiv:2205.09680. Confirmed via jmlr.org (paper id 22-0658). Safe to cite for ECCE.
    *(2026-07-22)*

15. **⚠ Miller, Hui & Tierney (1991) — verified; exact paper identified.** Miller, M. E.,
    Hui, S. L., Tierney, W. M. (1991). "Validation techniques for logistic regression models."
    *Statistics in Medicine* 10(8), 1213–1226. DOI 10.1002/sim.4780100805. Confirmed via PubMed
    (PMID 1925153) and Crossref. `calibration_test` may cite Miller et al. for the
    recalibration-test lineage alongside the Cox (1958) framing. *(2026-07-22)*

16. **⚠ van der Burgt (2008) — verified, record completed.** van der Burgt, M. (2008).
    "Calibrating low-default portfolios, using the cumulative accuracy profile." *Journal of
    Risk Model Validation* 1(4), 17–33. DOI 10.21314/JRMV.2008.016. The issue is labeled
    Winter 2007/08 but Crossref and risk.net date publication to 2008 — cite 2008. No arXiv or
    SSRN preprint could be found; cite the journal version only. *(2026-07-22)*

17. **⚠ Löfström et al. — verified; published record completed.** Löfström, H., Löfström, T.,
    Johansson, U., Sönströd, C. (2024). "Calibrated explanations: With uncertainty information
    and counterfactuals." *Expert Systems with Applications* 246, 123154.
    DOI 10.1016/j.eswa.2024.123154. arXiv:2305.02305 is confirmed to be the same paper.
    Safe to cite as related work in `shap-calibration.md`. *(2026-07-22)*

18. **⚠ ECB (2019) — verified; subtitle corrected.** The official title is *Instructions for
    reporting the validation results of internal models — IRB Pillar I models for credit
    risk*, European Central Bank Banking Supervision, February 2019 (earlier internal records
    omitted "for credit risk"). The Jeffreys PD-backtesting test is confirmed present in the
    document. Only the February 2019 edition exists at the official URL; do not conflate with
    the ECB *Guide to internal models* (a different publication). *(2026-07-22)*

19. **⚠ BCBS (2005) — verified; cite the revised version.** Basel Committee on Banking
    Supervision (2005). *Studies on the Validation of Internal Rating Systems.* Working Paper
    No. 14, revised version, May 2005. Bank for International Settlements. The original
    pre-revision date is not published on bis.org; cite the May 2005 revised version.
    *(2026-07-22)*

20. **⚠ Upadhyay et al. (2021) — verified as given.** Upadhyay, S., Joshi, S., Lakkaraju, H.
    (2021). "Towards Robust and Reliable Algorithmic Recourse." *Advances in Neural
    Information Processing Systems* 34 (NeurIPS 2021), 16926–16937. arXiv:2102.13620.
    Confirmed via the official proceedings page. Note: the page range comes from indexing
    metadata (the proceedings page shows none); no Crossref DOI exists for this proceedings
    entry. Safe to cite for recourse robustness in §10 docs. *(2026-07-22)*

21. **⚠ Rawal, Kamar & Lakkaraju — verified with a title-history caveat; cite as arXiv
    preprint.** arXiv:2012.11788 matches the given title and authors in its current (v2/v3,
    2021) form; v1 (Dec 2020) was titled "Can I Still Trust You?: Understanding the Impact of
    Distribution Shifts on Algorithmic Recourses". Never published at a peer-reviewed venue
    (dblp lists CoRR only). Cite as: Rawal, K., Kamar, E., Lakkaraju, H. (2020). "Algorithmic
    Recourse in the Wild: Understanding the Impact of Data and Model Shifts."
    arXiv:2012.11788. Do not confuse with entry 20. *(2026-07-22)*

22. **`norm_cdf` uses `math.erfc`, not `math.erf`.** The original design called for
    `norm_cdf` via `erf_vec`, but the erf form ``0.5*(1+erf(x/√2))`` loses relative accuracy
    to cancellation in the deep tails, which would break the Halley refinement of `norm_ppf`
    at quantiles like 1e-12. ``0.5*erfc(-x/√2)`` is tail-accurate; `erf_vec` remains available
    as specified. Reference test holds `norm_ppf` to 1e-11 absolute vs scipy on
    (1e-12, 1-1e-12). *(2026-07-22)*

23. **Separation heuristic in `irls_logistic`:** separation is declared when any fitted
    linear predictor exceeds 30 in absolute value during iteration or the Hessian solve
    fails; the routine then warns and returns a ridge-regularized refit with ridge = 1e-6.
    Thresholds are heuristics chosen to trigger long before float overflow while never firing
    on well-posed calibration fits. *(2026-07-22)*

24. **`loess` implementation details:** tricube weights over the ``ceil(frac·n)`` nearest
    neighbors, local linear (degree 1) by default with degree 0 supported, evaluated at the
    data points by default (the ICI use case) with an ``xeval`` override for grids. The
    statsmodels lowess comparison is documented loose (5% of response range): window and
    boundary handling differ by design. *(2026-07-22)*

25. **`_results` dataclass field sets are the Task-2 minimum.** `ReliabilityCurve`,
    `MetricReport`, `SelectionReport`, `Interpretation`, `BeltResult` are defined with the
    fields their downstream consumers need; extensions within 0.0.1 are
    allowed and will be recorded here. *(2026-07-22)*

26. **`test_no_forbidden_imports` runs in a subprocess.** The in-process variant is
    order-dependent — reference tests legitimately import scipy/sklearn/statsmodels into the
    session — so the invariant "importing probcal pulls no forbidden dependency" is asserted
    in a fresh interpreter. *(2026-07-22)*

27. **Beta variant semantics.** ``variant="abm"`` fits ``(a, b, c)`` unconstrained then
    applies the betacal refit strategy; ``"ab"`` ties ``a = b`` with a free intercept —
    exactly logistic recalibration on logits (Platt without target smoothing); ``"a"``
    additionally fixes ``c = 0``, leaving a single free exponent — the temperature family in
    a different parameterization, kept for completeness of the nested hierarchy and uniform
    constraint handling. The variant names were fixed before their exact tying; the
    theory chapter deliberately deferred to this entry. *(2026-07-22)*

28. **Platt target smoothing uses unweighted class counts.** ``N+`` and ``N-`` in the
    Lin–Lin–Weng targets are raw observation counts; sample weights enter the IRLS fit
    itself. Weighted counts would change the smoothing strength under reweighting, which is
    not what the stabilization is for. *(2026-07-22)*

29. **Temperature fitting bracket.** The NLL score equation is solved for ``u = 1/T`` on
    ``[1e-6, 1e6]``; if the score has no sign change on the bracket (degenerate data), the
    boundary with the smaller score magnitude is taken and a ``UserWarning`` is emitted.
    *(2026-07-22)*

30. **Venn–Abers batch prediction refits per unique query score.** Each unique query costs
    two PAVA fits on n+1 points after deduplication — exact and simple, fast at the
    calibration-set sizes this package targets. The O((n+m)log(n+m)) precomputed-envelope
    batch algorithm of Vovk & Petej is a planned optimization, to be adopted only if
    profiling on real workloads shows this path as a bottleneck (see the README
    performance note).
    The theory chapter's computational note was amended to match. *(2026-07-22)*

31. **CVAP `predict_interval` returns the conservative fold envelope** ``[min_k p0_k,
    max_k p1_k]``. Vovk & Petej define only the scalar geometric-mean merge for CVAP; the
    envelope is a deliberate, conservative summary, and the docstring points the validity
    guarantee at the per-fold IVAP intervals. *(2026-07-22)*

32. **Isotonic step semantics.** Tied scores are pooled (weighted) before PAVA; the step
    map is right-continuous with boundaries at each block's first score; out-of-range
    queries clamp to the terminal block levels; ``interpolation="linear"`` joins block
    midpoints ``(first_s + last_s)/2``. CIR interpolates through weight-centered block
    coordinates. *(2026-07-22)*

33. **Histogram binning defaults and edge cases.** Default ``n_bins=10``, ``strategy="mass"``,
    Jeffreys shrinkage on. Equal-mass quantile edges are deduplicated (heavy ties can reduce
    the effective bin count); empty bins under the ``"width"`` strategy fall back to the
    global weighted event rate. ``is_monotone_`` is computed after fitting — binning does not
    enforce monotonicity, so the flag reports the fitted rates' actual ordering. *(2026-07-22)*

34. **BBQ candidate range and prior.** Candidate bin counts default to
    ``B ∈ [2, ceil(sqrt(n))]`` capped at 50; per-bin prior is the Jeffreys Beta(1/2, 1/2),
    consistent with the package's other Jeffreys usages (shrinkage, grade test). Posterior
    weights are the softmax of Beta-Binomial log marginal likelihoods; predictions average
    posterior-mean bin rates. *(2026-07-22)*

35. **ENIR path and ensemble details.** Ties are aggregated before the path; the
    nearly-isotonic path is computed by modified PAVA with collision events for both
    violating pairs closing and non-violating pairs driven together by outer violations;
    solutions are recorded at every breakpoint from λ=0 (raw data) to the fully isotonic
    fit (verified equal to PAVA in tests). BIC uses the binomial log-likelihood with
    probabilities clipped to [1e-12, 1-1e-12] and k = number of distinct fitted levels.
    *(2026-07-22)*

36. **Spline calibrator defaults.** Knots at equally spaced quantiles of the logit scores,
    ``K = clip(ceil(n^(1/3)), 4, 12)`` by default; penalty grid ``logspace(-4, 4, 17)``;
    second-difference penalty on the coefficient sequence of the natural cubic basis;
    effective d.o.f. = trace((B'WB + λP)^{-1} B'WB) at convergence. Monotonicity is checked
    on a dense probe grid post-fit and a non-monotone fit warns rather than errors — the
    penalty does not enforce shape. *(2026-07-22)*

37. **`evaluate` lives in `metrics/__init__.py`.** Spec §4 annotates the metrics
    ``__init__`` as "flat re-exports"; `evaluate` aggregates every submodule, so the
    package root is its least-coupled home — the smallest deviation from the annotation.
    *(2026-07-23)*

38. **Debiased ECE estimator.** Per non-empty bin, the squared gap is corrected by the
    unbiased variance of the bin event rate, ``max(gap² − ȳ_b(1−ȳ_b)/(n_b−1), 0)``, and the
    reported value is the weight-averaged square root — a correction in the spirit of
    Bröcker (2009) / Ferro & Fricker (2012), floored at zero per bin. The same within-bin
    variance correction backs ``murphy_decomposition(bias_corrected=True)``. *(2026-07-23)*

39. **Log-loss calibration/refinement plug-in.** The recalibration curve c(p) is estimated
    by LOESS (frac 0.75, consistent with the ICI family); calibration is the mean
    KL(Bernoulli(c)‖Bernoulli(p)) and refinement the mean entropy of Bernoulli(c). *(2026-07-23)*

40. **smoothECE implementation.** Residuals are smoothed with a plain Gaussian kernel on
    the logit scale over a 257-point grid spanning the data ±5σ; the paper's reflected
    kernel is a boundary device for [0,1] and is a no-op on the unbounded logit scale. The
    bandwidth is the self-consistent fixed point smECE(σ) = σ found by 40-step bisection on
    [1e-4, 2]. ``ece_sweep`` scans equal-mass B from 2 to min(n, 100) and keeps the largest
    B with monotone bin event rates. *(2026-07-23)*

41. **Weight handling edge cases in metrics.** The LOESS stage of the ICI family is
    unweighted (weights enter the averaging of distances); e50/e90/emax use unweighted
    quantiles of the distances. Grade tests use raw integer counts — non-uniform sample
    weights are ignored with a UserWarning, because exact binomial and Jeffreys tests are
    defined on counts. Traffic lights: green > 0.05, amber > 0.01, red ≤ 0.01. *(2026-07-23)*

42. **Calibration-belt band is the information-matrix (Wald) ellipsoid approximation.**
    The Nattino construction inverts the LR region; probcal draws the pointwise
    band as η(t) ± sqrt(χ²_q(m+1) · x(t)ᵀ I⁻¹ x(t)) at the fitted polynomial — the standard
    Wald approximation of that inversion, asymptotically equivalent and numerically robust.
    Degree selection: forward LR (add a term while p < 0.05), capped at 4. The associated
    p-value is the LR test of the fitted degree-m polynomial against the identity map with
    df = m + 1 (a simplification of Nattino's selection-adjusted test, documented here).
    *(2026-07-23)*

43. **Curve grids span the 0.5%–99.5% quantiles of the predictions** (smoothed reliability
    curves and the belt): extrapolating a smoother beyond the observed score range invites
    overreading exactly where there is no data. `SmoothReliabilityCurve` added to
    `_results` per the extension clause of entry 25. *(2026-07-23)*

44. **`LogitOffset` is scores-only and lives outside the `BaseCalibrator` hierarchy.** Its
    `fit(p)` needs no outcomes (mode A is a constant; mode B matches a target mean), so
    forcing the `fit(s, y)` contract would demand a fake `y`. It duck-types the parts that
    matter downstream (`predict_proba`/`transform`, `affine_logit_coeffs_`, `interpret`,
    `is_monotone_`). The mode-B bisection bracket is ±40 log-odds — beyond the ±27.6 range
    that 1e-12 clipping permits, so the root is always interior. *(2026-07-23)*

45. **Attribution adjustment constants.** Degenerate rows (|s − s₀| < 1e-8 on the working
    scale) replace the ill-conditioned secant with a central-difference local slope at the
    base value, step h = 1e-4. In the Aumann–Shapley path the adjusted base is set to
    ``target − Σφ'`` (equal to g(s₀) on regular rows by telescoping), which zeroes the
    reconstruction error on degenerate rows as well. *(2026-07-23)*

46. **Venn–Abers inverse maps use monotone bisection, not searchsorted.** An earlier
    design note assigned scalarized Venn–Abers to the block-structure path, but the scalarized map
    has no *static* block structure — every query re-augments the two isotonic fits with
    the query point itself. An 80-step monotone bisection on the (tested-monotone)
    scalarized map locates the preimage boundary to ~1e-24 in score space instead. The
    same generic bisection serves beta and monotone splines. *(2026-07-23)*

47. **Inverse-map boundary and range conventions.** ``raw_lo = 0`` whenever the buffered
    lower target does not exceed the map's minimum (everything qualifies), and
    symmetrically ``raw_hi = 1``; on the logit scale these become ∓inf. The isotonic step
    map's ``raw_hi`` is the *next* block's left edge (the sup of the qualifying region,
    an exclusive boundary), or 1.0 for the terminal block. A Platt fit with slope ≤ 0
    (pathological data) now sets ``is_monotone_ = False`` and is refused by
    ``interval_inverse`` rather than inverted into nonsense. *(2026-07-23)*

48. **Wrapper duck-typing and offset anchoring.** Model cloning in the cv flow tries
    ``sklearn.base.clone`` through a guarded runtime import (sklearn is never a
    module-level import) and falls back to ``copy.deepcopy``. Models exposing only
    ``decision_function`` have their margins mapped through ``expit`` so the calibrator
    receives probabilities per the input convention (logit-based calibrators recover the
    margin exactly). ``offset_to`` without an explicit ``X`` anchors the target mean on the
    stored calibration scores (out-of-fold scores in the cv flow) — the portfolio the
    calibrator was fitted on. ``interval_inverse`` is undefined for ``ensemble=True``
    (K distinct maps have no single preimage) and raises NotImplementedError. *(2026-07-23)*

49. **Selector tie rule and parsimony ranks.** Candidates whose out-of-fold mean lies
    within one standard error (sd_best/√K) of the best mean are tied; the tie goes to the
    lowest parsimony rank (temperature 1 < beta_a 1.5 < platt 2 < beta_ab 2.5 < beta_abm 3
    < scaling_binning 4 < histogram 10 < spline 12 < BBQ 40 < isotonic/CIR 50 < IVAP/CVAP
    60 < ENIR 80; unknown names last). Ranks order model complexity classes; exact values
    are inert beyond their ordering. *(2026-07-23)*

50. **API reference split into three pages; notebook execution stack.** Rendering all 21
    mkdocstrings module blocks on one page triggers a superlinear blowup in the rendering
    stack (12 blocks ≈ 5 s, 16 ≈ 23 s, 18 ≈ 81 s, 20+ > 5 min — measured), independent of
    which modules; `docs/api/` therefore carries three pages of ≤ 8 blocks each (5 s total,
    strict-clean) with `api.md` as the index. This deviates from the originally planned
    single `api.md` page for cause. `ipykernel` was added to the docs extra to execute the committed
    tutorial notebook. *(2026-07-23)*

51. **`make_pd_portfolio` generative design.** Scores are logit-normal; the true
    probability follows the beta-calibration family with exponents
    ``a_lo = slope·(1+asymmetry)`` (low tail) and ``a_hi = slope``, and the intercept is
    anchored by bisection so that ``mean(p_true)`` equals ``event_rate`` exactly. With
    ``slope=1, asymmetry=0, intercept=0`` the scores are exactly calibrated. On the 3%
    default portfolio the high-tail exponent is deliberately unidentifiable (no data
    there) — the events-per-parameter lesson of the data-splitting chapter, exercised by
    the tests. *(2026-07-23)*

52. **Release 0.1.0 packaging decisions.** Entry 4's condition ("`publish.yml` is
    tag-gated and inactive until `0.1.0`") is now satisfied; the `v0.1.0` tag activates it.
    Four choices made at the release gate:
    (a) *PEP 639 license metadata* — `license = "MIT"` as an SPDX expression plus
    `license-files = ["LICENSE"]`, replacing the deprecated `license = { text = ... }`
    table, and the `License :: OSI Approved :: MIT License` classifier is dropped because
    PEP 639 deprecates license classifiers and build backends reject both forms together.
    This emits Metadata 2.4, which PyPI has accepted since 2024; hatchling has supported
    the SPDX form since 1.27 and `uv build` resolves it fresh in an isolated build env, so
    no floor pin is added to `[build-system]`.
    (b) *`Development Status :: 4 - Beta`* — the 248-test suite, the executed tutorial
    notebook, and the strict-mode docs build make Alpha an understatement; Beta claims a
    usable API without promising 1.0 stability.
    (c) *Python 3.13 classifier* — advertised only because `ci.yml` now runs the suite on
    3.13, not on the strength of the pure-numpy runtime alone.
    (d) *Tag-vs-version check* — the `build` job compares `uvx hatchling version` against
    `${GITHUB_REF_NAME#v}` and fails the run on a mismatch. A `v0.1.0` tag on a `0.0.1`
    `pyproject.toml` would otherwise publish a silently mislabelled artifact under a
    version number PyPI never lets you reuse. Reading the version through the build backend
    rather than grepping the file keeps the check correct if the version later becomes
    dynamic.
    A test job was also added ahead of `build`: before this release nothing in CI ran the
    suite, so a tag could publish code GitHub had never tested. *(2026-08-07)*

53. **SKCE design decisions (`probcal.metrics.kernel`).** Four choices made while
    implementing Widmann, Lindsten & Zachariah's (2019) squared kernel calibration error:
    (a) *Bootstrap fidelity* — `skce_test(method="bootstrap")` uses the Appendix G
    Arcones–Giné centered resampling, NOT the Rademacher wild bootstrap that other
    implementations substitute. probcal implements what its cited source states.
    (b) *Seeded permutation before the `"ul"` pairing* — the paper pairs consecutive
    observations of an i.i.d. sample, but a score-sorted array (routine in credit
    portfolios) silently breaks the independence of pair terms that Corollary G.3's CLT
    requires. probcal permutes with `default_rng(random_state)` first, then pairs
    `perm[0::2]` with `perm[1::2]` (dropping the odd tail). A fixed permutation of an
    i.i.d. sample is still i.i.d., so unbiasedness is unaffected — tested by matching the
    mean of `"ul"` over many pairings against `"uq"`.
    (c) *No `sample_weight`* — the cited U-statistic theory (unbiasedness, the degenerate
    limit of Theorem G.2, the H.3/H.4 bounds) is stated for unweighted i.i.d. samples.
    Refusing the argument is honest; improvising weighted inference is not. The parameter
    is not accepted at all, with the reason in the module docstring.
    (d) *Deterministic median heuristic* — `bandwidth=None` takes the median of pairwise
    absolute kernel-input differences; above 4096 points an evenly strided subsample of
    ≤ 4096 points (no RNG) keeps the O(n) `"ul"` path off an O(n²) distance matrix and
    results bit-reproducible. A zero median (heavy score ties) falls back to the mean
    pairwise distance; all-identical scores raise a `ValueError` instructing the user to
    pass `bandwidth` explicitly.
    `skce` joins the selection table as "never — it is a test" (alongside Spiegelhalter's
    z) and is NOT added to `evaluate()`'s bootstrap catalog: case-resampling CIs around an
    O(n²) statistic multiply cost without adding decision value over the test's calibrated
    p-values. MMCE (Kumar et al., 2018) is a special case of the SKCE (Example I.1) and is
    therefore not implemented separately. *(2026-08-08)*

54. **mypy targets 3.12 (config and CI).** `[tool.mypy] python_version` moves from
    `"3.11"` to `"3.12"`, and the CI mypy step's gate moves from the 3.11 to the 3.12
    matrix job. numpy ≥ 2.x type stubs use PEP 695 `type` statements, which mypy only
    parses when the *target* version is ≥ 3.12, so a 3.11-targeted run can die inside
    numpy's own `.pyi` before checking any probcal code (observed with mypy 1.x; mypy
    2.3.0 happens to tolerate it locally, but the target should not depend on that).
    Runtime 3.11 support is unaffected — the full test suite still runs on the 3.11
    matrix job, which is the guard against 3.12-only *runtime* constructs. The residual
    risk — 3.12-only *typing* constructs passing mypy — is accepted and stated.
    *(2026-08-08)*

55. **Visualization overhaul decisions (0.1.1 part 2).** Six choices:
    (a) *rc-context styling* — the house style (`_STYLE`, muted six-color palette) is
    applied via `matplotlib.pyplot.rc_context` inside each plot function and never by
    mutating global `rcParams`; a user's own matplotlib configuration survives calling
    probcal, verified by a test comparing `rcParams` before and after.
    (b) *Rug replaces count bars by default* — `plot_reliability` gains a per-class
    event/non-event rug and the twin-axis count margin flips to opt-in `counts=False`.
    The rug shows the same density information at the data's own coordinates without a
    second y-scale, and the margin remains one keyword away.
    (c) *Pointwise-band honesty* — `plot_ecce`'s grey envelope is ±2 pointwise SDs of
    the walk under calibration and is labeled "(pointwise)" on the canvas itself; a
    simultaneous band (the formal max-statistic test of Arrieta-Ibarra et al.) is out
    of scope for this release and the docstring and docs say so.
    (d) *Display intervals on grade results* — `ci_low`/`ci_high` are the central 90%
    Jeffreys posterior interval and the 90% Clopper–Pearson interval respectively,
    display-only companions for `plot_grade_backtest`; the traffic lights keep coming
    from the unchanged one-sided tests, and no p-values are printed on the canvas.
    (e) *`beta_ppf` by bisection* — the Beta quantile reuses the existing `bisect` on
    `betainc` (the `chi2_ppf` pattern), preserving the numpy-only runtime; verified
    against hand anchors and `scipy.stats.beta.ppf`.
    (f) *Deterministic reduction everywhere* — anywhere plotting thins data (the rug's
    ≤1000 marks per class) the subsample is sort-then-stride with no RNG, so identical
    inputs always render identical figures; `docs/scripts/generate_figures.py` extends
    the same principle to the committed documentation images. *(2026-08-08)*

56. **Repository governance after the 0.1.1 release.** `main` is protected by an active
    repository ruleset: changes land only via pull request with the full CI matrix
    (`Test on Python 3.11/3.12/3.13`) green and the branch up to date; force pushes and
    deletion are blocked. Required approvals are 0, not 1 — GitHub forbids approving
    one's own PR, so a solo maintainer would be locked out entirely; the count should be
    raised to 1 as soon as a second maintainer exists. The repository-admin role may
    bypass, and every bypass is visibly logged — the protection is against accidents and
    compromised automation, not against the owner. A second ruleset protects `v*` tags
    from deletion and movement with NO bypass: the publish workflow is tag-gated and
    PyPI never allows a version number to be reused, so moving a released tag is never
    correct. Also enabled: secret scanning with push protection, Dependabot alerts and
    security updates plus a weekly `dependabot.yml` (github-actions + pip), private
    vulnerability reporting with `SECURITY.md` pointing at it, auto-delete of merged PR
    branches, and squash/merge-commit as the allowed merge methods (rebase-merge off;
    squash inherits the conventional-commit PR title). *(2026-08-08)*

57. **`irls_logistic` descends monotonically and declares separation only for binary
    targets.** The `max|eta| > 30` abort conflated genuine separation with legitimately
    steep fitted maps: scores clipped to `[1e-12, 1-1e-12]` bound `|logit(s)|` by 27.6,
    so any slope above ~1.1 tripped the cap and the caller silently received an
    unconverged interior iterate (true a = 1.5 on z ~ N(0, 8) → v0.1.1 Platt returned
    a ≈ 1.18 with a spurious separation warning). Three choices, per `IRLS_SPEC.md`:
    (a) *eta-cap removal with step-halving* — Newton steps are halved until the
    overflow-safe softplus objective does not increase, so the iteration never diverges
    and steep maps are simply fitted; (b) *binary-target-only detection* — separation is
    declared only when all targets are within 1e-9 of {0, 1} on unpenalized fits, via
    every observation correct by > 10 log-odds *from the design's own contribution*
    (`eta - offset`: a discriminating offset must not trip the rule — the MLE it leaves
    for the coefficients can exist) with `‖grad‖∞ > 1e-8·(1+|nll|)` (the signature of a
    nonexistent MLE), a singular Hessian, or an unconverged exit on such a fit (the
    divergence signature of quasi-separation: tied boundary points hold the per-point
    margin near zero while the Hessian stays numerically nonsingular, so neither of the
    first two triggers fires at realistic sample sizes); soft targets (Platt's
    Lin–Lin–Weng smoothing) make the objective coercive, so for them separation is a
    category error and is never raised — the ridge-1e-6 fallback is coercive too and
    must report `converged=True`; (c) *convergence surfaced, not swallowed* — Platt and
    beta store `converged_` (beta also `separation_fallback_`), warn distinctly on
    non-convergence, record both in `interpret()`, and `calibration_belt` stops degree
    extension at a separated fit instead of consuming its coefficients.
    `spline._penalized_irls` is out of scope: its `lam * penalty` term already
    regularizes. *(2026-08-12)*

58. **The ICI family evaluates its LOESS smoother at 512 equal-mass anchors by
    default, not at every observation.** `loess`'s new `grid_size` argument fits at
    `grid_size` quantile points of the eval set (endpoints included) and linearly
    interpolates the rest; windows and bandwidths are still computed against the full
    data, so this is an interpolation device, not a subsampling one. The precedent is R
    `stats::lowess`'s `delta` parameter, which fits at points at least `delta` apart
    and interpolates between them by the same logic. `ici`/`e50`/`e90`/`emax` and
    `reliability_summary` all default to `grid_size=512`; `grid_size=None` recovers the
    exact pre-0.1.3 fit-at-every-point behavior and cost. Measured effect on
    `make_pd_portfolio(n=5000)`: `|Δici| ≈ 1.3e-6`, far below the bootstrap CI width at
    that sample size. Separately, `_loess_fit_sorted` was rewritten from an
    `argpartition`-based r-nearest-neighbor search to a sorted two-pointer window walk:
    for sorted 1-D eval points the r-nearest-neighbor window is contiguous, so the
    window start advances monotonically as the eval point advances, turning an
    O(n log n)-per-point search into amortized O(1). The rewrite changes results only
    at exact distance ties between the leftmost and rightmost candidate windows, which
    it resolves to the leftmost minimal-width window (strict `<` in the advance
    condition) — a deterministic tie rule, not a behavior regression. *(2026-08-16)*

59. **`smooth_ece` smooths a pre-binned residual measure instead of the raw
    per-observation one.** Each bisection step of the self-consistent bandwidth solve
    built a 257 x n kernel matrix; the new `bins` argument (default 8192) aggregates
    the weighted `y - p` residuals onto equal-width bins over the logit range before
    solving, cutting the per-step cost to 257 x bins. A small-bandwidth guard protects
    accuracy: if the fixed-point `sigma` found on the initial binning is smaller than
    8 bin widths (the kernel would be under-resolved by the bins), the solve repeats
    once on an 8x finer binning; if that guard still trips, the function silently falls
    back to the exact O(n)-per-step computation with no warning, matching pre-0.1.3
    worst-case cost. `bins=None`, or whenever `n <= bins`, or a degenerate logit range
    (`t.max() == t.min()`), is bit-identical to the pre-0.1.3 exact path — no binning
    is ever imposed where it wouldn't reduce cost. *(2026-08-16)*

60. **`evaluate` accepts a keyword-only `metrics=` subset of the catalog.** Passing a
    sequence of catalog names computes and bootstraps only those metrics — the
    dominant cost of `evaluate` is paying every metric's point-estimate cost `n_boot`
    times, so a caller who only needs `log_loss` and `ici` no longer pays for
    `ece_sweep`'s ~99-candidate scan on every replicate. `metrics=None` (the default)
    computes the full catalog. Unknown names raise `ValueError` listing the valid
    catalog rather than silently ignoring them. The returned `MetricReport` always
    follows catalog order, regardless of the order names were given in `metrics=`, so
    report layout stays stable across call sites. *(2026-08-16)*
