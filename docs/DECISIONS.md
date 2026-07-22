# Implementation Decisions

Resolved ambiguities and design choices made during implementation.
Each entry references the source that drove the decision.

---

1. **Python floor is `>=3.11`; development runs on 3.12.** Rationale: consistency with
   FlagGAM, `typing.Self` for fluent `fit() -> Self`, better tracebacks and performance.
   `mypy` runs with `python_version = "3.11"` to guard against 3.12-only constructs.
   *(spec §1.4; grooming 2026-07-22)*

2. **Input convention: probabilities only.** All calibrators accept scores
   `s ∈ (0,1)`; methods defined on logits (temperature, Platt-on-logits) convert internally
   via `z = logit(s)`. Users holding raw logits convert explicitly with the exported
   `expit`. No `input=` switch on calibrators. *(spec §3 left the logit path open;
   grooming 2026-07-22)*

3. **Task 1 executes as: reference verification first, then four writing chunks.**
   All ⚠ references of spec §15 are web-verified in one pass before any chapter is written;
   outcomes are logged here. Chunks: (1) why-calibration + methods-parametric,
   (2) methods-nonparametric + methods-distribution-free, (3) metrics + data-splitting +
   offset, (4) shap-calibration + inverse-maps + auto-selection + visualization.
   *(spec §14.1; grooming 2026-07-22)*

4. **GitHub repository `wlazlod/probcal` created private; public flip is the owner's
   decision.** While private, `docs.yml` (gh-pages deploy) is dormant; `publish.yml` is
   tag-gated and inactive until `0.1.0` per spec. *(spec §2; grooming 2026-07-22)*

5. **Repository is bound to the Obsidian project knowledge base at scaffold time.**
   Daily notes and reference-verification evidence live in the vault; canonical engineering
   decisions live here. *(grooming 2026-07-22)*

6. **Planning artifacts live in `.plan/` (git-ignored), one dated file per task unit.**
   Mirrors FlagGAM's untracked `plan/` convention, renamed per owner preference.
   *(grooming 2026-07-22)*

7. **Hyperparameters the spec leaves open** (spline knot count/placement and λ grid,
   histogram default bin count, LOESS evaluation grid, bootstrap internals beyond
   n_boot=1000/percentile/seeded) **are decided inside their implementing task**, each with
   its own entry here. *(spec §2 DECISIONS protocol; grooming 2026-07-22)*

8. **Tree deltas vs the FlagGAM template:** no `benchmarks/`, no `scripts/`, no `NOTICE`;
   the spec §2 tree is authoritative. *(spec §2; grooming 2026-07-22)*

9. **Commit conventions:** Conventional Commits; no AI-attribution trailers or taglines in
   any artifact. *(owner's global convention; grooming 2026-07-22)*

10. **⚠ Bröcker (2009) — verified, record completed.** Bröcker, J. (2009). "Reliability,
    sufficiency, and the decomposition of proper scores." *Quarterly Journal of the Royal
    Meteorological Society* 135(643), 1512–1519. DOI 10.1002/qj.456. Confirmed via Crossref
    and the Wiley landing page. *(spec §15; Task 1a, 2026-07-22)*

11. **⚠ Ferro & Fricker (2012) — verified, record completed.** Ferro, C. A. T., Fricker,
    T. E. (2012). "A bias-corrected decomposition of the Brier score." *Quarterly Journal of
    the Royal Meteorological Society* 138(668), 1954–1960. DOI 10.1002/qj.1924. Confirmed via
    Crossref, Wiley, and the author's manuscript copy. *(spec §15; Task 1a, 2026-07-22)*

12. **⚠ Murphy (1973) — verified, record completed.** Murphy, A. H. (1973). "A New Vector
    Partition of the Probability Score." *Journal of Applied Meteorology* 12(4), 595–600.
    DOI 10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2. Cite the journal under its 1973 name
    (now *Journal of Applied Meteorology and Climatology*). Confirmed via Crossref and the AMS
    journal archive. *(spec §15; Task 1a, 2026-07-22)*

13. **⚠ Tibshirani, Hoefling & Tibshirani (2011) — verified, record completed.**
    Tibshirani, R. J., Hoefling, H., Tibshirani, R. (2011). "Nearly-Isotonic Regression."
    *Technometrics* 53(1), 54–61. DOI 10.1198/TECH.2010.10111. Author order as given is
    correct; the published spelling is "Hoefling" (not "Höfling"), per the paper byline,
    Crossref, and Taylor & Francis. Safe to cite for ENIR's modified PAVA. *(spec §15; Task 1a,
    2026-07-22)*

14. **⚠ Arrieta-Ibarra et al. (2022) — verified as given; full author list confirmed.**
    Arrieta-Ibarra, I., Gujral, P., Tannen, J., Tygert, M., Xu, C. (2022). "Metrics of
    Calibration for Probabilistic Predictions." *Journal of Machine Learning Research* 23(351),
    1–54. arXiv:2205.09680. Confirmed via jmlr.org (paper id 22-0658). Safe to cite for ECCE.
    *(spec §15, §7 item 11; Task 1a, 2026-07-22)*

15. **⚠ Miller, Hui & Tierney (1991) — verified; exact paper identified.** Miller, M. E.,
    Hui, S. L., Tierney, W. M. (1991). "Validation techniques for logistic regression models."
    *Statistics in Medicine* 10(8), 1213–1226. DOI 10.1002/sim.4780100805. Confirmed via PubMed
    (PMID 1925153) and Crossref. `calibration_test` may cite Miller et al. for the
    recalibration-test lineage alongside the Cox (1958) framing. *(spec §15, §7 item 16;
    Task 1a, 2026-07-22)*

16. **⚠ van der Burgt (2008) — verified, record completed.** van der Burgt, M. (2008).
    "Calibrating low-default portfolios, using the cumulative accuracy profile." *Journal of
    Risk Model Validation* 1(4), 17–33. DOI 10.21314/JRMV.2008.016. The issue is labeled
    Winter 2007/08 but Crossref and risk.net date publication to 2008 — cite 2008. No arXiv or
    SSRN preprint could be found; cite the journal version only. *(spec §15; Task 1a,
    2026-07-22)*

17. **⚠ Löfström et al. — verified; published record completed.** Löfström, H., Löfström, T.,
    Johansson, U., Sönströd, C. (2024). "Calibrated explanations: With uncertainty information
    and counterfactuals." *Expert Systems with Applications* 246, 123154.
    DOI 10.1016/j.eswa.2024.123154. arXiv:2305.02305 is confirmed to be the same paper.
    Safe to cite as related work in `shap-calibration.md`. *(spec §15, §14.1 item 10; Task 1a,
    2026-07-22)*

18. **⚠ ECB (2019) — verified; subtitle corrected.** The official title is *Instructions for
    reporting the validation results of internal models — IRB Pillar I models for credit
    risk*, European Central Bank Banking Supervision, February 2019 (the spec's version
    omitted "for credit risk"). The Jeffreys PD-backtesting test is confirmed present in the
    document. Only the February 2019 edition exists at the official URL; do not conflate with
    the ECB *Guide to internal models* (a different publication). *(spec §15, §7 item 18;
    Task 1a, 2026-07-22)*

19. **⚠ BCBS (2005) — verified; cite the revised version.** Basel Committee on Banking
    Supervision (2005). *Studies on the Validation of Internal Rating Systems.* Working Paper
    No. 14, revised version, May 2005. Bank for International Settlements. The original
    pre-revision date is not published on bis.org; cite the May 2005 revised version.
    *(spec §15; Task 1a, 2026-07-22)*

20. **⚠ Upadhyay et al. (2021) — verified as given.** Upadhyay, S., Joshi, S., Lakkaraju, H.
    (2021). "Towards Robust and Reliable Algorithmic Recourse." *Advances in Neural
    Information Processing Systems* 34 (NeurIPS 2021), 16926–16937. arXiv:2102.13620.
    Confirmed via the official proceedings page. Note: the page range comes from indexing
    metadata (the proceedings page shows none); no Crossref DOI exists for this proceedings
    entry. Safe to cite for recourse robustness in §10 docs. *(spec §15; Task 1a, 2026-07-22)*

21. **⚠ Rawal, Kamar & Lakkaraju — verified with a title-history caveat; cite as arXiv
    preprint.** arXiv:2012.11788 matches the given title and authors in its current (v2/v3,
    2021) form; v1 (Dec 2020) was titled "Can I Still Trust You?: Understanding the Impact of
    Distribution Shifts on Algorithmic Recourses". Never published at a peer-reviewed venue
    (dblp lists CoRR only). Cite as: Rawal, K., Kamar, E., Lakkaraju, H. (2020). "Algorithmic
    Recourse in the Wild: Understanding the Impact of Data and Model Shifts."
    arXiv:2012.11788. Do not confuse with entry 20. *(spec §15; Task 1a, 2026-07-22)*

22. **`norm_cdf` uses `math.erfc`, not `math.erf`.** The spec's wording (§5.9) says
    "`norm_cdf` via `erf_vec`", but the erf form ``0.5*(1+erf(x/√2))`` loses relative accuracy
    to cancellation in the deep tails, which would break the Halley refinement of `norm_ppf`
    at quantiles like 1e-12. ``0.5*erfc(-x/√2)`` is tail-accurate; `erf_vec` remains available
    as specified. Reference test holds `norm_ppf` to 1e-11 absolute vs scipy on
    (1e-12, 1-1e-12). *(spec §5.9; Task 2, 2026-07-22)*

23. **Separation heuristic in `irls_logistic`:** separation is declared when any fitted
    linear predictor exceeds 30 in absolute value during iteration or the Hessian solve
    fails; the routine then warns and returns a ridge-regularized refit with ridge = 1e-6.
    Thresholds are heuristics chosen to trigger long before float overflow while never firing
    on well-posed calibration fits. *(spec §5.3; Task 2, 2026-07-22)*

24. **`loess` implementation details:** tricube weights over the ``ceil(frac·n)`` nearest
    neighbors, local linear (degree 1) by default with degree 0 supported, evaluated at the
    data points by default (the ICI use case) with an ``xeval`` override for grids. The
    statsmodels lowess comparison is documented loose (5% of response range): window and
    boundary handling differ by design. *(spec §5.10, §13; Task 2, 2026-07-22)*

25. **`_results` dataclass field sets are the Task-2 minimum.** `ReliabilityCurve`,
    `MetricReport`, `SelectionReport`, `Interpretation`, `BeltResult` are defined with the
    fields their consumers (Tasks 7–13) are specified to need; extensions within 0.0.1 are
    allowed and will be recorded here. *(spec §4; Task 2, 2026-07-22)*

26. **`test_no_forbidden_imports` runs in a subprocess.** The in-process variant is
    order-dependent — reference tests legitimately import scipy/sklearn/statsmodels into the
    session — so the invariant "importing probcal pulls no forbidden dependency" is asserted
    in a fresh interpreter. *(spec §13; Task 2, 2026-07-22)*

27. **Beta variant semantics.** ``variant="abm"`` fits ``(a, b, c)`` unconstrained then
    applies the betacal refit strategy; ``"ab"`` ties ``a = b`` with a free intercept —
    exactly logistic recalibration on logits (Platt without target smoothing); ``"a"``
    additionally fixes ``c = 0``, leaving a single free exponent — the temperature family in
    a different parameterization, kept for completeness of the nested hierarchy and uniform
    constraint handling. The spec (§6) named the variants without pinning the tying; the
    theory chapter deliberately deferred to this entry. *(spec §6 row 3; Task 3, 2026-07-22)*

28. **Platt target smoothing uses unweighted class counts.** ``N+`` and ``N-`` in the
    Lin–Lin–Weng targets are raw observation counts; sample weights enter the IRLS fit
    itself. Weighted counts would change the smoothing strength under reweighting, which is
    not what the stabilization is for. *(spec §6 row 1; Task 3, 2026-07-22)*

29. **Temperature fitting bracket.** The NLL score equation is solved for ``u = 1/T`` on
    ``[1e-6, 1e6]``; if the score has no sign change on the bracket (degenerate data), the
    boundary with the smaller score magnitude is taken and a ``UserWarning`` is emitted.
    *(spec §6 row 2; Task 3, 2026-07-22)*
