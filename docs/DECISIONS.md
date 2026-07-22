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
