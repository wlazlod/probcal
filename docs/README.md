# Writing docs

Notes for anyone editing the Markdown under `docs/`. This is not part of the
built site (mkdocs never lists it in `nav`); it is here for the same reason
a package keeps a `CONTRIBUTING.md` — instructions for the people editing
the source, not for the reader of the output.

## The snippet convention

Every fenced ` ```python ` block on a docs page is executed by
`tests/test_docs_snippets.py` (marked `slow`), page by page, in one
namespace pre-seeded with a fixed vocabulary of held-out data. That turns
the fragment style — showing a call without repeating a full fit every
time — from a rot risk into a tested contract. Two rules keep it that way:

1. **Every block imports what it names.** A block runs after the ones
   above it on the same page (one namespace per page), so it may reuse a
   name an earlier block on the same page defined, but never a name
   assumed from nowhere.
2. **Elided data uses the fixed vocabulary, named in a first-line
   comment.** When a block needs held-out calibration data without
   re-deriving it, it draws only from this set, and says so:

   ```python
   # docs: no-run — illustrates the convention, not a runnable snippet
   # s_cal, y_cal: held-out calibration scores and outcomes
   ...
   ```

   The vocabulary — and what the test harness seeds it with — is exactly:

   | Name | What it is |
   |---|---|
   | `s_cal` | Held-out calibration scores (`make_pd_portfolio(n=3000, random_state=0).scores`) |
   | `y_cal` | The matching outcomes (`.y`) |
   | `w_cal` | Uniform sample weights (`np.ones_like(y_cal)`) |
   | `model` | A sklearn-free stub with `predict_proba(X)` returning the score in column 0 |
   | `s_new` | Another portfolio's scores, for `predict_proba`/`point_inverse` calls on "new" data |
   | `mon` | A `CalibrationMonitor` with three seeded batches already applied |
   | `grades` | Rating labels derived from `s_cal` (`G1`/`G2`/`G3` by score band) |
   | `segments` | Three segment labels cycling over `s_cal`'s length |

   A page that needs a name outside this set either defines it locally in
   the block (e.g. a feature matrix built from `s_cal`) or, if the
   vocabulary should grow, extends the table above in this one place —
   never invents a page-local convention.

A block that is not meant to run — REPL transcripts (first non-blank line
starting with `>>>`), `--8<--` includes, and genuine pseudo-code that
cannot be made to run without a heavier fixture than the vocabulary
supports — is skipped by the harness automatically for the first two
cases; for pseudo-code, mark it explicitly and say why:

```python
# docs: no-run — target/treecf stand in for a real Explainer and Target
```

Use `# docs: no-run` sparingly — it opts a block out of the tested
contract, so prefer making the block actually run against the vocabulary
whenever that is feasible.

## Pages that need an optional extra

CI runs the harness in a `[dev,viz]` environment, so optbinning and treecf
are absent there. A page whose blocks genuinely need one declares it once,
anywhere in the file:

```markdown
<!-- docs: requires optbinning -->
```

The harness then `importorskip`s each named package and skips the whole
page where it is missing. Declare it only when the *page* is about that
integration (`guide/optbinning.md`); when one block on an otherwise
core page reaches for an extra — the scorecard and treecf blocks in
`guide/cutoffs.md` — mark that block `# docs: no-run` instead, so the rest
of the page stays under test.
