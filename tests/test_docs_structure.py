"""Structural acceptance for the docs reorganization.

Five properties the reader-oriented structure has to keep:

1. no URL disappeared — the v0.3.0 sitemap's page set is a subset of the new one;
2. the front page still names every capability it claims (a staleness tripwire);
3. every public plot function has a rendered, alt-texted figure on a prose page;
4. the sample validation report exists, is linked, and is self-contained;
5. the three workflow pages added by the reorganization are in nav and cross-link.

Plus one guard for `guide/choosing.md`, whose factual columns each name the test
that pins them: those node ids have to resolve; and one for the API reference,
which has to render every public symbol of every public module.
"""

import ast
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
_DOCS = _ROOT / "docs"
_INDEX = _DOCS / "index.md"
_SAMPLE_HTML = _DOCS / "assets" / "sample_validation_report.html"
_SAMPLE_PNG = _DOCS / "assets" / "sample_validation_report.png"
_NEW_PAGES = ("guide/choosing.md", "guide/cutoffs.md", "guide/auditability.md")

# Public modules the API reference must render in full (their `__all__`), plus the
# two `probcal._math` helpers exported flat from `probcal`.
_API_MODULES = (
    "probcal.base",
    "probcal.parametric",
    "probcal.isotonic",
    "probcal.binning",
    "probcal.bayesian",
    "probcal.vennabers",
    "probcal.spline",
    "probcal.segmented",
    "probcal.metrics",
    "probcal.offset",
    "probcal.wrapper",
    "probcal.selection",
    "probcal.curves",
    "probcal.report",
    "probcal.attribution",
    "probcal.thresholds",
    "probcal.masterscale",
    "probcal.datasets",
    "probcal.chain",
    "probcal.monitor",
    "probcal.sklearn",
    "probcal.integrations.optbinning",
)
_API_EXTRA_SYMBOLS = ("probcal._math.expit", "probcal._math.logit")

# The snippet vocabulary (docs/README.md, mirrored by tests/test_docs_snippets.py).
_VOCABULARY = frozenset({"s_cal", "y_cal", "w_cal", "model", "s_new", "mon", "grades", "segments"})
_VOCAB_INCLUDE = '--8<-- "docs/_snippets/vocab.md"'
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)

# Figure coverage: the noun each plot function's figure must be identifiable by,
# in the embed's filename stem or (as a whole word) in its alt text.
_PLOT_NOUNS = {
    "plot_reliability": ("reliability",),
    "plot_belt": ("belt",),
    "plot_comparison": ("comparison",),
    "plot_interval": ("interval",),
    "plot_selection": ("selection",),
    "plot_ecce": ("ecce",),
    "plot_grade_backtest": ("grade_backtest", "backtest"),
    "plot_offset_audit": ("offset_audit", "offset audit"),
    "plot_e_process": ("e_process", "e-process"),
    "plot_corp": ("corp",),
    "plot_mcb_dsc": ("mcb_dsc", "mcb-dsc"),
    "plot_attributes": ("attributes",),
    "plot_murphy": ("murphy",),
}

_EMBED_RE = re.compile(r"!\[([^\]]*)\]\(((?:[^)]*?/)?img/[^)]+\.png)\)")


def _prose_pages() -> list[pathlib.Path]:
    """Every built Markdown page (notebooks and the contributor note excluded)."""
    return sorted(p for p in _DOCS.rglob("*.md") if p.name != "README.md")


@pytest.fixture(scope="module")
def built_site(tmp_path_factory) -> pathlib.Path:
    if shutil.which("mkdocs") is None:
        pytest.skip("mkdocs is not installed (the [docs] extra)")
    out = tmp_path_factory.mktemp("site")
    subprocess.run(
        ["mkdocs", "build", "--strict", "-d", str(out)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    )
    return out


def test_every_v0_3_0_url_still_resolves(built_site: pathlib.Path) -> None:
    expected = {
        line.strip()
        for line in (_ROOT / "tests" / "data" / "docs_pages_v0_3_0.txt").read_text().splitlines()
        if line.strip()
    }
    sitemap = (built_site / "sitemap.xml").read_text()
    built = {
        re.sub(r"^https?://[^/]+", "", loc) for loc in re.findall(r"<loc>(.*?)</loc>", sitemap)
    }
    assert expected <= built, f"URLs lost by the reorganization: {sorted(expected - built)}"


@pytest.mark.parametrize(
    "term",
    ["monitoring", "serialization", "conservatism", "scikit-learn", "optbinning", "treecf"],
)
def test_front_page_names_every_capability(term: str) -> None:
    assert term in _INDEX.read_text().lower()


def test_every_plot_function_has_a_figure_on_a_prose_page() -> None:
    plots = pytest.importorskip("probcal.plots")
    functions = [name for name in plots.__all__ if name.startswith("plot_")]
    assert set(functions) == set(_PLOT_NOUNS), "extend _PLOT_NOUNS when adding a plot function"

    embeds = [match for page in _prose_pages() for match in _EMBED_RE.findall(page.read_text())]
    assert embeds, "no figure embeds found at all"

    for function in functions:
        nouns = _PLOT_NOUNS[function]
        found = any(
            noun in pathlib.PurePosixPath(target).stem.lower()
            or re.search(rf"\b{re.escape(noun)}\b", alt.lower())
            for alt, target in embeds
            for noun in nouns
        )
        assert found, f"{function} has no alt-texted figure embedded in a prose docs page"


def test_sample_validation_report_is_linked_and_self_contained() -> None:
    assert _SAMPLE_HTML.exists(), "run docs/scripts/generate_sample_report.py"
    assert _SAMPLE_PNG.exists(), "the thumbnail is written by the same script"
    report_md = (_DOCS / "guide" / "report.md").read_text()
    assert "assets/sample_validation_report.html" in report_md
    assert "assets/sample_validation_report.png" in report_md
    text = _SAMPLE_HTML.read_text()
    assert "http" not in text
    assert "<script" not in text


def test_new_workflow_pages_are_in_nav_and_cross_link() -> None:
    nav = (_ROOT / "mkdocs.yml").read_text()
    for page in _NEW_PAGES:
        assert page in nav, f"{page} is not in the mkdocs nav"

    for page in _NEW_PAGES:
        text = (_DOCS / page).read_text()
        for other in _NEW_PAGES:
            if other == page:
                continue
            assert pathlib.PurePosixPath(other).name in text, f"{page} does not link {other}"


def test_choosing_page_pinning_ids_resolve() -> None:
    ids = re.findall(r"<!--\s*pinned:\s*(.*?)-->", (_DOCS / "guide" / "choosing.md").read_text())
    node_ids = sorted({part.strip() for entry in ids for part in entry.split(";") if part.strip()})
    assert node_ids, "the catalog rows carry no pinning comments any more"

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    missing = [node_id for node_id in node_ids if node_id not in collected]
    assert not missing, f"guide/choosing.md cites tests that no longer exist: {missing}"


def _public_names(module_name: str) -> list[str]:
    """Public names of a module, read from its source without importing it.

    `__all__` when declared, else every public top-level class and function. Static
    on purpose: `probcal.sklearn` and `probcal.integrations.optbinning` raise
    ImportError without their extra, and the docs job does not install those.
    """
    path = _ROOT / "src" / pathlib.Path(*module_name.split("."))
    source = (path / "__init__.py") if path.is_dir() else path.with_suffix(".py")
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_")
    ]


def test_api_reference_renders_every_public_symbol(built_site: pathlib.Path) -> None:
    rendered = "".join(p.read_text() for p in (built_site / "api").rglob("index.html"))
    anchors = set(re.findall(r'id="(probcal(?:\.\w+)+)"', rendered))
    missing = []
    for module_name in _API_MODULES:
        for name in _public_names(module_name):
            # Re-exports (e.g. probcal.metrics.<name> defined in a submodule) render
            # under their defining module, so any probcal.* path ending in the name counts.
            if not any(anchor.rsplit(".", 1)[-1] == name for anchor in anchors):
                missing.append(f"{module_name}.{name}")
    missing += [s for s in _API_EXTRA_SYMBOLS if s not in anchors]
    assert not missing, f"public symbols absent from the rendered API reference: {missing}"


def _vocabulary_names_read_first(block: str) -> set[str]:
    """Vocabulary names the block reads before it assigns them (if it assigns them at all)."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return set()
    first: dict[str, tuple[tuple[int, int], bool]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _VOCABULARY:
            key = (node.lineno, node.col_offset)
            if node.id not in first or key < first[node.id][0]:
                first[node.id] = (key, isinstance(node.ctx, ast.Load))
    return {name for name, (_, is_load) in first.items() if is_load}


def _names_assigned(block: str) -> set[str]:
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return set()
    names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return names


def _leading_comment(block: str) -> str:
    head = []
    for line in block.strip().splitlines():
        if not line.lstrip().startswith("#"):
            break
        head.append(line)
    return "\n".join(head)


@pytest.mark.parametrize(
    "page",
    [p for p in _prose_pages() if p.parent.name != "_snippets"],
    ids=lambda p: str(p.relative_to(_DOCS)),
)
def test_vocabulary_names_are_declared(page: pathlib.Path) -> None:
    """Rule 2 of docs/README.md: a block that reads a vocabulary name it did not define
    names it in its leading comment lines, and the page carries the vocabulary include."""
    if page.name == "changelog.md":
        pytest.skip("not a snippet page")
    text = page.read_text()
    defined: set[str] = set()
    problems: list[str] = []
    needs_include = False
    for i, block in enumerate(_CODE_BLOCK_RE.findall(text)):
        skipped = "# docs: no-run" in block or "--8<--" in block or block.strip().startswith(">>>")
        needed = _vocabulary_names_read_first(block) - defined
        defined |= _names_assigned(block)
        if skipped or not needed:
            continue
        needs_include = True
        head = _leading_comment(block)
        unnamed = sorted(n for n in needed if not re.search(rf"\b{n}\b", head))
        if unnamed:
            problems.append(f"block {i} reads {unnamed} without naming them in its leading comment")
    if needs_include and page.name != "getting-started.md" and _VOCAB_INCLUDE not in text:
        problems.append(f"page uses the vocabulary but lacks {_VOCAB_INCLUDE}")
    assert not problems, f"{page.relative_to(_DOCS)}: " + "; ".join(problems)
