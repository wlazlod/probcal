"""Structural acceptance for the docs reorganization.

Five properties the reader-oriented structure has to keep:

1. no URL disappeared — the v0.3.0 sitemap's page set is a subset of the new one;
2. the front page still names every capability it claims (a staleness tripwire);
3. every public plot function has a rendered, alt-texted figure on a prose page;
4. the sample validation report exists, is linked, and is self-contained;
5. the three workflow pages added by the reorganization are in nav and cross-link.

Plus one guard for `guide/choosing.md`, whose factual columns each name the test
that pins them: those node ids have to resolve.
"""

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
