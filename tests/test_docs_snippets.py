"""Executable-snippet contract for the docs.

Every ```python fenced block in docs/**/*.md (except the changelog) is
executed, in order, in one namespace per page — pre-seeded with the fixed
vocabulary the snippet convention documents in docs/README.md. This turns
the fragment style from a rot risk into a tested contract: a block either
runs against that vocabulary, reuses a name an earlier block on the same
page defined, or is marked "# docs: no-run" as deliberate pseudo-code.

Printed results shown in a ``# -> <expected>`` comment are checked too: the
expected text must appear in what the block printed, with ``...`` matching
anything and each number matching the printed one to the precision shown
(``0.1144`` accepts 0.11443..., not 0.1150).
"""

import contextlib
import io
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from probcal import BetaCalibrator, Masterscale
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor

pytestmark = pytest.mark.slow

_DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
_EXCLUDED_PAGES = {"changelog.md"}
_NO_RUN_MARKER = "# docs: no-run"
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)
_REQUIRES_RE = re.compile(r"<!--\s*docs:\s*requires\s+([^>]*?)\s*-->")
_EXPECT_RE = re.compile(r"#\s*->\s*(.*?)\s*$", re.M)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_EXPECT_TOKEN_RE = re.compile(
    rf"(?P<ellipsis>\.\.\.)|(?P<number>{_NUMBER})|(?P<space>\s+)|(?P<other>.)"
)


class _StubModel:
    """Sklearn-free stand-in for a scoring model (mirrors tests/test_golden.py)."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def _make_vocabulary() -> dict:
    """The fixed vocabulary documented in docs/README.md."""
    cal_portfolio = make_pd_portfolio(n=3000, random_state=0)
    new_portfolio = make_pd_portfolio(n=1000, random_state=1)
    s_cal = cal_portfolio.scores
    y_cal = cal_portfolio.y
    w_cal = np.ones_like(y_cal)
    segments = np.array(["seg-a", "seg-b", "seg-c"])[np.arange(len(s_cal)) % 3]

    # The monitor watches a *calibrated* forecast, never the raw score —
    # feeding it uncalibrated s_cal would trip the alarm on batch one.
    p_cal = BetaCalibrator().fit(s_cal, y_cal).predict_proba(s_cal)
    ms = Masterscale.from_edges([0.01, 0.05], names=["G1", "G2", "G3"])
    grades = ms.assign(p_cal)
    mon = CalibrationMonitor(alpha=0.05, grades=ms.names)
    for k in range(3):
        idx = slice(k * 300, (k + 1) * 300)
        mon.update(y_cal[idx], p_cal[idx], grade=grades[idx], label=f"batch-{k}")

    return {
        "s_cal": s_cal,
        "y_cal": y_cal,
        "w_cal": w_cal,
        "model": _StubModel(),
        "s_new": new_portfolio.scores,
        "mon": mon,
        "ms": ms,
        "grades": grades,
        "segments": segments,
    }


def _discover_pages() -> list[pathlib.Path]:
    return sorted(p for p in _DOCS_DIR.rglob("*.md") if p.name not in _EXCLUDED_PAGES)


def _required_packages(text: str) -> list[str]:
    """Optional extras a page declares with `<!-- docs: requires ... -->`."""
    names: list[str] = []
    for match in _REQUIRES_RE.findall(text):
        names.extend(name.strip() for name in match.split(",") if name.strip())
    return names


def _extract_blocks(page: pathlib.Path) -> list[str]:
    text = page.read_text()
    blocks = []
    for raw in _CODE_BLOCK_RE.findall(text):
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        if "--8<--" in raw:
            continue
        if lines[0].lstrip().startswith(">>>"):
            continue
        if _NO_RUN_MARKER in raw:
            continue
        blocks.append(raw)
    return blocks


def _expected_outputs(block: str) -> list[str]:
    """The ``# -> <expected>`` printed-output claims of a block, in order."""
    return [m for m in _EXPECT_RE.findall(block) if m]


def _expectation_pattern(expected: str) -> tuple[re.Pattern[str], list[tuple[float, float]]]:
    """Regex for ``expected`` plus ``(value, tolerance)`` per number it shows.

    ``...`` matches anything, runs of whitespace match any whitespace, and
    every number becomes a capture group whose printed value must lie within
    half a unit of the last digit shown. Padding whitespace is allowed between
    any two tokens, so numpy's aligned array printing matches too.
    """
    parts: list[str] = []
    numbers: list[tuple[float, float]] = []
    for tok in _EXPECT_TOKEN_RE.finditer(expected):
        if tok.group("space"):
            parts.append(r"\s+")
            continue
        if parts and parts[-1] != r"\s+":
            parts.append(r"\s*")  # printed padding (e.g. numpy's) between any two tokens
        if tok.group("ellipsis"):
            parts.append(".*?")
        elif tok.group("number"):
            text = tok.group("number")
            mantissa, _, exponent = text.lower().partition("e")
            decimals = len(mantissa.partition(".")[2])
            tol = 0.5 * 10.0 ** (int(exponent or 0) - decimals)
            numbers.append((float(text), tol))
            parts.append(f"({_NUMBER})")
        else:
            parts.append(re.escape(tok.group("other")))
    return re.compile("".join(parts), re.S), numbers


def _output_matches(expected: str, printed: str) -> bool:
    """Whether ``printed`` shows ``expected`` (see :func:`_expectation_pattern`)."""
    pattern, numbers = _expectation_pattern(expected)
    for m in pattern.finditer(printed):
        got = [float(g) for g in m.groups()]
        if all(
            abs(v - want) <= tol + 1e-9 * abs(want)
            for v, (want, tol) in zip(got, numbers, strict=True)
        ):
            return True
    return False


def test_output_matcher_respects_shown_precision() -> None:
    printed = "OffsetEstimate(delta=-0.8765 +/- 0.11443, n=3000, events=82.0)\n"
    assert _output_matches("delta=-0.8765 +/- 0.1144, n=3000, events=82.0", printed)
    assert _output_matches("delta=-0.88 +/- ..., n=3000", printed)
    assert not _output_matches("delta=-0.8765 +/- 0.1150", printed)
    assert not _output_matches("delta=+0.1234", printed)
    assert _output_matches("{'A': (638.68, inf)", "{'A': (638.677940257227, inf), 'B': ...}")
    assert _output_matches("[0.00287 0.00328 0.00765]", "[0.00287409 0.003284   0.0076459 ]")
    assert not _output_matches("[0.00287 0.00329]", "[0.00287409 0.003284 ]")


_PAGES = _discover_pages()
_PAGE_IDS = [str(p.relative_to(_DOCS_DIR)) for p in _PAGES]


@pytest.mark.parametrize("page", _PAGES, ids=_PAGE_IDS)
def test_docs_page_snippets_execute(page: pathlib.Path, tmp_path, monkeypatch) -> None:
    blocks = _extract_blocks(page)
    if not blocks:
        pytest.skip("no runnable python blocks on this page")
    for package in _required_packages(page.read_text()):
        pytest.importorskip(package)

    # Fresh cwd per page, so to_json(path=...) writes land in an isolated dir.
    monkeypatch.chdir(tmp_path)
    namespace = _make_vocabulary()
    try:
        for i, block in enumerate(blocks):
            first_line = block.strip().splitlines()[0]
            printed = io.StringIO()
            try:
                with contextlib.redirect_stdout(printed):
                    exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
            except Exception as exc:
                raise AssertionError(
                    f"{page}: block {i} failed (first line: {first_line!r}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            for expected in _expected_outputs(block):
                assert _output_matches(expected, printed.getvalue()), (
                    f"{page}: block {i} (first line: {first_line!r}) claims output "
                    f"{expected!r} but printed:\n{printed.getvalue()}"
                )
    finally:
        plt.close("all")
