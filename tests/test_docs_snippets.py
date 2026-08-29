"""Executable-snippet contract for the docs (spec W7.3, DOCS_REORG_SPEC).

Every ```python fenced block in docs/**/*.md (except the changelog) is
executed, in order, in one namespace per page — pre-seeded with the fixed
vocabulary the snippet convention documents in docs/README.md. This turns
the fragment style from a rot risk into a tested contract: a block either
runs against that vocabulary, reuses a name an earlier block on the same
page defined, or is marked "# docs: no-run" as deliberate pseudo-code.
"""

import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from probcal import BetaCalibrator
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor

pytestmark = pytest.mark.slow

_DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
_EXCLUDED_PAGES = {"changelog.md"}
_NO_RUN_MARKER = "# docs: no-run"
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)


class _StubModel:
    """Sklearn-free stand-in for a scoring model (mirrors tests/test_golden.py)."""

    def fit(self, X, y):  # noqa: ARG002
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
    grades = np.array(["G1", "G2", "G3"])[np.searchsorted([0.01, 0.05], s_cal)]
    segments = np.array(["seg-a", "seg-b", "seg-c"])[np.arange(len(s_cal)) % 3]

    # The monitor watches a *calibrated* forecast, never the raw score —
    # feeding it uncalibrated s_cal would trip the alarm on batch one.
    p_cal = BetaCalibrator().fit(s_cal, y_cal).predict_proba(s_cal)
    mon = CalibrationMonitor(alpha=0.05)
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
        "grades": grades,
        "segments": segments,
    }


def _discover_pages() -> list[pathlib.Path]:
    return sorted(p for p in _DOCS_DIR.rglob("*.md") if p.name not in _EXCLUDED_PAGES)


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


_PAGES = _discover_pages()
_PAGE_IDS = [str(p.relative_to(_DOCS_DIR)) for p in _PAGES]


@pytest.mark.parametrize("page", _PAGES, ids=_PAGE_IDS)
def test_docs_page_snippets_execute(page: pathlib.Path, tmp_path, monkeypatch) -> None:
    blocks = _extract_blocks(page)
    if not blocks:
        pytest.skip("no runnable python blocks on this page")

    # Fresh cwd per page, so to_json(path=...) writes land in an isolated dir.
    monkeypatch.chdir(tmp_path)
    namespace = _make_vocabulary()
    try:
        for i, block in enumerate(blocks):
            try:
                exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
            except Exception as exc:
                first_line = block.strip().splitlines()[0]
                raise AssertionError(
                    f"{page}: block {i} failed (first line: {first_line!r}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
    finally:
        plt.close("all")
