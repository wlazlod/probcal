"""Tests for probcal.report.validation_report."""

import importlib.util
import re
import subprocess
import sys
import time

import numpy as np
import pytest

from probcal import BetaCalibrator, make_pd_portfolio
from probcal.monitor import CalibrationMonitor
from probcal.report import validation_report

HAS_MPL = importlib.util.find_spec("matplotlib") is not None

_TIMESTAMP_RE = re.compile(r"^.*Generated .* UTC.*$", re.MULTILINE)


def _portfolio(n: int = 600, random_state: int = 7):
    d = make_pd_portfolio(n=n, random_state=random_state)
    cal = BetaCalibrator().fit(d.scores, d.y)
    p = cal.predict_proba(d.scores)
    return d.y, p, cal


def _grades(p: np.ndarray) -> np.ndarray:
    names = np.array(["G1", "G2", "G3"])
    return names[np.searchsorted([0.01, 0.05], p)]


def _monitor(y: np.ndarray, p: np.ndarray) -> CalibrationMonitor:
    mon = CalibrationMonitor()
    for k in range(4):
        sl = slice(k * len(y) // 4, (k + 1) * len(y) // 4)
        mon.update(y[sl], p[sl], label=f"batch-{k}")
    return mon


def _strip_timestamp(text: str) -> str:
    return _TIMESTAMP_RE.sub("", text)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_all_sections_present() -> None:
    y, p, cal = _portfolio()
    grades = _grades(p)
    by = np.where(p < 0.02, "low", "high")
    mon = _monitor(y, p)

    start = time.monotonic()
    html = validation_report(
        y, p, calibrator=cal, monitor=mon, grades=grades, by=by, n_boot=30, seed=42
    )
    elapsed = time.monotonic() - start

    assert "Portfolio summary" in html
    assert "Reliability" in html
    assert "Metric report" in html
    assert "CORP decomposition" in html
    assert "Rating grades" in html
    assert "Grouped evaluation" in html
    assert "Calibration monitoring" in html
    assert "Appendix: calibrator" in html
    assert "data:image/png;base64," in html
    assert "http" not in html
    assert "<script" not in html
    assert elapsed < 10.0, f"all-sections report took {elapsed:.1f}s; mark this test slow"


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_minimal_report_omits_optional_sections() -> None:
    y, p, _ = _portfolio()
    html = validation_report(y, p, n_boot=20, seed=42)

    assert "Portfolio summary" in html
    assert "Reliability" in html
    assert "Metric report" in html
    assert "CORP decomposition" in html
    assert "Rating grades" not in html
    assert "Grouped evaluation" not in html
    assert "Calibration monitoring" not in html
    assert "Appendix: calibrator" not in html
    assert "http" not in html
    assert "<script" not in html


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_same_seed_is_byte_identical_after_stripping_timestamp() -> None:
    y, p, cal = _portfolio()
    mon = _monitor(y, p)
    kwargs = dict(y=y, p=p, calibrator=cal, monitor=mon, n_boot=20, seed=42)

    first = validation_report(**kwargs)
    second = validation_report(**kwargs)

    # Two builds can straddle a second boundary or not — don't assume the
    # timestamp line differs, just that each report has exactly one (the
    # "Generated ... UTC" text is embedded inline, not anchored to column 0).
    assert len(re.findall(r"Generated [^\n]*UTC", first)) == 1
    assert len(re.findall(r"Generated [^\n]*UTC", second)) == 1
    assert _strip_timestamp(first) == _strip_timestamp(second)


def test_missing_matplotlib_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import probcal.report as report_mod

    monkeypatch.setattr(report_mod, "_HAS_MPL", False)
    y, p, _ = _portfolio(n=100)
    with pytest.raises(ImportError, match=r"probcal\[viz\]"):
        validation_report(y, p, n_boot=10)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_markdown_requires_path() -> None:
    y, p, _ = _portfolio(n=100)
    with pytest.raises(ValueError, match="markdown"):
        validation_report(y, p, format="markdown", n_boot=10)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_markdown_writes_figures_and_has_no_data_uri(tmp_path) -> None:
    y, p, cal = _portfolio()
    grades = _grades(p)
    out = tmp_path / "report.md"

    text = validation_report(
        y, p, calibrator=cal, grades=grades, path=out, format="markdown", n_boot=20, seed=1
    )

    assert out.read_text(encoding="utf-8") == text
    assert "data:image" not in text
    fig_dir = tmp_path / "report_figures"
    assert fig_dir.is_dir()
    pngs = sorted(f.name for f in fig_dir.glob("*.png"))
    assert "reliability_corp.png" in pngs
    assert "reliability_kernel.png" in pngs
    assert "grade_backtest.png" in pngs
    assert "report_figures/reliability_corp.png" in text


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_path_writes_html_file(tmp_path) -> None:
    y, p, _ = _portfolio(n=100)
    out = tmp_path / "report.html"
    text = validation_report(y, p, path=out, n_boot=10)
    assert out.read_text(encoding="utf-8") == text
    assert text.startswith("<!doctype html>")


def test_import_report_module_stays_matplotlib_free() -> None:
    code = (
        "import sys, probcal.report\n"
        "assert 'matplotlib' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def _gfm_cells(line: str) -> list:
    """Split one GFM table row on unescaped ``|`` (``\\|`` is a literal pipe)."""
    parts = re.split(r"(?<!\\)\|", line)
    return parts[1:-1]


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_html_escapes_malicious_grade_and_group_labels() -> None:
    y, p, _ = _portfolio(n=400)
    n = len(y)
    evil = "<img src=x onerror=alert(1)>"
    labels = np.where(np.arange(n) % 2 == 0, evil, "A&B")

    html = validation_report(y, p, grades=labels, by=labels, n_boot=15, seed=3)

    assert evil not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "A&amp;B" in html
    assert "<script" not in html


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_html_title_with_dollar_and_tags_is_escaped() -> None:
    y, p, _ = _portfolio(n=100)
    html = validation_report(y, p, title="Cost $5 <b>bold</b>", n_boot=10)
    assert "Cost $5 &lt;b&gt;bold&lt;/b&gt;" in html
    assert "<b>bold</b>" not in html


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_markdown_table_survives_pipe_in_group_label(tmp_path) -> None:
    y, p, _ = _portfolio(n=400)
    n = len(y)
    by = np.where(np.arange(n) % 2 == 0, "a|b", "c|d")
    out = tmp_path / "report.md"

    text = validation_report(y, p, by=by, path=out, format="markdown", n_boot=15, seed=2)

    section = text.split("## Grouped evaluation", 1)[1]
    lines = [ln for ln in section.splitlines() if ln.startswith("| ")]
    assert lines, "grouped evaluation table not found"
    header_cols = len(_gfm_cells(lines[0]))
    assert header_cols == 5
    body_lines = [ln for ln in lines[1:] if set(ln.replace("|", "").strip()) - {"-"}]
    assert body_lines
    for ln in body_lines:
        assert len(_gfm_cells(ln)) == header_cols, ln
    assert "a\\|b" in text
    assert "c\\|d" in text
