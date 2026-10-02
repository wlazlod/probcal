"""Package-level invariants: version, exports, typing marker, dependency policy."""

import importlib.metadata
import importlib.resources
import json
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

FORBIDDEN_RUNTIME_IMPORTS = ("scipy", "sklearn", "pandas", "matplotlib", "shap")
_ROOT = pathlib.Path(__file__).resolve().parents[1]
# PEP 440 public version: release, optional pre/post/dev segments.
_PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def _is_editable_install() -> bool:
    dist = importlib.metadata.distribution("probcal")
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return False
    return bool(json.loads(direct_url).get("dir_info", {}).get("editable", False))


def test_version_single_source() -> None:
    """``probcal.__version__`` is the only hand-written version: pyproject reads it."""
    import probcal

    assert _PEP440.match(probcal.__version__), probcal.__version__
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    assert "version" not in pyproject["project"]
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/probcal/__init__.py"


def test_version_matches_installed_metadata() -> None:
    import probcal

    installed = importlib.metadata.version("probcal")
    if installed != probcal.__version__ and _is_editable_install():
        # Editable installs freeze the metadata at install time; a later bump of
        # __version__ is only picked up by reinstalling. Not a packaging bug.
        pytest.skip(
            f"stale editable-install metadata ({installed} vs {probcal.__version__}); "
            "reinstall with `uv pip install -e .`"
        )
    assert probcal.__version__ == installed


def test_citation_version_matches_release() -> None:
    """CITATION.cff tracks releases: on a final version it must equal __version__."""
    import probcal

    if re.search(r"(a|b|rc|\.dev)\d+", probcal.__version__):
        pytest.skip(f"pre-release {probcal.__version__}: CITATION.cff still names the last release")
    match = re.search(
        r"^version:\s*\"?([^\"\s]+)\"?\s*$", (_ROOT / "CITATION.cff").read_text(), re.M
    )
    assert match, "CITATION.cff has no version field"
    assert match.group(1) == probcal.__version__


def test_py_typed_shipped() -> None:
    marker = importlib.resources.files("probcal").joinpath("py.typed")
    assert marker.is_file()


def test_all_exports_resolve() -> None:
    import probcal

    for name in probcal.__all__:
        assert getattr(probcal, name, None) is not None, f"__all__ names missing attribute: {name}"


def test_no_forbidden_imports() -> None:
    # Run in a fresh interpreter: the test session itself legitimately imports
    # scipy/sklearn/statsmodels for reference tests, so sys.modules here is tainted.
    code = (
        "import sys, probcal\n"
        f"forbidden = {FORBIDDEN_RUNTIME_IMPORTS!r}\n"
        "hits = [m for m in sys.modules if m.split('.', 1)[0] in forbidden]\n"
        "assert not hits, f'importing probcal pulled in forbidden dependencies: {hits}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
