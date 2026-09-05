"""Package-level invariants: version, exports, typing marker, dependency policy."""

import importlib.metadata
import importlib.resources
import subprocess
import sys

FORBIDDEN_RUNTIME_IMPORTS = ("scipy", "sklearn", "pandas", "matplotlib", "shap")


def test_version_is_frozen() -> None:
    import probcal

    assert probcal.__version__ == "0.3.1"
    assert probcal.__version__ == importlib.metadata.version("probcal")


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
