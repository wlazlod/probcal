"""Package-level invariants: version, exports, typing marker, dependency policy."""

import importlib.resources
import sys

FORBIDDEN_RUNTIME_IMPORTS = ("scipy", "sklearn", "pandas", "matplotlib", "shap")


def test_version_is_frozen() -> None:
    import probcal

    assert probcal.__version__ == "0.0.1"


def test_py_typed_shipped() -> None:
    marker = importlib.resources.files("probcal").joinpath("py.typed")
    assert marker.is_file()


def test_all_exports_resolve() -> None:
    import probcal

    for name in probcal.__all__:
        assert getattr(probcal, name, None) is not None, f"__all__ names missing attribute: {name}"


def test_no_forbidden_imports() -> None:
    import probcal  # noqa: F401

    for module_name in list(sys.modules):
        top = module_name.split(".", 1)[0]
        assert (
            top not in FORBIDDEN_RUNTIME_IMPORTS
        ), f"importing probcal pulled in forbidden dependency: {module_name}"
