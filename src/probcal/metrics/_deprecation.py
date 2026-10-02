"""Deprecation helpers for renamed metric keywords and aliases (removal in 0.5.0)."""

import warnings

REMOVAL = "0.5.0"


class _Unset:
    """Sentinel type: a deprecated keyword that was not passed."""

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<unset>"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


def deprecated(message: str, *, stacklevel: int = 3) -> None:
    """Emit a ``DeprecationWarning`` that names the removal release."""
    warnings.warn(
        f"{message} (deprecated; removed in {REMOVAL})", DeprecationWarning, stacklevel=stacklevel
    )


def renamed_kwarg(
    func: str,
    old: str,
    new: str,
    old_value: object,
    new_value: object,
    new_default: object,
    *,
    stacklevel: int = 4,
) -> object:
    """Resolve a keyword renamed from ``old`` to ``new``.

    Returns ``new_value`` when ``old`` was not passed; otherwise warns and
    returns ``old_value``. Passing both (with ``new`` differing from its
    default) is a ``TypeError``: there is no right answer to pick silently.
    """
    if old_value is UNSET:
        return new_value
    if new_value is not new_default and new_value != new_default:
        raise TypeError(f"{func}() got both {old}= and {new}=; pass only {new}=")
    deprecated(f"{func}({old}=...) is renamed to {func}({new}=...)", stacklevel=stacklevel)
    return old_value
