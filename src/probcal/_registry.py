"""Registry of serializable classes for from_dict dispatch."""

from typing import TypeVar

from ._serialize import check_schema

T = TypeVar("T", bound=type)

SERIALIZABLE: dict[str, type] = {}
"""Class-name -> class map, filled by :func:`register` at import time."""


def register(cls: T) -> T:
    """Class decorator: make ``cls`` loadable by name through :func:`load`."""
    SERIALIZABLE[cls.__name__] = cls
    return cls


def load(d: dict) -> object:
    """Instantiate whatever registered class wrote ``d`` (schema-checked).

    Raises
    ------
    ValueError
        If the schema is unknown, or ``d["class"]`` is not registered.
    """
    check_schema(d)
    name = d.get("class")
    cls = SERIALIZABLE.get(str(name))
    if cls is None:
        raise ValueError(f"unknown serialized class {name!r}; registered: {sorted(SERIALIZABLE)}")
    return cls.from_dict(d)  # type: ignore[attr-defined]
