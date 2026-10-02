"""Registry of serializable classes for from_dict dispatch."""

from typing import TypeVar

from ._serialize import check_schema

T = TypeVar("T", bound=type)

SERIALIZABLE: dict[str, type] = {}
"""Class-name -> class map, filled by :func:`register` at import time."""


def register(cls: T) -> T:
    """Class decorator: make ``cls`` loadable by name through :func:`load`.

    Raises
    ------
    ValueError
        If a *different* class is already registered under the same name
        (payloads carry only the class name, so a collision is ambiguous).
    """
    existing = SERIALIZABLE.get(cls.__name__)
    if existing is not None and existing.__qualname__ + existing.__module__ != (
        cls.__qualname__ + cls.__module__
    ):
        raise ValueError(
            f"serializable class name {cls.__name__!r} already registered by "
            f"{existing.__module__}.{existing.__qualname__}"
        )
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
